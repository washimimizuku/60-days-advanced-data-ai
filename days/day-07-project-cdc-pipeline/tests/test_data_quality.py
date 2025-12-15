#!/usr/bin/env python3
"""
Integration tests for CDC Pipeline Data Quality
Tests the end-to-end data quality validation and processing
"""

import pytest
import json
import time
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError


class TestDataQuality:
    """Test suite for data quality validation in the CDC pipeline"""
    
    @pytest.fixture
    def kafka_producer(self):
        """Create Kafka producer for testing"""
        return KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
    
    @pytest.fixture
    def kafka_consumer(self):
        """Create Kafka consumer for testing"""
        return KafkaConsumer(
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='latest',
            consumer_timeout_ms=5000
        )
    
    def test_valid_order_processing(self, kafka_producer, kafka_consumer):
        """Test that valid orders are processed correctly"""
        # Subscribe to revenue analytics topic
        kafka_consumer.subscribe(['revenue-analytics'])
        
        # Send valid order
        valid_order = {
            "order_id": 12345,
            "user_id": 1,
            "total_amount": 99.99,
            "status": "completed",
            "created_at": "2023-01-01T12:00:00Z"
        }
        
        kafka_producer.send('orders', key='order-12345', value=valid_order)
        kafka_producer.flush()
        
        # Wait for processing
        time.sleep(2)
        
        # Check if analytics were generated
        messages = []
        for message in kafka_consumer:
            messages.append(message.value)
            break
        
        assert len(messages) > 0
        analytics = messages[0]
        assert 'total_revenue' in analytics
        assert analytics['total_revenue'] == 99.99
    
    def test_invalid_order_to_dlq(self, kafka_producer, kafka_consumer):
        """Test that invalid orders are sent to DLQ"""
        # Subscribe to DLQ topic
        kafka_consumer.subscribe(['dlq-events'])
        
        # Send invalid order (missing required fields)
        invalid_order = {
            "order_id": 12346,
            # Missing user_id, total_amount, status
            "created_at": "2023-01-01T12:00:00Z"
        }
        
        kafka_producer.send('orders', key='order-12346', value=invalid_order)
        kafka_producer.flush()
        
        # Wait for processing
        time.sleep(2)
        
        # Check if event was sent to DLQ
        messages = []
        for message in kafka_consumer:
            messages.append(message.value)
            break
        
        assert len(messages) > 0
        dlq_event = messages[0]
        assert dlq_event['source_topic'] == 'orders'
        assert 'errors' in dlq_event
        assert len(dlq_event['errors']) > 0
    
    def test_inventory_alert_generation(self, kafka_producer, kafka_consumer):
        """Test that low inventory generates alerts"""
        # Subscribe to inventory alerts topic
        kafka_consumer.subscribe(['inventory-alerts'])
        
        # Send product with low stock
        low_stock_product = {
            "product_id": 123,
            "name": "Test Product",
            "category": "Electronics",
            "price": 99.99,
            "stock_quantity": 2,  # Below threshold
            "min_threshold": 10,
            "updated_at": "2023-01-01T12:00:00Z"
        }
        
        kafka_producer.send('products', key='product-123', value=low_stock_product)
        kafka_producer.flush()
        
        # Wait for processing
        time.sleep(2)
        
        # Check if alert was generated
        messages = []
        for message in kafka_consumer:
            messages.append(message.value)
            break
        
        assert len(messages) > 0
        alert = messages[0]
        assert alert['product_id'] == 123
        assert alert['alert_level'] in ['low', 'critical', 'out_of_stock']
        assert alert['current_stock'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])