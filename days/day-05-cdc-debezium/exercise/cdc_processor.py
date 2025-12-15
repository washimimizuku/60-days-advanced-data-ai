#!/usr/bin/env python3
"""
Day 5: CDC with Debezium - Exercise
CDC Event Processor for Real-time Data Processing
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from kafka import KafkaConsumer, KafkaProducer
import redis
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CDCProcessor:
    """Main CDC event processor"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_connections()
        self.setup_processors()
        
    def setup_connections(self):
        """Initialize connections to Kafka, Redis, and PostgreSQL"""
        # Kafka consumer
        self.consumer = KafkaConsumer(
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            key_deserializer=lambda x: x.decode('utf-8') if x else None,
            group_id=self.config['kafka']['group_id'],
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )
        
        # Kafka producer for downstream events
        self.producer = KafkaProducer(
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None
        )
        
        # Redis for real-time analytics
        self.redis_client = redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            db=self.config['redis']['db'],
            decode_responses=True
        )
        
        # PostgreSQL for read models (optional)
        if 'postgres_read' in self.config:
            self.pg_conn = psycopg2.connect(**self.config['postgres_read'])
            
    def setup_processors(self):
        """Initialize specialized processors for different tables"""
        self.processors = {
            'customers': CustomerProcessor(self.redis_client, self.producer),
            'products': ProductProcessor(self.redis_client, self.producer),
            'orders': OrderProcessor(self.redis_client, self.producer),
            'order_items': OrderItemProcessor(self.redis_client, self.producer),
            'inventory_movements': InventoryProcessor(self.redis_client, self.producer),
            'user_events': EventProcessor(self.redis_client, self.producer)
        }
        
    def subscribe_to_topics(self, topics: list):
        """Subscribe to CDC topics"""
        self.consumer.subscribe(topics)
        logger.info(f"Subscribed to topics: {topics}")
        
    def process_events(self):
        """Main event processing loop"""
        logger.info("Starting CDC event processing...")
        
        try:
            for message in self.consumer:
                self.process_single_event(message)
        except KeyboardInterrupt:
            logger.info("Shutting down CDC processor...")
        except Exception as e:
            logger.error(f"Error in event processing: {e}")
        finally:
            self.cleanup()
            
    def process_single_event(self, message):
        """Process a single CDC event"""
        try:
            event = message.value
            topic = message.topic
            
            # Extract table name from topic
            # Topic format: ecommerce-db.public.table_name
            table_name = topic.split('.')[-1]
            
            # Get appropriate processor
            processor = self.processors.get(table_name)
            if processor:
                processor.process_event(event)
            else:
                logger.warning(f"No processor found for table: {table_name}")
                
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            logger.error(f"Event: {message.value}")
            
    def cleanup(self):
        """Clean up connections"""
        self.consumer.close()
        self.producer.close()
        if hasattr(self, 'pg_conn'):
            self.pg_conn.close()

class BaseProcessor:
    """Base class for table-specific processors"""
    
    def __init__(self, redis_client, producer):
        self.redis = redis_client
        self.producer = producer
        
    def process_event(self, event: Dict[str, Any]):
        """Process CDC event - to be implemented by subclasses"""
        operation = event.get('op')
        
        if operation == 'c':  # CREATE/INSERT
            self.handle_insert(event.get('after', {}))
        elif operation == 'u':  # UPDATE
            self.handle_update(event.get('before', {}), event.get('after', {}))
        elif operation == 'd':  # DELETE
            self.handle_delete(event.get('before', {}))
        elif operation == 'r':  # READ (snapshot)
            self.handle_snapshot(event.get('after', {}))
            
    def handle_insert(self, record: Dict[str, Any]):
        """Handle INSERT operations"""
        pass
        
    def handle_update(self, old_record: Dict[str, Any], new_record: Dict[str, Any]):
        """Handle UPDATE operations"""
        pass
        
    def handle_delete(self, record: Dict[str, Any]):
        """Handle DELETE operations"""
        pass
        
    def handle_snapshot(self, record: Dict[str, Any]):
        """Handle snapshot/initial load"""
        self.handle_insert(record)

class CustomerProcessor(BaseProcessor):
    """Processor for customer table changes"""
    
    def handle_insert(self, record: Dict[str, Any]):
        customer_id = record.get('customer_id')
        logger.info(f"New customer created: {customer_id}")
        
        # Update customer count by segment
        segment = record.get('customer_segment', 'Unknown')
        self.redis.hincrby('customer_counts', segment, 1)
        
        # Store customer data for quick lookup
        self.redis.hset(f'customer:{customer_id}', mapping={
            'name': f"{record.get('first_name', '')} {record.get('last_name', '')}",
            'email': record.get('email', ''),
            'segment': segment,
            'registration_date': record.get('registration_date', ''),
            'is_active': record.get('is_active', True)
        })
        
        # Publish customer event
        self.producer.send('customer-events', 
                          key=str(customer_id),
                          value={
                              'event_type': 'customer_created',
                              'customer_id': customer_id,
                              'customer_data': record,
                              'timestamp': datetime.now().isoformat()
                          })
        
    def handle_update(self, old_record: Dict[str, Any], new_record: Dict[str, Any]):
        customer_id = new_record.get('customer_id')
        logger.info(f"Customer updated: {customer_id}")
        
        # Check if segment changed
        old_segment = old_record.get('customer_segment')
        new_segment = new_record.get('customer_segment')
        
        if old_segment != new_segment:
            self.redis.hincrby('customer_counts', old_segment, -1)
            self.redis.hincrby('customer_counts', new_segment, 1)
            
        # Update customer data
        self.redis.hset(f'customer:{customer_id}', mapping={
            'name': f"{new_record.get('first_name', '')} {new_record.get('last_name', '')}",
            'email': new_record.get('email', ''),
            'segment': new_segment,
            'is_active': new_record.get('is_active', True)
        })
        
        # Publish update event
        self.producer.send('customer-events',
                          key=str(customer_id),
                          value={
                              'event_type': 'customer_updated',
                              'customer_id': customer_id,
                              'changes': self.get_changes(old_record, new_record),
                              'timestamp': datetime.now().isoformat()
                          })
        
    def handle_delete(self, record: Dict[str, Any]):
        customer_id = record.get('customer_id')
        logger.info(f"Customer deleted: {customer_id}")
        
        # Update counts
        segment = record.get('customer_segment', 'Unknown')
        self.redis.hincrby('customer_counts', segment, -1)
        
        # Remove customer data
        self.redis.delete(f'customer:{customer_id}')
        
    def get_changes(self, old_record: Dict[str, Any], new_record: Dict[str, Any]) -> Dict[str, Any]:
        """Identify what changed between old and new records"""
        changes = {}
        for key in new_record:
            if old_record.get(key) != new_record.get(key):
                changes[key] = {
                    'old': old_record.get(key),
                    'new': new_record.get(key)
                }
        return changes

class ProductProcessor(BaseProcessor):
    """Processor for product table changes"""
    
    def handle_insert(self, record: Dict[str, Any]):
        product_id = record.get('product_id')
        logger.info(f"New product created: {product_id}")
        
        # Update product count by category
        category = record.get('category', 'Unknown')
        self.redis.hincrby('product_counts', category, 1)
        
        # Store product data
        self.redis.hset(f'product:{product_id}', mapping={
            'name': record.get('product_name', ''),
            'category': category,
            'brand': record.get('brand', ''),
            'price': record.get('unit_price', 0),
            'inventory': record.get('inventory_quantity', 0),
            'is_active': record.get('is_active', True)
        })
        
    def handle_update(self, old_record: Dict[str, Any], new_record: Dict[str, Any]):
        product_id = new_record.get('product_id')
        
        # Check for inventory changes
        old_inventory = old_record.get('inventory_quantity', 0)
        new_inventory = new_record.get('inventory_quantity', 0)
        
        if old_inventory != new_inventory:
            logger.info(f"Product {product_id} inventory changed: {old_inventory} -> {new_inventory}")
            
            # Update inventory tracking
            self.redis.hset(f'product:{product_id}', 'inventory', new_inventory)
            
            # Check for low stock alert
            if new_inventory < 10:  # Low stock threshold
                self.producer.send('inventory-alerts',
                                  key=str(product_id),
                                  value={
                                      'alert_type': 'low_stock',
                                      'product_id': product_id,
                                      'current_inventory': new_inventory,
                                      'timestamp': datetime.now().isoformat()
                                  })

class OrderProcessor(BaseProcessor):
    """Processor for order table changes"""
    
    def handle_insert(self, record: Dict[str, Any]):
        order_id = record.get('order_id')
        customer_id = record.get('customer_id')
        total_amount = float(record.get('total_amount', 0))
        
        logger.info(f"New order created: {order_id} for customer {customer_id}")
        
        # Update daily revenue
        order_date = datetime.now().date()
        daily_key = f"revenue:daily:{order_date}"
        self.redis.incrbyfloat(daily_key, total_amount)
        
        # Update order count
        count_key = f"orders:daily:{order_date}"
        self.redis.incr(count_key)
        
        # Update customer metrics
        customer_key = f"customer:{customer_id}:metrics"
        self.redis.hincrby(customer_key, 'order_count', 1)
        self.redis.hincrbyfloat(customer_key, 'total_spent', total_amount)
        
        # Store order data
        self.redis.hset(f'order:{order_id}', mapping={
            'customer_id': customer_id,
            'total_amount': total_amount,
            'status': record.get('order_status', 'pending'),
            'created_at': record.get('created_at', '')
        })
        
        # Publish to real-time dashboard
        self.redis.publish('dashboard_updates', json.dumps({
            'type': 'new_order',
            'order_id': order_id,
            'customer_id': customer_id,
            'amount': total_amount,
            'timestamp': datetime.now().isoformat()
        }))
        
    def handle_update(self, old_record: Dict[str, Any], new_record: Dict[str, Any]):
        order_id = new_record.get('order_id')
        old_status = old_record.get('order_status')
        new_status = new_record.get('order_status')
        
        if old_status != new_status:
            logger.info(f"Order {order_id} status changed: {old_status} -> {new_status}")
            
            # Update order status
            self.redis.hset(f'order:{order_id}', 'status', new_status)
            
            # Track status changes
            status_key = f"order_status:{new_status}:daily:{datetime.now().date()}"
            self.redis.incr(status_key)
            
            # Publish status change event
            self.producer.send('order-status-changes',
                              key=str(order_id),
                              value={
                                  'order_id': order_id,
                                  'old_status': old_status,
                                  'new_status': new_status,
                                  'timestamp': datetime.now().isoformat()
                              })

class OrderItemProcessor(BaseProcessor):
    """Processor for order_items table changes"""
    
    def handle_insert(self, record: Dict[str, Any]):
        product_id = record.get('product_id')
        quantity = record.get('quantity', 0)
        
        # Update product sales metrics
        product_key = f"product:{product_id}:metrics"
        self.redis.hincrby(product_key, 'units_sold', quantity)
        self.redis.hincrbyfloat(product_key, 'revenue', float(record.get('total_price', 0)))

class InventoryProcessor(BaseProcessor):
    """Processor for inventory_movements table changes"""
    
    def handle_insert(self, record: Dict[str, Any]):
        product_id = record.get('product_id')
        movement_type = record.get('movement_type')
        quantity = record.get('quantity', 0)
        
        logger.info(f"Inventory movement: Product {product_id}, Type {movement_type}, Qty {quantity}")
        
        # Track inventory movements
        movement_key = f"inventory_movements:{movement_type}:daily:{datetime.now().date()}"
        self.redis.incr(movement_key)

class EventProcessor(BaseProcessor):
    """Processor for user_events table changes"""
    
    def handle_insert(self, record: Dict[str, Any]):
        user_id = record.get('user_id')
        event_type = record.get('event_type')
        
        # Track event counts
        event_key = f"events:{event_type}:daily:{datetime.now().date()}"
        self.redis.incr(event_key)
        
        # Track user activity
        if user_id:
            user_key = f"user:{user_id}:activity"
            self.redis.lpush(user_key, json.dumps({
                'event_type': event_type,
                'timestamp': record.get('created_at', ''),
                'data': record.get('event_data', {})
            }))
            self.redis.ltrim(user_key, 0, 99)  # Keep last 100 events

def main():
    """Main function to run the CDC processor"""
    
    # Configuration
    config = {
        'kafka': {
            'bootstrap_servers': ['localhost:9092'],
            'group_id': 'cdc-processor'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }
    }
    
    # CDC topics to subscribe to
    topics = [
        'ecommerce-db.public.customers',
        'ecommerce-db.public.products',
        'ecommerce-db.public.orders',
        'ecommerce-db.public.order_items',
        'ecommerce-db.public.inventory_movements',
        'ecommerce-db.public.user_events'
    ]
    
    # Create and run processor
    processor = CDCProcessor(config)
    processor.subscribe_to_topics(topics)
    processor.process_events()

if __name__ == "__main__":
    main()