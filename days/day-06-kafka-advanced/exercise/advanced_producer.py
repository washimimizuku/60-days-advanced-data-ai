#!/usr/bin/env python3
"""
Day 6: Advanced Kafka - Advanced Producer Patterns
Demonstrates idempotent, transactional, and high-performance producers
"""

import json
import time
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaTimeoutError
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedKafkaProducer:
    """Advanced Kafka producer with various patterns"""
    
    def __init__(self, bootstrap_servers=['localhost:9092']):
        self.bootstrap_servers = bootstrap_servers
        self.metrics = {
            'messages_sent': 0,
            'messages_failed': 0,
            'bytes_sent': 0,
            'start_time': None
        }
        
    def create_basic_producer(self) -> KafkaProducer:
        """Create basic high-performance producer"""
        return KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            # Performance optimizations
            acks='1',  # Wait for leader acknowledgment
            retries=3,
            batch_size=16384,  # 16KB batches
            linger_ms=10,      # Wait 10ms for batching
            buffer_memory=33554432,  # 32MB buffer
            compression_type='snappy',
            max_in_flight_requests_per_connection=5
        )
    
    def create_idempotent_producer(self) -> KafkaProducer:
        """Create idempotent producer to prevent duplicates"""
        return KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            # Idempotence configuration
            enable_idempotence=True,
            acks='all',  # Required for idempotence
            retries=2147483647,  # Max retries
            max_in_flight_requests_per_connection=5,
            # Performance settings
            batch_size=16384,
            linger_ms=10,
            compression_type='lz4'
        )
    
    def create_transactional_producer(self, transaction_id: str) -> KafkaProducer:
        """Create transactional producer for exactly-once semantics"""
        return KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            # Transaction configuration
            transactional_id=transaction_id,
            enable_idempotence=True,
            acks='all',
            retries=2147483647,
            max_in_flight_requests_per_connection=5,
            # Performance settings
            batch_size=16384,
            linger_ms=5,
            compression_type='lz4'
        )
    
    def send_with_callback(self, producer: KafkaProducer, topic: str, 
                          key: str, value: Dict[str, Any]):
        """Send message with success/error callbacks"""
        
        def on_success(record_metadata):
            self.metrics['messages_sent'] += 1
            logger.debug(f"Message sent to {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
        
        def on_error(exception):
            self.metrics['messages_failed'] += 1
            logger.error(f"Failed to send message: {exception}")
        
        try:
            future = producer.send(topic, key=key, value=value)
            future.add_callback(on_success)
            future.add_errback(on_error)
            return future
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.metrics['messages_failed'] += 1
    
    def demonstrate_basic_producer(self):
        """Demonstrate basic producer patterns"""
        logger.info("🚀 Demonstrating Basic Producer Patterns")
        
        producer = self.create_basic_producer()
        topic = 'user-events'
        
        try:
            # Send messages with different partitioning strategies
            messages = [
                # Key-based partitioning (same key goes to same partition)
                {'key': 'user_123', 'event': 'login', 'timestamp': datetime.now().isoformat()},
                {'key': 'user_123', 'event': 'page_view', 'timestamp': datetime.now().isoformat()},
                {'key': 'user_456', 'event': 'login', 'timestamp': datetime.now().isoformat()},
                
                # Round-robin partitioning (no key)
                {'event': 'system_event', 'type': 'health_check', 'timestamp': datetime.now().isoformat()},
                {'event': 'system_event', 'type': 'metrics', 'timestamp': datetime.now().isoformat()},
            ]
            
            futures = []
            for msg in messages:
                key = msg.pop('key', None)
                future = self.send_with_callback(producer, topic, key, msg)
                futures.append(future)
            
            # Wait for all messages to be sent
            for future in futures:
                try:
                    record_metadata = future.get(timeout=10)
                    logger.info(f"Message sent to partition {record_metadata.partition}")
                except KafkaTimeoutError:
                    logger.error("Message send timed out")
                except KafkaError as e:
                    logger.error(f"Message send failed: {e}")
            
        finally:
            producer.flush()
            producer.close()
    
    def demonstrate_idempotent_producer(self):
        """Demonstrate idempotent producer"""
        logger.info("🔒 Demonstrating Idempotent Producer")
        
        producer = self.create_idempotent_producer()
        topic = 'user-events'
        
        try:
            # Send the same message multiple times
            # Idempotent producer ensures no duplicates
            message = {
                'user_id': 'user_789',
                'event': 'purchase',
                'order_id': 'order_12345',
                'amount': 99.99,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Sending the same message 3 times (should not create duplicates)")
            
            for i in range(3):
                future = producer.send(topic, key='user_789', value=message)
                record_metadata = future.get(timeout=10)
                logger.info(f"Attempt {i+1}: Sent to partition {record_metadata.partition}, offset {record_metadata.offset}")
                time.sleep(0.1)  # Small delay
            
        finally:
            producer.flush()
            producer.close()
    
    def demonstrate_transactional_producer(self):
        """Demonstrate transactional producer"""
        logger.info("💳 Demonstrating Transactional Producer")
        
        transaction_id = f"tx-producer-{uuid.uuid4()}"
        producer = self.create_transactional_producer(transaction_id)
        
        try:
            # Initialize transactions
            producer.init_transactions()
            
            # Successful transaction
            logger.info("Starting successful transaction...")
            producer.begin_transaction()
            
            try:
                # Send multiple related messages in a transaction
                order_data = {
                    'order_id': 'order_67890',
                    'user_id': 'user_999',
                    'items': [
                        {'product_id': 'prod_1', 'quantity': 2, 'price': 29.99},
                        {'product_id': 'prod_2', 'quantity': 1, 'price': 49.99}
                    ],
                    'total': 109.97,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Send order event
                producer.send('user-events', key='user_999', value={
                    'event': 'order_created',
                    'order_id': order_data['order_id'],
                    'user_id': order_data['user_id'],
                    'total': order_data['total']
                })
                
                # Send inventory updates
                for item in order_data['items']:
                    producer.send('user-events', key=item['product_id'], value={
                        'event': 'inventory_reserved',
                        'product_id': item['product_id'],
                        'quantity': item['quantity'],
                        'order_id': order_data['order_id']
                    })
                
                # Send analytics event
                producer.send('analytics-results', key='daily_sales', value={
                    'event': 'sale_recorded',
                    'amount': order_data['total'],
                    'timestamp': order_data['timestamp']
                })
                
                # Commit transaction
                producer.commit_transaction()
                logger.info("✅ Transaction committed successfully")
                
            except Exception as e:
                logger.error(f"Error in transaction: {e}")
                producer.abort_transaction()
                logger.info("❌ Transaction aborted")
            
            # Demonstrate transaction abort
            logger.info("Starting transaction that will be aborted...")
            producer.begin_transaction()
            
            try:
                # Send some messages
                producer.send('user-events', key='user_error', value={
                    'event': 'test_event',
                    'should_be_aborted': True
                })
                
                # Simulate error condition
                raise Exception("Simulated error to demonstrate abort")
                
            except Exception as e:
                logger.error(f"Error occurred: {e}")
                producer.abort_transaction()
                logger.info("❌ Transaction aborted as expected")
                
        finally:
            producer.close()
    
    def demonstrate_custom_partitioner(self):
        """Demonstrate custom partitioning logic"""
        logger.info("🎯 Demonstrating Custom Partitioning")
        
        def custom_partitioner(key_bytes, all_partitions, available_partitions):
            """Custom partitioner that routes premium users to specific partitions"""
            if key_bytes:
                key = key_bytes.decode('utf-8')
                if key.startswith('premium_'):
                    # Premium users go to first partition
                    return all_partitions[0]
                elif key.startswith('vip_'):
                    # VIP users go to second partition
                    return all_partitions[1] if len(all_partitions) > 1 else all_partitions[0]
                else:
                    # Regular users distributed across remaining partitions
                    remaining_partitions = all_partitions[2:] if len(all_partitions) > 2 else all_partitions
                    return remaining_partitions[hash(key) % len(remaining_partitions)]
            
            # No key - round robin
            return available_partitions[0]
        
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            partitioner=custom_partitioner,
            acks='1'
        )
        
        try:
            # Send messages with different user types
            users = [
                ('premium_user_1', {'event': 'login', 'tier': 'premium'}),
                ('vip_user_1', {'event': 'login', 'tier': 'vip'}),
                ('regular_user_1', {'event': 'login', 'tier': 'regular'}),
                ('premium_user_2', {'event': 'purchase', 'tier': 'premium'}),
                ('regular_user_2', {'event': 'purchase', 'tier': 'regular'}),
            ]
            
            for user_key, event_data in users:
                future = producer.send('user-events', key=user_key, value=event_data)
                record_metadata = future.get(timeout=10)
                logger.info(f"User {user_key} -> Partition {record_metadata.partition}")
                
        finally:
            producer.flush()
            producer.close()
    
    def demonstrate_batch_sending(self):
        """Demonstrate optimized batch sending"""
        logger.info("📦 Demonstrating Batch Sending Optimization")
        
        # High-throughput producer configuration
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            # Batch optimization
            batch_size=65536,  # 64KB batches
            linger_ms=20,      # Wait 20ms for batching
            compression_type='lz4',
            buffer_memory=67108864,  # 64MB buffer
            # Throughput optimization
            acks='1',
            max_in_flight_requests_per_connection=5,
            request_timeout_ms=30000,
            retry_backoff_ms=100
        )
        
        try:
            start_time = time.time()
            num_messages = 1000
            
            # Generate and send messages
            futures = []
            for i in range(num_messages):
                message = {
                    'user_id': f'user_{i % 100}',  # 100 different users
                    'event_id': str(uuid.uuid4()),
                    'event_type': random.choice(['page_view', 'click', 'scroll', 'hover']),
                    'timestamp': datetime.now().isoformat(),
                    'session_id': f'session_{i // 10}',  # 10 events per session
                    'data': {
                        'page': f'/page_{random.randint(1, 50)}',
                        'duration': random.randint(1, 300),
                        'user_agent': 'Mozilla/5.0 (compatible; test-client)'
                    }
                }
                
                key = f"user_{i % 100}"
                future = producer.send('user-events', key=key, value=message)
                futures.append(future)
                
                if i % 100 == 0:
                    logger.info(f"Queued {i} messages...")
            
            # Wait for all messages to be sent
            logger.info("Waiting for all messages to be sent...")
            for future in futures:
                future.get(timeout=30)
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = num_messages / duration
            
            logger.info(f"✅ Sent {num_messages} messages in {duration:.2f} seconds")
            logger.info(f"📊 Throughput: {throughput:.2f} messages/second")
            
        finally:
            producer.flush()
            producer.close()
    
    def demonstrate_error_handling(self):
        """Demonstrate comprehensive error handling"""
        logger.info("⚠️  Demonstrating Error Handling")
        
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            acks='all',
            retries=3,
            retry_backoff_ms=1000,
            request_timeout_ms=5000,  # Short timeout for demo
            max_block_ms=5000
        )
        
        error_count = 0
        success_count = 0
        
        def on_success(record_metadata):
            nonlocal success_count
            success_count += 1
            logger.info(f"✅ Message sent successfully to {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
        
        def on_error(exception):
            nonlocal error_count
            error_count += 1
            logger.error(f"❌ Failed to send message: {exception}")
        
        try:
            # Send to valid topic
            future1 = producer.send('user-events', key='test', value={'test': 'valid_message'})
            future1.add_callback(on_success)
            future1.add_errback(on_error)
            
            # Send to non-existent topic (will create it or fail based on config)
            future2 = producer.send('non-existent-topic', key='test', value={'test': 'message_to_new_topic'})
            future2.add_callback(on_success)
            future2.add_errback(on_error)
            
            # Send message that's too large (if configured)
            large_message = {'data': 'x' * 1000000}  # 1MB message
            future3 = producer.send('user-events', key='large', value=large_message)
            future3.add_callback(on_success)
            future3.add_errback(on_error)
            
            # Wait for all futures
            for future in [future1, future2, future3]:
                try:
                    future.get(timeout=10)
                except Exception as e:
                    logger.error(f"Future failed: {e}")
            
            logger.info(f"📊 Results: {success_count} successful, {error_count} failed")
            
        finally:
            producer.flush()
            producer.close()

def main():
    """Run all advanced producer demonstrations"""
    producer_demo = AdvancedKafkaProducer()
    
    print("🚀 Advanced Kafka Producer Demonstrations")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        producer_demo.demonstrate_basic_producer()
        time.sleep(2)
        
        producer_demo.demonstrate_idempotent_producer()
        time.sleep(2)
        
        producer_demo.demonstrate_transactional_producer()
        time.sleep(2)
        
        producer_demo.demonstrate_custom_partitioner()
        time.sleep(2)
        
        producer_demo.demonstrate_batch_sending()
        time.sleep(2)
        
        producer_demo.demonstrate_error_handling()
        
        print("\n✅ All producer demonstrations completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demonstrations interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstrations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()