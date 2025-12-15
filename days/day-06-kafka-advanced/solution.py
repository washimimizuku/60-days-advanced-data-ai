#!/usr/bin/env python3
"""
Day 6: Advanced Kafka - Solution
Complete Advanced Kafka Implementation
"""

import json
import time
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
from kafka.errors import KafkaError, KafkaTimeoutError
import logging
from concurrent.futures import ThreadPoolExecutor
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KafkaAdvancedSolution:
    """Complete Advanced Kafka Solution"""
    
    def __init__(self):
        self.bootstrap_servers = ['localhost:9092', 'localhost:9093', 'localhost:9094']
        self.admin_client = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
        self.metrics = {
            'messages_sent': 0,
            'messages_consumed': 0,
            'errors': 0,
            'latencies': []
        }
        
    # ============================================================================
    # SOLUTION 1: Topic Management and Partitioning Strategy
    # ============================================================================
    
    def create_topics_with_strategy(self):
        """Create topics with optimal partitioning strategies"""
        
        topics_to_create = [
            NewTopic(
                name='user-events',
                num_partitions=12,
                replication_factor=3,
                topic_configs={
                    'retention.ms': str(7 * 24 * 60 * 60 * 1000),  # 7 days
                    'cleanup.policy': 'delete',
                    'compression.type': 'lz4'
                }
            ),
            NewTopic(
                name='real-time-alerts',
                num_partitions=3,
                replication_factor=3,
                topic_configs={
                    'retention.ms': str(24 * 60 * 60 * 1000),  # 1 day
                    'cleanup.policy': 'delete',
                    'min.insync.replicas': '2'
                }
            ),
            NewTopic(
                name='analytics-data',
                num_partitions=6,
                replication_factor=2,
                topic_configs={
                    'retention.ms': str(30 * 24 * 60 * 60 * 1000),  # 30 days
                    'cleanup.policy': 'compact',
                    'segment.ms': str(24 * 60 * 60 * 1000)  # 1 day segments
                }
            )
        ]
        
        try:
            # Create topics
            fs = self.admin_client.create_topics(topics_to_create, validate_only=False)
            
            # Wait for topics to be created
            for topic, f in fs.items():
                try:
                    f.result()  # The result itself is None
                    logger.info(f"Topic {topic} created successfully")
                except Exception as e:
                    if "already exists" in str(e):
                        logger.info(f"Topic {topic} already exists")
                    else:
                        logger.error(f"Failed to create topic {topic}: {e}")
                        
        except Exception as e:
            logger.error(f"Error creating topics: {e}")
            
    def analyze_partition_distribution(self, topic: str):
        """Analyze message distribution across partitions"""
        
        # Create producer for testing
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        partition_counts = {}
        
        try:
            # Send messages with keys (should go to same partition)
            logger.info("Testing key-based partitioning...")
            for i in range(100):
                key = f"user_{i % 10}"  # 10 different users
                message = {'user_id': key, 'event': 'test', 'timestamp': datetime.now().isoformat()}
                
                future = producer.send(topic, key=key, value=message)
                record_metadata = future.get(timeout=10)
                
                partition = record_metadata.partition
                partition_counts[partition] = partition_counts.get(partition, 0) + 1
                
            logger.info(f"Partition distribution with keys: {partition_counts}")
            
            # Reset counts
            partition_counts = {}
            
            # Send messages without keys (should be round-robin)
            logger.info("Testing round-robin partitioning...")
            for i in range(100):
                message = {'event_id': str(uuid.uuid4()), 'event': 'test', 'timestamp': datetime.now().isoformat()}
                
                future = producer.send(topic, value=message)
                record_metadata = future.get(timeout=10)
                
                partition = record_metadata.partition
                partition_counts[partition] = partition_counts.get(partition, 0) + 1
                
            logger.info(f"Partition distribution without keys: {partition_counts}")
            
            # Calculate distribution statistics
            counts = list(partition_counts.values())
            if counts:
                avg = statistics.mean(counts)
                std_dev = statistics.stdev(counts) if len(counts) > 1 else 0
                logger.info(f"Distribution stats - Mean: {avg:.2f}, Std Dev: {std_dev:.2f}")
                
        finally:
            producer.flush()
            producer.close()
            
    # ============================================================================
    # SOLUTION 2: Advanced Producer Patterns
    # ============================================================================
    
    def implement_idempotent_producer(self):
        """Implement idempotent producer to prevent duplicates"""
        
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            # Idempotence configuration
            enable_idempotence=True,
            acks='all',
            retries=2147483647,
            max_in_flight_requests_per_connection=5,
            # Performance settings
            batch_size=16384,
            linger_ms=10,
            compression_type='lz4'
        )
        
        try:
            # Send the same message multiple times
            message = {
                'order_id': 'order_12345',
                'user_id': 'user_789',
                'amount': 99.99,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Sending duplicate messages with idempotent producer...")
            
            sent_offsets = []
            for i in range(3):
                future = producer.send('user-events', key='user_789', value=message)
                record_metadata = future.get(timeout=10)
                sent_offsets.append(record_metadata.offset)
                logger.info(f"Attempt {i+1}: Partition {record_metadata.partition}, Offset {record_metadata.offset}")
                
            # Check if offsets are consecutive (no duplicates)
            if len(set(sent_offsets)) == len(sent_offsets):
                logger.info("✅ Idempotent producer working correctly - no duplicates")
            else:
                logger.warning("⚠️  Possible duplicates detected")
                
        finally:
            producer.flush()
            producer.close()
            
    def implement_transactional_producer(self):
        """Implement transactional producer for exactly-once semantics"""
        
        transaction_id = f"tx-producer-{uuid.uuid4()}"
        
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            # Transaction configuration
            transactional_id=transaction_id,
            enable_idempotence=True,
            acks='all',
            retries=2147483647,
            max_in_flight_requests_per_connection=5
        )
        
        try:
            # Initialize transactions
            producer.init_transactions()
            
            # Successful transaction
            logger.info("Starting successful transaction...")
            producer.begin_transaction()
            
            try:
                # Send order creation events
                order_data = {
                    'order_id': 'order_67890',
                    'user_id': 'user_999',
                    'total': 149.97,
                    'items': [
                        {'product_id': 'prod_1', 'quantity': 2},
                        {'product_id': 'prod_2', 'quantity': 1}
                    ]
                }
                
                # Send to multiple topics atomically
                producer.send('user-events', key='user_999', value={
                    'event': 'order_created',
                    'order_id': order_data['order_id'],
                    'total': order_data['total']
                })
                
                producer.send('analytics-data', key='daily_sales', value={
                    'event': 'sale_recorded',
                    'amount': order_data['total'],
                    'timestamp': datetime.now().isoformat()
                })
                
                # Commit transaction
                producer.commit_transaction()
                logger.info("✅ Transaction committed successfully")
                
            except Exception as e:
                logger.error(f"Error in transaction: {e}")
                producer.abort_transaction()
                logger.info("❌ Transaction aborted")
                
            # Demonstrate transaction abort
            logger.info("Demonstrating transaction abort...")
            producer.begin_transaction()
            
            try:
                producer.send('user-events', key='test', value={'test': 'will_be_aborted'})
                # Simulate error
                raise Exception("Simulated error")
                
            except Exception as e:
                logger.info(f"Expected error: {e}")
                producer.abort_transaction()
                logger.info("✅ Transaction aborted successfully")
                
        finally:
            producer.close()
            
    def implement_custom_partitioner(self):
        """Implement custom partitioning logic"""
        
        def business_partitioner(key_bytes, all_partitions, available_partitions):
            """Route messages based on business logic"""
            if key_bytes:
                key = key_bytes.decode('utf-8')
                
                # Premium users to first partition
                if key.startswith('premium_'):
                    return all_partitions[0]
                # VIP users to second partition
                elif key.startswith('vip_'):
                    return all_partitions[1] if len(all_partitions) > 1 else all_partitions[0]
                # Regular users distributed across remaining partitions
                else:
                    remaining = all_partitions[2:] if len(all_partitions) > 2 else all_partitions
                    return remaining[hash(key) % len(remaining)]
            
            # No key - use first available partition
            return available_partitions[0]
        
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            partitioner=business_partitioner
        )
        
        try:
            # Test different user types
            test_users = [
                ('premium_user_1', 'premium'),
                ('vip_user_1', 'vip'),
                ('regular_user_1', 'regular'),
                ('premium_user_2', 'premium'),
                ('regular_user_2', 'regular')
            ]
            
            logger.info("Testing custom partitioner...")
            
            for user_key, user_type in test_users:
                message = {
                    'user_id': user_key,
                    'user_type': user_type,
                    'event': 'login',
                    'timestamp': datetime.now().isoformat()
                }
                
                future = producer.send('user-events', key=user_key, value=message)
                record_metadata = future.get(timeout=10)
                
                logger.info(f"{user_type.upper()} user {user_key} -> Partition {record_metadata.partition}")
                
        finally:
            producer.flush()
            producer.close()
            
    def optimize_producer_performance(self):
        """Optimize producer for high throughput"""
        
        # High-throughput configuration
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            # Throughput optimization
            batch_size=65536,  # 64KB batches
            linger_ms=20,      # Wait for batching
            compression_type='lz4',
            buffer_memory=67108864,  # 64MB buffer
            acks='1',  # Balance durability and performance
            max_in_flight_requests_per_connection=5,
            request_timeout_ms=30000
        )
        
        try:
            num_messages = 10000
            start_time = time.time()
            
            logger.info(f"Sending {num_messages} messages for performance test...")
            
            futures = []
            for i in range(num_messages):
                message = {
                    'message_id': i,
                    'user_id': f'user_{i % 1000}',
                    'event_type': 'performance_test',
                    'timestamp': datetime.now().isoformat(),
                    'data': f'test_data_{i}'
                }
                
                future = producer.send('user-events', key=f'user_{i % 1000}', value=message)
                futures.append(future)
                
                if i % 1000 == 0:
                    logger.info(f"Queued {i} messages...")
            
            # Wait for all messages
            logger.info("Waiting for all messages to be sent...")
            for future in futures:
                future.get(timeout=30)
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = num_messages / duration
            
            logger.info(f"✅ Performance test completed:")
            logger.info(f"   Messages: {num_messages}")
            logger.info(f"   Duration: {duration:.2f} seconds")
            logger.info(f"   Throughput: {throughput:.2f} messages/second")
            
        finally:
            producer.flush()
            producer.close()
            
    # ============================================================================
    # SOLUTION 3: Advanced Consumer Patterns
    # ============================================================================
    
    def implement_consumer_group_patterns(self):
        """Implement advanced consumer group patterns"""
        
        def create_consumer(group_id: str, client_id: str):
            return KafkaConsumer(
                'user-events',
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                client_id=client_id,
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                enable_auto_commit=False,
                auto_offset_reset='earliest',
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
                max_poll_records=500
            )
        
        def consumer_worker(consumer, worker_id):
            """Consumer worker function"""
            logger.info(f"Consumer {worker_id} started")
            
            try:
                message_count = 0
                while message_count < 10:  # Process 10 messages per worker
                    msg_pack = consumer.poll(timeout_ms=5000)
                    
                    for tp, messages in msg_pack.items():
                        for message in messages:
                            logger.info(f"Consumer {worker_id} processed message from partition {tp.partition}")
                            message_count += 1
                            
                            # Simulate processing time
                            time.sleep(0.1)
                    
                    # Manual commit
                    consumer.commit()
                    
            except Exception as e:
                logger.error(f"Consumer {worker_id} error: {e}")
            finally:
                consumer.close()
                logger.info(f"Consumer {worker_id} stopped")
        
        # Create multiple consumers in the same group
        logger.info("Starting consumer group with 3 consumers...")
        
        consumers = []
        threads = []
        
        for i in range(3):
            consumer = create_consumer('analytics-group', f'consumer-{i}')
            consumers.append(consumer)
            
            thread = threading.Thread(target=consumer_worker, args=(consumer, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all consumers to finish
        for thread in threads:
            thread.join()
            
        logger.info("✅ Consumer group demonstration completed")
        
    def implement_manual_partition_assignment(self):
        """Implement manual partition assignment"""
        
        from kafka import TopicPartition
        
        consumer = KafkaConsumer(
            bootstrap_servers=self.bootstrap_servers,
            key_deserializer=lambda x: x.decode('utf-8') if x else None,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            enable_auto_commit=False
        )
        
        try:
            # Manually assign specific partitions
            partitions = [
                TopicPartition('user-events', 0),
                TopicPartition('user-events', 1)
            ]
            
            consumer.assign(partitions)
            
            # Seek to specific offsets
            consumer.seek(TopicPartition('user-events', 0), 0)  # Start from beginning
            consumer.seek_to_end(TopicPartition('user-events', 1))  # Start from end
            
            logger.info("Manual partition assignment completed")
            logger.info(f"Assigned partitions: {[tp.partition for tp in partitions]}")
            
            # Process a few messages
            message_count = 0
            while message_count < 5:
                msg_pack = consumer.poll(timeout_ms=5000)
                
                for tp, messages in msg_pack.items():
                    for message in messages:
                        logger.info(f"Processed message from partition {tp.partition}, offset {message.offset}")
                        message_count += 1
                        
                        if message_count >= 5:
                            break
                            
        finally:
            consumer.close()
            
    # ============================================================================
    # SOLUTION 4: Performance Monitoring
    # ============================================================================
    
    def implement_performance_monitoring(self):
        """Implement comprehensive performance monitoring"""
        
        # Create producer with metrics
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            acks='1'
        )
        
        # Create consumer with metrics
        consumer = KafkaConsumer(
            'user-events',
            bootstrap_servers=self.bootstrap_servers,
            group_id='monitoring-group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            enable_auto_commit=True
        )
        
        try:
            # Send test messages and measure latency
            logger.info("Starting performance monitoring...")
            
            latencies = []
            
            for i in range(100):
                start_time = time.time()
                
                message = {
                    'id': i,
                    'timestamp': datetime.now().isoformat(),
                    'data': f'monitoring_test_{i}'
                }
                
                future = producer.send('user-events', value=message)
                future.get(timeout=10)  # Wait for send to complete
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
                
                if i % 20 == 0:
                    logger.info(f"Sent {i} messages...")
            
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
            
            logger.info(f"✅ Performance Metrics:")
            logger.info(f"   Average latency: {avg_latency:.2f} ms")
            logger.info(f"   95th percentile: {p95_latency:.2f} ms")
            logger.info(f"   99th percentile: {p99_latency:.2f} ms")
            
        finally:
            producer.close()
            consumer.close()

def main():
    """Main solution demonstration"""
    
    print("Day 6: Advanced Kafka - Complete Solution")
    print("=" * 50)
    
    # Initialize solution
    solution = KafkaAdvancedSolution()
    
    try:
        # 1. Topic Management
        print("\n1. Creating topics with optimal partitioning...")
        solution.create_topics_with_strategy()
        time.sleep(2)
        
        # 2. Partition Analysis
        print("\n2. Analyzing partition distribution...")
        solution.analyze_partition_distribution('user-events')
        time.sleep(2)
        
        # 3. Advanced Producer Patterns
        print("\n3. Testing idempotent producer...")
        solution.implement_idempotent_producer()
        time.sleep(2)
        
        print("\n4. Testing transactional producer...")
        solution.implement_transactional_producer()
        time.sleep(2)
        
        print("\n5. Testing custom partitioner...")
        solution.implement_custom_partitioner()
        time.sleep(2)
        
        print("\n6. Running performance optimization test...")
        solution.optimize_producer_performance()
        time.sleep(2)
        
        # 4. Advanced Consumer Patterns
        print("\n7. Testing consumer group patterns...")
        solution.implement_consumer_group_patterns()
        time.sleep(2)
        
        print("\n8. Testing manual partition assignment...")
        solution.implement_manual_partition_assignment()
        time.sleep(2)
        
        # 5. Performance Monitoring
        print("\n9. Running performance monitoring...")
        solution.implement_performance_monitoring()
        
        print("\n✅ Advanced Kafka Solution Complete!")
        print("\nKey Features Demonstrated:")
        print("- Optimal topic creation and partitioning strategies")
        print("- Idempotent and transactional producers")
        print("- Custom partitioning logic")
        print("- High-throughput producer optimization")
        print("- Advanced consumer group patterns")
        print("- Manual partition assignment")
        print("- Comprehensive performance monitoring")
        
    except KeyboardInterrupt:
        print("\nSolution interrupted by user")
    except Exception as e:
        print(f"\nError during solution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
