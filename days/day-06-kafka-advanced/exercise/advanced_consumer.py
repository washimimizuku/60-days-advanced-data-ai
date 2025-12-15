#!/usr/bin/env python3
"""
Day 6: Advanced Kafka - Advanced Consumer Patterns
Demonstrates consumer groups, manual offset management, and rebalancing
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from kafka import KafkaConsumer, TopicPartition, OffsetAndMetadata
from kafka.errors import KafkaError, CommitFailedError
import logging
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedKafkaConsumer:
    """Advanced Kafka consumer with various patterns"""
    
    def __init__(self, bootstrap_servers=['localhost:9092']):
        self.bootstrap_servers = bootstrap_servers
        self.running = True
        self.metrics = {
            'messages_processed': 0,
            'processing_errors': 0,
            'commit_errors': 0,
            'rebalances': 0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def create_basic_consumer(self, group_id: str, topics: List[str]) -> KafkaConsumer:
        """Create basic consumer with auto-commit"""
        return KafkaConsumer(
            *topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            key_deserializer=lambda x: x.decode('utf-8') if x else None,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            # Performance settings
            fetch_min_bytes=1024,
            fetch_max_wait_ms=500,
            max_poll_records=500,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000
        )
    
    def create_manual_commit_consumer(self, group_id: str, topics: List[str]) -> KafkaConsumer:
        """Create consumer with manual offset management"""
        return KafkaConsumer(
            *topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            key_deserializer=lambda x: x.decode('utf-8') if x else None,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=False,  # Manual commit
            # Performance settings
            fetch_min_bytes=1024,
            fetch_max_wait_ms=500,
            max_poll_records=100,  # Smaller batches for manual commit
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            max_poll_interval_ms=300000  # 5 minutes max processing time
        )
    
    def create_specific_partition_consumer(self, topic: str, partitions: List[int]) -> KafkaConsumer:
        """Create consumer for specific partitions (no consumer group)"""
        consumer = KafkaConsumer(
            bootstrap_servers=self.bootstrap_servers,
            key_deserializer=lambda x: x.decode('utf-8') if x else None,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            fetch_min_bytes=1024,
            fetch_max_wait_ms=500
        )
        
        # Manually assign partitions
        topic_partitions = [TopicPartition(topic, p) for p in partitions]
        consumer.assign(topic_partitions)
        
        return consumer
    
    def demonstrate_basic_consumer(self):
        """Demonstrate basic consumer with auto-commit"""
        logger.info("📥 Demonstrating Basic Consumer with Auto-commit")
        
        consumer = self.create_basic_consumer('basic-consumer-group', ['user-events'])
        
        try:
            message_count = 0
            start_time = time.time()
            
            while self.running and message_count < 50:
                msg_pack = consumer.poll(timeout_ms=1000)
                
                if not msg_pack:
                    continue
                
                for topic_partition, messages in msg_pack.items():
                    for message in messages:
                        message_count += 1
                        
                        logger.info(f"Consumed message {message_count}: "
                                  f"Key={message.key}, "
                                  f"Partition={message.partition}, "
                                  f"Offset={message.offset}")
                        
                        # Simulate processing
                        self.process_message(message.value)
                        
                        if message_count >= 50:
                            break
                    
                    if message_count >= 50:
                        break
            
            duration = time.time() - start_time
            logger.info(f"✅ Processed {message_count} messages in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in basic consumer: {e}")
        finally:
            consumer.close()
    
    def demonstrate_manual_commit_consumer(self):
        """Demonstrate consumer with manual offset management"""
        logger.info("🎯 Demonstrating Manual Commit Consumer")
        
        consumer = self.create_manual_commit_consumer('manual-commit-group', ['user-events'])
        
        try:
            message_count = 0
            batch_size = 10
            processed_messages = []
            
            while self.running and message_count < 50:
                msg_pack = consumer.poll(timeout_ms=1000)
                
                if not msg_pack:
                    continue
                
                for topic_partition, messages in msg_pack.items():
                    for message in messages:
                        try:
                            # Process message
                            result = self.process_message(message.value)
                            processed_messages.append((message, result))
                            message_count += 1
                            
                            logger.info(f"Processed message {message_count}: "
                                      f"Key={message.key}, Offset={message.offset}")
                            
                            # Commit in batches
                            if len(processed_messages) >= batch_size:
                                self.commit_batch(consumer, processed_messages)
                                processed_messages.clear()
                            
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            self.metrics['processing_errors'] += 1
                            # Skip this message and continue
                            continue
                
                # Commit remaining messages
                if processed_messages:
                    self.commit_batch(consumer, processed_messages)
                    processed_messages.clear()
            
            logger.info(f"✅ Manual commit consumer processed {message_count} messages")
            
        except Exception as e:
            logger.error(f"Error in manual commit consumer: {e}")
        finally:
            consumer.close()
    
    def commit_batch(self, consumer: KafkaConsumer, processed_messages: List):
        """Commit offsets for a batch of processed messages"""
        try:
            # Build offset map
            offsets = {}
            for message, result in processed_messages:
                tp = TopicPartition(message.topic, message.partition)
                # Commit offset + 1 (next message to read)
                offsets[tp] = OffsetAndMetadata(message.offset + 1, None)
            
            # Commit offsets
            consumer.commit(offsets)
            logger.debug(f"Committed offsets for {len(processed_messages)} messages")
            
        except CommitFailedError as e:
            logger.error(f"Failed to commit offsets: {e}")
            self.metrics['commit_errors'] += 1
    
    def demonstrate_partition_assignment(self):
        """Demonstrate manual partition assignment"""
        logger.info("🎯 Demonstrating Manual Partition Assignment")
        
        topic = 'user-events'
        partitions = [0, 1]  # Consume only partitions 0 and 1
        
        consumer = self.create_specific_partition_consumer(topic, partitions)
        
        try:
            # Seek to specific offsets
            consumer.seek(TopicPartition(topic, 0), 0)  # Start from beginning of partition 0
            consumer.seek(TopicPartition(topic, 1), 10)  # Start from offset 10 of partition 1
            
            message_count = 0
            partition_counts = {0: 0, 1: 0}
            
            while self.running and message_count < 30:
                msg_pack = consumer.poll(timeout_ms=1000)
                
                if not msg_pack:
                    continue
                
                for topic_partition, messages in msg_pack.items():
                    for message in messages:
                        message_count += 1
                        partition_counts[message.partition] += 1
                        
                        logger.info(f"Message {message_count}: "
                                  f"Partition={message.partition}, "
                                  f"Offset={message.offset}, "
                                  f"Key={message.key}")
                        
                        self.process_message(message.value)
                        
                        if message_count >= 30:
                            break
                    
                    if message_count >= 30:
                        break
            
            logger.info(f"✅ Partition assignment results: {partition_counts}")
            
        except Exception as e:
            logger.error(f"Error in partition assignment demo: {e}")
        finally:
            consumer.close()
    
    def demonstrate_consumer_group_rebalancing(self):
        """Demonstrate consumer group rebalancing"""
        logger.info("⚖️  Demonstrating Consumer Group Rebalancing")
        
        def create_consumer_worker(consumer_id: str, group_id: str):
            """Worker function for consumer in a group"""
            consumer = self.create_manual_commit_consumer(group_id, ['user-events'])
            
            # Add rebalance listener
            def on_assign(assigned_partitions):
                logger.info(f"Consumer {consumer_id}: Assigned partitions {[tp.partition for tp in assigned_partitions]}")
                self.metrics['rebalances'] += 1
            
            def on_revoke(revoked_partitions):
                logger.info(f"Consumer {consumer_id}: Revoked partitions {[tp.partition for tp in revoked_partitions]}")
                # Commit offsets before rebalance
                try:
                    consumer.commit()
                except Exception as e:
                    logger.error(f"Error committing before rebalance: {e}")
            
            consumer.subscribe(['user-events'], 
                             listener=type('RebalanceListener', (), {
                                 'on_partitions_assigned': lambda self, assigned: on_assign(assigned),
                                 'on_partitions_revoked': lambda self, revoked: on_revoke(revoked)
                             })())
            
            try:
                message_count = 0
                while self.running and message_count < 20:
                    msg_pack = consumer.poll(timeout_ms=1000)
                    
                    if not msg_pack:
                        continue
                    
                    for topic_partition, messages in msg_pack.items():
                        for message in messages:
                            message_count += 1
                            logger.info(f"Consumer {consumer_id}: Message {message_count} "
                                      f"from partition {message.partition}")
                            
                            self.process_message(message.value)
                            
                            if message_count >= 20:
                                break
                        
                        if message_count >= 20:
                            break
                    
                    # Commit periodically
                    if message_count % 5 == 0:
                        consumer.commit()
                
                logger.info(f"Consumer {consumer_id} processed {message_count} messages")
                
            except Exception as e:
                logger.error(f"Error in consumer {consumer_id}: {e}")
            finally:
                consumer.close()
        
        # Start multiple consumers in the same group
        group_id = 'rebalancing-demo-group'
        threads = []
        
        # Start first consumer
        thread1 = threading.Thread(target=create_consumer_worker, args=('consumer-1', group_id))
        threads.append(thread1)
        thread1.start()
        
        time.sleep(5)  # Let first consumer get all partitions
        
        # Start second consumer (should trigger rebalance)
        logger.info("Starting second consumer - should trigger rebalance...")
        thread2 = threading.Thread(target=create_consumer_worker, args=('consumer-2', group_id))
        threads.append(thread2)
        thread2.start()
        
        time.sleep(5)  # Let rebalance happen
        
        # Start third consumer (another rebalance)
        logger.info("Starting third consumer - another rebalance...")
        thread3 = threading.Thread(target=create_consumer_worker, args=('consumer-3', group_id))
        threads.append(thread3)
        thread3.start()
        
        # Wait for all consumers to finish
        for thread in threads:
            thread.join()
        
        logger.info(f"✅ Rebalancing demo completed. Total rebalances: {self.metrics['rebalances']}")
    
    def demonstrate_offset_management(self):
        """Demonstrate advanced offset management"""
        logger.info("📍 Demonstrating Advanced Offset Management")
        
        consumer = self.create_manual_commit_consumer('offset-demo-group', ['user-events'])
        
        try:
            # Get current offsets
            partitions = consumer.assignment()
            if not partitions:
                # Subscribe and poll once to get assignment
                consumer.poll(timeout_ms=1000)
                partitions = consumer.assignment()
            
            logger.info("Current partition assignment and offsets:")
            for tp in partitions:
                try:
                    # Get committed offset
                    committed = consumer.committed(tp)
                    # Get current position
                    position = consumer.position(tp)
                    # Get high water mark
                    high_water_mark = consumer.end_offsets([tp])[tp]
                    
                    logger.info(f"Partition {tp.partition}: "
                              f"Committed={committed}, "
                              f"Position={position}, "
                              f"High Water Mark={high_water_mark}")
                    
                    # Calculate lag
                    if committed is not None:
                        lag = high_water_mark - committed
                        logger.info(f"  Lag: {lag} messages")
                    
                except Exception as e:
                    logger.error(f"Error getting offset info for {tp}: {e}")
            
            # Demonstrate seeking to specific offsets
            if partitions:
                tp = list(partitions)[0]  # Use first partition
                
                # Seek to beginning
                logger.info(f"Seeking to beginning of partition {tp.partition}")
                consumer.seek_to_beginning(tp)
                
                # Read a few messages
                for i in range(5):
                    msg_pack = consumer.poll(timeout_ms=1000)
                    if msg_pack:
                        for topic_partition, messages in msg_pack.items():
                            for message in messages:
                                logger.info(f"Message from beginning: Offset={message.offset}")
                                break
                        break
                
                # Seek to end
                logger.info(f"Seeking to end of partition {tp.partition}")
                consumer.seek_to_end(tp)
                
                # Seek to specific offset
                consumer.seek(tp, max(0, consumer.position(tp) - 10))
                logger.info(f"Seeked to offset {consumer.position(tp)}")
            
        except Exception as e:
            logger.error(f"Error in offset management demo: {e}")
        finally:
            consumer.close()
    
    def demonstrate_consumer_performance_tuning(self):
        """Demonstrate consumer performance optimization"""
        logger.info("🚀 Demonstrating Consumer Performance Tuning")
        
        # High-throughput consumer configuration
        consumer = KafkaConsumer(
            'user-events',
            bootstrap_servers=self.bootstrap_servers,
            group_id='performance-tuning-group',
            key_deserializer=lambda x: x.decode('utf-8') if x else None,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            # Performance optimizations
            fetch_min_bytes=50000,      # 50KB minimum fetch
            fetch_max_wait_ms=500,      # Max wait for fetch_min_bytes
            max_partition_fetch_bytes=1048576,  # 1MB per partition
            max_poll_records=1000,      # More records per poll
            # Session management
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            max_poll_interval_ms=300000
        )
        
        try:
            start_time = time.time()
            message_count = 0
            batch_count = 0
            
            while self.running and message_count < 500:
                msg_pack = consumer.poll(timeout_ms=1000)
                
                if not msg_pack:
                    continue
                
                batch_count += 1
                batch_size = sum(len(messages) for messages in msg_pack.values())
                
                for topic_partition, messages in msg_pack.items():
                    for message in messages:
                        message_count += 1
                        
                        # Simulate fast processing
                        self.process_message_fast(message.value)
                        
                        if message_count % 100 == 0:
                            logger.info(f"Processed {message_count} messages...")
                
                # Commit after each batch
                consumer.commit()
                
                if batch_size > 0:
                    logger.debug(f"Batch {batch_count}: {batch_size} messages")
            
            duration = time.time() - start_time
            throughput = message_count / duration if duration > 0 else 0
            
            logger.info(f"✅ Performance test results:")
            logger.info(f"   Messages processed: {message_count}")
            logger.info(f"   Duration: {duration:.2f} seconds")
            logger.info(f"   Throughput: {throughput:.2f} messages/second")
            logger.info(f"   Batches: {batch_count}")
            
        except Exception as e:
            logger.error(f"Error in performance tuning demo: {e}")
        finally:
            consumer.close()
    
    def process_message(self, message_data: Dict[str, Any]) -> bool:
        """Simulate message processing with some delay"""
        try:
            # Simulate processing time
            time.sleep(0.01)  # 10ms processing time
            
            # Simulate processing logic
            event_type = message_data.get('event', 'unknown')
            if event_type == 'purchase':
                # Simulate purchase processing
                amount = message_data.get('amount', 0)
                logger.debug(f"Processing purchase: ${amount}")
            elif event_type == 'login':
                # Simulate login processing
                user_id = message_data.get('user_id', 'unknown')
                logger.debug(f"Processing login for user: {user_id}")
            
            self.metrics['messages_processed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.metrics['processing_errors'] += 1
            return False
    
    def process_message_fast(self, message_data: Dict[str, Any]) -> bool:
        """Fast message processing for performance tests"""
        try:
            # Minimal processing for performance testing
            self.metrics['messages_processed'] += 1
            return True
        except Exception as e:
            self.metrics['processing_errors'] += 1
            return False

def main():
    """Run all advanced consumer demonstrations"""
    consumer_demo = AdvancedKafkaConsumer()
    
    print("📥 Advanced Kafka Consumer Demonstrations")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        consumer_demo.demonstrate_basic_consumer()
        time.sleep(2)
        
        consumer_demo.demonstrate_manual_commit_consumer()
        time.sleep(2)
        
        consumer_demo.demonstrate_partition_assignment()
        time.sleep(2)
        
        consumer_demo.demonstrate_offset_management()
        time.sleep(2)
        
        consumer_demo.demonstrate_consumer_group_rebalancing()
        time.sleep(2)
        
        consumer_demo.demonstrate_consumer_performance_tuning()
        
        print(f"\n✅ All consumer demonstrations completed!")
        print(f"📊 Final metrics: {consumer_demo.metrics}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demonstrations interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstrations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()