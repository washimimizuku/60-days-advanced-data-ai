#!/usr/bin/env python3
"""
Day 5: CDC with Debezium - Exercise
Build a Complete CDC Pipeline

Scenario: You're building a real-time e-commerce analytics system that needs to:
1. Capture changes from PostgreSQL transactional database
2. Process events in real-time for analytics
3. Update dashboards and alerts instantly
4. Maintain data consistency across systems
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any
from kafka import KafkaConsumer, KafkaProducer
import redis
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CDCExercise:
    """CDC Exercise Implementation"""
    
    def __init__(self):
        self.kafka_servers = ['localhost:9092']
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
    # ============================================================================
    # EXERCISE 1: Set up Debezium Connector
    # ============================================================================
    
    def setup_debezium_connector(self):
        """TODO: Configure and deploy Debezium PostgreSQL connector"""
        
        connector_config = {
            "name": "postgres-connector",
            "config": {
                # TODO: Complete the connector configuration
                "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
                "database.hostname": "localhost",
                "database.port": "5432",
                "database.user": "<debezium_user>",
                "database.password": "<debezium_password>",
                "database.dbname": "ecommerce",
                "database.server.name": "ecommerce-db",
                # TODO: Add table include list
                # TODO: Configure transforms
                # TODO: Set up error handling
            }
        }
        
        # TODO: Deploy connector using Kafka Connect REST API
        connect_url = "http://localhost:8083/connectors"
        
        try:
            # TODO: Send POST request to create connector
            pass
        except Exception as e:
            logger.error(f"Failed to create connector: {e}")
            
    def check_connector_status(self, connector_name: str):
        """TODO: Check if connector is running properly"""
        # TODO: Get connector status from Kafka Connect
        # TODO: Return health status
        pass
        
    # ============================================================================
    # EXERCISE 2: Process CDC Events
    # ============================================================================
    
    def create_cdc_consumer(self, topics: list):
        """TODO: Create Kafka consumer for CDC events"""
        
        consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=self.kafka_servers,
            # TODO: Configure deserializers
            # TODO: Set consumer group
            # TODO: Configure offset reset
        )
        
        return consumer
        
    def process_cdc_events(self):
        """TODO: Main event processing loop"""
        
        topics = [
            'ecommerce-db.public.customers',
            'ecommerce-db.public.orders',
            'ecommerce-db.public.products'
        ]
        
        consumer = self.create_cdc_consumer(topics)
        
        try:
            for message in consumer:
                # TODO: Process each CDC event
                event = message.value
                table = self.extract_table_name(message.topic)
                
                # TODO: Route to appropriate handler
                if table == 'customers':
                    self.handle_customer_event(event)
                elif table == 'orders':
                    self.handle_order_event(event)
                elif table == 'products':
                    self.handle_product_event(event)
                    
        except KeyboardInterrupt:
            logger.info("Stopping CDC processor...")
        finally:
            consumer.close()
            
    def extract_table_name(self, topic: str) -> str:
        """TODO: Extract table name from CDC topic"""
        # Topic format: ecommerce-db.public.table_name
        # TODO: Parse and return table name
        pass
        
    # ============================================================================
    # EXERCISE 3: Event Handlers
    # ============================================================================
    
    def handle_customer_event(self, event: Dict[str, Any]):
        """TODO: Process customer table changes"""
        
        operation = event.get('op')
        
        if operation == 'c':  # INSERT
            # TODO: Handle new customer creation
            customer = event.get('after', {})
            customer_id = customer.get('customer_id')
            
            # TODO: Update customer metrics in Redis
            # TODO: Send welcome email event
            # TODO: Update real-time dashboard
            
        elif operation == 'u':  # UPDATE
            # TODO: Handle customer updates
            old_customer = event.get('before', {})
            new_customer = event.get('after', {})
            
            # TODO: Check for important changes (email, status, etc.)
            # TODO: Update cached customer data
            # TODO: Trigger relevant workflows
            
        elif operation == 'd':  # DELETE
            # TODO: Handle customer deletion
            customer = event.get('before', {})
            
            # TODO: Clean up customer data
            # TODO: Update metrics
            
    def handle_order_event(self, event: Dict[str, Any]):
        """TODO: Process order table changes"""
        
        operation = event.get('op')
        
        if operation == 'c':  # New order
            order = event.get('after', {})
            order_id = order.get('order_id')
            customer_id = order.get('customer_id')
            total_amount = float(order.get('total_amount', 0))
            
            # TODO: Update real-time revenue metrics
            # TODO: Update customer lifetime value
            # TODO: Check for fraud patterns
            # TODO: Send order confirmation
            
        elif operation == 'u':  # Order status change
            old_order = event.get('before', {})
            new_order = event.get('after', {})
            
            old_status = old_order.get('status')
            new_status = new_order.get('status')
            
            if old_status != new_status:
                # TODO: Handle status transitions
                # TODO: Send notifications
                # TODO: Update fulfillment systems
                pass
                
    def handle_product_event(self, event: Dict[str, Any]):
        """TODO: Process product table changes"""
        
        operation = event.get('op')
        
        if operation == 'u':  # Product update
            old_product = event.get('before', {})
            new_product = event.get('after', {})
            
            # TODO: Check for inventory changes
            old_inventory = old_product.get('inventory_quantity', 0)
            new_inventory = new_product.get('inventory_quantity', 0)
            
            if old_inventory != new_inventory:
                # TODO: Update inventory alerts
                # TODO: Trigger reorder if low stock
                pass
                
            # TODO: Check for price changes
            # TODO: Update search indexes
            # TODO: Invalidate product caches
            
    # ============================================================================
    # EXERCISE 4: Real-time Analytics
    # ============================================================================
    
    def update_real_time_metrics(self, metric_type: str, value: float, dimensions: Dict[str, str] = None):
        """TODO: Update real-time metrics in Redis"""
        
        timestamp = datetime.now()
        date_key = timestamp.strftime('%Y-%m-%d')
        hour_key = timestamp.strftime('%Y-%m-%d-%H')
        
        # TODO: Update daily metrics
        daily_key = f"metrics:daily:{date_key}:{metric_type}"
        
        # TODO: Update hourly metrics
        hourly_key = f"metrics:hourly:{hour_key}:{metric_type}"
        
        # TODO: Add dimensional metrics if provided
        if dimensions:
            for dim_name, dim_value in dimensions.items():
                dim_key = f"metrics:daily:{date_key}:{metric_type}:{dim_name}:{dim_value}"
                # TODO: Update dimensional metric
                
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """TODO: Get current dashboard metrics"""
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        dashboard_data = {
            # TODO: Get today's revenue
            'daily_revenue': 0,
            # TODO: Get today's order count
            'daily_orders': 0,
            # TODO: Get active customers
            'active_customers': 0,
            # TODO: Get low stock alerts
            'low_stock_products': [],
            # TODO: Get recent orders
            'recent_orders': []
        }
        
        return dashboard_data
        
    # ============================================================================
    # EXERCISE 5: Data Warehouse Integration
    # ============================================================================
    
    def sync_to_data_warehouse(self, event: Dict[str, Any]):
        """TODO: Sync CDC events to data warehouse"""
        
        # TODO: Transform CDC event to warehouse format
        # TODO: Handle different operation types (INSERT, UPDATE, DELETE)
        # TODO: Implement upsert logic
        # TODO: Handle schema evolution
        
        pass
        
    def create_warehouse_record(self, table: str, record: Dict[str, Any], operation: str) -> Dict[str, Any]:
        """TODO: Create warehouse-compatible record"""
        
        warehouse_record = record.copy()
        
        # TODO: Add CDC metadata
        warehouse_record['_cdc_timestamp'] = datetime.now().isoformat()
        warehouse_record['_cdc_operation'] = operation
        warehouse_record['_cdc_source_table'] = table
        
        # TODO: Handle data type conversions
        # TODO: Add business logic transformations
        
        return warehouse_record
        
    # ============================================================================
    # EXERCISE 6: Monitoring and Alerting
    # ============================================================================
    
    def setup_monitoring(self):
        """TODO: Set up CDC pipeline monitoring"""
        
        # TODO: Monitor connector health
        # TODO: Track processing lag
        # TODO: Monitor error rates
        # TODO: Set up alerts for failures
        
        pass
        
    def check_pipeline_health(self) -> Dict[str, Any]:
        """TODO: Check overall pipeline health"""
        
        health_status = {
            'connector_status': 'unknown',
            'consumer_lag': 0,
            'error_rate': 0,
            'last_processed_event': None,
            'alerts': []
        }
        
        # TODO: Check connector status
        # TODO: Calculate consumer lag
        # TODO: Check error rates
        # TODO: Identify any alerts
        
        return health_status
        
    # ============================================================================
    # EXERCISE 7: Error Handling and Recovery
    # ============================================================================
    
    def handle_processing_error(self, event: Dict[str, Any], error: Exception):
        """TODO: Handle event processing errors"""
        
        error_info = {
            'event': event,
            'error': str(error),
            'timestamp': datetime.now().isoformat(),
            'retry_count': 0
        }
        
        # TODO: Log error details
        # TODO: Send to dead letter queue
        # TODO: Implement retry logic
        # TODO: Alert on critical errors
        
    def setup_dead_letter_queue(self):
        """TODO: Set up dead letter queue for failed events"""
        
        # TODO: Create DLQ topic
        # TODO: Configure error handling
        # TODO: Set up DLQ monitoring
        
        pass
        
    def replay_failed_events(self):
        """TODO: Replay events from dead letter queue"""
        
        # TODO: Read from DLQ
        # TODO: Retry processing
        # TODO: Track success/failure
        
        pass

def main():
    """Main exercise function"""
    
    print("Day 5: CDC with Debezium - Exercise")
    print("Building a complete CDC pipeline...")
    
    # Initialize exercise
    exercise = CDCExercise()
    
    # TODO: Complete the exercises in order:
    
    # 1. Set up Debezium connector
    print("\n1. Setting up Debezium connector...")
    # exercise.setup_debezium_connector()
    
    # 2. Process CDC events
    print("\n2. Processing CDC events...")
    # exercise.process_cdc_events()
    
    # 3. Monitor pipeline health
    print("\n3. Monitoring pipeline...")
    # health = exercise.check_pipeline_health()
    # print(f"Pipeline health: {health}")
    
    print("\nExercise setup complete! Follow the TODOs to implement each component.")
    print("\nNext steps:")
    print("1. Start the Docker environment: docker-compose up -d")
    print("2. Deploy the Debezium connector")
    print("3. Generate sample data in PostgreSQL")
    print("4. Run the CDC processor to handle events")
    print("5. Monitor the pipeline and verify data flow")

if __name__ == "__main__":
    main()
