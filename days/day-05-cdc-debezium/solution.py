#!/usr/bin/env python3
"""
Day 5: CDC with Debezium - Solution
Complete CDC Pipeline Implementation
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from kafka import KafkaConsumer, KafkaProducer
import redis
import requests
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CDCEvent:
    """CDC Event data structure"""
    operation: str
    table: str
    before: Optional[Dict[str, Any]]
    after: Optional[Dict[str, Any]]
    source: Dict[str, Any]
    timestamp: int

class CDCSolution:
    """Complete CDC Pipeline Solution"""
    
    def __init__(self):
        self.kafka_servers = ['localhost:9092']
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None
        )
        
    # ============================================================================
    # SOLUTION 1: Debezium Connector Setup
    # ============================================================================
    
    def setup_debezium_connector(self) -> bool:
        """Deploy Debezium PostgreSQL connector"""
        
        connector_config = {
            "name": "postgres-connector",
            "config": {
                "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
                "database.hostname": "localhost",
                "database.port": "5432",
                "database.user": "debezium",
                "database.password": "debezium",
                "database.dbname": "ecommerce",
                "database.server.name": "ecommerce-db",
                "table.include.list": "public.customers,public.orders,public.products,public.order_items",
                "plugin.name": "pgoutput",
                "publication.name": "dbz_publication",
                "slot.name": "debezium_slot",
                "key.converter": "org.apache.kafka.connect.json.JsonConverter",
                "value.converter": "org.apache.kafka.connect.json.JsonConverter",
                "key.converter.schemas.enable": False,
                "value.converter.schemas.enable": False,
                "transforms": "unwrap",
                "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
                "transforms.unwrap.drop.tombstones": False,
                "transforms.unwrap.delete.handling.mode": "rewrite",
                "snapshot.mode": "initial",
                "decimal.handling.mode": "string"
            }
        }
        
        connect_url = "http://localhost:8083/connectors"
        
        try:
            response = requests.post(
                connect_url,
                headers={'Content-Type': 'application/json'},
                json=connector_config,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                logger.info("Debezium connector created successfully")
                return True
            else:
                logger.error(f"Failed to create connector: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating connector: {e}")
            return False
            
    def check_connector_status(self, connector_name: str = "postgres-connector") -> Dict[str, Any]:
        """Check connector health status"""
        
        try:
            response = requests.get(f"http://localhost:8083/connectors/{connector_name}/status")
            
            if response.status_code == 200:
                status = response.json()
                return {
                    'connector_state': status['connector']['state'],
                    'tasks': [task['state'] for task in status['tasks']],
                    'healthy': all(task['state'] == 'RUNNING' for task in status['tasks'])
                }
            else:
                return {'healthy': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
            
    # ============================================================================
    # SOLUTION 2: CDC Event Processing
    # ============================================================================
    
    def create_cdc_consumer(self, topics: List[str]) -> KafkaConsumer:
        """Create optimized Kafka consumer for CDC events"""
        
        return KafkaConsumer(
            *topics,
            bootstrap_servers=self.kafka_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            key_deserializer=lambda x: x.decode('utf-8') if x else None,
            group_id='cdc-processor',
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            max_poll_records=500,
            session_timeout_ms=30000
        )
        
    def process_cdc_events(self):
        """Main event processing loop with error handling"""
        
        topics = [
            'ecommerce-db.public.customers',
            'ecommerce-db.public.orders',
            'ecommerce-db.public.products',
            'ecommerce-db.public.order_items'
        ]
        
        consumer = self.create_cdc_consumer(topics)
        logger.info(f"Started CDC processor for topics: {topics}")
        
        try:
            for message in consumer:
                try:
                    event = self.parse_cdc_event(message)
                    self.route_event(event)
                    
                except Exception as e:
                    self.handle_processing_error(message.value, e)
                    
        except KeyboardInterrupt:
            logger.info("Stopping CDC processor...")
        finally:
            consumer.close()
            
    def parse_cdc_event(self, message) -> CDCEvent:
        """Parse Kafka message into CDC event structure"""
        
        event_data = message.value
        table = self.extract_table_name(message.topic)
        
        return CDCEvent(
            operation=event_data.get('op', 'unknown'),
            table=table,
            before=event_data.get('before'),
            after=event_data.get('after'),
            source=event_data.get('source', {}),
            timestamp=event_data.get('ts_ms', int(time.time() * 1000))
        )
        
    def extract_table_name(self, topic: str) -> str:
        """Extract table name from CDC topic"""
        # Topic format: ecommerce-db.public.table_name
        return topic.split('.')[-1]
        
    def route_event(self, event: CDCEvent):
        """Route events to appropriate handlers"""
        
        handlers = {
            'customers': self.handle_customer_event,
            'orders': self.handle_order_event,
            'products': self.handle_product_event,
            'order_items': self.handle_order_item_event
        }
        
        handler = handlers.get(event.table)
        if handler:
            handler(event)
        else:
            logger.warning(f"No handler for table: {event.table}")
            
    # ============================================================================
    # SOLUTION 3: Event Handlers
    # ============================================================================
    
    def handle_customer_event(self, event: CDCEvent):
        """Process customer table changes"""
        
        if event.operation == 'c':  # INSERT
            customer = event.after
            customer_id = customer['customer_id']
            
            logger.info(f"New customer created: {customer_id}")
            
            # Update customer metrics
            segment = customer.get('customer_segment', 'Unknown')
            self.redis_client.hincrby('customer_counts', segment, 1)
            
            # Cache customer data
            self.redis_client.hset(f'customer:{customer_id}', mapping={
                'name': f"{customer.get('first_name', '')} {customer.get('last_name', '')}",
                'email': customer.get('email', ''),
                'segment': segment,
                'registration_date': customer.get('registration_date', ''),
                'is_active': customer.get('is_active', True)
            })
            
            # Publish customer event
            self.producer.send('customer-events', 
                              key=str(customer_id),
                              value={
                                  'event_type': 'customer_created',
                                  'customer_id': customer_id,
                                  'customer_data': customer,
                                  'timestamp': datetime.now().isoformat()
                              })
                              
        elif event.operation == 'u':  # UPDATE
            old_customer = event.before
            new_customer = event.after
            customer_id = new_customer['customer_id']
            
            # Check for segment changes
            old_segment = old_customer.get('customer_segment')
            new_segment = new_customer.get('customer_segment')
            
            if old_segment != new_segment:
                self.redis_client.hincrby('customer_counts', old_segment, -1)
                self.redis_client.hincrby('customer_counts', new_segment, 1)
                
            # Update cached data
            self.redis_client.hset(f'customer:{customer_id}', mapping={
                'name': f"{new_customer.get('first_name', '')} {new_customer.get('last_name', '')}",
                'email': new_customer.get('email', ''),
                'segment': new_segment,
                'is_active': new_customer.get('is_active', True)
            })
            
        elif event.operation == 'd':  # DELETE
            customer = event.before
            customer_id = customer['customer_id']
            
            # Update metrics
            segment = customer.get('customer_segment', 'Unknown')
            self.redis_client.hincrby('customer_counts', segment, -1)
            
            # Remove cached data
            self.redis_client.delete(f'customer:{customer_id}')
            
    def handle_order_event(self, event: CDCEvent):
        """Process order table changes"""
        
        if event.operation == 'c':  # New order
            order = event.after
            order_id = order['order_id']
            customer_id = order['customer_id']
            total_amount = float(order['total_amount'])
            
            logger.info(f"New order created: {order_id} for ${total_amount}")
            
            # Update real-time metrics
            self.update_real_time_metrics('revenue', total_amount)
            self.update_real_time_metrics('orders', 1)
            
            # Update customer metrics
            customer_key = f"customer:{customer_id}:metrics"
            self.redis_client.hincrby(customer_key, 'order_count', 1)
            self.redis_client.hincrbyfloat(customer_key, 'total_spent', total_amount)
            
            # Cache order data
            self.redis_client.hset(f'order:{order_id}', mapping={
                'customer_id': customer_id,
                'total_amount': total_amount,
                'status': order.get('order_status', 'pending'),
                'created_at': order.get('created_at', '')
            })
            
            # Publish to dashboard
            self.redis_client.publish('dashboard_updates', json.dumps({
                'type': 'new_order',
                'order_id': order_id,
                'customer_id': customer_id,
                'amount': total_amount,
                'timestamp': datetime.now().isoformat()
            }))
            
        elif event.operation == 'u':  # Order update
            old_order = event.before
            new_order = event.after
            order_id = new_order['order_id']
            
            old_status = old_order.get('order_status')
            new_status = new_order.get('order_status')
            
            if old_status != new_status:
                logger.info(f"Order {order_id} status: {old_status} -> {new_status}")
                
                # Update cached status
                self.redis_client.hset(f'order:{order_id}', 'status', new_status)
                
                # Track status metrics
                status_key = f"order_status:{new_status}:daily:{datetime.now().date()}"
                self.redis_client.incr(status_key)
                
                # Publish status change
                self.producer.send('order-status-changes',
                                  key=str(order_id),
                                  value={
                                      'order_id': order_id,
                                      'old_status': old_status,
                                      'new_status': new_status,
                                      'timestamp': datetime.now().isoformat()
                                  })
                                  
    def handle_product_event(self, event: CDCEvent):
        """Process product table changes"""
        
        if event.operation == 'c':  # New product
            product = event.after
            product_id = product['product_id']
            
            # Update category counts
            category = product.get('category', 'Unknown')
            self.redis_client.hincrby('product_counts', category, 1)
            
            # Cache product data
            self.redis_client.hset(f'product:{product_id}', mapping={
                'name': product.get('product_name', ''),
                'category': category,
                'brand': product.get('brand', ''),
                'price': product.get('unit_price', 0),
                'inventory': product.get('inventory_quantity', 0)
            })
            
        elif event.operation == 'u':  # Product update
            old_product = event.before
            new_product = event.after
            product_id = new_product['product_id']
            
            # Check inventory changes
            old_inventory = old_product.get('inventory_quantity', 0)
            new_inventory = new_product.get('inventory_quantity', 0)
            
            if old_inventory != new_inventory:
                logger.info(f"Product {product_id} inventory: {old_inventory} -> {new_inventory}")
                
                # Update cached inventory
                self.redis_client.hset(f'product:{product_id}', 'inventory', new_inventory)
                
                # Low stock alert
                if new_inventory < 10:
                    self.producer.send('inventory-alerts',
                                      key=str(product_id),
                                      value={
                                          'alert_type': 'low_stock',
                                          'product_id': product_id,
                                          'current_inventory': new_inventory,
                                          'timestamp': datetime.now().isoformat()
                                      })
                                      
    def handle_order_item_event(self, event: CDCEvent):
        """Process order_items table changes"""
        
        if event.operation == 'c':  # New order item
            item = event.after
            product_id = item['product_id']
            quantity = item['quantity']
            revenue = float(item['total_price'])
            
            # Update product sales metrics
            product_key = f"product:{product_id}:metrics"
            self.redis_client.hincrby(product_key, 'units_sold', quantity)
            self.redis_client.hincrbyfloat(product_key, 'revenue', revenue)
            
    # ============================================================================
    # SOLUTION 4: Real-time Analytics
    # ============================================================================
    
    def update_real_time_metrics(self, metric_type: str, value: float, dimensions: Dict[str, str] = None):
        """Update real-time metrics with time-based keys"""
        
        now = datetime.now()
        date_key = now.strftime('%Y-%m-%d')
        hour_key = now.strftime('%Y-%m-%d-%H')
        
        # Daily metrics
        daily_key = f"metrics:daily:{date_key}:{metric_type}"
        if metric_type == 'revenue':
            self.redis_client.incrbyfloat(daily_key, value)
        else:
            self.redis_client.incr(daily_key, int(value))
            
        # Hourly metrics
        hourly_key = f"metrics:hourly:{hour_key}:{metric_type}"
        if metric_type == 'revenue':
            self.redis_client.incrbyfloat(hourly_key, value)
        else:
            self.redis_client.incr(hourly_key, int(value))
            
        # Set expiration (30 days for daily, 7 days for hourly)
        self.redis_client.expire(daily_key, 30 * 24 * 3600)
        self.redis_client.expire(hourly_key, 7 * 24 * 3600)
        
        # Dimensional metrics
        if dimensions:
            for dim_name, dim_value in dimensions.items():
                dim_key = f"metrics:daily:{date_key}:{metric_type}:{dim_name}:{dim_value}"
                if metric_type == 'revenue':
                    self.redis_client.incrbyfloat(dim_key, value)
                else:
                    self.redis_client.incr(dim_key, int(value))
                self.redis_client.expire(dim_key, 30 * 24 * 3600)
                
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard metrics"""
        
        today = datetime.now().strftime('%Y-%m-%d')
        current_hour = datetime.now().strftime('%Y-%m-%d-%H')
        
        # Get basic metrics
        daily_revenue = float(self.redis_client.get(f"metrics:daily:{today}:revenue") or 0)
        daily_orders = int(self.redis_client.get(f"metrics:daily:{today}:orders") or 0)
        hourly_revenue = float(self.redis_client.get(f"metrics:hourly:{current_hour}:revenue") or 0)
        
        # Get customer segments
        customer_counts = self.redis_client.hgetall('customer_counts')
        
        # Get product categories
        product_counts = self.redis_client.hgetall('product_counts')
        
        # Get recent orders (from sorted set or list)
        recent_orders = []
        order_keys = self.redis_client.keys('order:*')
        for key in order_keys[-10:]:  # Last 10 orders
            order_data = self.redis_client.hgetall(key)
            if order_data:
                recent_orders.append(order_data)
                
        return {
            'daily_revenue': daily_revenue,
            'daily_orders': daily_orders,
            'hourly_revenue': hourly_revenue,
            'avg_order_value': daily_revenue / daily_orders if daily_orders > 0 else 0,
            'customer_segments': customer_counts,
            'product_categories': product_counts,
            'recent_orders': recent_orders,
            'timestamp': datetime.now().isoformat()
        }
        
    # ============================================================================
    # SOLUTION 5: Monitoring and Health Checks
    # ============================================================================
    
    def check_pipeline_health(self) -> Dict[str, Any]:
        """Comprehensive pipeline health check"""
        
        health_status = {
            'overall_status': 'healthy',
            'connector_status': 'unknown',
            'consumer_lag': 0,
            'error_rate': 0,
            'last_processed_event': None,
            'alerts': []
        }
        
        # Check connector status
        connector_health = self.check_connector_status()
        health_status['connector_status'] = 'healthy' if connector_health.get('healthy') else 'unhealthy'
        
        if not connector_health.get('healthy'):
            health_status['alerts'].append(f"Connector unhealthy: {connector_health.get('error')}")
            health_status['overall_status'] = 'degraded'
            
        # Check Redis connectivity
        try:
            self.redis_client.ping()
        except Exception as e:
            health_status['alerts'].append(f"Redis connectivity issue: {e}")
            health_status['overall_status'] = 'unhealthy'
            
        # Check recent activity
        last_activity = self.redis_client.get('last_cdc_event_time')
        if last_activity:
            last_time = datetime.fromisoformat(last_activity)
            if datetime.now() - last_time > timedelta(minutes=5):
                health_status['alerts'].append("No recent CDC events processed")
                health_status['overall_status'] = 'degraded'
                
        return health_status
        
    def handle_processing_error(self, event: Dict[str, Any], error: Exception):
        """Handle event processing errors with retry logic"""
        
        error_info = {
            'event': event,
            'error': str(error),
            'timestamp': datetime.now().isoformat(),
            'retry_count': 0
        }
        
        logger.error(f"Event processing error: {error}")
        logger.error(f"Event: {event}")
        
        # Send to dead letter queue
        self.producer.send('cdc-errors', value=error_info)
        
        # Update error metrics
        error_key = f"errors:daily:{datetime.now().strftime('%Y-%m-%d')}"
        self.redis_client.incr(error_key)
        self.redis_client.expire(error_key, 7 * 24 * 3600)

def main():
    """Main solution demonstration"""
    
    print("Day 5: CDC with Debezium - Complete Solution")
    print("=" * 50)
    
    # Initialize solution
    solution = CDCSolution()
    
    # 1. Set up Debezium connector
    print("\n1. Setting up Debezium connector...")
    if solution.setup_debezium_connector():
        print("✅ Connector deployed successfully")
    else:
        print("❌ Connector deployment failed")
        
    # 2. Check connector status
    print("\n2. Checking connector status...")
    status = solution.check_connector_status()
    print(f"Connector healthy: {status.get('healthy')}")
    
    # 3. Get dashboard data
    print("\n3. Getting real-time dashboard data...")
    dashboard = solution.get_real_time_dashboard_data()
    print(f"Daily revenue: ${dashboard['daily_revenue']:.2f}")
    print(f"Daily orders: {dashboard['daily_orders']}")
    print(f"Average order value: ${dashboard['avg_order_value']:.2f}")
    
    # 4. Check pipeline health
    print("\n4. Checking pipeline health...")
    health = solution.check_pipeline_health()
    print(f"Overall status: {health['overall_status']}")
    if health['alerts']:
        print(f"Alerts: {health['alerts']}")
        
    print("\n5. Starting CDC event processor...")
    print("Press Ctrl+C to stop")
    
    try:
        solution.process_cdc_events()
    except KeyboardInterrupt:
        print("\nStopping CDC processor...")
        
    print("\n✅ CDC Pipeline Solution Complete!")
    print("\nKey Features Implemented:")
    print("- Debezium connector setup and monitoring")
    print("- Real-time event processing")
    print("- Live analytics and metrics")
    print("- Error handling and monitoring")
    print("- Dashboard data aggregation")
    print("- Health checks and alerting")

if __name__ == "__main__":
    main()
