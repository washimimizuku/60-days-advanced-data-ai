#!/usr/bin/env python3
"""
Data Generator for CDC Pipeline Testing

This script generates realistic e-commerce data to test the CDC pipeline:
- Creates new users, products, and orders
- Updates existing records to trigger CDC events
- Simulates realistic business patterns and seasonality
- Provides metrics for monitoring data generation
"""

import os
import time
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from faker import Faker
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
events_generated = Counter('data_generator_events_total', 'Total events generated', ['event_type'])
generation_duration = Histogram('data_generator_duration_seconds', 'Time spent generating data')
active_users = Gauge('data_generator_active_users', 'Number of active users')
total_orders = Gauge('data_generator_total_orders', 'Total number of orders')
inventory_levels = Gauge('data_generator_inventory_levels', 'Product inventory levels', ['product_id'])

fake = Faker()

class DataGenerator:
    """Generates realistic e-commerce data for CDC pipeline testing"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'database': os.getenv('POSTGRES_DB', 'ecommerce'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'sslmode': os.getenv('POSTGRES_SSLMODE', 'prefer'),
            'connect_timeout': int(os.getenv('POSTGRES_TIMEOUT', '10'))
        }
        
        self.generation_rate = int(os.getenv('GENERATION_RATE', '10'))  # Events per second
        self.batch_size = int(os.getenv('BATCH_SIZE', '5'))
        
        # Product categories and their characteristics
        self.categories = {
            'Electronics': {'price_range': (50, 2000), 'popularity': 0.3},
            'Clothing': {'price_range': (20, 200), 'popularity': 0.25},
            'Books': {'price_range': (10, 50), 'popularity': 0.15},
            'Home': {'price_range': (25, 500), 'popularity': 0.2},
            'Sports': {'price_range': (15, 300), 'popularity': 0.1}
        }
        
        # Order status transitions
        self.status_transitions = {
            'pending': ['processing', 'cancelled'],
            'processing': ['completed', 'cancelled'],
            'completed': [],  # Terminal state
            'cancelled': []   # Terminal state
        }
        
        self.conn = None
        self.user_ids = []
        self.product_ids = []
        
    def connect_db(self):
        """Establish database connection with retry logic"""
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.conn = psycopg2.connect(**self.db_config)
                self.conn.autocommit = True
                logger.info("Connected to database successfully")
                return
            except psycopg2.Error as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
    
    def load_existing_data(self):
        """Load existing user and product IDs for realistic data generation"""
        try:
            with self.conn.cursor() as cursor:
                # Load user IDs
                cursor.execute("SELECT user_id FROM users WHERE status = 'active'")
                self.user_ids = [row[0] for row in cursor.fetchall()]
                
                # Load product IDs
                cursor.execute("SELECT product_id FROM products WHERE stock_quantity > 0")
                self.product_ids = [row[0] for row in cursor.fetchall()]
                
                logger.info(f"Loaded {len(self.user_ids)} users and {len(self.product_ids)} products")
                
        except psycopg2.Error as e:
            logger.error(f"Failed to load existing data: {e}")
            raise
    
    def generate_user(self) -> Dict[str, Any]:
        """Generate a new user"""
        return {
            'email': fake.email(),
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'status': random.choices(['active', 'inactive'], weights=[0.9, 0.1])[0]
        }
    
    def generate_product(self) -> Dict[str, Any]:
        """Generate a new product"""
        category = random.choices(
            list(self.categories.keys()),
            weights=[info['popularity'] for info in self.categories.values()]
        )[0]
        
        price_range = self.categories[category]['price_range']
        
        return {
            'name': fake.catch_phrase(),
            'category': category,
            'price': round(random.uniform(*price_range), 2),
            'stock_quantity': random.randint(10, 500),
            'min_threshold': random.randint(5, 20)
        }
    
    def generate_order(self) -> Dict[str, Any]:
        """Generate a new order with items"""
        if not self.user_ids or not self.product_ids:
            return None
            
        user_id = random.choice(self.user_ids)
        num_items = random.choices([1, 2, 3, 4, 5], weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]
        
        # Select random products for the order
        order_products = random.sample(self.product_ids, min(num_items, len(self.product_ids)))
        
        total_amount = 0
        order_items = []
        
        for product_id in order_products:
            quantity = random.randint(1, 3)
            # Get product price (simplified - in real scenario, fetch from DB)
            unit_price = round(random.uniform(10, 200), 2)
            total_amount += quantity * unit_price
            
            order_items.append({
                'product_id': product_id,
                'quantity': quantity,
                'unit_price': unit_price
            })
        
        return {
            'user_id': user_id,
            'total_amount': round(total_amount, 2),
            'status': 'pending',
            'items': order_items
        }
    
    def insert_user(self, user_data: Dict[str, Any]) -> int:
        """Insert a new user and return user_id"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO users (email, first_name, last_name, status)
                    VALUES (%(email)s, %(first_name)s, %(last_name)s, %(status)s)
                    RETURNING user_id
                """, user_data)
                
                user_id = cursor.fetchone()[0]
                self.user_ids.append(user_id)
                events_generated.labels(event_type='user_insert').inc()
                
                return user_id
                
        except psycopg2.Error as e:
            logger.error(f"Failed to insert user: {e}")
            return None
    
    def insert_product(self, product_data: Dict[str, Any]) -> int:
        """Insert a new product and return product_id"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO products (name, category, price, stock_quantity, min_threshold)
                    VALUES (%(name)s, %(category)s, %(price)s, %(stock_quantity)s, %(min_threshold)s)
                    RETURNING product_id
                """, product_data)
                
                product_id = cursor.fetchone()[0]
                self.product_ids.append(product_id)
                events_generated.labels(event_type='product_insert').inc()
                
                return product_id
                
        except psycopg2.Error as e:
            logger.error(f"Failed to insert product: {e}")
            return None
    
    def insert_order(self, order_data: Dict[str, Any]) -> int:
        """Insert a new order with items and return order_id"""
        try:
            with self.conn.cursor() as cursor:
                # Insert order
                cursor.execute("""
                    INSERT INTO orders (user_id, total_amount, status)
                    VALUES (%(user_id)s, %(total_amount)s, %(status)s)
                    RETURNING order_id
                """, {
                    'user_id': order_data['user_id'],
                    'total_amount': order_data['total_amount'],
                    'status': order_data['status']
                })
                
                order_id = cursor.fetchone()[0]
                
                # Insert order items
                for item in order_data['items']:
                    cursor.execute("""
                        INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                        VALUES (%s, %s, %s, %s)
                    """, (order_id, item['product_id'], item['quantity'], item['unit_price']))
                
                events_generated.labels(event_type='order_insert').inc()
                return order_id
                
        except psycopg2.Error as e:
            logger.error(f"Failed to insert order: {e}")
            return None
    
    def update_order_status(self):
        """Update random order statuses to simulate order processing"""
        try:
            with self.conn.cursor() as cursor:
                # Get orders that can be updated
                cursor.execute("""
                    SELECT order_id, status FROM orders 
                    WHERE status IN ('pending', 'processing')
                    ORDER BY RANDOM() LIMIT 5
                """)
                
                orders = cursor.fetchall()
                
                for order_id, current_status in orders:
                    possible_statuses = self.status_transitions.get(current_status, [])
                    if possible_statuses:
                        new_status = random.choice(possible_statuses)
                        
                        cursor.execute("""
                            UPDATE orders 
                            SET status = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE order_id = %s
                        """, (new_status, order_id))
                        
                        events_generated.labels(event_type='order_update').inc()
                        
        except psycopg2.Error as e:
            logger.error(f"Failed to update order status: {e}")
    
    def update_inventory(self):
        """Update product inventory to simulate sales and restocking"""
        try:
            with self.conn.cursor() as cursor:
                # Simulate inventory depletion (sales)
                cursor.execute("""
                    UPDATE products 
                    SET stock_quantity = GREATEST(0, stock_quantity - %s),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE product_id = %s AND stock_quantity > 0
                """, (random.randint(1, 5), random.choice(self.product_ids)))
                
                # Occasionally restock products
                if random.random() < 0.1:  # 10% chance
                    cursor.execute("""
                        UPDATE products 
                        SET stock_quantity = stock_quantity + %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE product_id = %s
                    """, (random.randint(50, 200), random.choice(self.product_ids)))
                
                events_generated.labels(event_type='inventory_update').inc()
                
        except psycopg2.Error as e:
            logger.error(f"Failed to update inventory: {e}")
    
    def update_metrics(self):
        """Update Prometheus metrics"""
        try:
            with self.conn.cursor() as cursor:
                # Count active users
                cursor.execute("SELECT COUNT(*) FROM users WHERE status = 'active'")
                active_users.set(cursor.fetchone()[0])
                
                # Count total orders
                cursor.execute("SELECT COUNT(*) FROM orders")
                total_orders.set(cursor.fetchone()[0])
                
                # Sample inventory levels for a few products
                cursor.execute("""
                    SELECT product_id, stock_quantity 
                    FROM products 
                    ORDER BY RANDOM() LIMIT 10
                """)
                
                for product_id, stock in cursor.fetchall():
                    inventory_levels.labels(product_id=str(product_id)).set(stock)
                    
        except psycopg2.Error as e:
            logger.error(f"Failed to update metrics: {e}")
    
    @generation_duration.time()
    def generate_batch(self):
        """Generate a batch of events"""
        events = []
        
        # Determine event types for this batch
        event_types = random.choices(
            ['user', 'product', 'order', 'order_update', 'inventory_update'],
            weights=[0.1, 0.05, 0.4, 0.25, 0.2],
            k=self.batch_size
        )
        
        for event_type in event_types:
            try:
                if event_type == 'user':
                    user_data = self.generate_user()
                    self.insert_user(user_data)
                    
                elif event_type == 'product':
                    product_data = self.generate_product()
                    self.insert_product(product_data)
                    
                elif event_type == 'order':
                    order_data = self.generate_order()
                    if order_data:
                        self.insert_order(order_data)
                    
                elif event_type == 'order_update':
                    self.update_order_status()
                    
                elif event_type == 'inventory_update':
                    self.update_inventory()
                    
            except Exception as e:
                logger.error(f"Failed to generate {event_type} event: {e}")
    
    def run(self):
        """Main execution loop"""
        logger.info("Starting data generator...")
        
        # Start Prometheus metrics server
        start_http_server(8000)
        logger.info("Prometheus metrics server started on port 8000")
        
        # Connect to database
        self.connect_db()
        
        # Load existing data
        self.load_existing_data()
        
        # Main generation loop
        try:
            while True:
                start_time = time.time()
                
                # Generate batch of events
                self.generate_batch()
                
                # Update metrics periodically
                if random.random() < 0.1:  # 10% chance
                    self.update_metrics()
                
                # Calculate sleep time to maintain target rate
                elapsed = time.time() - start_time
                sleep_time = max(0, (1.0 / self.generation_rate) - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Log progress periodically
                if random.random() < 0.01:  # 1% chance
                    logger.info(f"Generated batch in {elapsed:.3f}s, sleeping {sleep_time:.3f}s")
                    
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()
                logger.info("Database connection closed")

if __name__ == "__main__":
    generator = DataGenerator()
    generator.run()