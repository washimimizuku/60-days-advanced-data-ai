#!/bin/bash
# Day 5: CDC with Debezium - Setup Script

set -e

echo "🚀 Setting up CDC with Debezium Exercise Environment"
echo "=================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Create necessary directories
mkdir -p logs
mkdir -p data

echo "📁 Created necessary directories"

# Start the infrastructure
echo "🐳 Starting infrastructure services..."
docker-compose up -d zookeeper kafka postgres redis

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check if Kafka is ready
echo "🔍 Checking Kafka connectivity..."
timeout 60 bash -c 'until docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1; do sleep 2; done'

# Check if PostgreSQL is ready
echo "🔍 Checking PostgreSQL connectivity..."
timeout 60 bash -c 'until docker-compose exec postgres pg_isready -U postgres > /dev/null 2>&1; do sleep 2; done'

# Start Kafka Connect
echo "🔗 Starting Kafka Connect with Debezium..."
docker-compose up -d connect

# Wait for Kafka Connect to be ready
echo "⏳ Waiting for Kafka Connect to start..."
sleep 45

# Check if Kafka Connect is ready
echo "🔍 Checking Kafka Connect connectivity..."
timeout 120 bash -c 'until curl -s http://localhost:8083/connectors > /dev/null 2>&1; do sleep 5; done'

# Start Kafka UI
echo "🖥️  Starting Kafka UI..."
docker-compose up -d kafka-ui

# Deploy the PostgreSQL connector
echo "🔌 Deploying PostgreSQL Debezium connector..."
sleep 10

curl -X POST http://localhost:8083/connectors \
  -H "Content-Type: application/json" \
  -d @connectors/postgres-connector.json

# Check connector status
echo "📊 Checking connector status..."
sleep 10
curl -s http://localhost:8083/connectors/postgres-connector/status | jq '.'

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if command -v python3 &> /dev/null; then
    python3 -m pip install -r requirements.txt
elif command -v python &> /dev/null; then
    python -m pip install -r requirements.txt
else
    echo "⚠️  Python not found. Please install Python dependencies manually:"
    echo "   pip install -r requirements.txt"
fi

# Create sample data generation script
cat > generate_sample_data.py << 'EOF'
#!/usr/bin/env python3
"""Generate sample data to test CDC pipeline"""

import psycopg2
import random
import time
from datetime import datetime, timedelta

def connect_db():
    return psycopg2.connect(
        host='localhost',
        port=5432,
        database='ecommerce',
        user='postgres',
        password='postgres'
    )

def generate_orders():
    """Generate sample orders to trigger CDC events"""
    conn = connect_db()
    cursor = conn.cursor()
    
    # Get random customer and product IDs
    cursor.execute("SELECT customer_id FROM customers ORDER BY RANDOM() LIMIT 1")
    customer_id = cursor.fetchone()[0]
    
    cursor.execute("SELECT product_id, unit_price FROM products WHERE is_active = true ORDER BY RANDOM() LIMIT 3")
    products = cursor.fetchall()
    
    # Create order
    total_amount = sum(price * random.randint(1, 3) for _, price in products)
    
    cursor.execute("""
        INSERT INTO orders (customer_id, order_status, total_amount, tax_amount, shipping_amount, payment_method)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING order_id
    """, (customer_id, 'pending', total_amount, total_amount * 0.08, 9.99, 'Credit Card'))
    
    order_id = cursor.fetchone()[0]
    
    # Create order items
    for product_id, unit_price in products:
        quantity = random.randint(1, 3)
        cursor.execute("""
            INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price)
            VALUES (%s, %s, %s, %s, %s)
        """, (order_id, product_id, quantity, unit_price, unit_price * quantity))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"Created order {order_id} for customer {customer_id}")
    return order_id

def update_order_status(order_id):
    """Update order status to trigger CDC events"""
    conn = connect_db()
    cursor = conn.cursor()
    
    statuses = ['processing', 'shipped', 'completed']
    new_status = random.choice(statuses)
    
    cursor.execute("""
        UPDATE orders SET order_status = %s, updated_at = CURRENT_TIMESTAMP
        WHERE order_id = %s
    """, (new_status, order_id))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"Updated order {order_id} status to {new_status}")

def generate_user_events():
    """Generate user events"""
    conn = connect_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT customer_id FROM customers ORDER BY RANDOM() LIMIT 1")
    user_id = cursor.fetchone()[0]
    
    events = [
        ('page_view', {'page': '/products', 'referrer': 'google.com'}),
        ('search', {'query': 'wireless headphones', 'results': 15}),
        ('add_to_cart', {'product_id': random.randint(1, 10), 'quantity': 1}),
        ('checkout_start', {'cart_total': random.uniform(50, 500)})
    ]
    
    event_type, event_data = random.choice(events)
    
    cursor.execute("""
        INSERT INTO user_events (user_id, session_id, event_type, event_data, page_url)
        VALUES (%s, %s, %s, %s, %s)
    """, (user_id, f'sess_{random.randint(1000, 9999)}', event_type, event_data, '/test'))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"Generated {event_type} event for user {user_id}")

if __name__ == "__main__":
    print("Generating sample data to test CDC pipeline...")
    
    for i in range(5):
        try:
            order_id = generate_orders()
            time.sleep(2)
            
            if random.random() > 0.5:
                update_order_status(order_id)
                time.sleep(1)
            
            generate_user_events()
            time.sleep(3)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
    
    print("Sample data generation completed!")
EOF

chmod +x generate_sample_data.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Check services status:"
echo "   docker-compose ps"
echo ""
echo "2. View Kafka topics:"
echo "   docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list"
echo ""
echo "3. Monitor CDC events:"
echo "   docker-compose exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic ecommerce-db.public.orders --from-beginning"
echo ""
echo "4. Start the CDC processor:"
echo "   python3 cdc_processor.py"
echo ""
echo "5. Generate sample data:"
echo "   python3 generate_sample_data.py"
echo ""
echo "6. Access Kafka UI:"
echo "   http://localhost:8080"
echo ""
echo "7. Access Redis CLI:"
echo "   docker-compose exec redis redis-cli"
echo ""
echo "🔧 Troubleshooting:"
echo "- Check logs: docker-compose logs [service_name]"
echo "- Restart connector: curl -X POST http://localhost:8083/connectors/postgres-connector/restart"
echo "- Check connector status: curl http://localhost:8083/connectors/postgres-connector/status"