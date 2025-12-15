#!/bin/bash

# Day 30: Ensemble Methods - Setup Script
# Automated setup for production ensemble learning environment

set -e  # Exit on any error

echo "🚀 Setting up Day 30: Ensemble Methods Environment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if command -v docker &> /dev/null; then
        print_success "Docker is installed"
        docker --version
    else
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
}

# Check if Docker Compose is installed
check_docker_compose() {
    print_status "Checking Docker Compose installation..."
    if command -v docker-compose &> /dev/null; then
        print_success "Docker Compose is installed"
        docker-compose --version
    else
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    directories=(
        "data"
        "models"
        "logs"
        "grafana/dashboards"
        "grafana/datasources"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_warning "Directory already exists: $dir"
        fi
    done
}

# Create Prometheus configuration
create_prometheus_config() {
    print_status "Creating Prometheus configuration..."
    
    cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'ensemble-api'
    static_configs:
      - targets: ['ensemble-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF
    
    print_success "Created prometheus.yml"
}

# Create Grafana datasource configuration
create_grafana_datasource() {
    print_status "Creating Grafana datasource configuration..."
    
    cat > grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    
    print_success "Created Grafana datasource configuration"
}

# Create Grafana dashboard
create_grafana_dashboard() {
    print_status "Creating Grafana dashboard configuration..."
    
    cat > grafana/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
    
    # Create a simple dashboard JSON
    cat > grafana/dashboards/ensemble-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Ensemble Methods Dashboard",
    "tags": ["ensemble", "ml"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Prediction Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ensemble_predictions_total[5m])",
            "legendFormat": "Predictions/sec"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Prediction Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, ensemble_prediction_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      }
    ],
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "5s"
  }
}
EOF
    
    print_success "Created Grafana dashboard"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    if command -v python3 &> /dev/null; then
        if [ -f "requirements.txt" ]; then
            python3 -m pip install -r requirements.txt
            print_success "Python dependencies installed"
        else
            print_warning "requirements.txt not found, skipping Python dependencies"
        fi
    else
        print_warning "Python3 not found, skipping Python dependencies"
    fi
}

# Start Docker services
start_services() {
    print_status "Starting Docker services..."
    
    # Pull images first
    print_status "Pulling Docker images..."
    docker-compose pull
    
    # Start services
    print_status "Starting services in background..."
    docker-compose up -d
    
    print_success "Docker services started"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    services=(
        "postgres:5432"
        "redis:6379"
        "mlflow:5000"
    )
    
    for service in "${services[@]}"; do
        host=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        print_status "Waiting for $host:$port..."
        
        timeout=60
        while ! nc -z localhost $port 2>/dev/null; do
            sleep 2
            timeout=$((timeout - 2))
            if [ $timeout -le 0 ]; then
                print_warning "Timeout waiting for $host:$port"
                break
            fi
        done
        
        if nc -z localhost $port 2>/dev/null; then
            print_success "$host:$port is ready"
        fi
    done
}

# Generate sample data
generate_data() {
    print_status "Generating sample data..."
    
    if [ -f "data_generator.py" ]; then
        python3 data_generator.py
        print_success "Sample data generated"
    else
        print_warning "data_generator.py not found, skipping data generation"
    fi
}

# Train initial models
train_models() {
    print_status "Training initial models..."
    
    if [ -f "model_trainer.py" ]; then
        python3 model_trainer.py
        print_success "Initial models trained"
    else
        print_warning "model_trainer.py not found, skipping model training"
    fi
}

# Run health checks
health_check() {
    print_status "Running health checks..."
    
    # Check API health
    if curl -f http://localhost:8000/health &>/dev/null; then
        print_success "API server is healthy"
    else
        print_warning "API server health check failed"
    fi
    
    # Check MLflow
    if curl -f http://localhost:5000/health &>/dev/null; then
        print_success "MLflow is healthy"
    else
        print_warning "MLflow health check failed"
    fi
    
    # Check Prometheus
    if curl -f http://localhost:9090/-/healthy &>/dev/null; then
        print_success "Prometheus is healthy"
    else
        print_warning "Prometheus health check failed"
    fi
    
    # Check Grafana
    if curl -f http://localhost:3000/api/health &>/dev/null; then
        print_success "Grafana is healthy"
    else
        print_warning "Grafana health check failed"
    fi
}

# Display service URLs
show_urls() {
    echo ""
    echo "🎉 Setup completed! Access your services:"
    echo "=========================================="
    echo "📊 API Server:     http://localhost:8000"
    echo "📈 MLflow:         http://localhost:5000"
    echo "📊 Prometheus:     http://localhost:9090"
    echo "📈 Grafana:        http://localhost:3000 (admin/admin)"
    echo "🗄️  PostgreSQL:    localhost:5432 (ensemble_user/ensemble_pass)"
    echo "🔄 Redis:          localhost:6379"
    echo ""
    echo "📚 Quick Start Commands:"
    echo "========================"
    echo "# Test API:"
    echo "curl http://localhost:8000/health"
    echo ""
    echo "# Run demo:"
    echo "python3 demo.py"
    echo ""
    echo "# View logs:"
    echo "docker-compose logs -f ensemble-api"
    echo ""
    echo "# Stop services:"
    echo "docker-compose down"
    echo ""
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    docker-compose down
    print_success "Cleanup completed"
}

# Main setup function
main() {
    echo "Starting setup process..."
    
    # Check prerequisites
    check_docker
    check_docker_compose
    
    # Create directories and configs
    create_directories
    create_prometheus_config
    create_grafana_datasource
    create_grafana_dashboard
    
    # Install dependencies
    install_python_deps
    
    # Start services
    start_services
    
    # Wait for services
    wait_for_services
    
    # Generate data and train models
    sleep 10  # Give services more time to fully start
    generate_data
    train_models
    
    # Health checks
    sleep 5
    health_check
    
    # Show URLs
    show_urls
    
    print_success "🎉 Day 30 Ensemble Methods environment is ready!"
}

# Handle script interruption
trap cleanup EXIT

# Parse command line arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "start")
        print_status "Starting services..."
        docker-compose up -d
        wait_for_services
        health_check
        show_urls
        ;;
    "stop")
        print_status "Stopping services..."
        docker-compose down
        print_success "Services stopped"
        ;;
    "restart")
        print_status "Restarting services..."
        docker-compose restart
        wait_for_services
        health_check
        show_urls
        ;;
    "clean")
        print_status "Cleaning up everything..."
        docker-compose down -v
        docker system prune -f
        print_success "Cleanup completed"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        health_check
        ;;
    *)
        echo "Usage: $0 {setup|start|stop|restart|clean|logs|status}"
        echo ""
        echo "Commands:"
        echo "  setup   - Full setup (default)"
        echo "  start   - Start services"
        echo "  stop    - Stop services"
        echo "  restart - Restart services"
        echo "  clean   - Clean up everything"
        echo "  logs    - Show logs"
        echo "  status  - Show status"
        exit 1
        ;;
esac