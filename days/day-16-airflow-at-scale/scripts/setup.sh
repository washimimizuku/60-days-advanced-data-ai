#!/bin/bash

# Day 16: Airflow at Scale - Setup Script
# Production deployment setup for ScaleCorp

set -e

echo "🚀 Setting up Airflow at Scale for ScaleCorp..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
        exit 1
    fi
    
    # Check available resources
    AVAILABLE_MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')
    if [ "$AVAILABLE_MEMORY" -lt 8 ]; then
        echo -e "${YELLOW}Warning: Less than 8GB of available memory. Airflow at scale requires significant resources.${NC}"
    fi
    
    echo -e "${GREEN}Prerequisites check completed.${NC}"
}

# Setup environment
setup_environment() {
    echo -e "${YELLOW}Setting up environment...${NC}"
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        echo "Creating .env file from template..."
        cp .env.example .env
        
        # Generate secure passwords
        POSTGRES_PASSWORD=$(openssl rand -base64 32)
        REPLICATION_PASSWORD=$(openssl rand -base64 32)
        ADMIN_PASSWORD=$(openssl rand -base64 16)
        GRAFANA_PASSWORD=$(openssl rand -base64 16)
        
        # Update .env file
        sed -i "s/your_secure_postgres_password/$POSTGRES_PASSWORD/g" .env
        sed -i "s/your_secure_replication_password/$REPLICATION_PASSWORD/g" .env
        sed -i "s/your_secure_admin_password/$ADMIN_PASSWORD/g" .env
        sed -i "s/your_secure_grafana_password/$GRAFANA_PASSWORD/g" .env
        
        echo -e "${GREEN}Environment file created with secure passwords.${NC}"
        echo -e "${YELLOW}Please review and customize .env file before proceeding.${NC}"
    fi
    
    # Set proper permissions
    chmod 755 scripts/*.sh
    
    # Create necessary directories
    mkdir -p logs dags plugins config/certs
    
    # Set Airflow UID
    echo "AIRFLOW_UID=$(id -u)" >> .env
    
    echo -e "${GREEN}Environment setup completed.${NC}"
}

# Initialize Airflow
initialize_airflow() {
    echo -e "${YELLOW}Initializing Airflow...${NC}"
    
    # Initialize database and create admin user
    docker-compose up airflow-init
    
    echo -e "${GREEN}Airflow initialization completed.${NC}"
}

# Start services
start_services() {
    echo -e "${YELLOW}Starting services...${NC}"
    
    # Start infrastructure services first
    echo "Starting infrastructure services..."
    docker-compose up -d redis-master redis-replica postgres-primary postgres-replica
    
    # Wait for infrastructure to be ready
    echo "Waiting for infrastructure services to be ready..."
    sleep 30
    
    # Start monitoring services
    echo "Starting monitoring services..."
    docker-compose up -d prometheus prometheus-statsd-exporter grafana alertmanager
    
    # Start Airflow services
    echo "Starting Airflow services..."
    docker-compose up -d airflow-scheduler-1 airflow-scheduler-2 airflow-webserver-1 airflow-webserver-2 airflow-worker airflow-flower
    
    # Start load balancer
    echo "Starting load balancer..."
    docker-compose up -d nginx
    
    echo -e "${GREEN}All services started successfully.${NC}"
}

# Health check
health_check() {
    echo -e "${YELLOW}Performing health check...${NC}"
    
    # Wait for services to be ready
    sleep 60
    
    # Check Airflow webserver
    if curl -f http://localhost:80/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Airflow webserver is healthy${NC}"
    else
        echo -e "${RED}✗ Airflow webserver is not responding${NC}"
    fi
    
    # Check Prometheus
    if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Prometheus is healthy${NC}"
    else
        echo -e "${RED}✗ Prometheus is not responding${NC}"
    fi
    
    # Check Grafana
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Grafana is healthy${NC}"
    else
        echo -e "${RED}✗ Grafana is not responding${NC}"
    fi
    
    # Check Flower
    if curl -f http://localhost:5555/ > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Flower is healthy${NC}"
    else
        echo -e "${RED}✗ Flower is not responding${NC}"
    fi
}

# Display access information
display_access_info() {
    echo -e "${GREEN}"
    echo "=========================================="
    echo "🎉 Airflow at Scale Setup Complete!"
    echo "=========================================="
    echo -e "${NC}"
    
    echo "Access URLs:"
    echo "  • Airflow UI: http://localhost:80"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Grafana: http://localhost:3000"
    echo "  • Flower (Celery): http://localhost:5555"
    echo ""
    
    echo "Default Credentials:"
    echo "  • Airflow: admin / (check .env file)"
    echo "  • Grafana: admin / (check .env file)"
    echo ""
    
    echo "Monitoring:"
    echo "  • View metrics in Prometheus"
    echo "  • Import Airflow dashboards in Grafana"
    echo "  • Monitor Celery workers in Flower"
    echo ""
    
    echo "Scaling:"
    echo "  • Scale workers: docker-compose up -d --scale airflow-worker=5"
    echo "  • Monitor auto-scaling in Grafana dashboards"
    echo ""
    
    echo "Logs:"
    echo "  • View logs: docker-compose logs -f [service-name]"
    echo "  • Airflow logs: ./logs/"
    echo ""
    
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Review and customize configuration files"
    echo "2. Deploy your DAGs to ./dags/ directory"
    echo "3. Configure alerting endpoints in monitoring/alertmanager.yml"
    echo "4. Set up SSL certificates for production"
    echo "5. Configure backup strategies"
}

# Main execution
main() {
    echo "🚀 ScaleCorp Airflow at Scale Setup"
    echo "=================================="
    
    check_prerequisites
    setup_environment
    
    # Ask for confirmation before proceeding
    read -p "Continue with Airflow initialization and startup? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        initialize_airflow
        start_services
        health_check
        display_access_info
    else
        echo "Setup paused. Run this script again when ready to continue."
        exit 0
    fi
}

# Run main function
main "$@"