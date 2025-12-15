#!/bin/bash

# Day 16: Airflow at Scale - Worker Scaling Script
# Manual scaling script for Airflow workers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CURRENT_WORKERS=$(docker-compose ps -q airflow-worker | wc -l)
TARGET_WORKERS=${1:-$CURRENT_WORKERS}
MAX_WORKERS=50
MIN_WORKERS=2

# Help function
show_help() {
    echo "Usage: $0 [number_of_workers]"
    echo ""
    echo "Scale Airflow workers up or down"
    echo ""
    echo "Arguments:"
    echo "  number_of_workers    Target number of workers (default: current count)"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -s, --status        Show current worker status"
    echo "  -a, --auto          Enable auto-scaling mode"
    echo ""
    echo "Examples:"
    echo "  $0 10               Scale to 10 workers"
    echo "  $0 --status         Show current status"
    echo "  $0 --auto           Enable auto-scaling"
}

# Show current status
show_status() {
    echo -e "${YELLOW}Current Worker Status:${NC}"
    echo "  • Current workers: $CURRENT_WORKERS"
    echo "  • Min workers: $MIN_WORKERS"
    echo "  • Max workers: $MAX_WORKERS"
    echo ""
    
    # Show worker health
    echo -e "${YELLOW}Worker Health:${NC}"
    docker-compose ps airflow-worker
    echo ""
    
    # Show Celery worker status via Flower
    echo -e "${YELLOW}Celery Worker Status:${NC}"
    if curl -s http://localhost:5555/api/workers 2>/dev/null | jq -r 'keys[]' 2>/dev/null; then
        echo "Workers are registered with Celery"
    else
        echo "Unable to connect to Flower or no workers registered"
    fi
}

# Validate target workers
validate_target() {
    if ! [[ "$TARGET_WORKERS" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Error: Target workers must be a positive integer${NC}"
        exit 1
    fi
    
    if [ "$TARGET_WORKERS" -lt "$MIN_WORKERS" ]; then
        echo -e "${RED}Error: Target workers ($TARGET_WORKERS) is below minimum ($MIN_WORKERS)${NC}"
        exit 1
    fi
    
    if [ "$TARGET_WORKERS" -gt "$MAX_WORKERS" ]; then
        echo -e "${RED}Error: Target workers ($TARGET_WORKERS) exceeds maximum ($MAX_WORKERS)${NC}"
        exit 1
    fi
}

# Scale workers
scale_workers() {
    echo -e "${YELLOW}Scaling workers from $CURRENT_WORKERS to $TARGET_WORKERS...${NC}"
    
    if [ "$TARGET_WORKERS" -eq "$CURRENT_WORKERS" ]; then
        echo -e "${GREEN}Already at target worker count ($TARGET_WORKERS)${NC}"
        return 0
    fi
    
    # Scale using docker-compose
    docker-compose up -d --scale airflow-worker=$TARGET_WORKERS airflow-worker
    
    # Wait for workers to be ready
    echo "Waiting for workers to be ready..."
    sleep 30
    
    # Verify scaling
    NEW_WORKER_COUNT=$(docker-compose ps -q airflow-worker | wc -l)
    
    if [ "$NEW_WORKER_COUNT" -eq "$TARGET_WORKERS" ]; then
        echo -e "${GREEN}✓ Successfully scaled to $TARGET_WORKERS workers${NC}"
        
        # Send metrics to monitoring system
        send_scaling_metrics "manual_scale" "$TARGET_WORKERS" "success"
        
        # Show updated status
        show_status
    else
        echo -e "${RED}✗ Scaling failed. Expected $TARGET_WORKERS, got $NEW_WORKER_COUNT${NC}"
        send_scaling_metrics "manual_scale" "$TARGET_WORKERS" "failed"
        exit 1
    fi
}

# Send scaling metrics
send_scaling_metrics() {
    local action=$1
    local target_workers=$2
    local status=$3
    
    # Send to StatsD (if available)
    if command -v nc &> /dev/null; then
        echo "scalecorp_airflow.scaling.action:1|c|#action:$action,status:$status" | nc -u -w1 localhost 9125 2>/dev/null || true
        echo "scalecorp_airflow.scaling.target_workers:$target_workers|g" | nc -u -w1 localhost 9125 2>/dev/null || true
    fi
    
    # Log scaling event
    echo "$(date): Scaling action=$action target=$target_workers status=$status" >> logs/scaling.log
}

# Auto-scaling mode
auto_scaling_mode() {
    echo -e "${YELLOW}Enabling auto-scaling mode...${NC}"
    echo "This will monitor queue length and resource utilization to automatically scale workers."
    echo ""
    
    # Check if monitoring DAG is running
    if docker-compose exec -T airflow-scheduler-1 airflow dags state scalecorp_enterprise_monitoring 2>/dev/null | grep -q "running\|success"; then
        echo -e "${GREEN}✓ Auto-scaling DAG is already running${NC}"
    else
        echo -e "${YELLOW}Starting auto-scaling DAG...${NC}"
        docker-compose exec -T airflow-scheduler-1 airflow dags unpause scalecorp_enterprise_monitoring
        docker-compose exec -T airflow-scheduler-1 airflow dags trigger scalecorp_enterprise_monitoring
    fi
    
    echo ""
    echo "Auto-scaling is now active. Monitor progress in:"
    echo "  • Airflow UI: http://localhost:80"
    echo "  • Grafana: http://localhost:3000"
    echo "  • Logs: docker-compose logs -f airflow-scheduler-1"
}

# Get queue metrics
get_queue_metrics() {
    echo -e "${YELLOW}Current Queue Metrics:${NC}"
    
    # Try to get queue length from Redis
    if command -v redis-cli &> /dev/null; then
        QUEUE_LENGTH=$(docker-compose exec -T redis-master redis-cli llen celery 2>/dev/null || echo "N/A")
        echo "  • Queue length: $QUEUE_LENGTH"
    fi
    
    # Get task counts from Airflow
    if docker-compose exec -T postgres-primary psql -U airflow -d airflow -c "SELECT state, COUNT(*) FROM task_instance WHERE state IN ('queued', 'running') GROUP BY state;" 2>/dev/null; then
        echo "Task states retrieved from database"
    else
        echo "Unable to retrieve task states"
    fi
}

# Main execution
main() {
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--status)
            show_status
            get_queue_metrics
            exit 0
            ;;
        -a|--auto)
            auto_scaling_mode
            exit 0
            ;;
        "")
            show_status
            exit 0
            ;;
        *)
            TARGET_WORKERS=$1
            validate_target
            scale_workers
            ;;
    esac
}

# Run main function
main "$@"