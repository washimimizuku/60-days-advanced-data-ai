# Day 9: Data Lineage Tracking - Setup Guide

## Quick Start (10 minutes)

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 3. Database Setup
```bash
# Start services with Docker Compose
docker-compose up -d

# Or install individually (see detailed setup below)
```

### 4. Run Exercises
```bash
# Run all exercises
python exercise.py

# Run specific exercise
python -c "from exercise import LineageExerciseRunner; runner = LineageExerciseRunner(); runner.run_exercise_1_sql_parsing()"
```

---

## Detailed Setup

### Prerequisites
- Python 3.8+
- Docker and Docker Compose (recommended)
- 4GB+ RAM
- 2GB+ disk space

### Database Services

#### Option 1: Docker Compose (Recommended)
```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: lineage_db
      POSTGRES_USER: lineage_user
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  neo4j:
    image: neo4j:5.14
    environment:
      NEO4J_AUTH: neo4j/secure_password
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass secure_password
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  mongodb:
    image: mongo:7
    environment:
      MONGO_INITDB_ROOT_USERNAME: lineage_user
      MONGO_INITDB_ROOT_PASSWORD: secure_password
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  postgres_data:
  neo4j_data:
  redis_data:
  mongo_data:
```

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

#### Option 2: Local Installation

**PostgreSQL:**
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Start service
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS

# Create database and user
sudo -u postgres psql
CREATE DATABASE lineage_db;
CREATE USER lineage_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE lineage_db TO lineage_user;
```

**Neo4j:**
```bash
# Download and install Neo4j Community Edition
# https://neo4j.com/download/

# Start Neo4j
neo4j start

# Access browser interface
# http://localhost:7474
# Default credentials: neo4j/neo4j (change on first login)
```

**Redis:**
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Start Redis
sudo systemctl start redis  # Linux
brew services start redis   # macOS

# Set password
redis-cli
CONFIG SET requirepass "secure_password"
```

**MongoDB:**
```bash
# Ubuntu/Debian
sudo apt-get install mongodb

# macOS
brew install mongodb-community

# Start MongoDB
sudo systemctl start mongod  # Linux
brew services start mongodb-community  # macOS
```

### Data Catalog Services (Optional)

#### DataHub
```bash
# Install DataHub CLI
pip install acryl-datahub

# Start DataHub with Docker
git clone https://github.com/datahub-project/datahub.git
cd datahub/docker/quickstart
./quickstart.sh

# Access DataHub UI
# http://localhost:9002
```

#### Apache Atlas
```bash
# Download Atlas
wget https://downloads.apache.org/atlas/2.3.0/apache-atlas-2.3.0-bin.tar.gz
tar -xzf apache-atlas-2.3.0-bin.tar.gz
cd apache-atlas-2.3.0

# Configure Atlas
# Edit conf/atlas-application.properties

# Start Atlas
bin/atlas_start.py

# Access Atlas UI
# http://localhost:21000
# Default credentials: admin/admin
```

### Python Environment Setup

#### Virtual Environment
```bash
# Create virtual environment
python -m venv lineage_env
source lineage_env/bin/activate  # On Windows: lineage_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

#### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Configuration

#### Environment Variables
```bash
# Copy template
cp .env.example .env

# Required variables (edit .env file):
POSTGRES_HOST=localhost
POSTGRES_PASSWORD=your_secure_password
NEO4J_PASSWORD=your_neo4j_password
REDIS_PASSWORD=your_redis_password
```

#### Database Initialization
```bash
# Initialize PostgreSQL schema
python scripts/init_postgres.py

# Initialize Neo4j constraints and indexes
python scripts/init_neo4j.py

# Load sample data
python scripts/load_sample_data.py
```

### Verification

#### Test Database Connections
```bash
# Test all connections
python scripts/test_connections.py

# Expected output:
# ✅ PostgreSQL connection successful
# ✅ Neo4j connection successful
# ✅ Redis connection successful
# ✅ MongoDB connection successful
```

#### Run Sample Lineage Extraction
```bash
# Test SQL parsing
python -c "
from src.sql_extractor import SQLLineageExtractor
extractor = SQLLineageExtractor()
result = extractor.extract_table_lineage('SELECT * FROM users')
print('SQL parsing works:', result)
"

# Test graph operations
python -c "
from src.lineage_tracker import MultiHopLineageTracker
tracker = MultiHopLineageTracker()
tracker.add_node('users', 'table', {'schema': 'public'})
print('Graph operations work')
"
```

### Performance Optimization

#### Neo4j Optimization
```cypher
// Create indexes for better performance
CREATE INDEX node_id_index FOR (n:Dataset) ON (n.qualified_name);
CREATE INDEX column_index FOR (n:Column) ON (n.qualified_name);
CREATE INDEX transformation_index FOR (n:Transformation) ON (n.type);

// Configure memory settings in neo4j.conf
dbms.memory.heap.initial_size=1G
dbms.memory.heap.max_size=2G
dbms.memory.pagecache.size=1G
```

#### Redis Configuration
```bash
# Edit redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

#### PostgreSQL Tuning
```sql
-- Optimize for lineage queries
CREATE INDEX idx_lineage_source ON lineage_edges(source_id);
CREATE INDEX idx_lineage_target ON lineage_edges(target_id);
CREATE INDEX idx_lineage_created ON lineage_edges(created_at);

-- Update PostgreSQL configuration
-- shared_buffers = 256MB
-- effective_cache_size = 1GB
-- work_mem = 4MB
```

### Monitoring Setup

#### Prometheus Metrics
```bash
# Start Prometheus
docker run -d -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Start Grafana
docker run -d -p 3000:3000 grafana/grafana

# Import lineage dashboard
# Dashboard ID: lineage-monitoring.json
```

#### Health Checks
```bash
# Set up health check endpoint
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "services": {
    "postgres": "up",
    "neo4j": "up",
    "redis": "up"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Troubleshooting

#### Common Issues

**Connection Errors:**
```bash
# Check service status
docker-compose ps
systemctl status postgresql neo4j redis

# Check network connectivity
telnet localhost 5432  # PostgreSQL
telnet localhost 7687  # Neo4j
telnet localhost 6379  # Redis

# Check logs
docker-compose logs postgres
journalctl -u postgresql
```

**Memory Issues:**
```bash
# Check memory usage
docker stats
free -h

# Adjust memory limits in docker-compose.yml
services:
  neo4j:
    mem_limit: 2g
  postgres:
    mem_limit: 1g
```

**Performance Issues:**
```bash
# Check database performance
# PostgreSQL
SELECT * FROM pg_stat_activity;

# Neo4j
CALL dbms.listQueries();

# Redis
redis-cli info memory
```

#### Getting Help

1. **Check logs first:**
   ```bash
   tail -f logs/lineage.log
   docker-compose logs -f
   ```

2. **Verify configuration:**
   ```bash
   python scripts/verify_config.py
   ```

3. **Run diagnostics:**
   ```bash
   python scripts/diagnose.py
   ```

4. **Common solutions:**
   - Restart services: `docker-compose restart`
   - Clear cache: `redis-cli FLUSHALL`
   - Reset database: `python scripts/reset_db.py`

### Next Steps

1. **Complete the exercises** in `exercise.py`
2. **Review the solution** in `solution.py`
3. **Take the quiz** in `quiz.md`
4. **Explore advanced features:**
   - Custom SQL parsers
   - Real-time lineage updates
   - Integration with CI/CD pipelines
   - Advanced visualization dashboards

### Production Deployment

For production deployment, consider:
- **High availability** setup for databases
- **Load balancing** for API endpoints
- **Backup and recovery** procedures
- **Security hardening** (SSL, authentication)
- **Monitoring and alerting** setup
- **Disaster recovery** planning

Ready to track your data lineage! 🚀