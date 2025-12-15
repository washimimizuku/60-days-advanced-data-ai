# Day 11 Setup Guide: Access Control - RBAC, Row-Level Security

## Quick Start (5 minutes)

```bash
# 1. Navigate to day 11
cd days/day-11-access-control-rbac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env

# 4. Start services
docker-compose up -d

# 5. Run setup script
python scripts/setup_demo.py

# 6. Test installation
python scripts/test_setup.py
```

## Detailed Setup

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- PostgreSQL client (optional)
- Redis client (optional)

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (required changes marked with TODO)
nano .env
```

**Required Configuration Changes:**
- `SECRET_KEY`: Generate with `python -c "import secrets; print(secrets.token_hex(32))"`
- `JWT_SECRET_KEY`: Generate with `python -c "import secrets; print(secrets.token_hex(32))"`
- `ENCRYPTION_KEY`: Generate with `python scripts/generate_keys.py`

### 3. Database Setup

#### Option A: Docker (Recommended)
```bash
# Start PostgreSQL and Redis
docker-compose up -d

# Verify services
docker-compose ps
```

#### Option B: Local Installation
```bash
# PostgreSQL
sudo apt-get install postgresql postgresql-contrib  # Ubuntu
brew install postgresql                             # macOS

# Redis
sudo apt-get install redis-server                  # Ubuntu
brew install redis                                  # macOS

# Start services
sudo systemctl start postgresql redis-server       # Ubuntu
brew services start postgresql redis               # macOS
```

### 4. Database Initialization

```bash
# Create database and tables
python scripts/init_database.py

# Load sample data (optional)
python scripts/load_sample_data.py

# Verify setup
python scripts/test_setup.py
```

## Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   PostgreSQL    │    │     Redis       │
│                 │    │                 │    │                 │
│ • RBAC Engine   │◄──►│ • User Data     │    │ • Sessions      │
│ • RLS Manager   │    │ • Policies      │    │ • Cache         │
│ • ABAC Engine   │    │ • Audit Logs    │    │ • Temp Data     │
│ • Audit Logger  │    │ • RLS Policies  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Testing the Setup

### 1. Basic Functionality Test
```bash
python scripts/test_setup.py
```

### 2. RBAC System Test
```bash
python -c "
from solution import RBACSystem, Permission
rbac = RBACSystem()
rbac.create_role('test_role', {Permission.READ_CUSTOMER_DATA})
print('RBAC system working!')
"
```

### 3. Database Connection Test
```bash
python -c "
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
print('Database connection successful!')
conn.close()
"
```

### 4. Redis Connection Test
```bash
python -c "
import redis
import os
from dotenv import load_dotenv
load_dotenv()
r = redis.from_url(os.getenv('REDIS_URL'))
r.ping()
print('Redis connection successful!')
"
```

## Sample Data Overview

The setup includes sample data for testing:

### Users
- `alice` - Data Analyst (basic permissions)
- `bob` - Data Scientist (model permissions)
- `charlie` - Senior Data Scientist (deployment permissions)
- `diana` - Data Engineer (write permissions)
- `admin` - Administrator (full permissions)

### Tenants
- `acme_corp` - Premium plan tenant
- `beta_inc` - Basic plan tenant
- `gamma_ltd` - Enterprise plan tenant

### Sample Policies
- Regional data access restrictions
- Sensitivity-based filtering
- Time-based access controls
- Multi-tenant isolation

## Exercise Workflow

### 1. Start with Basic RBAC
```bash
python exercise.py  # Follow TODO instructions
```

### 2. Test Your Implementation
```bash
python scripts/test_rbac.py
```

### 3. Move to Advanced Features
- Row-Level Security
- ABAC Policies
- Multi-tenant Security
- Audit Logging

### 4. Verify Complete Solution
```bash
python solution.py  # See production implementation
```

## Troubleshooting

### Common Issues

#### Database Connection Failed
```bash
# Check if PostgreSQL is running
docker-compose ps
# or
sudo systemctl status postgresql

# Check connection string
echo $DATABASE_URL
```

#### Redis Connection Failed
```bash
# Check if Redis is running
docker-compose ps
# or
sudo systemctl status redis-server

# Test Redis connection
redis-cli ping
```

#### Permission Denied Errors
```bash
# Check file permissions
ls -la scripts/
chmod +x scripts/*.py

# Check database permissions
psql $DATABASE_URL -c "\du"
```

#### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

### Performance Issues

#### Slow Policy Evaluation
- Enable caching in `.env`: `RBAC_ENABLE_CACHING=true`
- Increase cache TTL: `RBAC_CACHE_TTL=1800`
- Check database indexes: `python scripts/check_indexes.py`

#### Memory Usage
- Reduce cache size: `ACCESS_CONTROL_CACHE_SIZE=5000`
- Enable batch processing: `AUDIT_BATCH_SIZE=500`

## Security Considerations

### Production Deployment

1. **Change Default Secrets**
   ```bash
   # Generate new secrets
   python scripts/generate_keys.py
   ```

2. **Enable SSL/TLS**
   ```bash
   # Update DATABASE_URL to use SSL
   DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
   ```

3. **Restrict Network Access**
   ```bash
   # Bind to localhost only
   REDIS_URL=redis://127.0.0.1:6379/0
   ```

4. **Enable Audit Logging**
   ```bash
   AUDIT_LOG_LEVEL=INFO
   AUDIT_ENABLE_REAL_TIME_ALERTS=true
   ```

### Security Checklist

- [ ] Changed all default passwords and secrets
- [ ] Enabled SSL/TLS for database connections
- [ ] Configured proper network security
- [ ] Set up audit logging and monitoring
- [ ] Implemented proper backup procedures
- [ ] Configured log rotation and retention
- [ ] Set up security alerting
- [ ] Performed security testing

## Next Steps

1. **Complete the exercises** in order (RBAC → RLS → ABAC → Multi-tenant → Audit)
2. **Test with sample data** to understand real-world scenarios
3. **Explore advanced features** like policy optimization and performance tuning
4. **Integrate with external systems** (LDAP, OAuth, SAML)
5. **Deploy to production** following security best practices

## Additional Resources

- [PostgreSQL Row Level Security Documentation](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [NIST RBAC Guidelines](https://csrc.nist.gov/projects/role-based-access-control)
- [OWASP Access Control Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html)
- [Redis Security Best Practices](https://redis.io/topics/security)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the sample implementations in `solution.py`
3. Test individual components with the provided test scripts
4. Refer to the official documentation links