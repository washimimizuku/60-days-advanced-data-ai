# Day 13 Setup Guide: dbt Basics

## Quick Start (5 minutes)

```bash
# 1. Navigate to day 13
cd days/day-13-dbt-basics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env

# 4. Start database
docker-compose up -d

# 5. Initialize dbt project
dbt deps
dbt seed
dbt run
dbt test

# 6. View documentation
dbt docs generate
dbt docs serve
```

## Detailed Setup

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- 2GB+ RAM available
- 5GB+ disk space

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv dbt-env
source dbt-env/bin/activate  # On Windows: dbt-env\Scripts\activate

# Install dbt and dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration if needed
nano .env
```

### 3. Database Setup

#### Option A: Docker (Recommended)
```bash
# Start PostgreSQL with sample data
docker-compose up -d

# Verify database is running
docker-compose ps
```

#### Option B: Local PostgreSQL
```bash
# Install PostgreSQL
brew install postgresql  # macOS
sudo apt-get install postgresql  # Ubuntu

# Create database and user
createdb ecommerce_analytics
createuser dbt_user
```

### 4. dbt Project Initialization

```bash
# Install dbt packages
dbt deps

# Load sample data
dbt seed

# Run all models
dbt run

# Run tests
dbt test

# Generate documentation
dbt docs generate
```

## Project Structure

```
ecommerce_analytics/
├── dbt_project.yml           # Project configuration
├── profiles.yml              # Connection settings
├── packages.yml              # dbt packages
│
├── models/
│   ├── sources.yml           # Source definitions
│   ├── schema.yml            # Model tests and docs
│   ├── staging/              # Staging models
│   │   ├── stg_customers.sql
│   │   ├── stg_orders.sql
│   │   ├── stg_order_items.sql
│   │   └── stg_products.sql
│   ├── intermediate/         # Business logic
│   │   ├── int_order_metrics.sql
│   │   ├── int_customer_metrics.sql
│   │   └── int_product_performance.sql
│   └── marts/                # Final tables
│       ├── core/
│       │   ├── dim_customers.sql
│       │   ├── dim_products.sql
│       │   └── fct_orders.sql
│       └── finance/
│           └── revenue_summary.sql
│
├── tests/                    # Custom tests
├── macros/                   # Reusable SQL
├── seeds/                    # CSV data files
└── snapshots/                # SCD tracking
```

## Testing the Setup

### 1. Verify dbt Installation
```bash
dbt --version
```

### 2. Test Database Connection
```bash
dbt debug
```

### 3. Run Sample Models
```bash
# Run staging models
dbt run --models staging

# Run all models
dbt run

# Test data quality
dbt test
```

### 4. View Documentation
```bash
# Generate docs
dbt docs generate

# Serve locally
dbt docs serve
# Open http://localhost:8080
```

## Sample Data Overview

The setup includes realistic e-commerce sample data:

### Raw Data Sources
- **customers**: 1,000 customer records with demographics
- **orders**: 5,000 order transactions over 12 months
- **order_items**: 15,000 line items with product details
- **products**: 500 products across multiple categories

### Generated Models
- **Staging**: Clean, standardized data (4 models)
- **Intermediate**: Business logic and metrics (3 models)
- **Marts**: Analytics-ready tables (4 models)

## dbt Workflow

### Development Cycle
```bash
# 1. Make changes to models
nano models/staging/stg_customers.sql

# 2. Compile to check syntax
dbt compile --models stg_customers

# 3. Run the model
dbt run --models stg_customers

# 4. Test the model
dbt test --models stg_customers

# 5. Update documentation
dbt docs generate
```

### Common Commands
```bash
# Run specific models
dbt run --models staging
dbt run --models +dim_customers
dbt run --models fct_orders+

# Test specific models
dbt test --models marts
dbt test --models tag:staging

# Full refresh incremental models
dbt run --full-refresh --models fct_orders

# Run with different target
dbt run --target prod
```

## Troubleshooting

### Common Issues

#### dbt Command Not Found
```bash
# Ensure virtual environment is activated
source dbt-env/bin/activate

# Reinstall dbt
pip install dbt-core dbt-postgres
```

#### Database Connection Failed
```bash
# Check database is running
docker-compose ps

# Test connection
dbt debug

# Check credentials in profiles.yml
```

#### Model Compilation Errors
```bash
# Check SQL syntax
dbt compile --models model_name

# View compiled SQL
cat target/compiled/ecommerce_analytics/models/model_name.sql

# Check for missing refs
dbt list --models model_name
```

#### Test Failures
```bash
# Run tests with details
dbt test --store-failures

# View failed test results
select * from analytics_dev.dbt_test_failures;

# Debug specific test
dbt test --models model_name --store-failures
```

### Performance Issues

#### Slow Model Execution
- Use `dbt run --models model_name --debug` for timing
- Check compiled SQL for efficiency
- Consider materialization changes
- Add appropriate indexes

#### Memory Issues
- Reduce `threads` in profiles.yml
- Use incremental materialization for large tables
- Process data in smaller batches

## Production Deployment

### Environment Setup
```bash
# Production profiles.yml
ecommerce_analytics:
  target: prod
  outputs:
    prod:
      type: postgres
      host: "{{ env_var('POSTGRES_HOST') }}"
      user: "{{ env_var('POSTGRES_USER') }}"
      password: "{{ env_var('POSTGRES_PASSWORD') }}"
      port: 5432
      dbname: "{{ env_var('POSTGRES_DATABASE') }}"
      schema: analytics
      threads: 8
      keepalives_idle: 0
```

### CI/CD Pipeline
```yaml
# .github/workflows/dbt.yml
name: dbt CI/CD
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dbt
        run: pip install -r requirements.txt
      - name: Run dbt
        run: |
          dbt deps
          dbt compile
          dbt run --target ci
          dbt test --target ci
```

### Monitoring
```bash
# Set up dbt artifacts collection
dbt run --store-failures
dbt test --store-failures

# Monitor model performance
select * from dbt_artifacts.model_executions
where execution_time > 300;  -- Models taking >5 minutes
```

## Next Steps

1. **Complete the exercises** - Build the e-commerce analytics project
2. **Experiment with materializations** - Try different strategies
3. **Add custom tests** - Implement business rule validation
4. **Explore advanced features** - Snapshots, macros, packages
5. **Deploy to production** - Set up CI/CD and monitoring

## Additional Resources

- [dbt Documentation](https://docs.getdbt.com/)
- [dbt Learn Courses](https://courses.getdbt.com/)
- [dbt Utils Package](https://github.com/dbt-labs/dbt-utils)
- [Analytics Engineering Guide](https://www.getdbt.com/analytics-engineering/)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `dbt debug` to diagnose connection issues
3. Review dbt logs in `logs/dbt.log`
4. Check the dbt community Slack for help