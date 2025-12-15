"""
Day 8: Data Catalogs - DataHub, Amundsen
Exercise: Build a comprehensive data catalog implementation

This exercise covers:
1. Setting up DataHub locally
2. Implementing custom metadata extractors
3. Creating data lineage tracking
4. Building data quality integration
5. Setting up search and discovery workflows
6. Implementing governance policies
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# DataHub Python SDK
try:
    from datahub.emitter.mce_builder import make_data_platform_urn, make_dataset_urn
    from datahub.emitter.mcp import MetadataChangeProposalWrapper
    from datahub.emitter.rest_emitter import DatahubRestEmitter
    from datahub.metadata.schema_classes import (
        DatasetPropertiesClass, 
        OwnershipClass, 
        OwnerClass,
        OwnershipTypeClass,
        GlobalTagsClass,
        TagAssociationClass,
        GlossaryTermAssociationClass
    )
    DATAHUB_AVAILABLE = True
except ImportError as e:
    print(f"DataHub SDK not available: {e}")
    print("Install with: pip install -r requirements.txt")
    DATAHUB_AVAILABLE = False

class DataCatalogExercise:
    """
    Comprehensive data catalog exercise implementation
    
    This class demonstrates:
    - Metadata extraction from various sources
    - DataHub integration and API usage
    - Data lineage tracking
    - Quality metrics integration
    - Search and discovery patterns
    """
    
    def __init__(self):
        self.datahub_url = os.getenv('DATAHUB_GMS_URL', 'http://localhost:8080')
        self.postgres_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'database': os.getenv('POSTGRES_DB', 'ecommerce'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
            'port': int(os.getenv('POSTGRES_PORT', '5432'))
        }
        
        # Initialize DataHub emitter if available
        if DATAHUB_AVAILABLE:
            self.emitter = DatahubRestEmitter(gms_server=self.datahub_url)
        else:
            self.emitter = None
    
    def exercise_1_setup_datahub(self):
        """
        Exercise 1: Set up DataHub locally using Docker
        
        TODO: Complete the Docker Compose setup for DataHub
        """
        print("=== Exercise 1: DataHub Setup ===")
        
        # TODO: Create docker-compose.yml for DataHub
        docker_compose_content = """
version: '3.8'
services:
  # TODO: Add Elasticsearch service
  elasticsearch:
    image: elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  # TODO: Add MySQL service for metadata storage
  mysql:
    # Complete the MySQL configuration
    pass

  # TODO: Add Kafka service for real-time updates
  kafka:
    # Complete the Kafka configuration
    pass

  # TODO: Add DataHub GMS (backend) service
  datahub-gms:
    # Complete the GMS configuration
    pass

  # TODO: Add DataHub Frontend service
  datahub-frontend:
    # Complete the frontend configuration
    pass

volumes:
  elasticsearch_data:
  mysql_data:
"""
        
        print("Docker Compose template created. Complete the TODOs to set up DataHub.")
        print("Run: docker-compose up -d to start DataHub")
        
        # TODO: Add health check function
        self._check_datahub_health()
    
    def _check_datahub_health(self):
        """Check if DataHub services are running"""
        try:
            # TODO: Implement health check for DataHub GMS
            response = requests.get(f"{self.datahub_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ DataHub is running and healthy")
                return True
            else:
                print("❌ DataHub health check failed")
                return False
        except requests.exceptions.RequestException:
            print("❌ DataHub is not accessible. Make sure it's running on localhost:8080")
            return False
    
    def exercise_2_metadata_extraction(self):
        """
        Exercise 2: Implement custom metadata extractors
        
        TODO: Create extractors for PostgreSQL, CSV files, and API endpoints
        """
        print("\n=== Exercise 2: Metadata Extraction ===")
        
        # TODO: Implement PostgreSQL metadata extractor
        postgres_metadata = self._extract_postgres_metadata()
        print(f"Extracted metadata for {len(postgres_metadata)} PostgreSQL tables")
        
        # TODO: Implement CSV file metadata extractor
        csv_metadata = self._extract_csv_metadata("./sample_data/")
        print(f"Extracted metadata for {len(csv_metadata)} CSV files")
        
        # Create sample data directory if it doesn't exist
        os.makedirs("./sample_data", exist_ok=True)
        
        # TODO: Implement API metadata extractor
        api_metadata = self._extract_api_metadata()
        print(f"Extracted metadata for {len(api_metadata)} API endpoints")
        
        return {
            'postgres': postgres_metadata,
            'csv': csv_metadata,
            'api': api_metadata
        }
    
    def _extract_postgres_metadata(self) -> List[Dict[str, Any]]:
        """Extract metadata from PostgreSQL database"""
        metadata = []
        
        try:
            # TODO: Connect to PostgreSQL and extract table metadata
            conn = psycopg2.connect(**self.postgres_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # TODO: Query for table information
            tables_query = """
            SELECT 
                table_schema,
                table_name,
                table_type,
                -- TODO: Add more metadata fields
                NULL as table_comment
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
            """
            
            cursor.execute(tables_query)
            tables = cursor.fetchall()
            
            for table in tables:
                # TODO: Extract column metadata for each table
                columns_metadata = self._extract_column_metadata(
                    cursor, table['table_schema'], table['table_name']
                )
                
                # TODO: Calculate table statistics
                stats = self._calculate_table_stats(
                    cursor, table['table_schema'], table['table_name']
                )
                
                table_metadata = {
                    'platform': 'postgres',
                    'database': self.postgres_config['database'],
                    'schema': table['table_schema'],
                    'name': table['table_name'],
                    'type': table['table_type'],
                    'description': table['table_comment'],
                    'columns': columns_metadata,
                    'stats': stats,
                    'extracted_at': datetime.now().isoformat()
                }
                
                metadata.append(table_metadata)
            
            conn.close()
            
        except Exception as e:
            print(f"Error extracting PostgreSQL metadata: {e}")
        
        return metadata
    
    def _extract_column_metadata(self, cursor, schema: str, table: str) -> List[Dict[str, Any]]:
        """Extract column metadata for a specific table"""
        # TODO: Implement column metadata extraction
        columns_query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            -- TODO: Add column comments and constraints
            NULL as column_comment
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """
        
        cursor.execute(columns_query, [schema, table])
        columns = cursor.fetchall()
        
        column_metadata = []
        for col in columns:
            # TODO: Process column information
            column_info = {
                'name': col['column_name'],
                'type': col['data_type'],
                'nullable': col['is_nullable'] == 'YES',
                'default': col['column_default'],
                'description': col['column_comment'],
                # TODO: Add more column properties
            }
            column_metadata.append(column_info)
        
        return column_metadata
    
    def _calculate_table_stats(self, cursor, schema: str, table: str) -> Dict[str, Any]:
        """Calculate basic statistics for a table"""
        try:
            # TODO: Implement table statistics calculation
            stats_query = f"""
            SELECT 
                COUNT(*) as row_count,
                -- TODO: Add more statistics
                pg_total_relation_size('{schema}.{table}') as size_bytes
            FROM {schema}.{table}
            """
            
            cursor.execute(stats_query)
            result = cursor.fetchone()
            
            return {
                'row_count': result['row_count'] if result else 0,
                'size_bytes': result['size_bytes'] if result else 0,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error calculating stats for {schema}.{table}: {e}")
            return {}
    
    def _extract_csv_metadata(self, directory: str) -> List[Dict[str, Any]]:
        """Extract metadata from CSV files in a directory"""
        # TODO: Implement CSV metadata extraction
        metadata = []
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return metadata
        
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                # TODO: Extract CSV file metadata
                file_path = os.path.join(directory, filename)
                
                try:
                    # Basic file information
                    file_stats = os.stat(file_path)
                    
                    # TODO: Add pandas-based column inference
                    # import pandas as pd
                    # df = pd.read_csv(file_path, nrows=100)
                    # columns = [{'name': col, 'type': str(df[col].dtype)} for col in df.columns]
                    
                    csv_metadata = {
                        'platform': 'file',
                        'name': filename.replace('.csv', ''),
                        'path': file_path,
                        'format': 'csv',
                        'size_bytes': file_stats.st_size,
                        'created_at': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                        'modified_at': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        # TODO: Add column inference and sample data
                        'columns': [],
                        'sample_data': [],
                        'extracted_at': datetime.now().isoformat()
                    }
                    metadata.append(csv_metadata)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return metadata
    
    def _extract_api_metadata(self) -> List[Dict[str, Any]]:
        """Extract metadata from API endpoints"""
        # TODO: Implement API metadata extraction
        # This could extract from OpenAPI specs, API documentation, etc.
        
        sample_apis = [
            {
                'platform': 'api',
                'name': 'users_api',
                'endpoint': '/api/v1/users',
                'method': 'GET',
                'description': 'Retrieve user information',
                'parameters': [
                    {'name': 'user_id', 'type': 'integer', 'required': True},
                    {'name': 'include_profile', 'type': 'boolean', 'required': False}
                ],
                'response_schema': {
                    'user_id': 'integer',
                    'email': 'string',
                    'created_at': 'timestamp'
                }
            }
        ]
        
        return sample_apis
    
    def exercise_3_datahub_integration(self, metadata: Dict[str, List[Dict[str, Any]]]):
        """
        Exercise 3: Integrate extracted metadata with DataHub
        
        TODO: Push metadata to DataHub using the Python SDK
        """
        print("\n=== Exercise 3: DataHub Integration ===")
        
        if not DATAHUB_AVAILABLE or not self.emitter:
            print("DataHub SDK not available. Skipping integration.")
            return
        
        # TODO: Push PostgreSQL metadata to DataHub
        for table_metadata in metadata['postgres']:
            self._push_table_to_datahub(table_metadata)
        
        # TODO: Push CSV metadata to DataHub
        for csv_metadata in metadata['csv']:
            self._push_csv_to_datahub(csv_metadata)
        
        print("Metadata successfully pushed to DataHub")
    
    def _push_table_to_datahub(self, table_metadata: Dict[str, Any]):
        """Push table metadata to DataHub"""
        if not self.emitter:
            return
        
        # TODO: Create dataset URN
        dataset_urn = make_dataset_urn(
            platform=table_metadata['platform'],
            name=f"{table_metadata['database']}.{table_metadata['schema']}.{table_metadata['name']}"
        )
        
        # TODO: Create dataset properties
        dataset_properties = DatasetPropertiesClass(
            description=table_metadata.get('description', ''),
            customProperties={
                'schema': table_metadata['schema'],
                'table_type': table_metadata['type'],
                'row_count': str(table_metadata['stats'].get('row_count', 0)),
                'size_bytes': str(table_metadata['stats'].get('size_bytes', 0)),
                'extracted_at': table_metadata['extracted_at']
            }
        )
        
        # TODO: Create ownership information
        ownership = OwnershipClass(
            owners=[
                OwnerClass(
                    owner=make_data_platform_urn('data-team'),
                    type=OwnershipTypeClass.DATAOWNER
                )
            ]
        )
        
        # TODO: Create tags
        tags = GlobalTagsClass(
            tags=[
                TagAssociationClass(tag=make_data_platform_urn('production')),
                TagAssociationClass(tag=make_data_platform_urn(table_metadata['schema']))
            ]
        )
        
        # TODO: Emit metadata to DataHub
        mcps = [
            MetadataChangeProposalWrapper(
                entityType="dataset",
                entityUrn=dataset_urn,
                aspectName="datasetProperties",
                aspect=dataset_properties
            ),
            MetadataChangeProposalWrapper(
                entityType="dataset", 
                entityUrn=dataset_urn,
                aspectName="ownership",
                aspect=ownership
            ),
            MetadataChangeProposalWrapper(
                entityType="dataset",
                entityUrn=dataset_urn, 
                aspectName="globalTags",
                aspect=tags
            )
        ]
        
        for mcp in mcps:
            self.emitter.emit_mcp(mcp)
    
    def _push_csv_to_datahub(self, csv_metadata: Dict[str, Any]):
        """Push CSV file metadata to DataHub"""
        # TODO: Implement CSV to DataHub integration
        pass
    
    def exercise_4_lineage_tracking(self):
        """
        Exercise 4: Implement data lineage tracking
        
        TODO: Create lineage relationships between datasets
        """
        print("\n=== Exercise 4: Data Lineage Tracking ===")
        
        # TODO: Define sample lineage relationships
        lineage_relationships = [
            {
                'upstream': 'postgres://ecommerce.public.users',
                'downstream': 'postgres://analytics.public.user_metrics',
                'transformation': 'SELECT user_id, email, created_at FROM users WHERE status = "active"'
            },
            {
                'upstream': 'postgres://ecommerce.public.orders',
                'downstream': 'postgres://analytics.public.user_metrics', 
                'transformation': 'SELECT user_id, COUNT(*) as order_count, SUM(total_amount) as total_spent FROM orders GROUP BY user_id'
            }
        ]
        
        # TODO: Implement lineage tracking
        for relationship in lineage_relationships:
            self._create_lineage_relationship(relationship)
        
        print(f"Created {len(lineage_relationships)} lineage relationships")
    
    def _create_lineage_relationship(self, relationship: Dict[str, Any]):
        """Create a lineage relationship in DataHub"""
        # TODO: Implement lineage relationship creation
        print(f"Creating lineage: {relationship['upstream']} -> {relationship['downstream']}")
        
        # This would use DataHub's lineage APIs to create relationships
        # For now, we'll just log the relationship
        pass
    
    def exercise_5_quality_integration(self):
        """
        Exercise 5: Integrate data quality metrics
        
        TODO: Add data quality scores and validation results to catalog
        """
        print("\n=== Exercise 5: Data Quality Integration ===")
        
        # TODO: Define sample quality metrics
        quality_metrics = {
            'postgres://ecommerce.public.users': {
                'completeness': 0.98,
                'uniqueness': 0.99,
                'validity': 0.95,
                'consistency': 0.97,
                'timeliness': 0.92,
                'overall_score': 0.96,
                'last_check': datetime.now().isoformat(),
                'rules_passed': 8,
                'rules_failed': 1
            },
            'postgres://ecommerce.public.orders': {
                'completeness': 0.99,
                'uniqueness': 1.0,
                'validity': 0.98,
                'consistency': 0.96,
                'timeliness': 0.94,
                'overall_score': 0.97,
                'last_check': datetime.now().isoformat(),
                'rules_passed': 12,
                'rules_failed': 0
            }
        }
        
        # TODO: Push quality metrics to DataHub
        for dataset_urn, metrics in quality_metrics.items():
            self._update_quality_metrics(dataset_urn, metrics)
        
        print(f"Updated quality metrics for {len(quality_metrics)} datasets")
    
    def _update_quality_metrics(self, dataset_urn: str, metrics: Dict[str, Any]):
        """Update quality metrics for a dataset in DataHub"""
        # TODO: Implement quality metrics update
        print(f"Updating quality metrics for {dataset_urn}: {metrics['overall_score']:.2f}")
        
        # This would update custom properties with quality information
        pass
    
    def exercise_6_search_discovery(self):
        """
        Exercise 6: Implement search and discovery workflows
        
        TODO: Create search queries and discovery patterns
        """
        print("\n=== Exercise 6: Search and Discovery ===")
        
        # TODO: Implement various search patterns
        search_queries = [
            "user data",
            "revenue analytics", 
            "customer orders",
            "tag:pii",
            "owner:data-team"
        ]
        
        for query in search_queries:
            results = self._search_catalog(query)
            print(f"Search '{query}': {len(results)} results")
        
        # TODO: Implement faceted search
        faceted_results = self._faceted_search({
            'domain': 'ecommerce',
            'platform': 'postgres',
            'tags': ['production']
        })
        print(f"Faceted search: {len(faceted_results)} results")
    
    def _search_catalog(self, query: str) -> List[Dict[str, Any]]:
        """Search the data catalog"""
        # TODO: Implement catalog search using DataHub APIs
        # This would call DataHub's search endpoints
        
        # Mock search results for demonstration
        mock_results = [
            {
                'urn': 'urn:li:dataset:(urn:li:dataPlatform:postgres,ecommerce.public.users,PROD)',
                'name': 'users',
                'description': 'User account information',
                'platform': 'postgres',
                'tags': ['pii', 'production']
            }
        ]
        
        return mock_results
    
    def _faceted_search(self, facets: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform faceted search with filters"""
        # TODO: Implement faceted search
        return []
    
    def exercise_7_governance_policies(self):
        """
        Exercise 7: Implement governance policies
        
        TODO: Create and enforce data governance rules
        """
        print("\n=== Exercise 7: Governance Policies ===")
        
        # TODO: Define governance policies
        policies = {
            'pii_data_policy': {
                'description': 'All PII data must have proper classification and access controls',
                'rules': [
                    'PII datasets must be tagged with "pii" tag',
                    'PII datasets must have designated data owner',
                    'PII datasets must have retention policy defined'
                ],
                'enforcement': 'automated'
            },
            'data_quality_policy': {
                'description': 'Production datasets must meet minimum quality thresholds',
                'rules': [
                    'Overall quality score must be >= 0.90',
                    'Completeness score must be >= 0.95',
                    'Quality checks must run daily'
                ],
                'enforcement': 'automated'
            }
        }
        
        # TODO: Implement policy enforcement
        for policy_name, policy in policies.items():
            violations = self._check_policy_compliance(policy)
            print(f"Policy '{policy_name}': {len(violations)} violations found")
    
    def _check_policy_compliance(self, policy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check compliance with a governance policy"""
        # TODO: Implement policy compliance checking
        violations = []
        
        # This would scan all datasets and check against policy rules
        # For demonstration, return empty violations
        
        return violations
    
    def run_all_exercises(self):
        """Run all exercises in sequence"""
        print("🚀 Starting Data Catalog Exercises")
        print("=" * 50)
        
        # Exercise 1: Setup
        self.exercise_1_setup_datahub()
        
        # Exercise 2: Metadata Extraction
        metadata = self.exercise_2_metadata_extraction()
        
        # Exercise 3: DataHub Integration
        self.exercise_3_datahub_integration(metadata)
        
        # Exercise 4: Lineage Tracking
        self.exercise_4_lineage_tracking()
        
        # Exercise 5: Quality Integration
        self.exercise_5_quality_integration()
        
        # Exercise 6: Search and Discovery
        self.exercise_6_search_discovery()
        
        # Exercise 7: Governance Policies
        self.exercise_7_governance_policies()
        
        print("\n✅ All exercises completed!")
        print("\nNext steps:")
        print("1. Complete the TODO items in each exercise")
        print("2. Set up DataHub locally using Docker Compose")
        print("3. Test the metadata extraction and ingestion")
        print("4. Explore the DataHub UI for search and discovery")
        print("5. Implement custom governance policies")

def main():
    """Run the data catalog exercises"""
    exercise = DataCatalogExercise()
    exercise.run_all_exercises()

if __name__ == "__main__":
    main()
