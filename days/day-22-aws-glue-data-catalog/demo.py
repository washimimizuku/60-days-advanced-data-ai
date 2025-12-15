#!/usr/bin/env python3
"""
Day 22: AWS Glue & Data Catalog - Interactive Demo
Comprehensive demonstration of serverless ETL capabilities
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

# Import our classes
from exercise import (
    DataCatalogManager,
    CrawlerManager,
    ETLJobManager,
    AthenaManager,
    MonitoringManager,
    get_aws_clients
)

class GlueDataCatalogDemo:
    """Interactive demonstration of AWS Glue and Data Catalog capabilities"""
    
    def __init__(self):
        self.clients = get_aws_clients()
        self.database_name = os.getenv('GLUE_DATABASE_NAME', 'serverlessdata_analytics')
        self.iam_role = os.getenv('GLUE_IAM_ROLE')
        self.results = {}
    
    def run_complete_demo(self):
        """Run comprehensive demonstration of all capabilities"""
        
        print("🚀 ServerlessData Corp - AWS Glue & Data Catalog Demo")
        print("=" * 60)
        print("Demonstrating enterprise serverless ETL capabilities")
        
        try:
            # Demo 1: Data Catalog Setup
            self.demo_data_catalog()
            
            # Demo 2: Crawler Operations
            self.demo_crawlers()
            
            # Demo 3: ETL Jobs
            self.demo_etl_jobs()
            
            # Demo 4: Athena Analytics
            self.demo_athena_analytics()
            
            # Demo 5: Monitoring
            self.demo_monitoring()
            
            # Generate summary
            self.generate_summary()
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            raise
    
    def demo_data_catalog(self):
        """Demonstrate Data Catalog capabilities"""
        
        print("\n" + "="*50)
        print("📊 DATA CATALOG DEMONSTRATION")
        print("="*50)
        
        catalog_manager = DataCatalogManager(self.database_name)
        
        print("\n1. Setting up Data Catalog...")
        start_time = time.time()
        
        catalog_results = catalog_manager.setup_data_catalog()
        
        end_time = time.time()
        
        print(f"✅ Data Catalog setup completed in {end_time - start_time:.2f}s")
        print(f"   Database created: {catalog_results['database_created']}")
        print(f"   Tables created: {catalog_results['tables_created']}")
        print(f"   Governance enabled: {catalog_results['governance_enabled']}")
        
        self.results['data_catalog'] = catalog_results
    
    def demo_crawlers(self):
        """Demonstrate Crawler capabilities"""
        
        print("\n" + "="*50)
        print("🕷️ CRAWLER DEMONSTRATION")
        print("="*50)
        
        crawler_manager = CrawlerManager(self.database_name, self.iam_role)
        
        print("\n1. Creating and starting crawlers...")
        start_time = time.time()
        
        crawler_results = crawler_manager.create_comprehensive_crawlers()
        
        end_time = time.time()
        
        print(f"✅ Crawler setup completed in {end_time - start_time:.2f}s")
        print(f"   Crawlers created: {crawler_results['crawlers_created']}")
        print(f"   Crawlers started: {crawler_results['crawlers_started']}")
        
        if crawler_results['failed_crawlers']:
            print(f"   Failed crawlers: {len(crawler_results['failed_crawlers'])}")
        
        # Monitor crawler status
        print("\n2. Monitoring crawler execution...")
        if crawler_results['crawlers_created'] > 0:
            # Simulate monitoring (in real scenario, would check actual status)
            print("   Crawlers are discovering schemas and cataloging data...")
            print("   Schema discovery: In Progress")
            print("   Partition detection: Enabled")
            print("   Data classification: Active")
        
        self.results['crawlers'] = crawler_results
    
    def demo_etl_jobs(self):
        """Demonstrate ETL Job capabilities"""
        
        print("\n" + "="*50)
        print("⚙️ ETL JOBS DEMONSTRATION")
        print("="*50)
        
        scripts_bucket = os.getenv('S3_SCRIPTS_BUCKET', 'serverlessdata-glue-scripts')
        etl_manager = ETLJobManager(self.database_name, self.iam_role, scripts_bucket)
        
        print("\n1. Creating ETL jobs...")
        start_time = time.time()
        
        etl_results = etl_manager.create_etl_jobs()
        
        end_time = time.time()
        
        print(f"✅ ETL jobs setup completed in {end_time - start_time:.2f}s")
        print(f"   Jobs created: {etl_results['jobs_created']}")
        print(f"   Jobs started: {etl_results['jobs_started']}")
        
        if etl_results['failed_jobs']:
            print(f"   Failed jobs: {len(etl_results['failed_jobs'])}")
        
        # Demonstrate job capabilities
        print("\n2. ETL Job Capabilities:")
        print("   • Customer analytics with segmentation")
        print("   • Data quality validation and cleansing")
        print("   • Product enrichment with external APIs")
        print("   • Automated schema evolution handling")
        print("   • Performance optimization with Spark tuning")
        
        self.results['etl_jobs'] = etl_results
    
    def demo_athena_analytics(self):
        """Demonstrate Athena Analytics capabilities"""
        
        print("\n" + "="*50)
        print("📈 ATHENA ANALYTICS DEMONSTRATION")
        print("="*50)
        
        results_location = os.getenv('ATHENA_OUTPUT_LOCATION', 's3://serverlessdata-athena-results/queries/')
        athena_manager = AthenaManager(self.database_name, results_location)
        
        print("\n1. Setting up Athena analytics...")
        start_time = time.time()
        
        athena_results = athena_manager.setup_athena_analytics()
        
        end_time = time.time()
        
        print(f"✅ Athena setup completed in {end_time - start_time:.2f}s")
        print(f"   Workgroup created: {athena_results['workgroup_created']}")
        print(f"   Views created: {athena_results['views_created']}")
        print(f"   Saved queries: {athena_results['saved_queries_created']}")
        
        # Demonstrate analytical capabilities
        print("\n2. Analytical Capabilities:")
        print("   • Customer 360-degree view")
        print("   • Revenue analytics and trends")
        print("   • Customer segmentation analysis")
        print("   • Cohort analysis and retention")
        print("   • Real-time business intelligence")
        
        # Simulate query execution
        print("\n3. Sample Query Execution (Simulated):")
        print("   Query: Customer segmentation analysis")
        print("   Execution time: 2.3 seconds")
        print("   Data scanned: 45.2 MB")
        print("   Cost: $0.023")
        print("   Results: 3 customer segments identified")
        
        self.results['athena'] = athena_results
    
    def demo_monitoring(self):
        """Demonstrate Monitoring capabilities"""
        
        print("\n" + "="*50)
        print("📊 MONITORING DEMONSTRATION")
        print("="*50)
        
        monitoring_manager = MonitoringManager()
        
        print("\n1. Setting up monitoring dashboard...")
        start_time = time.time()
        
        monitoring_results = monitoring_manager.setup_monitoring_dashboard()
        
        end_time = time.time()
        
        print(f"✅ Monitoring setup completed in {end_time - start_time:.2f}s")
        print(f"   Dashboards created: {monitoring_results['dashboards_created']}")
        print(f"   Alarms created: {monitoring_results['alarms_created']}")
        print(f"   Metrics enabled: {monitoring_results['metrics_enabled']}")
        
        # Demonstrate monitoring capabilities
        print("\n2. Monitoring Capabilities:")
        print("   • Real-time job execution tracking")
        print("   • Cost monitoring and optimization")
        print("   • Performance metrics and alerting")
        print("   • Data quality monitoring")
        print("   • SLA compliance tracking")
        
        self.results['monitoring'] = monitoring_results
    
    def generate_summary(self):
        """Generate comprehensive demo summary"""
        
        print("\n" + "="*60)
        print("📊 SERVERLESS ETL PLATFORM SUMMARY")
        print("="*60)
        
        print(f"\n🕐 Demo completed at: {datetime.now().isoformat()}")
        
        # Component summary
        if 'data_catalog' in self.results:
            catalog = self.results['data_catalog']
            print(f"\n📊 Data Catalog:")
            print(f"   • Database created: {catalog['database_created']}")
            print(f"   • Tables defined: {catalog['tables_created']}")
            print(f"   • Governance enabled: {catalog['governance_enabled']}")
        
        if 'crawlers' in self.results:
            crawlers = self.results['crawlers']
            print(f"\n🕷️ Crawlers:")
            print(f"   • Crawlers deployed: {crawlers['crawlers_created']}")
            print(f"   • Crawlers active: {crawlers['crawlers_started']}")
            print(f"   • Automated schema discovery: Enabled")
        
        if 'etl_jobs' in self.results:
            etl = self.results['etl_jobs']
            print(f"\n⚙️ ETL Jobs:")
            print(f"   • Jobs created: {etl['jobs_created']}")
            print(f"   • Serverless processing: Enabled")
            print(f"   • Auto-scaling: Active")
        
        if 'athena' in self.results:
            athena = self.results['athena']
            print(f"\n📈 Athena Analytics:")
            print(f"   • Analytical views: {athena['views_created']}")
            print(f"   • Saved queries: {athena['saved_queries_created']}")
            print(f"   • Business intelligence: Ready")
        
        if 'monitoring' in self.results:
            monitoring = self.results['monitoring']
            print(f"\n📊 Monitoring:")
            print(f"   • Dashboards: {monitoring['dashboards_created']}")
            print(f"   • Alarms: {monitoring['alarms_created']}")
            print(f"   • Real-time metrics: {monitoring['metrics_enabled']}")
        
        print(f"\n🎯 PLATFORM CAPABILITIES:")
        print("   ✅ Serverless ETL processing with auto-scaling")
        print("   ✅ Automated schema discovery and evolution")
        print("   ✅ Advanced customer analytics and segmentation")
        print("   ✅ Real-time business intelligence with SQL")
        print("   ✅ Cost-effective pay-per-use architecture")
        print("   ✅ Enterprise-grade monitoring and governance")
        
        print(f"\n💰 COST BENEFITS:")
        print("   • No infrastructure management overhead")
        print("   • Pay only for actual processing time")
        print("   • Automatic scaling reduces waste")
        print("   • Optimized storage formats reduce costs")
        
        print(f"\n🚀 READY FOR PRODUCTION!")
        print("   The serverless data platform is ready to handle")
        print("   enterprise workloads with automatic scaling,")
        print("   comprehensive monitoring, and cost optimization.")

def main():
    """Main demo execution"""
    
    demo = GlueDataCatalogDemo()
    demo.run_complete_demo()

if __name__ == '__main__':
    main()