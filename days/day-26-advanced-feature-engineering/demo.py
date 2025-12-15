#!/usr/bin/env python3
"""
Day 26: Advanced Feature Engineering - Interactive Demo

This demo showcases comprehensive feature engineering capabilities:
1. Time series feature engineering with financial data
2. NLP feature extraction from customer feedback
3. Automated feature selection and generation
4. Production pipeline with monitoring
5. Performance benchmarking and validation
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout

console = Console()

class AdvancedFeatureEngineeringDemo:
    """Interactive demo for advanced feature engineering"""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.console = Console()
        
    def print_header(self, title: str):
        """Print formatted header"""
        self.console.print(Panel(title, style="bold blue"))
    
    def print_success(self, message: str):
        """Print success message"""
        self.console.print(f"✅ {message}", style="green")
    
    def print_error(self, message: str):
        """Print error message"""
        self.console.print(f"❌ {message}", style="red")
    
    def print_info(self, message: str):
        """Print info message"""
        self.console.print(f"ℹ️  {message}", style="blue")
    
    def check_services(self) -> bool:
        """Check if all services are running"""
        
        self.print_header("🔍 Checking Service Health")
        
        services = {
            "Feature Engineering API": f"{self.api_base_url}/health",
            "PostgreSQL": "localhost:5432",
            "Redis": "localhost:6379",
            "Grafana": "http://localhost:3000",
            "Prometheus": "http://localhost:9090",
            "Jupyter Lab": "http://localhost:8888"
        }
        
        all_healthy = True
        
        for service, endpoint in services.items():
            try:
                if endpoint.startswith("http"):
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code == 200:
                        self.print_success(f"{service} is healthy")
                    else:
                        self.print_error(f"{service} returned status {response.status_code}")
                        all_healthy = False
                else:
                    # For non-HTTP services, just mark as assumed healthy
                    self.print_success(f"{service} is assumed healthy")
            except Exception as e:
                self.print_error(f"{service} is not accessible: {str(e)}")
                all_healthy = False
        
        return all_healthy
    
    def demonstrate_time_series_features(self):
        """Demonstrate time series feature engineering"""
        
        self.print_header("🕒 Time Series Feature Engineering Demo")
        
        # Load sample transaction data
        try:
            transactions = pd.read_parquet('data/transactions.parquet')
            self.print_success(f"Loaded {len(transactions)} transactions")
            
            # Show original data sample
            table = Table(title="Sample Transaction Data")
            table.add_column("Customer ID", style="cyan")
            table.add_column("Timestamp", style="magenta")
            table.add_column("Amount", style="green")
            table.add_column("Category", style="yellow")
            
            for _, row in transactions.head(5).iterrows():
                table.add_row(
                    str(row['customer_id']),
                    str(row['timestamp'])[:19],
                    f"${row['amount']:.2f}",
                    row['category']
                )
            
            self.console.print(table)
            
            # Demonstrate feature engineering
            from solution import AdvancedTimeSeriesFeatureEngineer
            
            ts_engineer = AdvancedTimeSeriesFeatureEngineer()
            
            # Create temporal features
            self.print_info("Creating temporal features...")
            df_temporal = ts_engineer.create_temporal_features(transactions.head(1000), 'timestamp')
            
            # Create lag features
            self.print_info("Creating lag features...")
            df_lags = ts_engineer.create_lag_features(
                df_temporal, 'amount', 'customer_id', [1, 6, 24]
            )
            
            # Create rolling features
            self.print_info("Creating rolling window features...")
            df_rolling = ts_engineer.create_rolling_features(
                df_lags, 'amount', 'customer_id', [6, 24, 168]
            )
            
            # Show results
            feature_summary = Table(title="Time Series Feature Engineering Results")
            feature_summary.add_column("Stage", style="cyan")
            feature_summary.add_column("Features Created", style="green")
            feature_summary.add_column("Total Features", style="magenta")
            
            feature_summary.add_row("Original Data", "0", str(len(transactions.columns)))
            feature_summary.add_row("Temporal Features", str(len(df_temporal.columns) - len(transactions.columns)), str(len(df_temporal.columns)))
            feature_summary.add_row("Lag Features", str(len(df_lags.columns) - len(df_temporal.columns)), str(len(df_lags.columns)))
            feature_summary.add_row("Rolling Features", str(len(df_rolling.columns) - len(df_lags.columns)), str(len(df_rolling.columns)))
            
            self.console.print(feature_summary)
            
        except Exception as e:
            self.print_error(f"Time series demo failed: {str(e)}")
    
    def demonstrate_nlp_features(self):
        """Demonstrate NLP feature engineering"""
        
        self.print_header("📝 NLP Feature Engineering Demo")
        
        try:
            feedback = pd.read_parquet('data/feedback.parquet')
            self.print_success(f"Loaded {len(feedback)} feedback records")
            
            # Show sample feedback
            table = Table(title="Sample Customer Feedback")
            table.add_column("Customer ID", style="cyan")
            table.add_column("Feedback Text", style="green", max_width=50)
            table.add_column("Sentiment", style="magenta")
            
            for _, row in feedback.head(3).iterrows():
                table.add_row(
                    str(row['customer_id']),
                    row['feedback_text'][:47] + "..." if len(row['feedback_text']) > 50 else row['feedback_text'],
                    row['sentiment']
                )
            
            self.console.print(table)
            
            # Demonstrate NLP feature engineering
            from solution import ProductionNLPFeatureEngineer
            
            nlp_engineer = ProductionNLPFeatureEngineer()
            
            # Extract linguistic features
            self.print_info("Extracting linguistic features...")
            sample_texts = feedback['feedback_text'].head(100).tolist()
            
            linguistic_features = []
            for text in sample_texts[:5]:  # Demo on small sample
                features = nlp_engineer.extract_linguistic_features(text)
                linguistic_features.append(features)
            
            linguistic_df = pd.DataFrame(linguistic_features)
            
            # Create TF-IDF features
            self.print_info("Creating TF-IDF features...")
            tfidf_features = nlp_engineer.create_tfidf_features(sample_texts, max_features=50)
            
            # Show results
            nlp_summary = Table(title="NLP Feature Engineering Results")
            nlp_summary.add_column("Feature Type", style="cyan")
            nlp_summary.add_column("Features Count", style="green")
            nlp_summary.add_column("Description", style="yellow")
            
            nlp_summary.add_row("Linguistic Features", str(len(linguistic_df.columns)), "Length, POS, sentiment, readability")
            nlp_summary.add_row("TF-IDF Features", str(tfidf_features.shape[1]), "Term frequency with SVD reduction")
            nlp_summary.add_row("Total NLP Features", str(len(linguistic_df.columns) + tfidf_features.shape[1]), "Combined feature set")
            
            self.console.print(nlp_summary)
            
            # Show sample linguistic features
            feature_details = Table(title="Sample Linguistic Features")
            feature_details.add_column("Feature", style="cyan")
            feature_details.add_column("Value", style="green")
            
            if linguistic_features:
                sample_features = linguistic_features[0]
                for key, value in list(sample_features.items())[:8]:
                    feature_details.add_row(key, f"{value:.3f}" if isinstance(value, float) else str(value))
            
            self.console.print(feature_details)
            
        except Exception as e:
            self.print_error(f"NLP demo failed: {str(e)}")
    
    def demonstrate_feature_selection(self):
        """Demonstrate automated feature selection"""
        
        self.print_header("🎯 Automated Feature Selection Demo")
        
        try:
            # Create synthetic dataset for demonstration
            np.random.seed(42)
            n_samples, n_features = 1000, 200
            
            # Create features with some being relevant
            X = pd.DataFrame(
                np.random.randn(n_samples, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            )
            
            # Create target with some features being predictive
            relevant_features = [0, 5, 10, 15, 20, 25, 30]
            y = (X.iloc[:, relevant_features].sum(axis=1) + 
                 np.random.randn(n_samples) * 0.1 > 0).astype(int)
            
            self.print_info(f"Created synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")
            self.print_info(f"Relevant features: {len(relevant_features)} out of {n_features}")
            
            # Demonstrate feature selection
            from solution import EnsembleFeatureSelector
            
            selector = EnsembleFeatureSelector(task_type='classification')
            
            # Test individual methods
            methods_results = {}
            
            self.print_info("Running individual selection methods...")
            
            # Univariate selection
            start_time = time.time()
            univariate_features = selector.univariate_selection(X, y, k=50)
            methods_results['Univariate'] = {
                'features': len(univariate_features),
                'time': time.time() - start_time,
                'relevant_captured': len(set(univariate_features) & set([f'feature_{i}' for i in relevant_features]))
            }
            
            # Mutual information selection
            start_time = time.time()
            mi_features = selector.mutual_information_selection(X, y, k=50)
            methods_results['Mutual Info'] = {
                'features': len(mi_features),
                'time': time.time() - start_time,
                'relevant_captured': len(set(mi_features) & set([f'feature_{i}' for i in relevant_features]))
            }
            
            # Ensemble selection
            start_time = time.time()
            ensemble_features = selector.ensemble_selection(X, y, voting_threshold=0.3)
            methods_results['Ensemble'] = {
                'features': len(ensemble_features),
                'time': time.time() - start_time,
                'relevant_captured': len(set(ensemble_features) & set([f'feature_{i}' for i in relevant_features]))
            }
            
            # Show results
            selection_table = Table(title="Feature Selection Results")
            selection_table.add_column("Method", style="cyan")
            selection_table.add_column("Features Selected", style="green")
            selection_table.add_column("Relevant Captured", style="magenta")
            selection_table.add_column("Time (s)", style="yellow")
            selection_table.add_column("Precision", style="blue")
            
            for method, results in methods_results.items():
                precision = results['relevant_captured'] / max(results['features'], 1)
                selection_table.add_row(
                    method,
                    str(results['features']),
                    f"{results['relevant_captured']}/{len(relevant_features)}",
                    f"{results['time']:.2f}",
                    f"{precision:.3f}"
                )
            
            self.console.print(selection_table)
            
        except Exception as e:
            self.print_error(f"Feature selection demo failed: {str(e)}")
    
    def demonstrate_feature_generation(self):
        """Demonstrate automated feature generation"""
        
        self.print_header("🔧 Automated Feature Generation Demo")
        
        try:
            # Load customer data for feature generation
            customers = pd.read_parquet('data/customers.parquet')
            
            # Select subset of numeric features
            base_features = customers[['age', 'income', 'credit_score', 'account_age_years']].head(1000)
            
            self.print_info(f"Starting with {len(base_features.columns)} base features")
            
            from solution import AdvancedFeatureGenerator
            
            generator = AdvancedFeatureGenerator(max_features=500)
            
            # Generate different types of features
            generation_results = {}
            
            # Ratio features
            self.print_info("Generating ratio features...")
            start_time = time.time()
            ratio_features = generator.generate_ratio_features(base_features, max_ratios=20)
            generation_results['Ratio Features'] = {
                'original': len(base_features.columns),
                'generated': len(ratio_features.columns) - len(base_features.columns),
                'time': time.time() - start_time
            }
            
            # Binning features
            self.print_info("Generating binning features...")
            start_time = time.time()
            binning_features = generator.generate_binning_features(base_features, n_bins=5)
            generation_results['Binning Features'] = {
                'original': len(base_features.columns),
                'generated': len(binning_features.columns) - len(base_features.columns),
                'time': time.time() - start_time
            }
            
            # Statistical features
            self.print_info("Generating statistical features...")
            start_time = time.time()
            statistical_features = generator.generate_statistical_features(base_features)
            generation_results['Statistical Features'] = {
                'original': len(base_features.columns),
                'generated': len(statistical_features.columns) - len(base_features.columns),
                'time': time.time() - start_time
            }
            
            # Show results
            generation_table = Table(title="Feature Generation Results")
            generation_table.add_column("Generation Type", style="cyan")
            generation_table.add_column("Original Features", style="green")
            generation_table.add_column("Generated Features", style="magenta")
            generation_table.add_column("Time (s)", style="yellow")
            generation_table.add_column("Expansion Ratio", style="blue")
            
            for gen_type, results in generation_results.items():
                expansion_ratio = results['generated'] / max(results['original'], 1)
                generation_table.add_row(
                    gen_type,
                    str(results['original']),
                    str(results['generated']),
                    f"{results['time']:.2f}",
                    f"{expansion_ratio:.1f}x"
                )
            
            self.console.print(generation_table)
            
        except Exception as e:
            self.print_error(f"Feature generation demo failed: {str(e)}")
    
    def demonstrate_production_pipeline(self):
        """Demonstrate production feature engineering pipeline"""
        
        self.print_header("🏭 Production Pipeline Demo")
        
        try:
            # Load sample data
            customers = pd.read_parquet('data/customers.parquet')
            
            # Select features for pipeline demo
            pipeline_data = customers[['age', 'income', 'credit_score', 'account_type', 'account_age_years']].head(1000)
            
            # Create synthetic target
            np.random.seed(42)
            y = (pipeline_data['credit_score'] < 600).astype(int)
            
            self.print_info(f"Pipeline input: {pipeline_data.shape}")
            
            from solution import ProductionFeatureEngineeringPipeline
            
            config = {
                'task_type': 'classification',
                'max_features': 50
            }
            
            pipeline = ProductionFeatureEngineeringPipeline(config)
            
            # Fit and transform
            self.print_info("Fitting production pipeline...")
            start_time = time.time()
            X_transformed = pipeline.fit_transform(pipeline_data, y)
            pipeline_time = time.time() - start_time
            
            # Show pipeline results
            pipeline_table = Table(title="Production Pipeline Results")
            pipeline_table.add_column("Metric", style="cyan")
            pipeline_table.add_column("Value", style="green")
            
            pipeline_table.add_row("Input Shape", f"{pipeline_data.shape}")
            pipeline_table.add_row("Output Shape", f"{X_transformed.shape}")
            pipeline_table.add_row("Feature Reduction", f"{len(pipeline_data.columns)} → {X_transformed.shape[1]}")
            pipeline_table.add_row("Processing Time", f"{pipeline_time:.2f}s")
            pipeline_table.add_row("Throughput", f"{len(pipeline_data) / pipeline_time:.0f} samples/sec")
            
            self.console.print(pipeline_table)
            
            self.print_success("Production pipeline successfully processed data")
            
        except Exception as e:
            self.print_error(f"Production pipeline demo failed: {str(e)}")
    
    def show_monitoring_dashboard_info(self):
        """Show information about monitoring dashboards"""
        
        self.print_header("📊 Monitoring Dashboard Information")
        
        dashboard_info = [
            {
                "Service": "Grafana Dashboard",
                "URL": "http://localhost:3000",
                "Credentials": "admin / admin123",
                "Description": "Feature engineering metrics, performance monitoring, and quality tracking"
            },
            {
                "Service": "Prometheus Metrics",
                "URL": "http://localhost:9090",
                "Credentials": "None required",
                "Description": "Raw metrics collection and querying interface"
            },
            {
                "Service": "Jupyter Lab",
                "URL": "http://localhost:8888",
                "Credentials": "None required",
                "Description": "Interactive feature engineering development and analysis"
            },
            {
                "Service": "Feature Engineering API",
                "URL": "http://localhost:8000",
                "Credentials": "None required",
                "Description": "Production feature engineering endpoints and health checks"
            }
        ]
        
        for info in dashboard_info:
            table = Table(title=info["Service"])
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in info.items():
                if key != "Service":
                    table.add_row(key, value)
            
            self.console.print(table)
            self.console.print()
    
    def run_complete_demo(self):
        """Run the complete feature engineering demo"""
        
        self.console.print(Panel("🚀 Advanced Feature Engineering - Complete Demo", style="bold blue"))
        
        # Check services
        if not self.check_services():
            self.print_error("Some services are not healthy. Please run 'docker-compose up -d' first.")
            return
        
        # Wait for user input between sections
        input("\nPress Enter to continue to Time Series Feature Engineering Demo...")
        self.demonstrate_time_series_features()
        
        input("\nPress Enter to continue to NLP Feature Engineering Demo...")
        self.demonstrate_nlp_features()
        
        input("\nPress Enter to continue to Feature Selection Demo...")
        self.demonstrate_feature_selection()
        
        input("\nPress Enter to continue to Feature Generation Demo...")
        self.demonstrate_feature_generation()
        
        input("\nPress Enter to continue to Production Pipeline Demo...")
        self.demonstrate_production_pipeline()
        
        input("\nPress Enter to see Monitoring Dashboard Information...")
        self.show_monitoring_dashboard_info()
        
        # Final summary
        self.print_header("🎉 Demo Complete!")
        
        self.console.print("""
✅ Advanced Feature Engineering Demo Summary:
   • Time series feature engineering with temporal, lag, and rolling features
   • NLP feature extraction with linguistic analysis and TF-IDF
   • Automated feature selection with ensemble voting
   • Feature generation with ratios, binning, and statistical transforms
   • Production pipeline with preprocessing and monitoring

🚀 Next Steps:
   1. Explore Jupyter notebooks for interactive development
   2. Complete the exercises in exercise.py
   3. Monitor feature quality with Grafana dashboards
   4. Build your own feature engineering pipelines

📚 Resources:
   • README.md - Complete documentation and theory
   • solution.py - Full production implementation
   • notebooks/ - Interactive examples and tutorials
        """)

if __name__ == "__main__":
    demo = AdvancedFeatureEngineeringDemo()
    demo.run_complete_demo()