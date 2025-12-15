#!/usr/bin/env python3
"""
Day 30: Ensemble Methods - Realistic Data Generator

Generates realistic credit risk datasets for ensemble learning experiments.
Includes various data scenarios for testing ensemble robustness.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Tuple, Dict, Any, Optional
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import psycopg2
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CreditRiskDataGenerator:
    """
    Advanced credit risk data generator with realistic patterns and scenarios
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Database connection
        self.db_url = os.getenv('DATABASE_URL')
        self.redis_url = os.getenv('REDIS_URL')
        
        # Feature definitions
        self.feature_names = [
            'annual_income', 'monthly_income', 'debt_to_income_ratio',
            'credit_score', 'credit_history_length', 'number_of_accounts',
            'credit_utilization', 'payment_history_score', 'employment_length',
            'loan_amount', 'loan_to_income_ratio', 'loan_term',
            'property_value', 'down_payment_ratio', 'liquid_assets',
            'investment_accounts', 'savings_balance', 'checking_balance',
            'age', 'education_level', 'marital_status', 'dependents',
            'residence_type', 'years_at_residence', 'previous_defaults'
        ]
        
        # Risk factors and correlations
        self.risk_correlations = {
            'high_debt_to_income': 0.4,
            'low_credit_score': 0.5,
            'short_employment': 0.3,
            'high_utilization': 0.35,
            'previous_defaults': 0.6,
            'low_income': 0.25
        }
    
    def generate_base_dataset(self, n_samples: int = 10000, 
                            default_rate: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate base synthetic credit dataset with realistic characteristics
        """
        logger.info(f"Generating base dataset: {n_samples} samples, {default_rate:.1%} default rate")
        
        # Generate base features using make_classification
        X_base, y = make_classification(
            n_samples=n_samples,
            n_features=len(self.feature_names),
            n_informative=int(len(self.feature_names) * 0.7),
            n_redundant=int(len(self.feature_names) * 0.15),
            n_clusters_per_class=3,
            weights=[1-default_rate, default_rate],
            flip_y=0.01,  # 1% label noise
            class_sep=0.8,
            random_state=self.random_state
        )
        
        # Transform features to realistic ranges
        X_realistic = self._transform_to_realistic_features(X_base)
        
        return X_realistic, y
    
    def _transform_to_realistic_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform synthetic features to realistic credit risk feature ranges
        """
        X_transformed = X.copy()
        
        # Income features (log-normal distribution)
        X_transformed[:, 0] = np.exp(X[:, 0] * 0.5 + 11) * 1000  # annual_income: 30k-200k+
        X_transformed[:, 1] = X_transformed[:, 0] / 12  # monthly_income
        
        # Ratios (bounded between 0 and 1, or reasonable ranges)
        X_transformed[:, 2] = np.clip(np.abs(X[:, 2]) * 0.3 + 0.2, 0, 1.5)  # debt_to_income_ratio
        X_transformed[:, 6] = np.clip(np.abs(X[:, 6]) * 0.4, 0, 1)  # credit_utilization
        X_transformed[:, 13] = np.clip(np.abs(X[:, 13]) * 0.3 + 0.1, 0, 0.5)  # down_payment_ratio
        
        # Credit score (300-850 range)
        X_transformed[:, 3] = np.clip(X[:, 3] * 100 + 650, 300, 850)
        
        # Count features (non-negative integers)
        X_transformed[:, 5] = np.clip(np.abs(X[:, 5]) * 5 + 3, 1, 25).astype(int)  # number_of_accounts
        X_transformed[:, 21] = np.clip(np.abs(X[:, 21]) * 2, 0, 5).astype(int)  # dependents
        X_transformed[:, 24] = np.clip(np.abs(X[:, 24]) * 0.5, 0, 3).astype(int)  # previous_defaults
        
        # Time-based features (years)
        X_transformed[:, 4] = np.clip(np.abs(X[:, 4]) * 10 + 2, 0, 30)  # credit_history_length
        X_transformed[:, 8] = np.clip(np.abs(X[:, 8]) * 8 + 1, 0, 30)  # employment_length
        X_transformed[:, 18] = np.clip(np.abs(X[:, 18]) * 20 + 25, 18, 80)  # age
        X_transformed[:, 23] = np.clip(np.abs(X[:, 23]) * 10 + 1, 0, 20)  # years_at_residence
        
        # Loan features
        X_transformed[:, 9] = np.exp(X[:, 9] * 0.8 + 10) * 1000  # loan_amount: 10k-500k+
        X_transformed[:, 10] = X_transformed[:, 9] / X_transformed[:, 0]  # loan_to_income_ratio
        X_transformed[:, 11] = np.clip(np.abs(X[:, 11]) * 20 + 15, 10, 30)  # loan_term
        
        # Asset features
        X_transformed[:, 12] = X_transformed[:, 9] * (1 + np.abs(X[:, 12]) * 0.5)  # property_value
        X_transformed[:, 14] = np.exp(X[:, 14] * 0.6 + 8) * 1000  # liquid_assets
        X_transformed[:, 16] = np.exp(X[:, 16] * 0.5 + 7) * 1000  # savings_balance
        X_transformed[:, 17] = np.exp(X[:, 17] * 0.4 + 6) * 1000  # checking_balance
        
        # Categorical features (encoded as integers)
        X_transformed[:, 19] = np.clip(np.abs(X[:, 19]) * 4, 0, 4).astype(int)  # education_level
        X_transformed[:, 20] = np.clip(np.abs(X[:, 20]) * 3, 0, 3).astype(int)  # marital_status
        X_transformed[:, 22] = np.clip(np.abs(X[:, 22]) * 3, 0, 3).astype(int)  # residence_type
        
        # Score features (0-100 scale)
        X_transformed[:, 7] = np.clip(X[:, 7] * 20 + 70, 0, 100)  # payment_history_score
        X_transformed[:, 15] = np.clip(np.abs(X[:, 15]) * 3, 0, 10).astype(int)  # investment_accounts
        
        return X_transformed
    
    def add_realistic_correlations(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add realistic correlations between features and target
        """
        logger.info("Adding realistic correlations to dataset")
        
        X_corr = X.copy()
        
        # Increase default probability for high-risk profiles
        high_risk_mask = (
            (X[:, 2] > 0.5) |  # High debt-to-income
            (X[:, 3] < 600) |  # Low credit score
            (X[:, 6] > 0.8) |  # High credit utilization
            (X[:, 24] > 0)     # Previous defaults
        )
        
        # Adjust labels based on risk factors
        risk_adjustment = np.random.random(len(y))
        for i, is_high_risk in enumerate(high_risk_mask):
            if is_high_risk and y[i] == 0 and risk_adjustment[i] < 0.3:
                y[i] = 1  # Convert some low-risk to high-risk
            elif not is_high_risk and y[i] == 1 and risk_adjustment[i] < 0.2:
                y[i] = 0  # Convert some high-risk to low-risk
        
        return X_corr, y
    
    def generate_scenario_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate multiple datasets for different scenarios
        """
        logger.info("Generating scenario datasets")
        
        scenarios = {}
        
        # Scenario 1: Balanced dataset
        X_balanced, y_balanced = self.generate_base_dataset(n_samples=8000, default_rate=0.15)
        X_balanced, y_balanced = self.add_realistic_correlations(X_balanced, y_balanced)
        scenarios['balanced'] = (X_balanced, y_balanced)
        
        # Scenario 2: Imbalanced dataset (high default rate)
        X_imbalanced, y_imbalanced = self.generate_base_dataset(n_samples=6000, default_rate=0.25)
        X_imbalanced, y_imbalanced = self.add_realistic_correlations(X_imbalanced, y_imbalanced)
        scenarios['imbalanced'] = (X_imbalanced, y_imbalanced)
        
        # Scenario 3: Low default rate (conservative lending)
        X_conservative, y_conservative = self.generate_base_dataset(n_samples=10000, default_rate=0.08)
        X_conservative, y_conservative = self.add_realistic_correlations(X_conservative, y_conservative)
        scenarios['conservative'] = (X_conservative, y_conservative)
        
        # Scenario 4: Economic stress scenario (higher defaults)
        X_stress, y_stress = self.generate_base_dataset(n_samples=7000, default_rate=0.30)
        X_stress, y_stress = self.add_realistic_correlations(X_stress, y_stress)
        # Add economic stress factors
        X_stress[:, 0] *= 0.9  # Reduce income
        X_stress[:, 2] *= 1.2  # Increase debt ratios
        X_stress[:, 3] -= 20   # Reduce credit scores
        scenarios['economic_stress'] = (X_stress, y_stress)
        
        return scenarios
    
    def save_datasets(self, scenarios: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                     output_dir: str = './data'):
        """
        Save generated datasets to files and database
        """
        logger.info(f"Saving datasets to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for scenario_name, (X, y) in scenarios.items():
            # Create DataFrame
            df = pd.DataFrame(X, columns=self.feature_names)
            df['default'] = y
            df['scenario'] = scenario_name
            df['generated_at'] = datetime.now()
            
            # Save to CSV
            csv_path = os.path.join(output_dir, f'credit_risk_{scenario_name}.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {scenario_name} dataset: {len(df)} samples to {csv_path}")
            
            # Save metadata
            metadata = {
                'scenario': scenario_name,
                'n_samples': len(df),
                'n_features': len(self.feature_names),
                'default_rate': float(np.mean(y)),
                'feature_names': self.feature_names,
                'generated_at': datetime.now().isoformat(),
                'statistics': {
                    'mean_income': float(np.mean(X[:, 0])),
                    'mean_credit_score': float(np.mean(X[:, 3])),
                    'mean_debt_ratio': float(np.mean(X[:, 2])),
                    'mean_age': float(np.mean(X[:, 18]))
                }
            }
            
            metadata_path = os.path.join(output_dir, f'metadata_{scenario_name}.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save combined dataset
        all_data = []
        for scenario_name, (X, y) in scenarios.items():
            df_scenario = pd.DataFrame(X, columns=self.feature_names)
            df_scenario['default'] = y
            df_scenario['scenario'] = scenario_name
            all_data.append(df_scenario)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_path = os.path.join(output_dir, 'credit_risk_combined.csv')
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"Saved combined dataset: {len(combined_df)} samples to {combined_path}")
    
    def store_in_database(self, scenarios: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """
        Store datasets in PostgreSQL database
        """
        if not self.db_url:
            logger.warning("No database URL provided, skipping database storage")
            return
        
        try:
            import psycopg2
            from psycopg2.extras import execute_values
            
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            # Create datasets table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id SERIAL PRIMARY KEY,
                    scenario VARCHAR(100) NOT NULL,
                    features JSONB NOT NULL,
                    target INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert data
            for scenario_name, (X, y) in scenarios.items():
                logger.info(f"Storing {scenario_name} dataset in database")
                
                # Prepare data for insertion
                data_to_insert = []
                for i in range(len(X)):
                    features_dict = {name: float(X[i, j]) for j, name in enumerate(self.feature_names)}
                    data_to_insert.append((scenario_name, json.dumps(features_dict), int(y[i])))
                
                # Batch insert
                execute_values(
                    cur,
                    "INSERT INTO datasets (scenario, features, target) VALUES %s",
                    data_to_insert,
                    template=None,
                    page_size=1000
                )
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info("Successfully stored datasets in database")
            
        except Exception as e:
            logger.error(f"Error storing datasets in database: {e}")
    
    def cache_in_redis(self, scenarios: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """
        Cache dataset metadata in Redis
        """
        if not self.redis_url:
            logger.warning("No Redis URL provided, skipping Redis caching")
            return
        
        try:
            r = redis.from_url(self.redis_url)
            
            for scenario_name, (X, y) in scenarios.items():
                metadata = {
                    'scenario': scenario_name,
                    'n_samples': len(X),
                    'n_features': X.shape[1],
                    'default_rate': float(np.mean(y)),
                    'generated_at': datetime.now().isoformat()
                }
                
                # Cache metadata
                r.setex(f"dataset_metadata:{scenario_name}", 3600, json.dumps(metadata))
                
                # Cache sample statistics
                stats = {
                    'feature_means': [float(np.mean(X[:, i])) for i in range(X.shape[1])],
                    'feature_stds': [float(np.std(X[:, i])) for i in range(X.shape[1])],
                    'class_distribution': {
                        '0': int(np.sum(y == 0)),
                        '1': int(np.sum(y == 1))
                    }
                }
                r.setex(f"dataset_stats:{scenario_name}", 3600, json.dumps(stats))
            
            logger.info("Successfully cached dataset metadata in Redis")
            
        except Exception as e:
            logger.error(f"Error caching in Redis: {e}")

def main():
    """
    Main function to generate and store credit risk datasets
    """
    logger.info("Starting credit risk data generation")
    
    # Initialize generator
    generator = CreditRiskDataGenerator(random_state=42)
    
    # Generate scenarios
    scenarios = generator.generate_scenario_datasets()
    
    # Save datasets
    generator.save_datasets(scenarios)
    
    # Store in database
    generator.store_in_database(scenarios)
    
    # Cache in Redis
    generator.cache_in_redis(scenarios)
    
    # Print summary
    print("\n" + "="*60)
    print("CREDIT RISK DATA GENERATION COMPLETED")
    print("="*60)
    
    total_samples = sum(len(y) for _, (_, y) in scenarios.items())
    print(f"Total samples generated: {total_samples:,}")
    
    for scenario_name, (X, y) in scenarios.items():
        default_rate = np.mean(y)
        print(f"{scenario_name:15}: {len(y):6,} samples, {default_rate:.1%} default rate")
    
    print("\nDatasets saved to:")
    print("- CSV files: ./data/")
    print("- Database: PostgreSQL")
    print("- Cache: Redis")
    
    logger.info("Data generation completed successfully")

if __name__ == "__main__":
    main()