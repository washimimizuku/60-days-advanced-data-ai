#!/usr/bin/env python3
"""
Day 30: Ensemble Methods - Interactive Demo

Interactive demonstration of ensemble methods with real-time predictions,
model comparison, and performance visualization.
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

# Database and utilities
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnsembleDemo:
    """
    Interactive demonstration of ensemble methods
    """
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.db_url = os.getenv('DATABASE_URL')
        self.redis_url = os.getenv('REDIS_URL')
        
        # Demo data
        self.X_demo = None
        self.y_demo = None
        self.feature_names = None
        
        # Results storage
        self.demo_results = {}
        
    def generate_demo_data(self, n_samples: int = 1000):
        """Generate demonstration dataset"""
        print("Generating demonstration dataset...")
        
        # Generate realistic credit risk data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=25,
            n_informative=18,
            n_redundant=4,
            n_clusters_per_class=3,
            weights=[0.82, 0.18],
            flip_y=0.01,
            class_sep=0.8,
            random_state=42
        )
        
        # Transform to realistic ranges
        X_realistic = self._transform_to_realistic_ranges(X)
        
        self.X_demo = X_realistic
        self.y_demo = y
        
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
        
        print(f"Generated {n_samples} samples with {X.shape[1]} features")
        print(f"Default rate: {np.mean(y):.2%}")
        
    def _transform_to_realistic_ranges(self, X: np.ndarray) -> np.ndarray:
        """Transform features to realistic credit risk ranges"""
        X_transformed = X.copy()
        
        # Income features
        X_transformed[:, 0] = np.exp(X[:, 0] * 0.5 + 11) * 1000  # annual_income
        X_transformed[:, 1] = X_transformed[:, 0] / 12  # monthly_income
        
        # Ratios
        X_transformed[:, 2] = np.clip(np.abs(X[:, 2]) * 0.3 + 0.2, 0, 1.5)  # debt_to_income
        X_transformed[:, 6] = np.clip(np.abs(X[:, 6]) * 0.4, 0, 1)  # credit_utilization
        X_transformed[:, 10] = np.clip(np.abs(X[:, 10]) * 2, 0.5, 5)  # loan_to_income
        X_transformed[:, 13] = np.clip(np.abs(X[:, 13]) * 0.3, 0.1, 0.5)  # down_payment
        
        # Credit score
        X_transformed[:, 3] = np.clip(X[:, 3] * 100 + 650, 300, 850)
        
        # Count features
        X_transformed[:, 5] = np.clip(np.abs(X[:, 5]) * 5 + 3, 1, 25).astype(int)  # accounts
        X_transformed[:, 21] = np.clip(np.abs(X[:, 21]) * 2, 0, 5).astype(int)  # dependents
        X_transformed[:, 24] = np.clip(np.abs(X[:, 24]) * 0.5, 0, 3).astype(int)  # defaults
        
        # Time features
        X_transformed[:, 4] = np.clip(np.abs(X[:, 4]) * 10 + 2, 0, 30)  # credit_history
        X_transformed[:, 8] = np.clip(np.abs(X[:, 8]) * 8 + 1, 0, 30)  # employment
        X_transformed[:, 18] = np.clip(np.abs(X[:, 18]) * 20 + 25, 18, 80)  # age
        X_transformed[:, 23] = np.clip(np.abs(X[:, 23]) * 10 + 1, 0, 20)  # residence
        
        # Loan features
        X_transformed[:, 9] = np.exp(X[:, 9] * 0.8 + 10) * 1000  # loan_amount
        X_transformed[:, 11] = np.clip(np.abs(X[:, 11]) * 20 + 15, 10, 30)  # loan_term
        
        # Asset features
        X_transformed[:, 12] = X_transformed[:, 9] * (1 + np.abs(X[:, 12]) * 0.5)  # property
        X_transformed[:, 14] = np.exp(X[:, 14] * 0.6 + 8) * 1000  # liquid_assets
        X_transformed[:, 16] = np.exp(X[:, 16] * 0.5 + 7) * 1000  # savings
        X_transformed[:, 17] = np.exp(X[:, 17] * 0.4 + 6) * 1000  # checking
        
        # Categorical features
        X_transformed[:, 19] = np.clip(np.abs(X[:, 19]) * 4, 0, 4).astype(int)  # education
        X_transformed[:, 20] = np.clip(np.abs(X[:, 20]) * 3, 0, 3).astype(int)  # marital
        X_transformed[:, 22] = np.clip(np.abs(X[:, 22]) * 3, 0, 3).astype(int)  # residence_type
        
        # Score features
        X_transformed[:, 7] = np.clip(X[:, 7] * 20 + 70, 0, 100)  # payment_history
        X_transformed[:, 15] = np.clip(np.abs(X[:, 15]) * 3, 0, 10).astype(int)  # investments
        
        return X_transformed
    
    def check_api_health(self) -> bool:
        """Check if API server is running"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_single_prediction(self):
        """Test single prediction via API"""
        print("\n" + "="*60)
        print("TESTING SINGLE PREDICTION")
        print("="*60)
        
        if not self.check_api_health():
            print("❌ API server is not running. Please start with: python api.py")
            return
        
        # Select a random sample
        sample_idx = np.random.randint(0, len(self.X_demo))
        sample_features = self.X_demo[sample_idx].tolist()
        actual_label = self.y_demo[sample_idx]
        
        print(f"Testing sample {sample_idx}:")
        print(f"Actual label: {'Default' if actual_label == 1 else 'No Default'}")
        
        # Display key features
        key_features = {
            'Annual Income': f"${sample_features[0]:,.0f}",
            'Credit Score': f"{sample_features[3]:.0f}",
            'Debt-to-Income': f"{sample_features[2]:.2%}",
            'Credit Utilization': f"{sample_features[6]:.2%}",
            'Age': f"{sample_features[18]:.0f} years",
            'Previous Defaults': f"{sample_features[24]:.0f}"
        }
        
        print("\nKey Features:")
        for feature, value in key_features.items():
            print(f"  {feature}: {value}")
        
        # Make prediction
        try:
            payload = {
                "features": sample_features,
                "return_probabilities": True,
                "request_id": f"demo_{int(time.time())}"
            }
            
            response = requests.post(f"{self.api_base_url}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                prediction = result['prediction']
                probability = result.get('probability', 0)
                latency = result['latency_ms']
                model_used = result['model_used']
                
                print(f"\n✅ Prediction Results:")
                print(f"  Predicted: {'Default' if prediction == 1 else 'No Default'}")
                print(f"  Probability: {probability:.3f}")
                print(f"  Model Used: {model_used}")
                print(f"  Latency: {latency:.2f}ms")
                print(f"  Correct: {'✅' if prediction == actual_label else '❌'}")
                
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
    
    def test_batch_prediction(self, batch_size: int = 10):
        """Test batch prediction via API"""
        print("\n" + "="*60)
        print(f"TESTING BATCH PREDICTION ({batch_size} samples)")
        print("="*60)
        
        if not self.check_api_health():
            print("❌ API server is not running. Please start with: python api.py")
            return
        
        # Select random samples
        sample_indices = np.random.choice(len(self.X_demo), batch_size, replace=False)
        batch_features = [self.X_demo[i].tolist() for i in sample_indices]
        actual_labels = [self.y_demo[i] for i in sample_indices]
        
        try:
            payload = {
                "features_batch": batch_features,
                "return_probabilities": True,
                "batch_id": f"demo_batch_{int(time.time())}"
            }
            
            start_time = time.time()
            response = requests.post(f"{self.api_base_url}/predict/batch", json=payload)
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                predictions = result['predictions']
                probabilities = result.get('probabilities', [0] * batch_size)
                latency = result['latency_ms']
                model_used = result['model_used']
                
                # Calculate accuracy
                accuracy = accuracy_score(actual_labels, predictions)
                
                print(f"✅ Batch Prediction Results:")
                print(f"  Batch Size: {batch_size}")
                print(f"  Model Used: {model_used}")
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  API Latency: {latency:.2f}ms")
                print(f"  Total Time: {total_time*1000:.2f}ms")
                print(f"  Avg per Sample: {latency/batch_size:.2f}ms")
                
                # Show sample results
                print(f"\nSample Results:")
                for i in range(min(5, batch_size)):
                    actual = actual_labels[i]
                    pred = predictions[i]
                    prob = probabilities[i]
                    status = "✅" if actual == pred else "❌"
                    print(f"  Sample {i+1}: Actual={actual}, Pred={pred}, Prob={prob:.3f} {status}")
                
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Error making batch prediction: {e}")
    
    def compare_local_vs_api(self, n_samples: int = 100):
        """Compare local ensemble vs API predictions"""
        print("\n" + "="*60)
        print("COMPARING LOCAL ENSEMBLE VS API")
        print("="*60)
        
        # Create local ensemble
        print("Training local ensemble...")
        
        # Split demo data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_demo, self.y_demo, test_size=0.3, random_state=42, stratify=self.y_demo
        )
        
        # Create simple ensemble
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        
        local_ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft'
        )
        
        # Train local ensemble
        local_ensemble.fit(X_train, y_train)
        
        # Select test samples
        test_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        test_features = X_test[test_indices]
        test_labels = y_test[test_indices]
        
        # Local predictions
        local_predictions = local_ensemble.predict(test_features)
        local_probabilities = local_ensemble.predict_proba(test_features)[:, 1]
        local_accuracy = accuracy_score(test_labels, local_predictions)
        
        print(f"Local Ensemble Accuracy: {local_accuracy:.3f}")
        
        # API predictions (if available)
        if self.check_api_health():
            print("Getting API predictions...")
            
            try:
                # Batch API prediction
                payload = {
                    "features_batch": test_features.tolist(),
                    "return_probabilities": True
                }
                
                response = requests.post(f"{self.api_base_url}/predict/batch", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    api_predictions = result['predictions']
                    api_probabilities = result.get('probabilities', [0] * len(test_features))
                    api_accuracy = accuracy_score(test_labels, api_predictions)
                    
                    print(f"API Ensemble Accuracy: {api_accuracy:.3f}")
                    
                    # Compare predictions
                    agreement = accuracy_score(local_predictions, api_predictions)
                    print(f"Local vs API Agreement: {agreement:.3f}")
                    
                    # Correlation of probabilities
                    prob_correlation = np.corrcoef(local_probabilities, api_probabilities)[0, 1]
                    print(f"Probability Correlation: {prob_correlation:.3f}")
                    
                else:
                    print(f"❌ API prediction failed: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error comparing with API: {e}")
        else:
            print("❌ API server not available for comparison")
    
    def demonstrate_ensemble_diversity(self):
        """Demonstrate the importance of ensemble diversity"""
        print("\n" + "="*60)
        print("DEMONSTRATING ENSEMBLE DIVERSITY")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_demo, self.y_demo, test_size=0.3, random_state=42, stratify=self.y_demo
        )
        
        # Create diverse models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'Random Forest 2': RandomForestClassifier(n_estimators=50, random_state=123),
            'Random Forest 3': RandomForestClassifier(n_estimators=50, random_state=456)
        }
        
        # Train models and get predictions
        model_predictions = {}
        model_accuracies = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            model_predictions[name] = predictions
            model_accuracies[name] = accuracy
            
            print(f"{name}: {accuracy:.3f}")
        
        # Create ensembles
        print("\nEnsemble Combinations:")
        
        # Diverse ensemble (RF + GB)
        diverse_ensemble = VotingClassifier(
            estimators=[('rf', models['Random Forest']), ('gb', models['Gradient Boosting'])],
            voting='hard'
        )
        diverse_ensemble.fit(X_train, y_train)
        diverse_accuracy = diverse_ensemble.score(X_test, y_test)
        print(f"Diverse Ensemble (RF + GB): {diverse_accuracy:.3f}")
        
        # Similar ensemble (3 Random Forests)
        similar_ensemble = VotingClassifier(
            estimators=[
                ('rf1', models['Random Forest']),
                ('rf2', models['Random Forest 2']),
                ('rf3', models['Random Forest 3'])
            ],
            voting='hard'
        )
        similar_ensemble.fit(X_train, y_train)
        similar_accuracy = similar_ensemble.score(X_test, y_test)
        print(f"Similar Ensemble (3 RFs): {similar_accuracy:.3f}")
        
        # Calculate diversity metrics
        print("\nDiversity Analysis:")
        
        # Pairwise disagreement
        rf_gb_disagreement = np.mean(model_predictions['Random Forest'] != model_predictions['Gradient Boosting'])
        rf_rf_disagreement = np.mean(model_predictions['Random Forest'] != model_predictions['Random Forest 2'])
        
        print(f"RF vs GB Disagreement: {rf_gb_disagreement:.3f}")
        print(f"RF vs RF2 Disagreement: {rf_rf_disagreement:.3f}")
        
        print(f"\nKey Insight: Diverse models (RF vs GB) disagree more ({rf_gb_disagreement:.3f}) than similar models ({rf_rf_disagreement:.3f})")
        print(f"This diversity often leads to better ensemble performance!")
    
    def visualize_model_performance(self):
        """Create visualizations of model performance"""
        print("\n" + "="*60)
        print("CREATING PERFORMANCE VISUALIZATIONS")
        print("="*60)
        
        # Get model information from API
        if not self.check_api_health():
            print("❌ API server not available for visualization")
            return
        
        try:
            # Get models info
            response = requests.get(f"{self.api_base_url}/models")
            
            if response.status_code == 200:
                models_info = response.json()
                
                if models_info:
                    # Extract performance metrics
                    model_names = []
                    accuracies = []
                    aucs = []
                    
                    for model in models_info:
                        model_names.append(model['name'])
                        metrics = model.get('performance_metrics', {})
                        accuracies.append(metrics.get('accuracy', 0))
                        aucs.append(metrics.get('auc', 0))
                    
                    # Create visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Accuracy comparison
                    bars1 = ax1.bar(range(len(model_names)), accuracies, alpha=0.7, color='skyblue')
                    ax1.set_xticks(range(len(model_names)))
                    ax1.set_xticklabels(model_names, rotation=45, ha='right')
                    ax1.set_ylabel('Accuracy')
                    ax1.set_title('Model Accuracy Comparison')
                    ax1.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, acc in zip(bars1, accuracies):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{acc:.3f}', ha='center', va='bottom')
                    
                    # AUC comparison
                    bars2 = ax2.bar(range(len(model_names)), aucs, alpha=0.7, color='lightcoral')
                    ax2.set_xticks(range(len(model_names)))
                    ax2.set_xticklabels(model_names, rotation=45, ha='right')
                    ax2.set_ylabel('AUC Score')
                    ax2.set_title('Model AUC Comparison')
                    ax2.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, auc in zip(bars2, aucs):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{auc:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    print("✅ Performance visualization created")
                    
                else:
                    print("❌ No model information available")
            else:
                print(f"❌ Failed to get model info: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
    
    def show_feature_importance_demo(self):
        """Demonstrate feature importance in ensemble models"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE DEMONSTRATION")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_demo, self.y_demo, test_size=0.3, random_state=42, stratify=self.y_demo
        )
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get feature importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Display top features
        print("Top 10 Most Important Features:")
        print("-" * 40)
        
        for i in range(min(10, len(self.feature_names))):
            idx = indices[i]
            print(f"{i+1:2d}. {self.feature_names[idx]:25} {importances[idx]:.4f}")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot top 15 features
        top_n = min(15, len(self.feature_names))
        plt.barh(range(top_n), importances[indices[:top_n]], alpha=0.7)
        plt.yticks(range(top_n), [self.feature_names[i] for i in indices[:top_n]])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances (Random Forest)')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("\n✅ Feature importance analysis completed")
    
    def interactive_prediction_demo(self):
        """Interactive prediction with user input"""
        print("\n" + "="*60)
        print("INTERACTIVE PREDICTION DEMO")
        print("="*60)
        
        if not self.check_api_health():
            print("❌ API server not available. Starting local demo...")
            return self._local_interactive_demo()
        
        print("Enter customer information for credit risk assessment:")
        print("(Press Enter to use default values)")
        
        try:
            # Collect user input
            features = {}
            
            # Key features with defaults
            key_inputs = {
                'annual_income': (50000, "Annual Income ($)"),
                'credit_score': (650, "Credit Score (300-850)"),
                'debt_to_income_ratio': (0.3, "Debt-to-Income Ratio (0-1)"),
                'credit_utilization': (0.5, "Credit Utilization (0-1)"),
                'age': (35, "Age (years)"),
                'employment_length': (5, "Employment Length (years)"),
                'loan_amount': (200000, "Loan Amount ($)"),
                'previous_defaults': (0, "Previous Defaults (count)")
            }
            
            for key, (default, prompt) in key_inputs.items():
                user_input = input(f"{prompt} [{default}]: ").strip()
                features[key] = float(user_input) if user_input else default
            
            # Create full feature vector with defaults for missing features
            full_features = [
                features['annual_income'],  # annual_income
                features['annual_income'] / 12,  # monthly_income
                features['debt_to_income_ratio'],  # debt_to_income_ratio
                features['credit_score'],  # credit_score
                10,  # credit_history_length
                8,  # number_of_accounts
                features['credit_utilization'],  # credit_utilization
                75,  # payment_history_score
                features['employment_length'],  # employment_length
                features['loan_amount'],  # loan_amount
                features['loan_amount'] / features['annual_income'],  # loan_to_income_ratio
                30,  # loan_term
                features['loan_amount'] * 1.2,  # property_value
                0.2,  # down_payment_ratio
                features['annual_income'] * 0.1,  # liquid_assets
                2,  # investment_accounts
                features['annual_income'] * 0.05,  # savings_balance
                features['annual_income'] * 0.02,  # checking_balance
                features['age'],  # age
                2,  # education_level
                1,  # marital_status
                1,  # dependents
                1,  # residence_type
                5,  # years_at_residence
                features['previous_defaults']  # previous_defaults
            ]
            
            # Make prediction
            payload = {
                "features": full_features,
                "return_probabilities": True,
                "request_id": f"interactive_{int(time.time())}"
            }
            
            response = requests.post(f"{self.api_base_url}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                prediction = result['prediction']
                probability = result.get('probability', 0)
                model_used = result['model_used']
                
                print(f"\n🎯 CREDIT RISK ASSESSMENT RESULTS")
                print("=" * 50)
                print(f"Risk Level: {'HIGH RISK' if prediction == 1 else 'LOW RISK'}")
                print(f"Default Probability: {probability:.1%}")
                print(f"Model Used: {model_used}")
                
                # Risk interpretation
                if probability < 0.2:
                    risk_msg = "Very Low Risk - Excellent candidate"
                elif probability < 0.4:
                    risk_msg = "Low Risk - Good candidate"
                elif probability < 0.6:
                    risk_msg = "Medium Risk - Requires careful evaluation"
                elif probability < 0.8:
                    risk_msg = "High Risk - Additional scrutiny needed"
                else:
                    risk_msg = "Very High Risk - Consider rejection"
                
                print(f"Interpretation: {risk_msg}")
                
                # Key factors
                print(f"\nKey Input Factors:")
                print(f"  Annual Income: ${features['annual_income']:,.0f}")
                print(f"  Credit Score: {features['credit_score']:.0f}")
                print(f"  Debt-to-Income: {features['debt_to_income_ratio']:.1%}")
                print(f"  Credit Utilization: {features['credit_utilization']:.1%}")
                
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user")
        except Exception as e:
            print(f"❌ Error in interactive demo: {e}")
    
    def _local_interactive_demo(self):
        """Fallback local interactive demo"""
        print("Running local ensemble demo...")
        
        # Create simple local model
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_demo, self.y_demo, test_size=0.3, random_state=42
        )
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        
        print(f"Local model accuracy: {rf.score(X_test, y_test):.3f}")
        print("Local interactive prediction would work here...")
    
    def run_comprehensive_demo(self):
        """Run all demonstration components"""
        print("🚀 ENSEMBLE METHODS COMPREHENSIVE DEMO")
        print("=" * 80)
        
        # Generate demo data
        self.generate_demo_data()
        
        # Run all demos
        demos = [
            ("Single Prediction Test", self.test_single_prediction),
            ("Batch Prediction Test", lambda: self.test_batch_prediction(20)),
            ("Local vs API Comparison", lambda: self.compare_local_vs_api(50)),
            ("Ensemble Diversity Demo", self.demonstrate_ensemble_diversity),
            ("Performance Visualization", self.visualize_model_performance),
            ("Feature Importance Demo", self.show_feature_importance_demo),
            ("Interactive Prediction", self.interactive_prediction_demo)
        ]
        
        for demo_name, demo_func in demos:
            try:
                print(f"\n🎯 Running: {demo_name}")
                demo_func()
                print(f"✅ Completed: {demo_name}")
                
                # Pause between demos
                input("\nPress Enter to continue to next demo...")
                
            except KeyboardInterrupt:
                print(f"\n❌ Demo interrupted: {demo_name}")
                break
            except Exception as e:
                print(f"❌ Error in {demo_name}: {e}")
                continue
        
        print("\n🎉 COMPREHENSIVE DEMO COMPLETED!")
        print("=" * 80)

def main():
    """
    Main function to run the ensemble methods demo
    """
    print("Day 30: Ensemble Methods - Interactive Demo")
    print("Production-ready ensemble learning demonstration")
    print()
    
    demo = EnsembleDemo()
    
    # Check if user wants comprehensive demo or specific tests
    print("Demo Options:")
    print("1. Comprehensive Demo (all features)")
    print("2. Quick API Test")
    print("3. Local Ensemble Demo")
    print("4. Interactive Prediction Only")
    
    try:
        choice = input("\nSelect option (1-4) [1]: ").strip() or "1"
        
        if choice == "1":
            demo.run_comprehensive_demo()
        elif choice == "2":
            demo.generate_demo_data()
            demo.test_single_prediction()
            demo.test_batch_prediction()
        elif choice == "3":
            demo.generate_demo_data()
            demo.demonstrate_ensemble_diversity()
            demo.show_feature_importance_demo()
        elif choice == "4":
            demo.generate_demo_data()
            demo.interactive_prediction_demo()
        else:
            print("Invalid choice, running comprehensive demo...")
            demo.run_comprehensive_demo()
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nError running demo: {e}")

if __name__ == "__main__":
    main()