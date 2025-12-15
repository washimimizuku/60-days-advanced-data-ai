#!/usr/bin/env python3
"""
Day 31: Model Explainability - Interactive Demo

Interactive demonstration of model explainability with SHAP, LIME, and clinical
explanations for healthcare applications.
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

# ML and explainability libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ExplainabilityDemo:
    """
    Interactive demonstration of model explainability
    """
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        
        # Demo data
        self.X_demo = None
        self.y_demo = None
        self.feature_names = None
        self.models = {}
        self.explainers = {}
        
        # Clinical mappings
        self.clinical_mappings = {
            'age': 'Patient Age',
            'length_of_stay': 'Length of Hospital Stay',
            'num_medications': 'Number of Medications',
            'charlson_comorbidity_index': 'Comorbidity Burden',
            'diabetes_severity': 'Diabetes Severity',
            'heart_failure_severity': 'Heart Failure Severity',
            'renal_disease_severity': 'Kidney Disease Severity',
            'liver_disease_severity': 'Liver Disease Severity',
            'hemoglobin_level': 'Hemoglobin Level',
            'creatinine_level': 'Creatinine Level',
            'sodium_level': 'Sodium Level',
            'glucose_level': 'Glucose Level',
            'emergency_admission': 'Emergency Admission',
            'icu_stay': 'ICU Stay',
            'surgical_procedure': 'Surgical Procedure'
        }
    
    def generate_demo_data(self, n_samples: int = 1000):
        """Generate demonstration dataset"""
        print("Generating patient readmission demonstration dataset...")
        
        # Generate realistic patient data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=15,
            n_informative=12,
            n_redundant=2,
            n_clusters_per_class=2,
            weights=[0.75, 0.25],  # 25% readmission rate
            flip_y=0.02,
            class_sep=0.7,
            random_state=42
        )
        
        # Transform to realistic ranges
        X_realistic = self._transform_to_realistic_ranges(X)
        
        self.X_demo = X_realistic
        self.y_demo = y
        
        self.feature_names = [
            'age', 'length_of_stay', 'num_medications', 'charlson_comorbidity_index',
            'diabetes_severity', 'heart_failure_severity', 'renal_disease_severity',
            'liver_disease_severity', 'hemoglobin_level', 'creatinine_level',
            'sodium_level', 'glucose_level', 'emergency_admission', 'icu_stay',
            'surgical_procedure'
        ]
        
        print(f"Generated {n_samples} patient records with {X.shape[1]} features")
        print(f"Readmission rate: {np.mean(y):.2%}")
        
        # Display sample statistics
        df = pd.DataFrame(X_realistic, columns=self.feature_names)
        print("\nSample Patient Statistics:")
        print(df.describe().round(2))
    
    def _transform_to_realistic_ranges(self, X: np.ndarray) -> np.ndarray:
        """Transform features to realistic medical ranges"""
        X_transformed = X.copy()
        
        # Age (20-90 years)
        X_transformed[:, 0] = np.clip(np.abs(X[:, 0]) * 20 + 50, 20, 90)
        
        # Length of stay (1-30 days)
        X_transformed[:, 1] = np.clip(np.abs(X[:, 1]) * 5 + 3, 1, 30)
        
        # Number of medications (0-20)
        X_transformed[:, 2] = np.clip(np.abs(X[:, 2]) * 3 + 2, 0, 20)
        
        # Comorbidity scores (0-10)
        for i in range(3, 8):
            X_transformed[:, i] = np.clip(np.abs(X[:, i]) * 2, 0, 10)
        
        # Lab values (normalized)
        for i in range(8, 12):
            X_transformed[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
        
        # Binary indicators (0 or 1)
        for i in range(12, 15):
            X_transformed[:, i] = (X[:, i] > 0).astype(int)
        
        return X_transformed
    
    def check_api_health(self) -> bool:
        """Check if API server is running"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_api_explanation(self):
        """Test explanation via API"""
        print("\n" + "="*60)
        print("TESTING API EXPLANATION SERVICE")
        print("="*60)
        
        if not self.check_api_health():
            print("❌ API server is not running. Please start with: python api.py")
            return
        
        # Select a sample patient
        sample_idx = np.random.randint(0, len(self.X_demo))
        sample_patient = self.X_demo[sample_idx]
        actual_label = self.y_demo[sample_idx]
        
        # Create patient data dictionary
        patient_data = {
            feature: float(value) 
            for feature, value in zip(self.feature_names, sample_patient)
        }
        
        print(f"Testing explanation for Patient {sample_idx}:")
        print(f"Actual outcome: {'Readmission' if actual_label == 1 else 'No Readmission'}")
        
        # Display key patient characteristics
        print("\nKey Patient Characteristics:")
        key_features = ['age', 'length_of_stay', 'num_medications', 'charlson_comorbidity_index']
        for feature in key_features:
            value = patient_data[feature]
            clinical_name = self.clinical_mappings[feature]
            print(f"  {clinical_name}: {value:.1f}")
        
        # Test different explanation types
        explanation_types = ['shap', 'lime', 'permutation']
        
        for exp_type in explanation_types:
            print(f"\n--- {exp_type.upper()} Explanation ---")
            
            try:
                payload = {
                    "model_name": "patient_readmission_rf",
                    "instance_data": patient_data,
                    "explanation_type": exp_type,
                    "num_features": 10,
                    "return_probabilities": True
                }
                
                response = requests.post(f"{self.api_base_url}/explain", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print(f"✅ Prediction: {'Readmission' if result['prediction'] == 1 else 'No Readmission'}")
                    print(f"   Probability: {result['probability']:.3f}")
                    print(f"   Latency: {result['computation_time_ms']:.2f}ms")
                    print(f"   Cached: {result['cached']}")
                    
                    print("   Top Risk Factors:")
                    for i, exp in enumerate(result['explanations'][:5]):
                        feature = exp['feature']
                        clinical_name = self.clinical_mappings.get(feature, feature)
                        importance = exp['importance']
                        impact = exp['impact']
                        print(f"     {i+1}. {clinical_name}: {importance:.3f} ({impact})")
                    
                else:
                    print(f"❌ API Error: {response.status_code}")
                    print(response.text)
                    
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def test_clinical_explanation(self):
        """Test clinical explanation interface"""
        print("\n" + "="*60)
        print("TESTING CLINICAL EXPLANATION INTERFACE")
        print("="*60)
        
        if not self.check_api_health():
            print("❌ API server is not running")
            return
        
        # Select a high-risk patient
        sample_idx = np.random.randint(0, len(self.X_demo))
        sample_patient = self.X_demo[sample_idx]
        
        # Create patient data dictionary
        patient_data = {
            feature: float(value) 
            for feature, value in zip(self.feature_names, sample_patient)
        }
        
        patient_id = f"DEMO_PATIENT_{sample_idx:04d}"
        
        try:
            payload = {
                "model_name": "patient_readmission_rf",
                "patient_data": patient_data,
                "patient_id": patient_id,
                "include_recommendations": True
            }
            
            response = requests.post(f"{self.api_base_url}/explain/clinical", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"🏥 CLINICAL RISK ASSESSMENT")
                print("=" * 50)
                print(f"Patient ID: {result['patient_id']}")
                print(f"Risk Level: {result['risk_level']} ({result['risk_probability']:.1%})")
                print(f"Model Used: {result['model_used']}")
                
                print(f"\n📊 TOP RISK FACTORS:")
                for i, contrib in enumerate(result['feature_contributions'][:5]):
                    clinical_name = contrib['clinical_name']
                    importance = contrib['importance']
                    value = contrib['feature_value']
                    impact = contrib['impact']
                    print(f"  {i+1}. {clinical_name}: {value:.1f}")
                    print(f"     Impact: {impact} (Score: {importance:.3f})")
                
                print(f"\n💡 CLINICAL RECOMMENDATIONS:")
                for i, rec in enumerate(result['recommendations']):
                    category = rec['category']
                    recommendation = rec['recommendation']
                    priority = rec['priority']
                    print(f"  {i+1}. [{priority}] {category}")
                    print(f"     {recommendation}")
                
            else:
                print(f"❌ Clinical API Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def create_local_models(self):
        """Create local models for comparison"""
        print("\n" + "="*60)
        print("CREATING LOCAL MODELS FOR COMPARISON")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_demo, self.y_demo, test_size=0.3, random_state=42, stratify=self.y_demo
        )
        
        # Create models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'Logistic Regression':
                # Scale features for logistic regression
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'model': model
            }
            
            print(f"  Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        self.models = {name: result['model'] for name, result in results.items()}
        
        return results
    
    def demonstrate_shap_explanations(self):
        """Demonstrate SHAP explanations locally"""
        print("\n" + "="*60)
        print("DEMONSTRATING SHAP EXPLANATIONS")
        print("="*60)
        
        if not SHAP_AVAILABLE:
            print("❌ SHAP not available. Install with: pip install shap")
            return
        
        if not self.models:
            print("Creating local models first...")
            self.create_local_models()
        
        # Use Random Forest for SHAP demo
        model = self.models['Random Forest']
        
        # Split data for explanation
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_demo, self.y_demo, test_size=0.3, random_state=42
        )
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:100])
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        print("✅ SHAP values calculated successfully")
        
        # Create visualizations
        try:
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test[:100], feature_names=self.feature_names, show=False)
            plt.title('SHAP Summary Plot - Patient Readmission Risk Factors')
            plt.tight_layout()
            plt.show()
            
            # Feature importance
            plt.figure(figsize=(10, 6))
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=True)
            
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), 
                      [self.clinical_mappings.get(f, f) for f in importance_df['feature']])
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Feature Importance (SHAP)')
            plt.tight_layout()
            plt.show()
            
            print("✅ SHAP visualizations created")
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
    
    def demonstrate_lime_explanations(self):
        """Demonstrate LIME explanations locally"""
        print("\n" + "="*60)
        print("DEMONSTRATING LIME EXPLANATIONS")
        print("="*60)
        
        if not LIME_AVAILABLE:
            print("❌ LIME not available. Install with: pip install lime")
            return
        
        if not self.models:
            print("Creating local models first...")
            self.create_local_models()
        
        # Use Random Forest for LIME demo
        model = self.models['Random Forest']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_demo, self.y_demo, test_size=0.3, random_state=42
        )
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=['No Readmission', 'Readmission'],
            mode='classification'
        )
        
        # Explain a few instances
        print("Generating LIME explanations for sample patients...")
        
        for i in range(3):
            print(f"\n--- Patient {i+1} ---")
            
            # Generate explanation
            explanation = explainer.explain_instance(
                X_test[i],
                model.predict_proba,
                num_features=10
            )
            
            # Get prediction
            prediction = model.predict([X_test[i]])[0]
            probability = model.predict_proba([X_test[i]])[0, 1]
            
            print(f"Prediction: {'Readmission' if prediction == 1 else 'No Readmission'}")
            print(f"Probability: {probability:.3f}")
            
            print("Top factors:")
            for feature, importance in explanation.as_list()[:5]:
                print(f"  {feature}: {importance:.3f}")
        
        print("✅ LIME explanations generated")
    
    def compare_explanation_methods(self):
        """Compare different explanation methods"""
        print("\n" + "="*60)
        print("COMPARING EXPLANATION METHODS")
        print("="*60)
        
        if not self.check_api_health():
            print("❌ API server not available for comparison")
            return
        
        # Select a sample patient
        sample_idx = 0
        patient_data = {
            feature: float(value) 
            for feature, value in zip(self.feature_names, self.X_demo[sample_idx])
        }
        
        print(f"Comparing explanations for Patient {sample_idx}")
        
        # Test different explanation methods
        methods = ['shap', 'lime', 'permutation']
        results = {}
        
        for method in methods:
            print(f"\nTesting {method.upper()}...")
            
            try:
                payload = {
                    "model_name": "patient_readmission_rf",
                    "instance_data": patient_data,
                    "explanation_type": method,
                    "num_features": 10
                }
                
                start_time = time.time()
                response = requests.post(f"{self.api_base_url}/explain", json=payload)
                api_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    results[method] = {
                        'explanations': result['explanations'],
                        'computation_time': result['computation_time_ms'],
                        'api_time': api_time * 1000,
                        'cached': result['cached']
                    }
                    
                    print(f"  ✅ Success - Computation: {result['computation_time_ms']:.2f}ms, "
                          f"API: {api_time*1000:.2f}ms, Cached: {result['cached']}")
                else:
                    print(f"  ❌ Failed: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Compare top features across methods
        if results:
            print(f"\n📊 COMPARISON OF TOP 5 FEATURES:")
            print("-" * 80)
            
            for method, data in results.items():
                print(f"\n{method.upper()}:")
                for i, exp in enumerate(data['explanations'][:5]):
                    feature = exp['feature']
                    clinical_name = self.clinical_mappings.get(feature, feature)
                    importance = exp['importance']
                    print(f"  {i+1}. {clinical_name}: {importance:.3f}")
            
            # Performance comparison
            print(f"\n⚡ PERFORMANCE COMPARISON:")
            print("-" * 40)
            for method, data in results.items():
                comp_time = data['computation_time']
                cached = data['cached']
                status = "(cached)" if cached else "(computed)"
                print(f"{method.upper():12}: {comp_time:6.2f}ms {status}")
    
    def interactive_patient_explanation(self):
        """Interactive patient explanation demo"""
        print("\n" + "="*60)
        print("INTERACTIVE PATIENT EXPLANATION")
        print("="*60)
        
        if not self.check_api_health():
            print("❌ API server not available. Starting local demo...")
            return self._local_interactive_demo()
        
        print("Enter patient information for readmission risk assessment:")
        print("(Press Enter to use default values)")
        
        try:
            # Collect patient input
            patient_data = {}
            
            # Key clinical inputs with defaults
            clinical_inputs = {
                'age': (65, "Patient Age (years)"),
                'length_of_stay': (5, "Length of Stay (days)"),
                'num_medications': (8, "Number of Medications"),
                'charlson_comorbidity_index': (3, "Comorbidity Index (0-10)"),
                'diabetes_severity': (2, "Diabetes Severity (0-10)"),
                'heart_failure_severity': (1, "Heart Failure Severity (0-10)"),
                'renal_disease_severity': (1, "Kidney Disease Severity (0-10)"),
                'liver_disease_severity': (0, "Liver Disease Severity (0-10)"),
                'hemoglobin_level': (0.0, "Hemoglobin Level (normalized)"),
                'creatinine_level': (0.5, "Creatinine Level (normalized)"),
                'sodium_level': (-0.2, "Sodium Level (normalized)"),
                'glucose_level': (0.3, "Glucose Level (normalized)"),
                'emergency_admission': (1, "Emergency Admission (0/1)"),
                'icu_stay': (0, "ICU Stay (0/1)"),
                'surgical_procedure': (1, "Surgical Procedure (0/1)")
            }
            
            for key, (default, prompt) in clinical_inputs.items():
                user_input = input(f"{prompt} [{default}]: ").strip()
                patient_data[key] = float(user_input) if user_input else default
            
            patient_id = f"INTERACTIVE_{int(time.time())}"
            
            # Get clinical explanation
            payload = {
                "model_name": "patient_readmission_rf",
                "patient_data": patient_data,
                "patient_id": patient_id,
                "include_recommendations": True
            }
            
            response = requests.post(f"{self.api_base_url}/explain/clinical", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n🏥 READMISSION RISK ASSESSMENT")
                print("=" * 60)
                print(f"Patient ID: {result['patient_id']}")
                print(f"Risk Level: {result['risk_level']} ({result['risk_probability']:.1%})")
                
                # Risk interpretation
                risk_prob = result['risk_probability']
                if risk_prob < 0.2:
                    interpretation = "Very Low Risk - Standard discharge planning"
                elif risk_prob < 0.4:
                    interpretation = "Low Risk - Basic follow-up recommended"
                elif risk_prob < 0.6:
                    interpretation = "Moderate Risk - Enhanced discharge planning"
                elif risk_prob < 0.8:
                    interpretation = "High Risk - Intensive intervention needed"
                else:
                    interpretation = "Very High Risk - Immediate intervention required"
                
                print(f"Interpretation: {interpretation}")
                
                print(f"\n📊 KEY RISK FACTORS:")
                for i, contrib in enumerate(result['feature_contributions'][:5]):
                    clinical_name = contrib['clinical_name']
                    value = contrib['feature_value']
                    importance = contrib['importance']
                    impact = contrib['impact']
                    print(f"  {i+1}. {clinical_name}: {value:.1f}")
                    print(f"     {impact} (Contribution: {importance:+.3f})")
                
                print(f"\n💡 CLINICAL RECOMMENDATIONS:")
                for i, rec in enumerate(result['recommendations']):
                    category = rec['category']
                    recommendation = rec['recommendation']
                    priority = rec['priority']
                    print(f"  {i+1}. [{priority}] {category}")
                    print(f"     {recommendation}")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _local_interactive_demo(self):
        """Fallback local interactive demo"""
        print("Running local explainability demo...")
        
        if not self.models:
            self.create_local_models()
        
        model = self.models['Random Forest']
        sample_patient = self.X_demo[0]
        
        prediction = model.predict([sample_patient])[0]
        probability = model.predict_proba([sample_patient])[0, 1]
        
        print(f"Sample prediction: {'Readmission' if prediction == 1 else 'No Readmission'}")
        print(f"Risk probability: {probability:.3f}")
    
    def run_comprehensive_demo(self):
        """Run all demonstration components"""
        print("🏥 MODEL EXPLAINABILITY COMPREHENSIVE DEMO")
        print("=" * 80)
        
        # Generate demo data
        self.generate_demo_data()
        
        # Run all demos
        demos = [
            ("API Explanation Test", self.test_api_explanation),
            ("Clinical Explanation Test", self.test_clinical_explanation),
            ("Local Model Creation", self.create_local_models),
            ("SHAP Demonstrations", self.demonstrate_shap_explanations),
            ("LIME Demonstrations", self.demonstrate_lime_explanations),
            ("Explanation Method Comparison", self.compare_explanation_methods),
            ("Interactive Patient Assessment", self.interactive_patient_explanation)
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
    Main function to run the explainability demo
    """
    print("Day 31: Model Explainability - Interactive Demo")
    print("Healthcare AI explainability demonstration")
    print()
    
    demo = ExplainabilityDemo()
    
    # Check if user wants comprehensive demo or specific tests
    print("Demo Options:")
    print("1. Comprehensive Demo (all features)")
    print("2. Quick API Test")
    print("3. Local Explainability Demo")
    print("4. Interactive Patient Assessment Only")
    
    try:
        choice = input("\nSelect option (1-4) [1]: ").strip() or "1"
        
        if choice == "1":
            demo.run_comprehensive_demo()
        elif choice == "2":
            demo.generate_demo_data()
            demo.test_api_explanation()
            demo.test_clinical_explanation()
        elif choice == "3":
            demo.generate_demo_data()
            demo.create_local_models()
            demo.demonstrate_shap_explanations()
            demo.demonstrate_lime_explanations()
        elif choice == "4":
            demo.generate_demo_data()
            demo.interactive_patient_explanation()
        else:
            print("Invalid choice, running comprehensive demo...")
            demo.run_comprehensive_demo()
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nError running demo: {e}")

if __name__ == "__main__":
    main()