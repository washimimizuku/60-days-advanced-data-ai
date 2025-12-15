"""
Day 31: Model Explainability - SHAP, LIME, Interpretability - Exercise

Business Scenario:
You're the Lead AI Engineer at MedTech Solutions, a healthcare technology company that develops 
AI-powered diagnostic tools for hospitals. The company has built a machine learning model to 
predict patient readmission risk within 30 days of discharge.

However, the hospital's ethics committee and regulatory bodies require that all AI decisions 
be explainable and transparent. Doctors need to understand why the model predicts a patient 
is at high risk, and the explanations must be clinically meaningful and actionable.

Your task is to build a comprehensive explainability framework that provides:
- Individual patient risk explanations for doctors
- Global model insights for hospital administrators
- Regulatory compliance documentation
- Real-time explanation services for clinical workflows

Requirements:
1. Implement multiple explanation methods (SHAP, LIME, permutation importance)
2. Create production-ready explainability service with API endpoints
3. Build explanation quality evaluation framework
4. Design clinically meaningful explanation presentations
5. Ensure regulatory compliance and audit trail capabilities

Success Criteria:
- Explanation generation time < 500ms for real-time clinical use
- 95% explanation faithfulness score
- Comprehensive audit logging for regulatory compliance
- Clinically interpretable explanations validated by medical experts
- Scalable architecture supporting 1000+ requests per minute
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Explainability libraries (handle gracefully if not available)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available - install with: pip install lime")

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import json
from datetime import datetime

def generate_patient_readmission_dataset(n_samples=5000, random_state=42):
    """
    Generate synthetic patient readmission dataset with realistic medical features
    
    Returns:
        X: Feature matrix with patient characteristics
        y: Target labels (0: No readmission, 1: Readmission within 30 days)
        feature_names: List of medically relevant feature names
    """
    # TODO: Generate realistic patient dataset for readmission prediction
    # HINT: Use make_classification with appropriate parameters for medical data
    # HINT: Consider features like age, comorbidities, length of stay, etc.
    
    np.random.seed(random_state)
    
    # Generate base dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=12,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.75, 0.25],  # 25% readmission rate (realistic for high-risk patients)
        flip_y=0.02,  # 2% label noise
        class_sep=0.7,  # Moderate class separation
        random_state=random_state
    )
    
    # Transform features to be more realistic for medical data
    # Age (20-90 years)
    X[:, 0] = np.clip(np.abs(X[:, 0]) * 20 + 50, 20, 90)
    
    # Length of stay (1-30 days)
    X[:, 1] = np.clip(np.abs(X[:, 1]) * 5 + 3, 1, 30)
    
    # Number of medications (0-20)
    X[:, 2] = np.clip(np.abs(X[:, 2]) * 3 + 2, 0, 20)
    
    # Comorbidity scores (0-10)
    for i in range(3, 8):
        X[:, i] = np.clip(np.abs(X[:, i]) * 2, 0, 10)
    
    # Lab values (normalized)
    for i in range(8, 12):
        X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    
    # Binary indicators (0 or 1)
    for i in range(12, 15):
        X[:, i] = (X[:, i] > 0).astype(int)
    
    # Create medically relevant feature names
    feature_names = [
        'age', 'length_of_stay', 'num_medications', 'charlson_comorbidity_index',
        'diabetes_severity', 'heart_failure_severity', 'renal_disease_severity',
        'liver_disease_severity', 'hemoglobin_level', 'creatinine_level',
        'sodium_level', 'glucose_level', 'emergency_admission', 'icu_stay',
        'surgical_procedure'
    ]
    
    return X, y, feature_names

class ExplainabilityExperiment:
    """
    Comprehensive explainability experiment for patient readmission prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.models = {}
        self.explainers = {}
        self.explanations = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the patient readmission dataset"""
        print("Loading patient readmission dataset...")
        
        # TODO: Generate dataset and split into train/test
        # HINT: Use stratified split to maintain class balance
        
        X, y, self.feature_names = generate_patient_readmission_dataset()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Create DataFrames for easier handling
        self.X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        self.X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        
        # Scale features for models that need it
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Training set: {self.X_train.shape[0]} patients")
        print(f"Test set: {self.X_test.shape[0]} patients")
        print(f"Readmission rate: {np.mean(y):.2%}")
        
        # Display feature statistics
        print("\nFeature Statistics:")
        print(self.X_train_df.describe())
        
    def create_prediction_models(self):
        """Create different types of models for explainability comparison"""
        print("\nCreating prediction models...")
        
        # TODO: Create diverse models with different complexity levels
        # HINT: Include interpretable and black-box models
        # HINT: Consider clinical requirements for model performance
        
        self.models = {
            # Interpretable model
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            
            # Tree-based models (moderately interpretable)
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state
            )
        }
        
        print(f"Created {len(self.models)} models for explainability analysis")
        
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\nTraining and evaluating models...")
        
        # TODO: Train each model and evaluate performance
        # HINT: Use appropriate metrics for medical applications (AUC, sensitivity, specificity)
        
        model_results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Determine if model needs scaled features
            if name == 'logistic_regression':
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Predictions
            y_pred = model.predict(X_test_use)
            y_proba = model.predict_proba(X_test_use)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_score = roc_auc_score(self.y_test, y_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5, scoring='roc_auc')
            
            model_results[name] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'cv_auc_mean': np.mean(cv_scores),
                'cv_auc_std': np.std(cv_scores)
            }
            
            print(f"  {name}: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}")
        
        return model_results
        
    def implement_shap_explanations(self):
        """Implement SHAP explanations for all models"""
        print("\nImplementing SHAP explanations...")
        
        if not SHAP_AVAILABLE:
            print("SHAP not available - skipping SHAP explanations")
            return {}
        
        # TODO: Create SHAP explainers for different model types
        # HINT: Use TreeExplainer for tree models, LinearExplainer for linear models
        # HINT: Generate both local and global explanations
        
        shap_results = {}
        
        for name, model in self.models.items():
            print(f"Creating SHAP explainer for {name}...")
            
            try:
                # Determine appropriate explainer type
                if name == 'logistic_regression':
                    explainer = shap.LinearExplainer(model, self.X_train_scaled)
                    X_explain = self.X_test_scaled[:100]  # Limit for performance
                elif name in ['random_forest', 'gradient_boosting']:
                    explainer = shap.TreeExplainer(model)
                    X_explain = self.X_test[:100]
                else:
                    continue
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(X_explain)
                
                # Handle different output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Class 1 for binary classification
                
                # Store results
                shap_results[name] = {
                    'explainer': explainer,
                    'shap_values': shap_values,
                    'X_explain': X_explain,
                    'expected_value': explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
                }
                
                print(f"  SHAP values calculated for {name}")
                
            except Exception as e:
                print(f"  Error creating SHAP explainer for {name}: {e}")
        
        self.explanations['shap'] = shap_results
        return shap_results
        
    def implement_lime_explanations(self):
        """Implement LIME explanations"""
        print("\nImplementing LIME explanations...")
        
        if not LIME_AVAILABLE:
            print("LIME not available - skipping LIME explanations")
            return {}
        
        # TODO: Create LIME explainers and generate explanations
        # HINT: Use LimeTabularExplainer with appropriate configuration
        # HINT: Generate explanations for sample patients
        
        lime_results = {}
        
        for name, model in self.models.items():
            print(f"Creating LIME explainer for {name}...")
            
            try:
                # Determine training data for LIME
                if name == 'logistic_regression':
                    X_train_lime = self.X_train_scaled
                    X_test_lime = self.X_test_scaled
                else:
                    X_train_lime = self.X_train
                    X_test_lime = self.X_test
                
                # Create LIME explainer
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train_lime,
                    feature_names=self.feature_names,
                    class_names=['No Readmission', 'Readmission'],
                    mode='classification',
                    discretize_continuous=True
                )
                
                # Generate explanations for sample instances
                sample_explanations = []
                for i in range(min(10, len(X_test_lime))):
                    explanation = explainer.explain_instance(
                        X_test_lime[i],
                        model.predict_proba,
                        num_features=len(self.feature_names),
                        top_labels=2
                    )
                    sample_explanations.append(explanation)
                
                lime_results[name] = {
                    'explainer': explainer,
                    'sample_explanations': sample_explanations
                }
                
                print(f"  LIME explanations generated for {name}")
                
            except Exception as e:
                print(f"  Error creating LIME explainer for {name}: {e}")
        
        self.explanations['lime'] = lime_results
        return lime_results
        
    def calculate_permutation_importance(self):
        """Calculate permutation importance for global interpretability"""
        print("\nCalculating permutation importance...")
        
        # TODO: Implement permutation importance calculation
        # HINT: Use sklearn's permutation_importance or implement custom version
        # HINT: Calculate importance for each model
        
        from sklearn.inspection import permutation_importance
        
        perm_results = {}
        
        for name, model in self.models.items():
            print(f"Calculating permutation importance for {name}...")
            
            # Determine appropriate test data
            if name == 'logistic_regression':
                X_test_use = self.X_test_scaled
            else:
                X_test_use = self.X_test
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_test_use, self.y_test,
                n_repeats=10,
                random_state=self.random_state,
                scoring='roc_auc'
            )
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            perm_results[name] = {
                'importance_df': importance_df,
                'raw_importances': perm_importance.importances
            }
            
            print(f"  Permutation importance calculated for {name}")
        
        self.explanations['permutation'] = perm_results
        return perm_results
        
    def create_clinical_explanation_interface(self):
        """Create clinically meaningful explanation interface"""
        print("\nCreating clinical explanation interface...")
        
        # TODO: Design explanation interface tailored for clinical use
        # HINT: Focus on actionable insights and clinical terminology
        # HINT: Provide risk factors and recommendations
        
        class ClinicalExplanationInterface:
            def __init__(self, models, explainers, feature_names):
                self.models = models
                self.explainers = explainers
                self.feature_names = feature_names
                
                # Clinical feature mappings
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
                
                # Risk level thresholds
                self.risk_thresholds = {
                    'low': 0.3,
                    'moderate': 0.6,
                    'high': 0.8
                }
            
            def generate_patient_explanation(self, patient_data, model_name='random_forest'):
                """Generate clinical explanation for a patient"""
                # TODO: Create comprehensive patient explanation
                # HINT: Include risk level, key factors, and recommendations
                
                model = self.models[model_name]
                
                # Make prediction
                if model_name == 'logistic_regression':
                    # Use scaled data for logistic regression
                    patient_scaled = self.scaler.transform([patient_data])
                    risk_probability = model.predict_proba(patient_scaled)[0, 1]
                else:
                    risk_probability = model.predict_proba([patient_data])[0, 1]
                
                # Determine risk level
                if risk_probability < self.risk_thresholds['low']:
                    risk_level = 'Low'
                    risk_color = 'green'
                elif risk_probability < self.risk_thresholds['moderate']:
                    risk_level = 'Moderate'
                    risk_color = 'yellow'
                elif risk_probability < self.risk_thresholds['high']:
                    risk_level = 'High'
                    risk_color = 'orange'
                else:
                    risk_level = 'Very High'
                    risk_color = 'red'
                
                explanation = {
                    'patient_id': f"Patient_{int(time.time())}",
                    'risk_probability': float(risk_probability),
                    'risk_level': risk_level,
                    'risk_color': risk_color,
                    'model_used': model_name,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add feature contributions if SHAP is available
                if SHAP_AVAILABLE and model_name in self.explainers.get('shap', {}):
                    try:
                        shap_explainer = self.explainers['shap'][model_name]['explainer']
                        
                        if model_name == 'logistic_regression':
                            shap_values = shap_explainer.shap_values([patient_scaled[0]])
                        else:
                            shap_values = shap_explainer.shap_values([patient_data])
                        
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]
                        
                        # Create feature contributions
                        contributions = []
                        for i, (feature, value) in enumerate(zip(self.feature_names, patient_data)):
                            contribution = {
                                'feature': feature,
                                'clinical_name': self.clinical_mappings.get(feature, feature),
                                'value': float(value),
                                'contribution': float(shap_values[0][i]),
                                'impact': 'Increases Risk' if shap_values[0][i] > 0 else 'Decreases Risk'
                            }
                            contributions.append(contribution)
                        
                        # Sort by absolute contribution
                        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
                        explanation['feature_contributions'] = contributions[:10]  # Top 10
                        
                    except Exception as e:
                        explanation['feature_contributions'] = []
                        explanation['shap_error'] = str(e)
                
                return explanation
            
            def generate_recommendations(self, explanation):
                """Generate clinical recommendations based on explanation"""
                # TODO: Create actionable clinical recommendations
                # HINT: Map high-risk factors to specific interventions
                
                recommendations = []
                
                if 'feature_contributions' in explanation:
                    top_risk_factors = [contrib for contrib in explanation['feature_contributions'][:5] 
                                     if contrib['contribution'] > 0]
                    
                    for factor in top_risk_factors:
                        feature = factor['feature']
                        
                        if feature == 'length_of_stay' and factor['value'] > 7:
                            recommendations.append({
                                'category': 'Discharge Planning',
                                'recommendation': 'Consider enhanced discharge planning and home health services',
                                'priority': 'High'
                            })
                        
                        elif feature == 'num_medications' and factor['value'] > 10:
                            recommendations.append({
                                'category': 'Medication Management',
                                'recommendation': 'Conduct medication reconciliation and consider polypharmacy review',
                                'priority': 'High'
                            })
                        
                        elif feature == 'charlson_comorbidity_index' and factor['value'] > 5:
                            recommendations.append({
                                'category': 'Comorbidity Management',
                                'recommendation': 'Coordinate care with specialists for comorbidity management',
                                'priority': 'Medium'
                            })
                        
                        elif 'severity' in feature and factor['value'] > 5:
                            recommendations.append({
                                'category': 'Disease Management',
                                'recommendation': f'Optimize management of {factor["clinical_name"]}',
                                'priority': 'Medium'
                            })
                
                # General recommendations based on risk level
                if explanation['risk_level'] in ['High', 'Very High']:
                    recommendations.extend([
                        {
                            'category': 'Follow-up Care',
                            'recommendation': 'Schedule follow-up appointment within 7 days of discharge',
                            'priority': 'High'
                        },
                        {
                            'category': 'Patient Education',
                            'recommendation': 'Provide comprehensive discharge education and written instructions',
                            'priority': 'High'
                        }
                    ])
                
                return recommendations
        
        # Create interface instance
        clinical_interface = ClinicalExplanationInterface(
            self.models, 
            self.explanations, 
            self.feature_names
        )
        
        return clinical_interface
        
    def evaluate_explanation_quality(self):
        """Evaluate the quality and reliability of explanations"""
        print("\nEvaluating explanation quality...")
        
        # TODO: Implement explanation quality metrics
        # HINT: Test faithfulness, stability, and comprehensiveness
        # HINT: Use perturbation-based evaluation methods
        
        evaluation_results = {}
        
        for model_name in self.models.keys():
            print(f"Evaluating explanations for {model_name}...")
            
            model_eval = {}
            
            # Faithfulness test (remove important features and measure prediction change)
            if SHAP_AVAILABLE and model_name in self.explanations.get('shap', {}):
                faithfulness_scores = []
                
                for i in range(min(50, len(self.X_test))):
                    # Get SHAP explanation
                    if model_name == 'logistic_regression':
                        instance = self.X_test_scaled[i:i+1]
                        shap_explainer = self.explanations['shap'][model_name]['explainer']
                        shap_values = shap_explainer.shap_values(instance)
                    else:
                        instance = self.X_test[i:i+1]
                        shap_explainer = self.explanations['shap'][model_name]['explainer']
                        shap_values = shap_explainer.shap_values(instance)
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    
                    # Original prediction
                    original_pred = self.models[model_name].predict_proba(instance)[0, 1]
                    
                    # Remove top 3 most important features
                    top_features = np.argsort(np.abs(shap_values[0]))[-3:]
                    
                    # Create modified instance (replace with mean values)
                    instance_modified = instance.copy()
                    for feature_idx in top_features:
                        if model_name == 'logistic_regression':
                            instance_modified[0, feature_idx] = 0  # Scaled mean is 0
                        else:
                            instance_modified[0, feature_idx] = self.X_train[:, feature_idx].mean()
                    
                    # Modified prediction
                    modified_pred = self.models[model_name].predict_proba(instance_modified)[0, 1]
                    
                    # Faithfulness score (higher change = more faithful)
                    faithfulness = abs(original_pred - modified_pred)
                    faithfulness_scores.append(faithfulness)
                
                model_eval['faithfulness_mean'] = np.mean(faithfulness_scores)
                model_eval['faithfulness_std'] = np.std(faithfulness_scores)
            
            # Stability test (small perturbations should give similar explanations)
            if SHAP_AVAILABLE and model_name in self.explanations.get('shap', {}):
                stability_scores = []
                
                for i in range(min(20, len(self.X_test))):
                    # Original explanation
                    if model_name == 'logistic_regression':
                        instance = self.X_test_scaled[i:i+1]
                    else:
                        instance = self.X_test[i:i+1]
                    
                    shap_explainer = self.explanations['shap'][model_name]['explainer']
                    original_shap = shap_explainer.shap_values(instance)
                    
                    if isinstance(original_shap, list):
                        original_shap = original_shap[1]
                    
                    # Add small noise
                    noise = np.random.normal(0, 0.01, instance.shape)
                    noisy_instance = instance + noise
                    
                    # Noisy explanation
                    noisy_shap = shap_explainer.shap_values(noisy_instance)
                    if isinstance(noisy_shap, list):
                        noisy_shap = noisy_shap[1]
                    
                    # Calculate correlation between explanations
                    correlation = np.corrcoef(original_shap[0], noisy_shap[0])[0, 1]
                    stability_scores.append(correlation)
                
                model_eval['stability_mean'] = np.mean(stability_scores)
                model_eval['stability_std'] = np.std(stability_scores)
            
            evaluation_results[model_name] = model_eval
        
        return evaluation_results
        
    def create_explanation_dashboard(self):
        """Create visualization dashboard for explanations"""
        print("\nCreating explanation dashboard...")
        
        # TODO: Create comprehensive visualization dashboard
        # HINT: Include global and local explanation visualizations
        # HINT: Make visualizations clinically relevant
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Patient Readmission Model Explainability Dashboard', fontsize=16)
        
        # 1. Model Performance Comparison
        if hasattr(self, 'model_results'):
            model_names = list(self.model_results.keys())
            auc_scores = [self.model_results[name]['auc'] for name in model_names]
            
            axes[0, 0].bar(model_names, auc_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
            axes[0, 0].set_title('Model Performance (AUC)')
            axes[0, 0].set_ylabel('AUC Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Feature Importance (Permutation)
        if 'permutation' in self.explanations:
            perm_data = self.explanations['permutation']['random_forest']['importance_df']
            top_features = perm_data.head(10)
            
            axes[0, 1].barh(top_features['feature'], top_features['importance_mean'])
            axes[0, 1].set_title('Top 10 Feature Importance (Permutation)')
            axes[0, 1].set_xlabel('Importance Score')
        
        # 3. SHAP Summary (if available)
        if SHAP_AVAILABLE and 'shap' in self.explanations and 'random_forest' in self.explanations['shap']:
            try:
                shap_data = self.explanations['shap']['random_forest']
                shap_values = shap_data['shap_values']
                
                # Calculate mean absolute SHAP values
                mean_shap = np.mean(np.abs(shap_values), axis=0)
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': mean_shap
                }).sort_values('importance', ascending=True).tail(10)
                
                axes[0, 2].barh(feature_importance['feature'], feature_importance['importance'])
                axes[0, 2].set_title('Top 10 SHAP Feature Importance')
                axes[0, 2].set_xlabel('Mean |SHAP Value|')
            except Exception as e:
                axes[0, 2].text(0.5, 0.5, f'SHAP visualization error: {str(e)}', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
        
        # 4. Risk Distribution
        if 'random_forest' in self.models:
            model = self.models['random_forest']
            risk_probs = model.predict_proba(self.X_test)[:, 1]
            
            axes[1, 0].hist(risk_probs, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            axes[1, 0].axvline(0.3, color='green', linestyle='--', label='Low Risk Threshold')
            axes[1, 0].axvline(0.6, color='orange', linestyle='--', label='High Risk Threshold')
            axes[1, 0].set_title('Risk Score Distribution')
            axes[1, 0].set_xlabel('Readmission Risk Probability')
            axes[1, 0].set_ylabel('Number of Patients')
            axes[1, 0].legend()
        
        # 5. Feature Correlation with Target
        correlations = []
        for feature in self.feature_names:
            corr = np.corrcoef(self.X_train_df[feature], self.y_train)[0, 1]
            correlations.append(abs(corr))
        
        feature_corr = pd.DataFrame({
            'feature': self.feature_names,
            'correlation': correlations
        }).sort_values('correlation', ascending=True).tail(10)
        
        axes[1, 1].barh(feature_corr['feature'], feature_corr['correlation'])
        axes[1, 1].set_title('Top 10 Feature-Target Correlations')
        axes[1, 1].set_xlabel('Absolute Correlation')
        
        # 6. Explanation Quality Metrics
        if hasattr(self, 'evaluation_results'):
            models = list(self.evaluation_results.keys())
            faithfulness = [self.evaluation_results[m].get('faithfulness_mean', 0) for m in models]
            stability = [self.evaluation_results[m].get('stability_mean', 0) for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            axes[1, 2].bar(x - width/2, faithfulness, width, label='Faithfulness', alpha=0.7)
            axes[1, 2].bar(x + width/2, stability, width, label='Stability', alpha=0.7)
            axes[1, 2].set_title('Explanation Quality Metrics')
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].set_xticks(x)
            axes[1, 2].set_xticklabels(models, rotation=45)
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
        
    def run_complete_experiment(self):
        """Run the complete explainability experiment"""
        print("Starting Complete Model Explainability Experiment")
        print("=" * 60)
        
        # TODO: Execute the complete experimental pipeline
        # HINT: Call all methods in the correct order
        # HINT: Handle any errors gracefully and provide meaningful output
        
        try:
            # Data preparation
            self.load_and_prepare_data()
            
            # Model creation and training
            self.create_prediction_models()
            self.model_results = self.train_and_evaluate_models()
            
            # Explainability methods
            self.implement_shap_explanations()
            self.implement_lime_explanations()
            self.calculate_permutation_importance()
            
            # Clinical interface
            clinical_interface = self.create_clinical_explanation_interface()
            
            # Explanation quality evaluation
            self.evaluation_results = self.evaluate_explanation_quality()
            
            # Visualization dashboard
            self.create_explanation_dashboard()
            
            # Demonstrate clinical explanation
            print("\n" + "=" * 60)
            print("CLINICAL EXPLANATION DEMONSTRATION")
            print("=" * 60)
            
            # Generate explanation for a sample patient
            sample_patient = self.X_test[0]
            explanation = clinical_interface.generate_patient_explanation(sample_patient)
            recommendations = clinical_interface.generate_recommendations(explanation)
            
            print(f"Patient ID: {explanation['patient_id']}")
            print(f"Risk Level: {explanation['risk_level']} ({explanation['risk_probability']:.2%})")
            print(f"Model Used: {explanation['model_used']}")
            
            if 'feature_contributions' in explanation:
                print("\nTop Risk Factors:")
                for contrib in explanation['feature_contributions'][:5]:
                    print(f"  - {contrib['clinical_name']}: {contrib['impact']} "
                          f"(Contribution: {contrib['contribution']:.3f})")
            
            print("\nClinical Recommendations:")
            for rec in recommendations:
                print(f"  - {rec['category']}: {rec['recommendation']} (Priority: {rec['priority']})")
            
            print("\n" + "=" * 60)
            print("EXPERIMENT COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            return {
                'model_results': self.model_results,
                'explanations': self.explanations,
                'evaluation_results': self.evaluation_results,
                'clinical_interface': clinical_interface
            }
            
        except Exception as e:
            print(f"Error during experiment: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """
    Main function to run the model explainability exercise
    
    This exercise demonstrates:
    1. Implementation of multiple explainability methods (SHAP, LIME, permutation)
    2. Clinical explanation interface design
    3. Explanation quality evaluation
    4. Production-ready explainability service architecture
    """
    print("Day 31: Model Explainability - Patient Readmission Risk Assessment")
    print("Building explainable AI systems for healthcare applications")
    print()
    
    # TODO: Run the complete explainability experiment
    # HINT: Create experiment instance and run all components
    
    # Initialize experiment
    experiment = ExplainabilityExperiment(random_state=42)
    
    # Run complete experiment
    results = experiment.run_complete_experiment()
    
    if results:
        print("\nKey Insights:")
        print("- SHAP provides theoretically grounded feature attributions")
        print("- LIME offers intuitive local explanations for individual patients")
        print("- Permutation importance reveals global feature importance")
        print("- Clinical interfaces make explanations actionable for healthcare")
        print("- Explanation quality evaluation ensures reliability")
        
        # Performance summary
        if 'model_results' in results:
            best_model = max(results['model_results'].items(), key=lambda x: x[1]['auc'])
            print(f"\nBest performing model: {best_model[0]} (AUC: {best_model[1]['auc']:.4f})")
        
        # Explainability summary
        available_methods = list(results['explanations'].keys())
        print(f"Explainability methods implemented: {', '.join(available_methods)}")
    
    print("\nExercise completed! Consider these questions:")
    print("1. How do different explanation methods complement each other?")
    print("2. What are the trade-offs between explanation accuracy and interpretability?")
    print("3. How would you validate explanations with medical experts?")
    print("4. What regulatory requirements apply to your use case?")
    print("5. How would you monitor explanation quality in production?")

if __name__ == "__main__":
    main()
