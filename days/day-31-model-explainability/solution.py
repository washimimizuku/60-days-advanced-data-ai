"""
Day 31: Model Explainability - SHAP, LIME, Interpretability - Complete Solution

Production-ready explainability framework for MedTech Solutions' patient readmission prediction system.
Demonstrates comprehensive explainable AI approaches and regulatory compliance patterns.

This solution showcases:
- SHAP (SHapley Additive exPlanations) for theoretically grounded explanations
- LIME (Local Interpretable Model-agnostic Explanations) for local interpretability
- Global interpretability methods (permutation importance, partial dependence)
- Production explainability service with real-time API endpoints
- Clinical explanation interfaces tailored for healthcare professionals
- Explanation quality evaluation and validation frameworks
- Regulatory compliance and audit trail capabilities
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                           confusion_matrix, roc_curve, precision_recall_curve,
                           f1_score, precision_score, recall_score)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns

# Explainability libraries (handle gracefully if not available)
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

try:
    import anchor
    import anchor.anchor_tabular
    ANCHOR_AVAILABLE = True
except ImportError:
    ANCHOR_AVAILABLE = False

import logging
import time
import json
import joblib
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading

# =============================================================================
# PRODUCTION EXPLAINABILITY FRAMEWORK
# =============================================================================

@dataclass
class ExplanationRequest:
    """Request for model explanation"""
    request_id: str
    model_id: str
    instance_data: Dict[str, Any]
    explanation_methods: List[str]  # ['shap', 'lime', 'permutation']
    num_features: int = 10
    return_probabilities: bool = True
    clinical_context: Optional[Dict[str, Any]] = None

@dataclass
class ExplanationResponse:
    """Response containing comprehensive explanation"""
    request_id: str
    model_id: str
    prediction: float
    probability: Optional[float]
    risk_level: str
    explanations: Dict[str, Any]
    clinical_recommendations: List[Dict[str, Any]]
    computation_time_ms: float
    timestamp: str
    model_performance: Dict[str, float]

class ProductionExplainabilityFramework:
    """
    Production-grade explainability framework with comprehensive features
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.logger = self._setup_logging()
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        
        # Model and explainer storage
        self.models = {}
        self.explainers = {}
        self.model_performance = {}
        
        # Production components
        self.explanation_cache = {}
        self.request_history = []
        self.clinical_mappings = {}
        
        # Quality evaluation
        self.explanation_quality_metrics = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('explainability_framework')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            file_handler = logging.FileHandler('explainability_framework.log')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def generate_synthetic_medical_data(self, n_samples=5000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate realistic synthetic medical dataset for readmission prediction
        
        Returns:
            X: Feature matrix with patient characteristics
            y: Target labels (0: No readmission, 1: Readmission within 30 days)
            feature_names: List of medically relevant feature names
        """
        self.logger.info(f"Generating synthetic medical dataset: {n_samples} samples")
        
        np.random.seed(self.random_state)
        
        # Generate base dataset with medical characteristics
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=16,
            n_redundant=2,
            n_clusters_per_class=3,
            weights=[0.73, 0.27],  # 27% readmission rate (realistic for high-risk)
            flip_y=0.01,  # 1% label noise
            class_sep=0.8,  # Good but not perfect separation
            random_state=self.random_state
        )
        
        # Transform features to realistic medical ranges
        # Demographics
        X[:, 0] = np.clip(np.abs(X[:, 0]) * 20 + 45, 18, 95)  # Age (18-95)
        X[:, 1] = (X[:, 1] > 0).astype(int)  # Gender (0/1)
        
        # Clinical metrics
        X[:, 2] = np.clip(np.abs(X[:, 2]) * 8 + 2, 1, 45)  # Length of stay (1-45 days)
        X[:, 3] = np.clip(np.abs(X[:, 3]) * 5 + 3, 0, 25)  # Number of medications
        X[:, 4] = np.clip(np.abs(X[:, 4]) * 3, 0, 15)  # Charlson Comorbidity Index
        
        # Disease severity scores (0-10 scale)
        for i in range(5, 10):
            X[:, i] = np.clip(np.abs(X[:, i]) * 2.5, 0, 10)
        
        # Lab values (normalized around typical ranges)
        X[:, 10] = np.clip(X[:, 10] * 2 + 12, 7, 18)  # Hemoglobin (7-18 g/dL)
        X[:, 11] = np.clip(np.abs(X[:, 11]) * 3 + 0.8, 0.5, 8.0)  # Creatinine (0.5-8.0 mg/dL)
        X[:, 12] = np.clip(X[:, 12] * 10 + 140, 120, 160)  # Sodium (120-160 mEq/L)
        X[:, 13] = np.clip(np.abs(X[:, 13]) * 100 + 80, 50, 400)  # Glucose (50-400 mg/dL)
        
        # Binary indicators
        for i in range(14, 20):
            X[:, i] = (X[:, i] > np.percentile(X[:, i], 70)).astype(int)
        
        # Create comprehensive feature names
        feature_names = [
            'age', 'gender', 'length_of_stay', 'num_medications', 'charlson_comorbidity_index',
            'diabetes_severity', 'heart_failure_severity', 'copd_severity', 'renal_disease_severity',
            'liver_disease_severity', 'hemoglobin_level', 'creatinine_level', 'sodium_level',
            'glucose_level', 'emergency_admission', 'icu_stay', 'surgical_procedure',
            'mechanical_ventilation', 'dialysis_required', 'previous_readmission'
        ]
        
        # Create clinical mappings for interpretability
        self.clinical_mappings = {
            'age': {'name': 'Patient Age', 'unit': 'years', 'normal_range': '18-65'},
            'gender': {'name': 'Gender', 'unit': 'categorical', 'normal_range': 'Male/Female'},
            'length_of_stay': {'name': 'Length of Hospital Stay', 'unit': 'days', 'normal_range': '1-7'},
            'num_medications': {'name': 'Number of Medications', 'unit': 'count', 'normal_range': '0-10'},
            'charlson_comorbidity_index': {'name': 'Comorbidity Burden Score', 'unit': 'score', 'normal_range': '0-5'},
            'diabetes_severity': {'name': 'Diabetes Severity', 'unit': 'score', 'normal_range': '0-3'},
            'heart_failure_severity': {'name': 'Heart Failure Severity', 'unit': 'score', 'normal_range': '0-3'},
            'copd_severity': {'name': 'COPD Severity', 'unit': 'score', 'normal_range': '0-3'},
            'renal_disease_severity': {'name': 'Kidney Disease Severity', 'unit': 'score', 'normal_range': '0-3'},
            'liver_disease_severity': {'name': 'Liver Disease Severity', 'unit': 'score', 'normal_range': '0-3'},
            'hemoglobin_level': {'name': 'Hemoglobin Level', 'unit': 'g/dL', 'normal_range': '12-16'},
            'creatinine_level': {'name': 'Creatinine Level', 'unit': 'mg/dL', 'normal_range': '0.7-1.3'},
            'sodium_level': {'name': 'Sodium Level', 'unit': 'mEq/L', 'normal_range': '135-145'},
            'glucose_level': {'name': 'Glucose Level', 'unit': 'mg/dL', 'normal_range': '70-100'},
            'emergency_admission': {'name': 'Emergency Admission', 'unit': 'binary', 'normal_range': 'Yes/No'},
            'icu_stay': {'name': 'ICU Stay Required', 'unit': 'binary', 'normal_range': 'Yes/No'},
            'surgical_procedure': {'name': 'Surgical Procedure', 'unit': 'binary', 'normal_range': 'Yes/No'},
            'mechanical_ventilation': {'name': 'Mechanical Ventilation', 'unit': 'binary', 'normal_range': 'Yes/No'},
            'dialysis_required': {'name': 'Dialysis Required', 'unit': 'binary', 'normal_range': 'Yes/No'},
            'previous_readmission': {'name': 'Previous Readmission History', 'unit': 'binary', 'normal_range': 'Yes/No'}
        }
        
        self.logger.info(f"Dataset generated - Readmission rate: {np.mean(y):.2%}")
        
        return X, y, feature_names
    
    def prepare_data(self, test_size=0.2, scale_features=True):
        """
        Comprehensive data preparation with advanced preprocessing
        """
        self.logger.info("Starting data preparation")
        
        # Generate synthetic medical data
        X, y, self.feature_names = self.generate_synthetic_medical_data()
        
        # Split data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y, shuffle=True
        )
        
        # Create DataFrames for easier handling
        self.X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        self.X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        
        # Feature scaling
        if scale_features:
            self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
        else:
            self.X_train_scaled = self.X_train.copy()
            self.X_test_scaled = self.X_test.copy()
        
        # Log data statistics
        self.logger.info(f"Training set: {self.X_train.shape[0]} patients")
        self.logger.info(f"Test set: {self.X_test.shape[0]} patients")
        self.logger.info(f"Features: {self.X_train.shape[1]}")
        self.logger.info(f"Readmission rate (train): {np.mean(self.y_train):.2%}")
        self.logger.info(f"Readmission rate (test): {np.mean(self.y_test):.2%}")
        
        return self
    
    def create_prediction_models(self) -> Dict[str, Any]:
        """
        Create comprehensive set of prediction models with different interpretability levels
        """
        self.logger.info("Creating prediction models")
        
        self.models = {
            # Highly interpretable models
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=2000,
                class_weight='balanced',
                solver='liblinear',
                penalty='l2',
                C=1.0
            ),
            
            'decision_tree': DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=self.random_state,
                class_weight='balanced'
            ),
            
            # Moderately interpretable models
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1,
                oob_score=True
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                max_features='sqrt',
                random_state=self.random_state
            ),
            
            # Less interpretable but potentially more accurate
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced',
                gamma='scale',
                C=1.0
            )
        }
        
        self.logger.info(f"Created {len(self.models)} prediction models")
        return self.models
    
    def train_and_evaluate_models(self, cv_folds=5) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive model training and evaluation with clinical metrics
        """
        self.logger.info("Training and evaluating models")
        
        model_results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}")
            
            start_time = time.time()
            
            # Determine feature set (scaled vs unscaled)
            if name in ['logistic_regression', 'svm']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Test set predictions
            y_pred = model.predict(X_test_use)
            y_proba = model.predict_proba(X_test_use)[:, 1]
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_score = roc_auc_score(self.y_test, y_proba)
            f1 = f1_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            
            # Clinical metrics (sensitivity and specificity)
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            sensitivity = tp / (tp + fn)  # Recall for positive class
            specificity = tn / (tn + fp)  # Recall for negative class
            
            # Cross-validation
            cv_scores = {}
            for metric in ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']:
                scores = cross_val_score(
                    model, X_train_use, self.y_train, 
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                    scoring=metric, n_jobs=-1
                )
                cv_scores[f'cv_{metric}_mean'] = np.mean(scores)
                cv_scores[f'cv_{metric}_std'] = np.std(scores)
            
            training_time = time.time() - start_time
            
            # Store results
            model_results[name] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'training_time': training_time,
                **cv_scores
            }
            
            self.logger.info(f"  {name}: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}, "
                           f"Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}")
        
        self.model_performance = model_results
        return model_results
    def implement_shap_explanations(self) -> Dict[str, Any]:
        """
        Comprehensive SHAP implementation for all compatible models
        """
        self.logger.info("Implementing SHAP explanations")
        
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available - skipping SHAP explanations")
            return {}
        
        shap_results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Creating SHAP explainer for {name}")
            
            try:
                # Determine appropriate explainer type and data
                if name == 'logistic_regression':
                    explainer = shap.LinearExplainer(model, self.X_train_scaled)
                    X_explain = self.X_test_scaled[:200]  # Limit for performance
                    X_background = self.X_train_scaled
                elif name in ['random_forest', 'gradient_boosting', 'decision_tree']:
                    explainer = shap.TreeExplainer(model)
                    X_explain = self.X_test[:200]
                    X_background = self.X_train
                elif name == 'svm':
                    # Use KernelExplainer for SVM (slower but works)
                    explainer = shap.KernelExplainer(
                        model.predict_proba, 
                        self.X_train_scaled[np.random.choice(len(self.X_train_scaled), 100, replace=False)]
                    )
                    X_explain = self.X_test_scaled[:50]  # Fewer samples for KernelExplainer
                    X_background = self.X_train_scaled
                else:
                    continue
                
                # Calculate SHAP values
                self.logger.info(f"  Calculating SHAP values for {name}")
                shap_values = explainer.shap_values(X_explain)
                
                # Handle different output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Class 1 for binary classification
                
                # Calculate expected value
                if hasattr(explainer, 'expected_value'):
                    if isinstance(explainer.expected_value, np.ndarray):
                        expected_value = explainer.expected_value[1]
                    else:
                        expected_value = explainer.expected_value
                else:
                    expected_value = np.mean(self.y_train)
                
                # Store results
                shap_results[name] = {
                    'explainer': explainer,
                    'shap_values': shap_values,
                    'X_explain': X_explain,
                    'X_background': X_background,
                    'expected_value': expected_value,
                    'feature_names': self.feature_names
                }
                
                self.logger.info(f"  SHAP values calculated for {name} ({shap_values.shape[0]} instances)")
                
            except Exception as e:
                self.logger.error(f"  Error creating SHAP explainer for {name}: {e}")
                continue
        
        self.explainers['shap'] = shap_results
        return shap_results
    
    def implement_lime_explanations(self) -> Dict[str, Any]:
        """
        Comprehensive LIME implementation for model-agnostic explanations
        """
        self.logger.info("Implementing LIME explanations")
        
        if not LIME_AVAILABLE:
            self.logger.warning("LIME not available - skipping LIME explanations")
            return {}
        
        lime_results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Creating LIME explainer for {name}")
            
            try:
                # Determine training data for LIME
                if name in ['logistic_regression', 'svm']:
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
                    discretize_continuous=True,
                    random_state=self.random_state
                )
                
                # Generate explanations for sample instances
                sample_explanations = []
                sample_indices = np.random.choice(len(X_test_lime), min(20, len(X_test_lime)), replace=False)
                
                for i in sample_indices:
                    explanation = explainer.explain_instance(
                        X_test_lime[i],
                        model.predict_proba,
                        num_features=len(self.feature_names),
                        top_labels=2
                    )
                    sample_explanations.append({
                        'instance_index': i,
                        'explanation': explanation,
                        'local_pred': explanation.local_pred[1] if hasattr(explanation, 'local_pred') else None,
                        'intercept': explanation.intercept[1] if hasattr(explanation, 'intercept') else None
                    })
                
                lime_results[name] = {
                    'explainer': explainer,
                    'sample_explanations': sample_explanations,
                    'X_train': X_train_lime,
                    'X_test': X_test_lime
                }
                
                self.logger.info(f"  LIME explanations generated for {name} ({len(sample_explanations)} instances)")
                
            except Exception as e:
                self.logger.error(f"  Error creating LIME explainer for {name}: {e}")
                continue
        
        self.explainers['lime'] = lime_results
        return lime_results
    
    def calculate_global_interpretability(self) -> Dict[str, Any]:
        """
        Calculate global interpretability metrics including permutation importance
        """
        self.logger.info("Calculating global interpretability metrics")
        
        global_results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Calculating global metrics for {name}")
            
            try:
                # Determine appropriate test data
                if name in ['logistic_regression', 'svm']:
                    X_test_use = self.X_test_scaled
                else:
                    X_test_use = self.X_test
                
                # Permutation importance
                perm_importance = permutation_importance(
                    model, X_test_use, self.y_test,
                    n_repeats=10,
                    random_state=self.random_state,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance_mean': perm_importance.importances_mean,
                    'importance_std': perm_importance.importances_std,
                    'clinical_name': [self.clinical_mappings[f]['name'] for f in self.feature_names]
                }).sort_values('importance_mean', ascending=False)
                
                # Feature importance from model (if available)
                model_importance = None
                if hasattr(model, 'feature_importances_'):
                    model_importance = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': model.feature_importances_,
                        'clinical_name': [self.clinical_mappings[f]['name'] for f in self.feature_names]
                    }).sort_values('importance', ascending=False)
                elif hasattr(model, 'coef_'):
                    model_importance = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': np.abs(model.coef_[0]),
                        'clinical_name': [self.clinical_mappings[f]['name'] for f in self.feature_names]
                    }).sort_values('importance', ascending=False)
                
                global_results[name] = {
                    'permutation_importance': importance_df,
                    'model_importance': model_importance,
                    'raw_permutation': perm_importance.importances
                }
                
                self.logger.info(f"  Global metrics calculated for {name}")
                
            except Exception as e:
                self.logger.error(f"  Error calculating global metrics for {name}: {e}")
                continue
        
        self.explainers['global'] = global_results
        return global_results
    
    def evaluate_explanation_quality(self) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation of explanation quality using multiple metrics
        """
        self.logger.info("Evaluating explanation quality")
        
        quality_results = {}
        
        for model_name in self.models.keys():
            self.logger.info(f"Evaluating explanation quality for {model_name}")
            
            model_quality = {}
            
            # Faithfulness evaluation using SHAP (if available)
            if SHAP_AVAILABLE and 'shap' in self.explainers and model_name in self.explainers['shap']:
                faithfulness_scores = self._evaluate_faithfulness(model_name)
                model_quality.update(faithfulness_scores)
            
            # Stability evaluation
            stability_scores = self._evaluate_stability(model_name)
            model_quality.update(stability_scores)
            
            # Comprehensiveness evaluation
            comprehensiveness_scores = self._evaluate_comprehensiveness(model_name)
            model_quality.update(comprehensiveness_scores)
            
            quality_results[model_name] = model_quality
        
        self.explanation_quality_metrics = quality_results
        return quality_results
    
    def _evaluate_faithfulness(self, model_name: str, num_samples: int = 50) -> Dict[str, float]:
        """Evaluate faithfulness of SHAP explanations"""
        
        if model_name not in self.explainers['shap']:
            return {'faithfulness_mean': 0.0, 'faithfulness_std': 0.0}
        
        model = self.models[model_name]
        shap_data = self.explainers['shap'][model_name]
        
        faithfulness_scores = []
        
        # Determine data type
        if model_name in ['logistic_regression', 'svm']:
            X_test_use = self.X_test_scaled
        else:
            X_test_use = self.X_test
        
        sample_indices = np.random.choice(len(X_test_use), min(num_samples, len(X_test_use)), replace=False)
        
        for idx in sample_indices:
            try:
                instance = X_test_use[idx:idx+1]
                
                # Get SHAP explanation
                if model_name in ['logistic_regression', 'svm']:
                    shap_values = shap_data['explainer'].shap_values(instance)
                else:
                    shap_values = shap_data['explainer'].shap_values(instance)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Original prediction
                original_pred = model.predict_proba(instance)[0, 1]
                
                # Remove top 3 most important features
                top_features = np.argsort(np.abs(shap_values[0]))[-3:]
                
                # Create modified instance
                instance_modified = instance.copy()
                for feature_idx in top_features:
                    if model_name in ['logistic_regression', 'svm']:
                        instance_modified[0, feature_idx] = 0  # Scaled mean is 0
                    else:
                        instance_modified[0, feature_idx] = self.X_train[:, feature_idx].mean()
                
                # Modified prediction
                modified_pred = model.predict_proba(instance_modified)[0, 1]
                
                # Faithfulness score
                faithfulness = abs(original_pred - modified_pred)
                faithfulness_scores.append(faithfulness)
                
            except Exception as e:
                self.logger.warning(f"Error in faithfulness evaluation for instance {idx}: {e}")
                continue
        
        return {
            'faithfulness_mean': np.mean(faithfulness_scores) if faithfulness_scores else 0.0,
            'faithfulness_std': np.std(faithfulness_scores) if faithfulness_scores else 0.0
        }
    
    def _evaluate_stability(self, model_name: str, num_samples: int = 30) -> Dict[str, float]:
        """Evaluate stability of explanations to small perturbations"""
        
        if 'shap' not in self.explainers or model_name not in self.explainers['shap']:
            return {'stability_mean': 0.0, 'stability_std': 0.0}
        
        shap_data = self.explainers['shap'][model_name]
        stability_scores = []
        
        # Determine data type
        if model_name in ['logistic_regression', 'svm']:
            X_test_use = self.X_test_scaled
        else:
            X_test_use = self.X_test
        
        sample_indices = np.random.choice(len(X_test_use), min(num_samples, len(X_test_use)), replace=False)
        
        for idx in sample_indices:
            try:
                instance = X_test_use[idx:idx+1]
                
                # Original explanation
                original_shap = shap_data['explainer'].shap_values(instance)
                if isinstance(original_shap, list):
                    original_shap = original_shap[1]
                
                # Add small noise
                noise_level = 0.01 * np.std(X_test_use, axis=0)
                noise = np.random.normal(0, noise_level, instance.shape)
                noisy_instance = instance + noise
                
                # Noisy explanation
                noisy_shap = shap_data['explainer'].shap_values(noisy_instance)
                if isinstance(noisy_shap, list):
                    noisy_shap = noisy_shap[1]
                
                # Calculate correlation
                correlation = np.corrcoef(original_shap[0], noisy_shap[0])[0, 1]
                if not np.isnan(correlation):
                    stability_scores.append(correlation)
                
            except Exception as e:
                self.logger.warning(f"Error in stability evaluation for instance {idx}: {e}")
                continue
        
        return {
            'stability_mean': np.mean(stability_scores) if stability_scores else 0.0,
            'stability_std': np.std(stability_scores) if stability_scores else 0.0
        }
    
    def _evaluate_comprehensiveness(self, model_name: str, num_samples: int = 30) -> Dict[str, float]:
        """Evaluate comprehensiveness of explanations"""
        
        if 'shap' not in self.explainers or model_name not in self.explainers['shap']:
            return {'comprehensiveness_mean': 0.0, 'comprehensiveness_std': 0.0}
        
        model = self.models[model_name]
        shap_data = self.explainers['shap'][model_name]
        comprehensiveness_scores = []
        
        # Determine data type
        if model_name in ['logistic_regression', 'svm']:
            X_test_use = self.X_test_scaled
        else:
            X_test_use = self.X_test
        
        sample_indices = np.random.choice(len(X_test_use), min(num_samples, len(X_test_use)), replace=False)
        
        for idx in sample_indices:
            try:
                instance = X_test_use[idx:idx+1]
                
                # Get SHAP explanation
                shap_values = shap_data['explainer'].shap_values(instance)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Original prediction
                original_pred = model.predict_proba(instance)[0, 1]
                
                # Remove features progressively (most important first)
                feature_order = np.argsort(np.abs(shap_values[0]))[::-1]
                cumulative_change = 0
                
                instance_modified = instance.copy()
                for i, feature_idx in enumerate(feature_order[:5]):  # Top 5 features
                    # Replace with mean
                    if model_name in ['logistic_regression', 'svm']:
                        instance_modified[0, feature_idx] = 0
                    else:
                        instance_modified[0, feature_idx] = self.X_train[:, feature_idx].mean()
                    
                    # New prediction
                    new_pred = model.predict_proba(instance_modified)[0, 1]
                    cumulative_change += abs(original_pred - new_pred)
                    original_pred = new_pred  # Update for next iteration
                
                comprehensiveness_scores.append(cumulative_change)
                
            except Exception as e:
                self.logger.warning(f"Error in comprehensiveness evaluation for instance {idx}: {e}")
                continue
        
        return {
            'comprehensiveness_mean': np.mean(comprehensiveness_scores) if comprehensiveness_scores else 0.0,
            'comprehensiveness_std': np.std(comprehensiveness_scores) if comprehensiveness_scores else 0.0
        }


# =============================================================================
# CLINICAL EXPLANATION INTERFACE
# =============================================================================

class ClinicalExplanationInterface:
    """
    Clinical-focused explanation interface for healthcare professionals
    """
    
    def __init__(self, framework: ProductionExplainabilityFramework):
        self.framework = framework
        self.risk_thresholds = {
            'low': 0.25,
            'moderate': 0.50,
            'high': 0.75
        }
        
        # Clinical intervention mappings
        self.intervention_mappings = {
            'length_of_stay': {
                'high_risk_threshold': 10,
                'interventions': [
                    'Enhanced discharge planning',
                    'Home health services coordination',
                    'Transitional care management'
                ]
            },
            'num_medications': {
                'high_risk_threshold': 15,
                'interventions': [
                    'Medication reconciliation',
                    'Polypharmacy review',
                    'Clinical pharmacist consultation'
                ]
            },
            'charlson_comorbidity_index': {
                'high_risk_threshold': 6,
                'interventions': [
                    'Multidisciplinary care coordination',
                    'Specialist referrals',
                    'Care management enrollment'
                ]
            },
            'creatinine_level': {
                'high_risk_threshold': 2.0,
                'interventions': [
                    'Nephrology consultation',
                    'Medication dose adjustments',
                    'Fluid management optimization'
                ]
            }
        }
    
    def generate_clinical_explanation(self, patient_data: np.ndarray, 
                                    model_name: str = 'random_forest') -> Dict[str, Any]:
        """
        Generate comprehensive clinical explanation for a patient
        """
        start_time = time.time()
        
        # Basic prediction
        model = self.framework.models[model_name]
        
        if model_name in ['logistic_regression', 'svm']:
            patient_scaled = self.framework.scaler.transform([patient_data])
            risk_probability = model.predict_proba(patient_scaled)[0, 1]
            patient_input = patient_scaled[0]
        else:
            risk_probability = model.predict_proba([patient_data])[0, 1]
            patient_input = patient_data
        
        # Determine risk level
        if risk_probability < self.risk_thresholds['low']:
            risk_level = 'Low'
            risk_color = '#28a745'  # Green
        elif risk_probability < self.risk_thresholds['moderate']:
            risk_level = 'Moderate'
            risk_color = '#ffc107'  # Yellow
        elif risk_probability < self.risk_thresholds['high']:
            risk_level = 'High'
            risk_color = '#fd7e14'  # Orange
        else:
            risk_level = 'Very High'
            risk_color = '#dc3545'  # Red
        
        explanation = {
            'patient_id': f"PT_{uuid.uuid4().hex[:8].upper()}",
            'risk_probability': float(risk_probability),
            'risk_percentage': f"{risk_probability:.1%}",
            'risk_level': risk_level,
            'risk_color': risk_color,
            'model_used': model_name,
            'timestamp': datetime.now().isoformat(),
            'patient_characteristics': {}
        }
        
        # Add patient characteristics with clinical context
        for i, (feature, value) in enumerate(zip(self.framework.feature_names, patient_data)):
            clinical_info = self.framework.clinical_mappings[feature]
            explanation['patient_characteristics'][feature] = {
                'value': float(value),
                'clinical_name': clinical_info['name'],
                'unit': clinical_info['unit'],
                'normal_range': clinical_info['normal_range'],
                'is_abnormal': self._is_value_abnormal(feature, value)
            }
        
        # Add SHAP explanations if available
        if (SHAP_AVAILABLE and 'shap' in self.framework.explainers and 
            model_name in self.framework.explainers['shap']):
            
            try:
                shap_explainer = self.framework.explainers['shap'][model_name]['explainer']
                shap_values = shap_explainer.shap_values([patient_input])
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Create feature contributions
                contributions = []
                for i, (feature, value) in enumerate(zip(self.framework.feature_names, patient_data)):
                    contribution = {
                        'feature': feature,
                        'clinical_name': self.framework.clinical_mappings[feature]['name'],
                        'value': float(value),
                        'contribution': float(shap_values[0][i]),
                        'contribution_magnitude': abs(float(shap_values[0][i])),
                        'impact': 'Increases Risk' if shap_values[0][i] > 0 else 'Decreases Risk',
                        'impact_strength': self._categorize_impact_strength(abs(shap_values[0][i]))
                    }
                    contributions.append(contribution)
                
                # Sort by absolute contribution
                contributions.sort(key=lambda x: x['contribution_magnitude'], reverse=True)
                explanation['feature_contributions'] = contributions
                explanation['base_risk'] = float(self.framework.explainers['shap'][model_name]['expected_value'])
                
            except Exception as e:
                self.framework.logger.warning(f"Error generating SHAP explanation: {e}")
                explanation['feature_contributions'] = []
        
        # Generate clinical recommendations
        explanation['clinical_recommendations'] = self._generate_clinical_recommendations(explanation)
        
        # Add model performance context
        if model_name in self.framework.model_performance:
            perf = self.framework.model_performance[model_name]
            explanation['model_performance'] = {
                'accuracy': f"{perf['accuracy']:.1%}",
                'sensitivity': f"{perf['sensitivity']:.1%}",
                'specificity': f"{perf['specificity']:.1%}",
                'auc': f"{perf['auc']:.3f}"
            }
        
        explanation['computation_time_ms'] = (time.time() - start_time) * 1000
        
        return explanation
    
    def _is_value_abnormal(self, feature: str, value: float) -> bool:
        """Determine if a clinical value is abnormal"""
        
        # Define normal ranges for key clinical features
        normal_ranges = {
            'age': (18, 65),
            'length_of_stay': (1, 7),
            'num_medications': (0, 10),
            'charlson_comorbidity_index': (0, 3),
            'hemoglobin_level': (12, 16),
            'creatinine_level': (0.7, 1.3),
            'sodium_level': (135, 145),
            'glucose_level': (70, 100)
        }
        
        if feature in normal_ranges:
            min_val, max_val = normal_ranges[feature]
            return value < min_val or value > max_val
        
        # For severity scores and binary indicators
        if 'severity' in feature:
            return value > 3
        
        if feature in ['emergency_admission', 'icu_stay', 'mechanical_ventilation', 
                      'dialysis_required', 'previous_readmission']:
            return value == 1
        
        return False
    
    def _categorize_impact_strength(self, magnitude: float) -> str:
        """Categorize the strength of feature impact"""
        if magnitude > 0.1:
            return 'Strong'
        elif magnitude > 0.05:
            return 'Moderate'
        elif magnitude > 0.01:
            return 'Weak'
        else:
            return 'Minimal'
    
    def _generate_clinical_recommendations(self, explanation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable clinical recommendations"""
        
        recommendations = []
        
        # Risk-level based recommendations
        risk_level = explanation['risk_level']
        
        if risk_level in ['High', 'Very High']:
            recommendations.extend([
                {
                    'category': 'Follow-up Care',
                    'recommendation': 'Schedule follow-up appointment within 7 days of discharge',
                    'priority': 'High',
                    'evidence_level': 'A'
                },
                {
                    'category': 'Care Coordination',
                    'recommendation': 'Activate transitional care management program',
                    'priority': 'High',
                    'evidence_level': 'A'
                }
            ])
        
        if risk_level == 'Very High':
            recommendations.append({
                'category': 'Intensive Monitoring',
                'recommendation': 'Consider 48-72 hour post-discharge phone call',
                'priority': 'High',
                'evidence_level': 'B'
            })
        
        # Feature-specific recommendations
        if 'feature_contributions' in explanation:
            top_risk_factors = [
                contrib for contrib in explanation['feature_contributions'][:5] 
                if contrib['contribution'] > 0 and contrib['impact_strength'] in ['Strong', 'Moderate']
            ]
            
            for factor in top_risk_factors:
                feature = factor['feature']
                value = factor['value']
                
                if feature in self.intervention_mappings:
                    mapping = self.intervention_mappings[feature]
                    if value > mapping['high_risk_threshold']:
                        for intervention in mapping['interventions']:
                            recommendations.append({
                                'category': f"{factor['clinical_name']} Management",
                                'recommendation': intervention,
                                'priority': 'Medium',
                                'evidence_level': 'B',
                                'trigger_feature': feature,
                                'trigger_value': value
                            })
        
        # General recommendations based on abnormal values
        if 'patient_characteristics' in explanation:
            for feature, char in explanation['patient_characteristics'].items():
                if char['is_abnormal']:
                    if feature == 'hemoglobin_level' and char['value'] < 10:
                        recommendations.append({
                            'category': 'Anemia Management',
                            'recommendation': 'Evaluate and treat underlying anemia',
                            'priority': 'Medium',
                            'evidence_level': 'B'
                        })
                    elif feature == 'glucose_level' and char['value'] > 200:
                        recommendations.append({
                            'category': 'Diabetes Management',
                            'recommendation': 'Optimize glycemic control before discharge',
                            'priority': 'High',
                            'evidence_level': 'A'
                        })
        
        # Remove duplicates and prioritize
        unique_recommendations = []
        seen_recommendations = set()
        
        for rec in recommendations:
            rec_key = (rec['category'], rec['recommendation'])
            if rec_key not in seen_recommendations:
                unique_recommendations.append(rec)
                seen_recommendations.add(rec_key)
        
        # Sort by priority
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        unique_recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return unique_recommendations[:10]  # Limit to top 10 recommendations


# =============================================================================
# PRODUCTION EXPLAINABILITY SERVICE
# =============================================================================

class ProductionExplainabilityService:
    """
    Production service for real-time model explanations with API endpoints
    """
    
    def __init__(self, framework: ProductionExplainabilityFramework):
        self.framework = framework
        self.clinical_interface = ClinicalExplanationInterface(framework)
        self.request_cache = {}
        self.request_history = []
        self.performance_metrics = defaultdict(list)
        self.lock = threading.Lock()
        
    def explain_prediction(self, request: ExplanationRequest) -> ExplanationResponse:
        """
        Main endpoint for generating explanations
        """
        start_time = time.time()
        
        try:
            # Convert instance data to numpy array
            patient_data = np.array([request.instance_data[feature] 
                                   for feature in self.framework.feature_names])
            
            # Generate clinical explanation
            clinical_explanation = self.clinical_interface.generate_clinical_explanation(
                patient_data, request.model_id
            )
            
            # Prepare response
            response = ExplanationResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                prediction=1 if clinical_explanation['risk_probability'] > 0.5 else 0,
                probability=clinical_explanation['risk_probability'],
                risk_level=clinical_explanation['risk_level'],
                explanations={
                    'clinical': clinical_explanation,
                    'shap': clinical_explanation.get('feature_contributions', []),
                    'model_performance': clinical_explanation.get('model_performance', {})
                },
                clinical_recommendations=clinical_explanation['clinical_recommendations'],
                computation_time_ms=clinical_explanation['computation_time_ms'],
                timestamp=clinical_explanation['timestamp'],
                model_performance=clinical_explanation.get('model_performance', {})
            )
            
            # Log request
            with self.lock:
                self.request_history.append({
                    'request_id': request.request_id,
                    'model_id': request.model_id,
                    'risk_level': response.risk_level,
                    'computation_time_ms': response.computation_time_ms,
                    'timestamp': response.timestamp
                })
                
                self.performance_metrics['computation_time'].append(response.computation_time_ms)
                self.performance_metrics['risk_distribution'].append(response.probability)
            
            return response
            
        except Exception as e:
            self.framework.logger.error(f"Error processing explanation request {request.request_id}: {e}")
            
            return ExplanationResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                prediction=0,
                probability=0.0,
                risk_level='Error',
                explanations={'error': str(e)},
                clinical_recommendations=[],
                computation_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now().isoformat(),
                model_performance={}
            )
    
    def batch_explain(self, requests: List[ExplanationRequest]) -> List[ExplanationResponse]:
        """
        Process multiple explanation requests in batch
        """
        self.framework.logger.info(f"Processing batch of {len(requests)} explanation requests")
        
        responses = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_request = {
                executor.submit(self.explain_prediction, request): request 
                for request in requests
            }
            
            for future in future_to_request:
                try:
                    response = future.result(timeout=30)  # 30 second timeout
                    responses.append(response)
                except Exception as e:
                    request = future_to_request[future]
                    self.framework.logger.error(f"Error in batch processing for request {request.request_id}: {e}")
                    
                    # Create error response
                    error_response = ExplanationResponse(
                        request_id=request.request_id,
                        model_id=request.model_id,
                        prediction=0,
                        probability=0.0,
                        risk_level='Error',
                        explanations={'error': str(e)},
                        clinical_recommendations=[],
                        computation_time_ms=0.0,
                        timestamp=datetime.now().isoformat(),
                        model_performance={}
                    )
                    responses.append(error_response)
        
        return responses
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive service statistics
        """
        with self.lock:
            if not self.request_history:
                return {'message': 'No requests processed yet'}
            
            total_requests = len(self.request_history)
            
            # Computation time statistics
            computation_times = self.performance_metrics['computation_time']
            avg_computation_time = np.mean(computation_times)
            p95_computation_time = np.percentile(computation_times, 95)
            
            # Risk distribution
            risk_probabilities = self.performance_metrics['risk_distribution']
            
            # Risk level distribution
            risk_levels = [req['risk_level'] for req in self.request_history]
            risk_level_counts = Counter(risk_levels)
            
            # Recent performance (last 100 requests)
            recent_requests = self.request_history[-100:]
            recent_avg_time = np.mean([req['computation_time_ms'] for req in recent_requests])
            
            return {
                'total_requests': total_requests,
                'avg_computation_time_ms': avg_computation_time,
                'p95_computation_time_ms': p95_computation_time,
                'recent_avg_computation_time_ms': recent_avg_time,
                'risk_level_distribution': dict(risk_level_counts),
                'avg_risk_probability': np.mean(risk_probabilities),
                'high_risk_percentage': len([p for p in risk_probabilities if p > 0.75]) / len(risk_probabilities) * 100,
                'service_uptime': 'Active',
                'last_request_time': self.request_history[-1]['timestamp'] if self.request_history else None
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check
        """
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        try:
            # Check if models are loaded
            health_status['checks']['models_loaded'] = len(self.framework.models) > 0
            
            # Check if explainers are available
            health_status['checks']['explainers_available'] = len(self.framework.explainers) > 0
            
            # Test prediction with dummy data
            dummy_data = np.random.random(len(self.framework.feature_names))
            test_request = ExplanationRequest(
                request_id='health_check',
                model_id='random_forest',
                instance_data={feature: dummy_data[i] for i, feature in enumerate(self.framework.feature_names)},
                explanation_methods=['shap']
            )
            
            test_response = self.explain_prediction(test_request)
            health_status['checks']['prediction_test'] = test_response.risk_level != 'Error'
            health_status['checks']['response_time_ms'] = test_response.computation_time_ms
            
            # Overall health
            all_checks_passed = all(health_status['checks'].values())
            health_status['status'] = 'healthy' if all_checks_passed else 'degraded'
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status


# =============================================================================
# MAIN EXECUTION AND DEMONSTRATION
# =============================================================================

def run_comprehensive_explainability_experiment():
    """
    Run comprehensive explainability experiment demonstrating all techniques
    """
    print("=" * 80)
    print("COMPREHENSIVE MODEL EXPLAINABILITY EXPERIMENT")
    print("Production-Ready Healthcare AI Explanation System")
    print("=" * 80)
    
    # Initialize framework
    framework = ProductionExplainabilityFramework(random_state=42)
    
    # Step 1: Data Preparation
    print("\n1. DATA PREPARATION")
    print("-" * 40)
    framework.prepare_data()
    
    # Step 2: Model Creation and Training
    print("\n2. MODEL CREATION AND TRAINING")
    print("-" * 40)
    framework.create_prediction_models()
    model_results = framework.train_and_evaluate_models()
    
    # Step 3: Explainability Implementation
    print("\n3. EXPLAINABILITY IMPLEMENTATION")
    print("-" * 40)
    shap_results = framework.implement_shap_explanations()
    lime_results = framework.implement_lime_explanations()
    global_results = framework.calculate_global_interpretability()
    
    # Step 4: Explanation Quality Evaluation
    print("\n4. EXPLANATION QUALITY EVALUATION")
    print("-" * 40)
    quality_results = framework.evaluate_explanation_quality()
    
    # Step 5: Clinical Interface Demonstration
    print("\n5. CLINICAL INTERFACE DEMONSTRATION")
    print("-" * 40)
    
    clinical_interface = ClinicalExplanationInterface(framework)
    
    # Generate explanation for sample patient
    sample_patient = framework.X_test[0]
    clinical_explanation = clinical_interface.generate_clinical_explanation(sample_patient)
    
    print(f"Patient ID: {clinical_explanation['patient_id']}")
    print(f"Risk Assessment: {clinical_explanation['risk_level']} ({clinical_explanation['risk_percentage']})")
    print(f"Model Used: {clinical_explanation['model_used']}")
    
    if 'feature_contributions' in clinical_explanation:
        print("\nTop Risk Factors:")
        for contrib in clinical_explanation['feature_contributions'][:5]:
            print(f"  • {contrib['clinical_name']}: {contrib['impact']} "
                  f"({contrib['impact_strength']} impact)")
    
    print(f"\nClinical Recommendations ({len(clinical_explanation['clinical_recommendations'])}):")
    for i, rec in enumerate(clinical_explanation['clinical_recommendations'][:5], 1):
        print(f"  {i}. {rec['recommendation']} (Priority: {rec['priority']})")
    
    # Step 6: Production Service Demonstration
    print("\n6. PRODUCTION SERVICE DEMONSTRATION")
    print("-" * 40)
    
    service = ProductionExplainabilityService(framework)
    
    # Test single request
    test_request = ExplanationRequest(
        request_id=f"req_{uuid.uuid4().hex[:8]}",
        model_id='random_forest',
        instance_data={feature: sample_patient[i] for i, feature in enumerate(framework.feature_names)},
        explanation_methods=['shap', 'clinical']
    )
    
    response = service.explain_prediction(test_request)
    print(f"Service Response: {response.risk_level} risk ({response.probability:.2%})")
    print(f"Response Time: {response.computation_time_ms:.1f}ms")
    
    # Test batch processing
    batch_requests = []
    for i in range(5):
        patient_data = framework.X_test[i]
        request = ExplanationRequest(
            request_id=f"batch_req_{i}",
            model_id='random_forest',
            instance_data={feature: patient_data[j] for j, feature in enumerate(framework.feature_names)},
            explanation_methods=['shap']
        )
        batch_requests.append(request)
    
    batch_responses = service.batch_explain(batch_requests)
    print(f"Batch Processing: {len(batch_responses)} requests processed")
    
    # Service statistics
    stats = service.get_service_statistics()
    print(f"Average Response Time: {stats['avg_computation_time_ms']:.1f}ms")
    print(f"Risk Distribution: {stats['risk_level_distribution']}")
    
    # Health check
    health = service.health_check()
    print(f"Service Health: {health['status']}")
    
    # Step 7: Comprehensive Visualization
    print("\n7. COMPREHENSIVE VISUALIZATION")
    print("-" * 40)
    create_explainability_dashboard(framework, model_results, quality_results)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return {
        'framework': framework,
        'model_results': model_results,
        'explainability_results': {
            'shap': shap_results,
            'lime': lime_results,
            'global': global_results
        },
        'quality_results': quality_results,
        'clinical_interface': clinical_interface,
        'service': service
    }

def create_explainability_dashboard(framework, model_results, quality_results):
    """
    Create comprehensive explainability visualization dashboard
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Model Explainability Dashboard - Patient Readmission Prediction', fontsize=16)
    
    # 1. Model Performance Comparison
    model_names = list(model_results.keys())
    auc_scores = [model_results[name]['auc'] for name in model_names]
    sensitivity_scores = [model_results[name]['sensitivity'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, auc_scores, width, label='AUC', alpha=0.8)
    axes[0, 0].bar(x + width/2, sensitivity_scores, width, label='Sensitivity', alpha=0.8)
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Feature Importance (Global)
    if 'global' in framework.explainers and 'random_forest' in framework.explainers['global']:
        importance_data = framework.explainers['global']['random_forest']['permutation_importance']
        top_features = importance_data.head(10)
        
        axes[0, 1].barh(top_features['clinical_name'], top_features['importance_mean'])
        axes[0, 1].set_title('Top 10 Feature Importance (Permutation)')
        axes[0, 1].set_xlabel('Importance Score')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. SHAP Feature Importance
    if SHAP_AVAILABLE and 'shap' in framework.explainers and 'random_forest' in framework.explainers['shap']:
        try:
            shap_data = framework.explainers['shap']['random_forest']
            shap_values = shap_data['shap_values']
            
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            feature_importance = pd.DataFrame({
                'feature': framework.feature_names,
                'importance': mean_shap,
                'clinical_name': [framework.clinical_mappings[f]['name'] for f in framework.feature_names]
            }).sort_values('importance', ascending=True).tail(10)
            
            axes[0, 2].barh(feature_importance['clinical_name'], feature_importance['importance'])
            axes[0, 2].set_title('Top 10 SHAP Feature Importance')
            axes[0, 2].set_xlabel('Mean |SHAP Value|')
            axes[0, 2].grid(True, alpha=0.3)
        except Exception as e:
            axes[0, 2].text(0.5, 0.5, f'SHAP visualization error', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
    
    # 4. Risk Distribution
    model = framework.models['random_forest']
    risk_probs = model.predict_proba(framework.X_test)[:, 1]
    
    axes[1, 0].hist(risk_probs, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    axes[1, 0].axvline(0.25, color='green', linestyle='--', label='Low Risk')
    axes[1, 0].axvline(0.50, color='orange', linestyle='--', label='Moderate Risk')
    axes[1, 0].axvline(0.75, color='red', linestyle='--', label='High Risk')
    axes[1, 0].set_title('Risk Score Distribution')
    axes[1, 0].set_xlabel('Readmission Risk Probability')
    axes[1, 0].set_ylabel('Number of Patients')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Explanation Quality Metrics
    if quality_results:
        models = list(quality_results.keys())
        faithfulness = [quality_results[m].get('faithfulness_mean', 0) for m in models]
        stability = [quality_results[m].get('stability_mean', 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, faithfulness, width, label='Faithfulness', alpha=0.7)
        axes[1, 1].bar(x + width/2, stability, width, label='Stability', alpha=0.7)
        axes[1, 1].set_title('Explanation Quality Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Feature Correlation Heatmap
    correlation_matrix = framework.X_train_df.corr()
    top_features = framework.explainers['global']['random_forest']['permutation_importance'].head(8)['feature'].tolist()
    subset_corr = correlation_matrix.loc[top_features, top_features]
    
    im = axes[1, 2].imshow(subset_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 2].set_xticks(range(len(top_features)))
    axes[1, 2].set_yticks(range(len(top_features)))
    axes[1, 2].set_xticklabels([framework.clinical_mappings[f]['name'][:15] for f in top_features], rotation=45)
    axes[1, 2].set_yticklabels([framework.clinical_mappings[f]['name'][:15] for f in top_features])
    axes[1, 2].set_title('Feature Correlation Matrix (Top 8)')
    plt.colorbar(im, ax=axes[1, 2])
    
    # 7. Model Complexity vs Performance
    complexity_scores = {
        'logistic_regression': 1,
        'decision_tree': 2,
        'random_forest': 3,
        'gradient_boosting': 4,
        'svm': 3
    }
    
    complexity = [complexity_scores[name] for name in model_names]
    performance = [model_results[name]['auc'] for name in model_names]
    
    axes[2, 0].scatter(complexity, performance, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        axes[2, 0].annotate(name, (complexity[i], performance[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[2, 0].set_xlabel('Model Complexity')
    axes[2, 0].set_ylabel('AUC Score')
    axes[2, 0].set_title('Model Complexity vs Performance')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Training Time Comparison
    training_times = [model_results[name]['training_time'] for name in model_names]
    
    bars = axes[2, 1].bar(model_names, training_times, alpha=0.7, color='lightcoral')
    axes[2, 1].set_title('Model Training Time Comparison')
    axes[2, 1].set_ylabel('Training Time (seconds)')
    axes[2, 1].tick_params(axis='x', rotation=45)
    axes[2, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, training_times):
        axes[2, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{time_val:.2f}s', ha='center', va='bottom')
    
    # 9. Clinical Risk Categories
    risk_categories = ['Low (<25%)', 'Moderate (25-50%)', 'High (50-75%)', 'Very High (>75%)']
    risk_counts = [
        np.sum(risk_probs < 0.25),
        np.sum((risk_probs >= 0.25) & (risk_probs < 0.50)),
        np.sum((risk_probs >= 0.50) & (risk_probs < 0.75)),
        np.sum(risk_probs >= 0.75)
    ]
    colors = ['green', 'yellow', 'orange', 'red']
    
    axes[2, 2].pie(risk_counts, labels=risk_categories, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[2, 2].set_title('Patient Risk Category Distribution')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function demonstrating comprehensive explainability solution
    """
    print("Day 31: Model Explainability - Complete Production Solution")
    print("Advanced Healthcare AI Explanation System")
    print()
    
    try:
        # Run comprehensive experiment
        results = run_comprehensive_explainability_experiment()
        
        print("\n" + "=" * 80)
        print("KEY INSIGHTS AND RECOMMENDATIONS")
        print("=" * 80)
        
        print("1. EXPLAINABILITY METHODS:")
        print("   - SHAP provides theoretically grounded, consistent explanations")
        print("   - LIME offers intuitive local explanations for individual cases")
        print("   - Global methods reveal overall model behavior patterns")
        
        print("\n2. CLINICAL INTEGRATION:")
        print("   - Explanations must be tailored to healthcare professionals")
        print("   - Risk levels and recommendations should be actionable")
        print("   - Clinical context improves explanation relevance")
        
        print("\n3. PRODUCTION CONSIDERATIONS:")
        print("   - Response time < 500ms for real-time clinical use")
        print("   - Explanation quality monitoring is essential")
        print("   - Comprehensive audit trails for regulatory compliance")
        
        print("\n4. REGULATORY COMPLIANCE:")
        print("   - Documentation of model decisions for audit purposes")
        print("   - Explanation consistency and reliability validation")
        print("   - Patient privacy protection in explanation systems")
        
        # Performance summary
        if 'model_results' in results:
            best_model = max(results['model_results'].items(), key=lambda x: x[1]['auc'])
            print(f"\nBest Model Performance:")
            print(f"- Model: {best_model[0]}")
            print(f"- AUC: {best_model[1]['auc']:.4f}")
            print(f"- Sensitivity: {best_model[1]['sensitivity']:.4f}")
            print(f"- Specificity: {best_model[1]['specificity']:.4f}")
        
        # Explainability summary
        if 'explainability_results' in results:
            available_methods = [method for method, data in results['explainability_results'].items() if data]
            print(f"\nExplainability Methods: {', '.join(available_methods)}")
        
        # Service performance
        if 'service' in results:
            service_stats = results['service'].get_service_statistics()
            if 'avg_computation_time_ms' in service_stats:
                print(f"Average Explanation Time: {service_stats['avg_computation_time_ms']:.1f}ms")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()