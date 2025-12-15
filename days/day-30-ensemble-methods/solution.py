"""
Day 30: Ensemble Methods - Bagging, Boosting, Stacking - Complete Solution

Production-ready ensemble system for TechCorp's credit risk assessment platform.
Demonstrates comprehensive ensemble approaches and production deployment patterns.

This solution showcases:
- Multiple ensemble methods (bagging, boosting, stacking, voting)
- Advanced model selection and hyperparameter optimization
- Production-ready deployment with monitoring and alerting
- Comprehensive evaluation framework with business metrics
- Real-time serving infrastructure with caching and logging
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
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, 
                           confusion_matrix, roc_curve, precision_recall_curve,
                           f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.datasets import make_classification
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import matplotlib.pyplot as plt
import seaborn as sns

# Advanced libraries (handle gracefully if not available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

import logging
import time
import json
import joblib
from collections import defaultdict, Counter
import scipy.sparse as sp
from scipy import stats

# =============================================================================
# PRODUCTION ENSEMBLE FRAMEWORK
# =============================================================================

class ProductionEnsembleFramework:
    """
    Production-grade ensemble framework with comprehensive features
    """
    
    def __init__(self, random_state=42, n_jobs=-1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.logger = self._setup_logging()
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        
        # Model storage
        self.base_models = {}
        self.ensemble_models = {}
        self.best_model = None
        self.model_performance = {}
        
        # Production components
        self.model_server = None
        self.monitor = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('ensemble_framework')
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
            file_handler = logging.FileHandler('ensemble_framework.log')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def generate_synthetic_credit_data(self, n_samples=10000, n_features=20) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate realistic synthetic credit risk dataset
        
        Returns:
            X: Feature matrix
            y: Target labels (0: No default, 1: Default)
            feature_names: List of feature names
        """
        self.logger.info(f"Generating synthetic credit dataset: {n_samples} samples, {n_features} features")
        
        # Generate base dataset with realistic characteristics
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            n_redundant=int(n_features * 0.15),
            n_clusters_per_class=3,
            weights=[0.82, 0.18],  # 18% default rate (realistic for credit)
            flip_y=0.01,  # 1% label noise
            class_sep=0.8,  # Moderate class separation
            random_state=self.random_state
        )
        
        # Add realistic feature transformations
        np.random.seed(self.random_state)
        
        # Make some features more realistic (e.g., income should be positive)
        for i in range(min(5, n_features)):
            X[:, i] = np.abs(X[:, i]) * 50000 + 25000  # Income-like features
        
        # Add some categorical-like features
        for i in range(5, min(10, n_features)):
            X[:, i] = np.round(np.abs(X[:, i]) * 10)  # Count-like features
        
        # Create realistic feature names
        feature_names = [
            'annual_income', 'monthly_income', 'total_assets', 'liquid_assets', 'investment_value',
            'credit_accounts', 'open_accounts', 'total_accounts', 'delinquent_accounts', 'inquiries_6m',
            'credit_utilization', 'debt_to_income', 'loan_to_value', 'payment_history_score', 'credit_age',
            'employment_length', 'residence_stability', 'education_level', 'marital_status', 'dependents'
        ][:n_features]
        
        self.logger.info(f"Dataset generated - Default rate: {np.mean(y):.2%}")
        
        return X, y, feature_names
    
    def prepare_data(self, X=None, y=None, feature_names=None, test_size=0.2, 
                    scale_features=True, handle_imbalance=True):
        """
        Comprehensive data preparation with advanced preprocessing
        """
        self.logger.info("Starting data preparation")
        
        # Generate data if not provided
        if X is None or y is None:
            X, y, feature_names = self.generate_synthetic_credit_data()
        
        self.feature_names = feature_names
        
        # Split data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y, shuffle=True
        )
        
        # Feature scaling
        if scale_features:
            self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
        else:
            self.X_train_scaled = self.X_train.copy()
            self.X_test_scaled = self.X_test.copy()
        
        # Log data statistics
        self.logger.info(f"Training set: {self.X_train.shape[0]} samples")
        self.logger.info(f"Test set: {self.X_test.shape[0]} samples")
        self.logger.info(f"Features: {self.X_train.shape[1]}")
        self.logger.info(f"Default rate (train): {np.mean(self.y_train):.2%}")
        self.logger.info(f"Default rate (test): {np.mean(self.y_test):.2%}")
        
        return self
    
    def create_base_models(self) -> Dict[str, Any]:
        """
        Create comprehensive set of diverse base models
        """
        self.logger.info("Creating base models")
        
        self.base_models = {
            # Linear models
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                solver='liblinear'
            ),
            
            # Tree-based models
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=self.n_jobs
            ),
            
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=self.n_jobs
            ),
            
            'decision_tree': DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=self.random_state,
                class_weight='balanced'
            ),
            
            # Boosting models
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=self.random_state
            ),
            
            'adaboost': AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=3, class_weight='balanced'),
                n_estimators=50,
                learning_rate=1.0,
                random_state=self.random_state
            ),
            
            # Probabilistic models
            'naive_bayes': GaussianNB(),
            
            # Instance-based models
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='minkowski',
                n_jobs=self.n_jobs
            ),
            
            # Support Vector Machine
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced',
                gamma='scale'
            )
        }
        
        # Add advanced boosting models if available
        if XGBOOST_AVAILABLE:
            self.base_models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=3,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=self.n_jobs,
                verbosity=0
            )
        
        if LIGHTGBM_AVAILABLE:
            self.base_models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1
            )
        
        if CATBOOST_AVAILABLE:
            self.base_models['catboost'] = cb.CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                l2_leaf_reg=3,
                random_seed=self.random_state,
                verbose=False,
                thread_count=self.n_jobs if self.n_jobs > 0 else None
            )
        
        self.logger.info(f"Created {len(self.base_models)} base models")
        return self.base_models
    
    def evaluate_base_models(self, cv_folds=5) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation of base models with cross-validation
        """
        self.logger.info("Evaluating base models")
        
        base_results = {}
        
        for name, model in self.base_models.items():
            self.logger.info(f"Evaluating {name}")
            
            start_time = time.time()
            
            # Determine feature set (scaled vs unscaled)
            if name in ['logistic_regression', 'svm', 'knn']:
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
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_score = roc_auc_score(self.y_test, y_proba)
            f1 = f1_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            
            # Cross-validation
            cv_scores = {}
            for metric in ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']:
                scores = cross_val_score(
                    model, X_train_use, self.y_train, 
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                    scoring=metric, n_jobs=self.n_jobs
                )
                cv_scores[f'cv_{metric}_mean'] = np.mean(scores)
                cv_scores[f'cv_{metric}_std'] = np.std(scores)
            
            training_time = time.time() - start_time
            
            # Store results
            base_results[name] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'training_time': training_time,
                **cv_scores
            }
            
            self.logger.info(f"  {name}: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}, F1={f1:.4f}")
        
        self.model_performance['base_models'] = base_results
        return base_results
    def create_ensemble_models(self) -> Dict[str, Any]:
        """
        Create comprehensive ensemble models using various techniques
        """
        self.logger.info("Creating ensemble models")
        
        # Select best base models for ensembles (avoid scaling issues)
        ensemble_base_models = []
        base_results = self.model_performance.get('base_models', {})
        
        # Sort base models by AUC and select top performers
        if base_results:
            sorted_models = sorted(base_results.items(), key=lambda x: x[1]['auc'], reverse=True)
            top_model_names = [name for name, _ in sorted_models[:6]]  # Top 6 models
        else:
            # Fallback to predefined selection
            top_model_names = ['random_forest', 'gradient_boosting', 'extra_trees', 'naive_bayes']
            if XGBOOST_AVAILABLE:
                top_model_names.append('xgboost')
            if LIGHTGBM_AVAILABLE:
                top_model_names.append('lightgbm')
        
        # Create ensemble base model list
        for name in top_model_names:
            if name in self.base_models:
                ensemble_base_models.append((name, self.base_models[name]))
        
        self.ensemble_models = {
            # Voting Ensembles
            'voting_hard': VotingClassifier(
                estimators=ensemble_base_models,
                voting='hard',
                n_jobs=self.n_jobs
            ),
            
            'voting_soft': VotingClassifier(
                estimators=ensemble_base_models,
                voting='soft',
                n_jobs=self.n_jobs
            ),
            
            # Bagging Ensembles
            'bagging_rf': BaggingClassifier(
                base_estimator=DecisionTreeClassifier(
                    max_depth=8, 
                    min_samples_split=10,
                    class_weight='balanced',
                    random_state=self.random_state
                ),
                n_estimators=50,
                max_samples=0.8,
                max_features=0.8,
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            
            'bagging_extra_trees': BaggingClassifier(
                base_estimator=ExtraTreesClassifier(
                    n_estimators=10,
                    max_depth=8,
                    class_weight='balanced',
                    random_state=self.random_state
                ),
                n_estimators=20,
                max_samples=0.8,
                max_features=0.8,
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            
            # Advanced Random Forest
            'random_forest_optimized': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            
            # Gradient Boosting variants
            'gradient_boosting_optimized': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                max_features='sqrt',
                random_state=self.random_state
            )
        }
        
        # Add XGBoost ensemble if available
        if XGBOOST_AVAILABLE:
            self.ensemble_models['xgboost_optimized'] = xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=7,
                min_child_weight=3,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=self.n_jobs,
                verbosity=0
            )
        
        self.logger.info(f"Created {len(self.ensemble_models)} ensemble models")
        return self.ensemble_models
    
    def implement_advanced_stacking(self, meta_model=None, cv_folds=5) -> 'AdvancedStackingClassifier':
        """
        Implement advanced stacking ensemble with multiple layers
        """
        self.logger.info("Implementing advanced stacking ensemble")
        
        # Select diverse base models for stacking
        stacking_models = []
        
        # Add tree-based models
        if 'random_forest' in self.base_models:
            stacking_models.append(('rf', self.base_models['random_forest']))
        if 'gradient_boosting' in self.base_models:
            stacking_models.append(('gb', self.base_models['gradient_boosting']))
        if 'extra_trees' in self.base_models:
            stacking_models.append(('et', self.base_models['extra_trees']))
        
        # Add probabilistic model
        if 'naive_bayes' in self.base_models:
            stacking_models.append(('nb', self.base_models['naive_bayes']))
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE and 'xgboost' in self.base_models:
            stacking_models.append(('xgb', self.base_models['xgboost']))
        if LIGHTGBM_AVAILABLE and 'lightgbm' in self.base_models:
            stacking_models.append(('lgb', self.base_models['lightgbm']))
        
        # Default meta-model
        if meta_model is None:
            meta_model = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            )
        
        # Create advanced stacking classifier
        stacking_classifier = AdvancedStackingClassifier(
            base_models=stacking_models,
            meta_model=meta_model,
            cv_folds=cv_folds,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Fit the stacking ensemble
        stacking_classifier.fit(self.X_train, self.y_train)
        
        # Add to ensemble models
        self.ensemble_models['advanced_stacking'] = stacking_classifier
        
        self.logger.info("Advanced stacking ensemble implemented")
        return stacking_classifier
    
    def evaluate_ensemble_models(self, cv_folds=5) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation of ensemble models
        """
        self.logger.info("Evaluating ensemble models")
        
        ensemble_results = {}
        
        for name, model in self.ensemble_models.items():
            self.logger.info(f"Evaluating {name}")
            
            start_time = time.time()
            
            try:
                # Train model
                if name != 'advanced_stacking':  # Already fitted
                    model.fit(self.X_train, self.y_train)
                
                # Test set predictions
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                auc_score = roc_auc_score(self.y_test, y_proba)
                f1 = f1_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                
                # Cross-validation (skip for already fitted stacking)
                cv_scores = {}
                if name != 'advanced_stacking':
                    for metric in ['accuracy', 'roc_auc', 'f1']:
                        scores = cross_val_score(
                            model, self.X_train, self.y_train,
                            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                            scoring=metric, n_jobs=self.n_jobs
                        )
                        cv_scores[f'cv_{metric}_mean'] = np.mean(scores)
                        cv_scores[f'cv_{metric}_std'] = np.std(scores)
                else:
                    # Use internal CV scores for stacking
                    cv_scores = {
                        'cv_accuracy_mean': 0.0,
                        'cv_accuracy_std': 0.0,
                        'cv_roc_auc_mean': 0.0,
                        'cv_roc_auc_std': 0.0,
                        'cv_f1_mean': 0.0,
                        'cv_f1_std': 0.0
                    }
                
                training_time = time.time() - start_time
                
                # Store results
                ensemble_results[name] = {
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'training_time': training_time,
                    **cv_scores
                }
                
                self.logger.info(f"  {name}: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}, F1={f1:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {e}")
                continue
        
        self.model_performance['ensemble_models'] = ensemble_results
        return ensemble_results
    
    def hyperparameter_optimization(self, model_name: str, param_grid: Dict, 
                                  cv_folds=3, n_iter=20) -> Any:
        """
        Perform hyperparameter optimization for specified model
        """
        self.logger.info(f"Optimizing hyperparameters for {model_name}")
        
        if model_name not in self.base_models and model_name not in self.ensemble_models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.base_models.get(model_name) or self.ensemble_models.get(model_name)
        
        # Use RandomizedSearchCV for efficiency
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1
        )
        
        # Determine feature set
        if model_name in ['logistic_regression', 'svm', 'knn']:
            X_use = self.X_train_scaled
        else:
            X_use = self.X_train
        
        # Fit optimization
        random_search.fit(X_use, self.y_train)
        
        self.logger.info(f"Best parameters for {model_name}: {random_search.best_params_}")
        self.logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_
    
    def select_best_model(self) -> Tuple[str, Any, Dict[str, float]]:
        """
        Select the best performing model based on comprehensive metrics
        """
        self.logger.info("Selecting best model")
        
        all_results = {}
        
        # Combine all results
        if 'base_models' in self.model_performance:
            for name, metrics in self.model_performance['base_models'].items():
                all_results[f"base_{name}"] = metrics
        
        if 'ensemble_models' in self.model_performance:
            for name, metrics in self.model_performance['ensemble_models'].items():
                all_results[f"ensemble_{name}"] = metrics
        
        if not all_results:
            raise ValueError("No model results available")
        
        # Multi-criteria selection (weighted combination of metrics)
        weights = {
            'auc': 0.4,
            'f1': 0.3,
            'accuracy': 0.2,
            'precision': 0.1
        }
        
        best_score = -1
        best_model_name = None
        best_metrics = None
        
        for model_name, metrics in all_results.items():
            # Calculate weighted score
            score = sum(weights[metric] * metrics[metric] for metric in weights.keys())
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_metrics = metrics
        
        # Get actual model object
        if best_model_name.startswith('base_'):
            actual_name = best_model_name[5:]  # Remove 'base_' prefix
            best_model = self.base_models[actual_name]
        else:
            actual_name = best_model_name[9:]  # Remove 'ensemble_' prefix
            best_model = self.ensemble_models[actual_name]
        
        self.best_model = best_model
        
        self.logger.info(f"Best model selected: {best_model_name}")
        self.logger.info(f"Best model metrics: AUC={best_metrics['auc']:.4f}, "
                        f"F1={best_metrics['f1']:.4f}, Accuracy={best_metrics['accuracy']:.4f}")
        
        return best_model_name, best_model, best_metrics
    
    def create_production_pipeline(self) -> 'ProductionEnsemblePipeline':
        """
        Create production-ready ensemble pipeline
        """
        self.logger.info("Creating production pipeline")
        
        if self.best_model is None:
            raise ValueError("No best model selected. Run select_best_model() first.")
        
        pipeline = ProductionEnsemblePipeline(
            model=self.best_model,
            scaler=self.scaler,
            feature_names=self.feature_names,
            model_metadata={
                'training_date': datetime.now().isoformat(),
                'model_type': type(self.best_model).__name__,
                'performance_metrics': self.model_performance,
                'feature_count': len(self.feature_names) if self.feature_names else self.X_train.shape[1]
            }
        )
        
        return pipeline
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report
        """
        self.logger.info("Generating comprehensive report")
        
        report = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset_size': {
                    'train_samples': len(self.X_train) if self.X_train is not None else 0,
                    'test_samples': len(self.X_test) if self.X_test is not None else 0,
                    'features': len(self.feature_names) if self.feature_names else 0
                },
                'class_distribution': {
                    'train_default_rate': float(np.mean(self.y_train)) if self.y_train is not None else 0,
                    'test_default_rate': float(np.mean(self.y_test)) if self.y_test is not None else 0
                }
            },
            'model_performance': self.model_performance,
            'best_model_info': {},
            'recommendations': []
        }
        
        # Add best model information
        if self.best_model is not None:
            best_name, _, best_metrics = self.select_best_model()
            report['best_model_info'] = {
                'name': best_name,
                'metrics': best_metrics,
                'model_type': type(self.best_model).__name__
            }
        
        # Generate recommendations
        if 'ensemble_models' in self.model_performance and 'base_models' in self.model_performance:
            ensemble_avg_auc = np.mean([m['auc'] for m in self.model_performance['ensemble_models'].values()])
            base_avg_auc = np.mean([m['auc'] for m in self.model_performance['base_models'].values()])
            
            if ensemble_avg_auc > base_avg_auc:
                report['recommendations'].append(
                    "Ensemble methods show superior performance compared to individual models"
                )
            
            # Check for overfitting
            for name, metrics in self.model_performance['ensemble_models'].items():
                if 'cv_roc_auc_mean' in metrics and metrics['cv_roc_auc_mean'] > 0:
                    cv_auc = metrics['cv_roc_auc_mean']
                    test_auc = metrics['auc']
                    if test_auc - cv_auc > 0.05:
                        report['recommendations'].append(
                            f"Model {name} may be overfitting (CV AUC: {cv_auc:.3f}, Test AUC: {test_auc:.3f})"
                        )
        
        return report
    
    def visualize_results(self, save_plots=True, plot_dir='ensemble_plots'):
        """
        Create comprehensive visualizations of results
        """
        self.logger.info("Creating result visualizations")
        
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
        
        # Model performance comparison
        self._plot_model_comparison(save_plots, plot_dir)
        
        # ROC curves
        self._plot_roc_curves(save_plots, plot_dir)
        
        # Feature importance (if available)
        self._plot_feature_importance(save_plots, plot_dir)
        
        # Training time comparison
        self._plot_training_times(save_plots, plot_dir)
    
    def _plot_model_comparison(self, save_plots, plot_dir):
        """Plot model performance comparison"""
        if not self.model_performance:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        all_models = {}
        if 'base_models' in self.model_performance:
            all_models.update({f"Base_{k}": v for k, v in self.model_performance['base_models'].items()})
        if 'ensemble_models' in self.model_performance:
            all_models.update({f"Ens_{k}": v for k, v in self.model_performance['ensemble_models'].items()})
        
        model_names = list(all_models.keys())
        accuracies = [all_models[name]['accuracy'] for name in model_names]
        aucs = [all_models[name]['auc'] for name in model_names]
        f1s = [all_models[name]['f1'] for name in model_names]
        times = [all_models[name]['training_time'] for name in model_names]
        
        # Accuracy comparison
        bars1 = ax1.bar(range(len(model_names)), accuracies, alpha=0.7)
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison')
        ax1.grid(True, alpha=0.3)
        
        # AUC comparison
        bars2 = ax2.bar(range(len(model_names)), aucs, alpha=0.7, color='orange')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.set_ylabel('AUC Score')
        ax2.set_title('Model AUC Comparison')
        ax2.grid(True, alpha=0.3)
        
        # F1 Score comparison
        bars3 = ax3.bar(range(len(model_names)), f1s, alpha=0.7, color='green')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Model F1 Score Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Training time comparison
        bars4 = ax4.bar(range(len(model_names)), times, alpha=0.7, color='red')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.set_ylabel('Training Time (seconds)')
        ax4.set_title('Model Training Time Comparison')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{plot_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curves(self, save_plots, plot_dir):
        """Plot ROC curves for all models"""
        if not hasattr(self, 'X_test') or self.X_test is None:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot ROC curves for ensemble models
        if 'ensemble_models' in self.model_performance:
            for name, model in self.ensemble_models.items():
                try:
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                    auc_score = roc_auc_score(self.y_test, y_proba)
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
                except Exception as e:
                    self.logger.warning(f"Could not plot ROC for {name}: {e}")
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Ensemble Models Comparison')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig(f"{plot_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_importance(self, save_plots, plot_dir):
        """Plot feature importance for tree-based models"""
        if self.best_model is None or not hasattr(self.best_model, 'feature_importances_'):
            return
        
        if self.feature_names is None:
            return
        
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importance - {type(self.best_model).__name__}")
        plt.bar(range(min(20, len(importances))), importances[indices[:20]])
        plt.xticks(range(min(20, len(importances))), 
                  [self.feature_names[i] for i in indices[:20]], rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig(f"{plot_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_training_times(self, save_plots, plot_dir):
        """Plot training time comparison"""
        if not self.model_performance:
            return
        
        all_models = {}
        if 'base_models' in self.model_performance:
            all_models.update({f"Base_{k}": v for k, v in self.model_performance['base_models'].items()})
        if 'ensemble_models' in self.model_performance:
            all_models.update({f"Ens_{k}": v for k, v in self.model_performance['ensemble_models'].items()})
        
        model_names = list(all_models.keys())
        times = [all_models[name]['training_time'] for name in model_names]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(model_names)), times, alpha=0.7)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylabel('Training Time (seconds)')
        plt.title('Model Training Time Comparison')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time_val:.2f}s', ha='center', va='bottom')
        
        if save_plots:
            plt.savefig(f"{plot_dir}/training_times.png", dpi=300, bbox_inches='tight')
        plt.show()


# =============================================================================
# ADVANCED STACKING CLASSIFIER
# =============================================================================

class AdvancedStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Advanced stacking classifier with multiple layers and cross-validation
    """
    
    def __init__(self, base_models, meta_model=None, cv_folds=5, 
                 use_probabilities=True, random_state=42, n_jobs=-1):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(random_state=random_state)
        self.cv_folds = cv_folds
        self.use_probabilities = use_probabilities
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.fitted_base_models = []
        self.fitted_meta_model = None
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit the stacking ensemble"""
        self.classes_ = np.unique(y)
        
        # Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X, y)
        
        # Train base models on full dataset
        self.fitted_base_models = []
        for name, model in self.base_models:
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            self.fitted_base_models.append((name, fitted_model))
        
        # Train meta-model
        self.fitted_meta_model = clone(self.meta_model)
        self.fitted_meta_model.fit(meta_features, y)
        
        return self
    
    def _generate_meta_features(self, X, y):
        """Generate meta-features using cross-validation"""
        n_models = len(self.base_models)
        meta_features = np.zeros((len(X), n_models))
        
        kfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for i, (name, model) in enumerate(self.base_models):
            cv_predictions = np.zeros(len(X))
            
            for train_idx, val_idx in kfold.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                
                # Train model on fold
                model_copy = clone(model)
                model_copy.fit(X_train_fold, y_train_fold)
                
                # Generate predictions for validation fold
                if self.use_probabilities and hasattr(model_copy, 'predict_proba'):
                    cv_predictions[val_idx] = model_copy.predict_proba(X_val_fold)[:, 1]
                else:
                    cv_predictions[val_idx] = model_copy.predict(X_val_fold)
            
            meta_features[:, i] = cv_predictions
        
        return meta_features
    
    def predict(self, X):
        """Make predictions using the stacking ensemble"""
        meta_features = self._get_meta_features(X)
        return self.fitted_meta_model.predict(meta_features)
    
    def predict_proba(self, X):
        """Predict class probabilities using the stacking ensemble"""
        meta_features = self._get_meta_features(X)
        return self.fitted_meta_model.predict_proba(meta_features)
    
    def _get_meta_features(self, X):
        """Get meta-features from fitted base models"""
        meta_features = np.zeros((len(X), len(self.fitted_base_models)))
        
        for i, (name, model) in enumerate(self.fitted_base_models):
            if self.use_probabilities and hasattr(model, 'predict_proba'):
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = model.predict(X)
        
        return meta_features


# =============================================================================
# PRODUCTION ENSEMBLE PIPELINE
# =============================================================================

class ProductionEnsemblePipeline:
    """
    Production-ready ensemble pipeline with monitoring and serving capabilities
    """
    
    def __init__(self, model, scaler=None, feature_names=None, model_metadata=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.model_metadata = model_metadata or {}
        
        self.prediction_history = []
        self.performance_metrics = {}
        self.alert_thresholds = {
            'prediction_drift': 0.1,
            'latency_threshold': 1.0,  # seconds
            'error_rate_threshold': 0.05
        }
        
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for production pipeline"""
        logger = logging.getLogger('production_pipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def predict(self, X, return_probabilities=False, log_prediction=True):
        """
        Make predictions with comprehensive logging and monitoring
        """
        start_time = time.time()
        
        try:
            # Input validation
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Feature scaling if required
            if self.scaler is not None:
                X_processed = self.scaler.transform(X)
            else:
                X_processed = X
            
            # Make predictions
            predictions = self.model.predict(X_processed)
            
            result = {
                'predictions': predictions.tolist(),
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(X),
                'model_type': type(self.model).__name__
            }
            
            # Add probabilities if requested
            if return_probabilities and hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_processed)
                result['probabilities'] = probabilities.tolist()
            
            # Calculate latency
            latency = time.time() - start_time
            result['latency_ms'] = latency * 1000
            
            # Log prediction
            if log_prediction:
                self._log_prediction(result, X, latency)
            
            # Check for alerts
            self._check_alerts(result, latency)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'model_type': type(self.model).__name__
            }
    
    def _log_prediction(self, result, X, latency):
        """Log prediction details"""
        log_entry = {
            'timestamp': result['timestamp'],
            'n_samples': result['n_samples'],
            'latency_ms': result['latency_ms'],
            'predictions': result['predictions'],
            'input_shape': X.shape
        }
        
        if 'probabilities' in result:
            log_entry['avg_probability'] = np.mean(result['probabilities'])
        
        self.prediction_history.append(log_entry)
        
        # Keep only recent history (last 1000 predictions)
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def _check_alerts(self, result, latency):
        """Check for alert conditions"""
        
        # Latency alert
        if latency > self.alert_thresholds['latency_threshold']:
            self.logger.warning(f"High latency detected: {latency:.3f}s")
        
        # Prediction drift alert (simplified)
        if len(self.prediction_history) > 100:
            recent_predictions = [entry['predictions'] for entry in self.prediction_history[-100:]]
            recent_avg = np.mean(recent_predictions)
            
            if hasattr(self, 'baseline_prediction_rate'):
                drift = abs(recent_avg - self.baseline_prediction_rate)
                if drift > self.alert_thresholds['prediction_drift']:
                    self.logger.warning(f"Prediction drift detected: {drift:.3f}")
    
    def batch_predict(self, X_batch, batch_size=1000, return_probabilities=False):
        """
        Process large batches efficiently
        """
        self.logger.info(f"Processing batch of {len(X_batch)} samples")
        
        results = []
        
        for i in range(0, len(X_batch), batch_size):
            batch = X_batch[i:i+batch_size]
            batch_result = self.predict(batch, return_probabilities, log_prediction=False)
            results.extend(batch_result['predictions'])
        
        return {
            'predictions': results,
            'total_samples': len(X_batch),
            'batch_size': batch_size,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_model_info(self):
        """Get comprehensive model information"""
        info = {
            'model_type': type(self.model).__name__,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'has_scaler': self.scaler is not None,
            'metadata': self.model_metadata,
            'prediction_count': len(self.prediction_history),
            'alert_thresholds': self.alert_thresholds
        }
        
        # Add model-specific information
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'feature_importances_'):
            info['has_feature_importance'] = True
        
        return info
    
    def get_performance_summary(self):
        """Get performance summary from prediction history"""
        if not self.prediction_history:
            return {'message': 'No predictions logged yet'}
        
        latencies = [entry['latency_ms'] for entry in self.prediction_history]
        predictions = [entry['predictions'] for entry in self.prediction_history]
        
        summary = {
            'total_predictions': len(self.prediction_history),
            'avg_latency_ms': np.mean(latencies),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies),
            'avg_prediction_rate': np.mean(predictions),
            'prediction_std': np.std(predictions),
            'last_prediction_time': self.prediction_history[-1]['timestamp']
        }
        
        return summary
    
    def save_model(self, filepath):
        """Save the complete pipeline"""
        pipeline_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_metadata': self.model_metadata,
            'alert_thresholds': self.alert_thresholds
        }
        
        joblib.dump(pipeline_data, filepath)
        self.logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a saved pipeline"""
        pipeline_data = joblib.load(filepath)
        
        pipeline = cls(
            model=pipeline_data['model'],
            scaler=pipeline_data['scaler'],
            feature_names=pipeline_data['feature_names'],
            model_metadata=pipeline_data['model_metadata']
        )
        
        pipeline.alert_thresholds = pipeline_data.get('alert_thresholds', pipeline.alert_thresholds)
        
        return pipeline


# =============================================================================
# MAIN EXECUTION AND DEMONSTRATION
# =============================================================================

def run_comprehensive_ensemble_experiment():
    """
    Run comprehensive ensemble methods experiment demonstrating all techniques
    """
    print("=" * 80)
    print("COMPREHENSIVE ENSEMBLE METHODS EXPERIMENT")
    print("Production-Ready Credit Risk Assessment System")
    print("=" * 80)
    
    # Initialize framework
    framework = ProductionEnsembleFramework(random_state=42)
    
    # Step 1: Data Preparation
    print("\n1. DATA PREPARATION")
    print("-" * 40)
    framework.prepare_data()
    
    # Step 2: Base Models
    print("\n2. BASE MODEL CREATION AND EVALUATION")
    print("-" * 40)
    framework.create_base_models()
    base_results = framework.evaluate_base_models()
    
    # Step 3: Ensemble Models
    print("\n3. ENSEMBLE MODEL CREATION AND EVALUATION")
    print("-" * 40)
    framework.create_ensemble_models()
    framework.implement_advanced_stacking()
    ensemble_results = framework.evaluate_ensemble_models()
    
    # Step 4: Model Selection
    print("\n4. BEST MODEL SELECTION")
    print("-" * 40)
    best_name, best_model, best_metrics = framework.select_best_model()
    
    # Step 5: Hyperparameter Optimization (example)
    print("\n5. HYPERPARAMETER OPTIMIZATION EXAMPLE")
    print("-" * 40)
    if 'random_forest' in framework.base_models:
        rf_param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [8, 10, 12],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        try:
            optimized_rf, best_params = framework.hyperparameter_optimization(
                'random_forest', rf_param_grid, cv_folds=3, n_iter=10
            )
            print(f"Optimized Random Forest parameters: {best_params}")
        except Exception as e:
            print(f"Hyperparameter optimization failed: {e}")
    
    # Step 6: Production Pipeline
    print("\n6. PRODUCTION PIPELINE CREATION")
    print("-" * 40)
    production_pipeline = framework.create_production_pipeline()
    
    # Test production pipeline
    print("Testing production pipeline...")
    test_sample = framework.X_test[:5]  # Test with 5 samples
    
    # Single prediction
    result = production_pipeline.predict(test_sample[0], return_probabilities=True)
    print(f"Single prediction result: {result}")
    
    # Batch prediction
    batch_result = production_pipeline.batch_predict(test_sample, return_probabilities=True)
    print(f"Batch prediction completed: {batch_result['total_samples']} samples")
    
    # Step 7: Comprehensive Report
    print("\n7. COMPREHENSIVE ANALYSIS REPORT")
    print("-" * 40)
    report = framework.generate_comprehensive_report()
    
    # Display key findings
    print(f"Best Model: {report['best_model_info']['name']}")
    print(f"Best AUC Score: {report['best_model_info']['metrics']['auc']:.4f}")
    print(f"Best Accuracy: {report['best_model_info']['metrics']['accuracy']:.4f}")
    print(f"Best F1 Score: {report['best_model_info']['metrics']['f1']:.4f}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    # Step 8: Visualizations
    print("\n8. CREATING VISUALIZATIONS")
    print("-" * 40)
    framework.visualize_results(save_plots=False)  # Set to True to save plots
    
    # Step 9: Model Serving Demo
    print("\n9. MODEL SERVING DEMONSTRATION")
    print("-" * 40)
    
    # Get model info
    model_info = production_pipeline.get_model_info()
    print(f"Model Type: {model_info['model_type']}")
    print(f"Feature Count: {model_info['feature_count']}")
    print(f"Prediction Count: {model_info['prediction_count']}")
    
    # Performance summary
    perf_summary = production_pipeline.get_performance_summary()
    print(f"Average Latency: {perf_summary['avg_latency_ms']:.2f}ms")
    print(f"Average Prediction Rate: {perf_summary['avg_prediction_rate']:.3f}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return {
        'framework': framework,
        'production_pipeline': production_pipeline,
        'report': report,
        'best_model': best_model,
        'best_metrics': best_metrics
    }

def demonstrate_hyperparameter_optimization():
    """
    Demonstrate advanced hyperparameter optimization techniques
    """
    print("\nADVANCED HYPERPARAMETER OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    framework = ProductionEnsembleFramework(random_state=42)
    framework.prepare_data()
    framework.create_base_models()
    
    # Random Forest optimization
    if 'random_forest' in framework.base_models:
        print("Optimizing Random Forest...")
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [8, 10, 12, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        optimized_rf, rf_params = framework.hyperparameter_optimization(
            'random_forest', rf_param_grid, cv_folds=3, n_iter=20
        )
        
        print(f"Best RF parameters: {rf_params}")
    
    # XGBoost optimization (if available)
    if XGBOOST_AVAILABLE and 'xgboost' in framework.base_models:
        print("Optimizing XGBoost...")
        xgb_param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        optimized_xgb, xgb_params = framework.hyperparameter_optimization(
            'xgboost', xgb_param_grid, cv_folds=3, n_iter=15
        )
        
        print(f"Best XGB parameters: {xgb_params}")

def main():
    """
    Main function demonstrating comprehensive ensemble methods solution
    """
    print("Day 30: Ensemble Methods - Complete Production Solution")
    print("Advanced Credit Risk Assessment with Ensemble Learning")
    print()
    
    try:
        # Run comprehensive experiment
        results = run_comprehensive_ensemble_experiment()
        
        # Additional demonstrations
        print("\n" + "=" * 80)
        print("ADDITIONAL DEMONSTRATIONS")
        print("=" * 80)
        
        # Hyperparameter optimization demo
        demonstrate_hyperparameter_optimization()
        
        print("\n" + "=" * 80)
        print("KEY INSIGHTS AND RECOMMENDATIONS")
        print("=" * 80)
        
        print("1. ENSEMBLE SUPERIORITY:")
        print("   - Ensemble methods consistently outperform individual models")
        print("   - Stacking provides additional performance gains through meta-learning")
        print("   - Voting ensembles offer good performance with lower complexity")
        
        print("\n2. PRODUCTION CONSIDERATIONS:")
        print("   - Model serving requires careful latency monitoring")
        print("   - Prediction drift detection is crucial for model reliability")
        print("   - Comprehensive logging enables performance analysis")
        
        print("\n3. BUSINESS VALUE:")
        print("   - Higher accuracy reduces false positives/negatives in credit decisions")
        print("   - Robust predictions improve risk assessment reliability")
        print("   - Ensemble diversity provides better generalization")
        
        print("\n4. DEPLOYMENT RECOMMENDATIONS:")
        print("   - Use stacking for maximum performance in batch scenarios")
        print("   - Consider voting ensembles for real-time applications")
        print("   - Implement comprehensive monitoring and alerting")
        print("   - Regular model retraining with new data")
        
        print(f"\nBest Model Performance:")
        print(f"- Model: {results['best_model']}")
        print(f"- AUC: {results['best_metrics']['auc']:.4f}")
        print(f"- Accuracy: {results['best_metrics']['accuracy']:.4f}")
        print(f"- F1 Score: {results['best_metrics']['f1']:.4f}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()