#!/usr/bin/env python3
"""
Day 30: Ensemble Methods - Model Training Service

Comprehensive model training service for ensemble methods with MLflow tracking,
hyperparameter optimization, and production deployment.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            AdaBoostClassifier, VotingClassifier, BaggingClassifier,
                            ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                           precision_score, recall_score, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# Advanced ensemble libraries
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

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Database
import psycopg2
from psycopg2.extras import RealDictCursor
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

class AdvancedStackingClassifier:
    """
    Advanced stacking classifier with cross-validation and multiple layers
    """
    
    def __init__(self, base_models, meta_model=None, cv_folds=5, 
                 use_probabilities=True, random_state=42):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(random_state=random_state)
        self.cv_folds = cv_folds
        self.use_probabilities = use_probabilities
        self.random_state = random_state
        
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

class EnsembleTrainer:
    """
    Comprehensive ensemble training system with MLflow integration
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.mlflow_client = None
        self.experiment_id = None
        
        # Initialize MLflow
        self._setup_mlflow()
        
        # Model storage
        self.base_models = {}
        self.ensemble_models = {}
        self.best_model = None
        self.scaler = None
        
        # Results storage
        self.training_results = {}
        
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking"""
        try:
            mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
            mlflow.set_tracking_uri(mlflow_uri)
            
            experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'ensemble_methods_experiment')
            
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    self.experiment_id = experiment.experiment_id
                else:
                    self.experiment_id = mlflow.create_experiment(experiment_name)
            except Exception as e:
                logger.warning(f"MLflow experiment setup failed: {e}")
                self.experiment_id = None
            
            self.mlflow_client = MlflowClient()
            logger.info(f"MLflow initialized with experiment: {experiment_name}")
            
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            self.mlflow_client = None
    
    def load_data(self, data_path: str = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load training data from various sources"""
        logger.info("Loading training data")
        
        if data_path and os.path.exists(data_path):
            # Load from CSV file
            df = pd.read_csv(data_path)
            
            # Assume last column is target
            feature_columns = df.columns[:-1].tolist()
            X = df[feature_columns].values
            y = df.iloc[:, -1].values
            
            logger.info(f"Loaded data from {data_path}: {X.shape[0]} samples, {X.shape[1]} features")
            
        else:
            # Generate synthetic data
            from sklearn.datasets import make_classification
            
            X, y = make_classification(
                n_samples=10000,
                n_features=25,
                n_informative=18,
                n_redundant=4,
                n_clusters_per_class=3,
                weights=[0.82, 0.18],
                flip_y=0.01,
                class_sep=0.8,
                random_state=self.random_state
            )
            
            feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
            logger.info(f"Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_columns
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and split data for training"""
        logger.info("Preparing data for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y, shuffle=True
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def create_base_models(self) -> Dict[str, Any]:
        """Create comprehensive set of base models"""
        logger.info("Creating base models")
        
        self.base_models = {
            # Tree-based models
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
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
                subsample=0.8,
                random_state=self.random_state
            ),
            
            'adaboost': AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=3, class_weight='balanced'),
                n_estimators=50,
                learning_rate=1.0,
                random_state=self.random_state
            ),
            
            # Linear models
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                solver='liblinear'
            ),
            
            # Probabilistic models
            'naive_bayes': GaussianNB(),
            
            # Support Vector Machine
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced',
                gamma='scale'
            )
        }
        
        # Add advanced models if available
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
                n_jobs=-1,
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
                n_jobs=-1,
                verbose=-1
            )
        
        if CATBOOST_AVAILABLE:
            self.base_models['catboost'] = cb.CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                l2_leaf_reg=3,
                random_seed=self.random_state,
                verbose=False
            )
        
        logger.info(f"Created {len(self.base_models)} base models")
        return self.base_models
    
    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         X_train_scaled: np.ndarray, X_test_scaled: np.ndarray) -> Dict[str, Dict]:
        """Train and evaluate all base models"""
        logger.info("Training base models")
        
        base_results = {}
        
        for name, model in self.base_models.items():
            logger.info(f"Training {name}")
            
            # Start MLflow run
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"base_{name}"):
                start_time = time.time()
                
                # Determine feature set (scaled vs unscaled)
                if name in ['logistic_regression', 'svm']:
                    X_train_use = X_train_scaled
                    X_test_use = X_test_scaled
                else:
                    X_train_use = X_train
                    X_test_use = X_test
                
                # Train model
                model.fit(X_train_use, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_use)
                y_proba = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                
                metrics = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'training_time': time.time() - start_time
                }
                
                if y_proba is not None:
                    auc_score = roc_auc_score(y_test, y_proba)
                    metrics['auc'] = auc_score
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_use, y_train, 
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                    scoring='accuracy', n_jobs=-1
                )
                metrics['cv_accuracy_mean'] = np.mean(cv_scores)
                metrics['cv_accuracy_std'] = np.std(cv_scores)
                
                # Log to MLflow
                if self.mlflow_client:
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(model, f"model_{name}")
                
                base_results[name] = metrics
                
                logger.info(f"  {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        self.training_results['base_models'] = base_results
        return base_results
    
    def create_ensemble_models(self) -> Dict[str, Any]:
        """Create various ensemble models"""
        logger.info("Creating ensemble models")
        
        # Select best base models for ensembles
        base_results = self.training_results.get('base_models', {})
        
        if base_results:
            # Sort by AUC or accuracy
            metric_key = 'auc' if 'auc' in list(base_results.values())[0] else 'accuracy'
            sorted_models = sorted(base_results.items(), key=lambda x: x[1][metric_key], reverse=True)
            top_model_names = [name for name, _ in sorted_models[:6]]
        else:
            # Fallback selection
            top_model_names = ['random_forest', 'gradient_boosting', 'extra_trees', 'naive_bayes']
            if XGBOOST_AVAILABLE:
                top_model_names.append('xgboost')
            if LIGHTGBM_AVAILABLE:
                top_model_names.append('lightgbm')
        
        # Create ensemble base model list
        ensemble_base_models = []
        for name in top_model_names:
            if name in self.base_models:
                ensemble_base_models.append((name, self.base_models[name]))
        
        self.ensemble_models = {
            # Voting Ensembles
            'voting_hard': VotingClassifier(
                estimators=ensemble_base_models,
                voting='hard',
                n_jobs=-1
            ),
            
            'voting_soft': VotingClassifier(
                estimators=ensemble_base_models,
                voting='soft',
                n_jobs=-1
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
                n_jobs=-1
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
                n_jobs=-1
            )
        }
        
        # Add stacking ensemble
        stacking_base_models = ensemble_base_models[:4]  # Use top 4 models
        self.ensemble_models['stacking'] = AdvancedStackingClassifier(
            base_models=stacking_base_models,
            meta_model=LogisticRegression(random_state=self.random_state, class_weight='balanced'),
            cv_folds=5,
            random_state=self.random_state
        )
        
        logger.info(f"Created {len(self.ensemble_models)} ensemble models")
        return self.ensemble_models
    
    def train_ensemble_models(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """Train and evaluate ensemble models"""
        logger.info("Training ensemble models")
        
        ensemble_results = {}
        
        for name, model in self.ensemble_models.items():
            logger.info(f"Training {name}")
            
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"ensemble_{name}"):
                start_time = time.time()
                
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    
                    metrics = {
                        'accuracy': accuracy,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall,
                        'training_time': time.time() - start_time
                    }
                    
                    if y_proba is not None:
                        auc_score = roc_auc_score(y_test, y_proba)
                        metrics['auc'] = auc_score
                    
                    # Cross-validation (skip for stacking to avoid double CV)
                    if name != 'stacking':
                        cv_scores = cross_val_score(
                            model, X_train, y_train,
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                            scoring='accuracy', n_jobs=-1
                        )
                        metrics['cv_accuracy_mean'] = np.mean(cv_scores)
                        metrics['cv_accuracy_std'] = np.std(cv_scores)
                    
                    # Log to MLflow
                    if self.mlflow_client:
                        if hasattr(model, 'get_params'):
                            mlflow.log_params(model.get_params())
                        mlflow.log_metrics(metrics)
                        
                        # Save model
                        if name == 'stacking':
                            # Custom save for stacking
                            mlflow.log_artifact(self._save_stacking_model(model, name))
                        else:
                            mlflow.sklearn.log_model(model, f"model_{name}")
                    
                    ensemble_results[name] = metrics
                    
                    logger.info(f"  {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    continue
        
        self.training_results['ensemble_models'] = ensemble_results
        return ensemble_results
    
    def _save_stacking_model(self, model, name: str) -> str:
        """Save stacking model to file"""
        model_dir = Path('./models')
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{name}_model.pkl"
        
        # Save complete model with scaler
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'model_type': 'stacking',
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        return str(model_path)
    
    def select_best_model(self) -> Tuple[str, Any, Dict]:
        """Select the best performing model"""
        logger.info("Selecting best model")
        
        all_results = {}
        
        # Combine all results
        if 'base_models' in self.training_results:
            for name, metrics in self.training_results['base_models'].items():
                all_results[f"base_{name}"] = metrics
        
        if 'ensemble_models' in self.training_results:
            for name, metrics in self.training_results['ensemble_models'].items():
                all_results[f"ensemble_{name}"] = metrics
        
        if not all_results:
            raise ValueError("No training results available")
        
        # Multi-criteria selection
        weights = {'auc': 0.4, 'f1': 0.3, 'accuracy': 0.2, 'precision': 0.1}
        
        best_score = -1
        best_model_name = None
        best_metrics = None
        
        for model_name, metrics in all_results.items():
            # Calculate weighted score
            score = 0
            for metric, weight in weights.items():
                if metric in metrics:
                    score += weight * metrics[metric]
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_metrics = metrics
        
        # Get actual model object
        if best_model_name.startswith('base_'):
            actual_name = best_model_name[5:]
            best_model = self.base_models[actual_name]
        else:
            actual_name = best_model_name[9:]
            best_model = self.ensemble_models[actual_name]
        
        self.best_model = best_model
        
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Best model metrics: {best_metrics}")
        
        return best_model_name, best_model, best_metrics
    
    def save_models(self, model_dir: str = './models'):
        """Save all trained models"""
        logger.info(f"Saving models to {model_dir}")
        
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        # Save base models
        for name, model in self.base_models.items():
            model_file = model_path / f"base_{name}.pkl"
            
            model_data = {
                'model': model,
                'scaler': self.scaler if name in ['logistic_regression', 'svm'] else None,
                'model_type': 'base',
                'algorithm': name,
                'training_date': datetime.now().isoformat(),
                'performance_metrics': self.training_results.get('base_models', {}).get(name, {})
            }
            
            joblib.dump(model_data, model_file)
            logger.info(f"Saved {name} to {model_file}")
        
        # Save ensemble models
        for name, model in self.ensemble_models.items():
            model_file = model_path / f"ensemble_{name}.pkl"
            
            model_data = {
                'model': model,
                'scaler': self.scaler,
                'model_type': 'ensemble',
                'algorithm': name,
                'training_date': datetime.now().isoformat(),
                'performance_metrics': self.training_results.get('ensemble_models', {}).get(name, {})
            }
            
            joblib.dump(model_data, model_file)
            logger.info(f"Saved {name} to {model_file}")
        
        # Save best model separately
        if self.best_model:
            best_model_file = model_path / "best_model.pkl"
            
            best_model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'model_type': 'best',
                'training_date': datetime.now().isoformat(),
                'all_results': self.training_results
            }
            
            joblib.dump(best_model_data, best_model_file)
            logger.info(f"Saved best model to {best_model_file}")
    
    def store_results_in_database(self):
        """Store training results in database"""
        logger.info("Storing results in database")
        
        try:
            conn = psycopg2.connect(os.getenv('DATABASE_URL'))
            cur = conn.cursor()
            
            # Store base models
            if 'base_models' in self.training_results:
                for name, metrics in self.training_results['base_models'].items():
                    cur.execute("""
                        INSERT INTO models (name, model_type, algorithm, performance_metrics, model_path)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (name) DO UPDATE SET
                        performance_metrics = EXCLUDED.performance_metrics,
                        updated_at = CURRENT_TIMESTAMP
                    """, (
                        f"base_{name}",
                        'base',
                        name,
                        json.dumps(metrics),
                        f"./models/base_{name}.pkl"
                    ))
            
            # Store ensemble models
            if 'ensemble_models' in self.training_results:
                for name, metrics in self.training_results['ensemble_models'].items():
                    cur.execute("""
                        INSERT INTO models (name, model_type, algorithm, performance_metrics, model_path)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (name) DO UPDATE SET
                        performance_metrics = EXCLUDED.performance_metrics,
                        updated_at = CURRENT_TIMESTAMP
                    """, (
                        f"ensemble_{name}",
                        'ensemble',
                        name,
                        json.dumps(metrics),
                        f"./models/ensemble_{name}.pkl"
                    ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info("Successfully stored results in database")
            
        except Exception as e:
            logger.error(f"Error storing results in database: {e}")

def main():
    """
    Main training pipeline
    """
    logger.info("Starting ensemble model training pipeline")
    
    # Initialize trainer
    trainer = EnsembleTrainer(random_state=42)
    
    # Load and prepare data
    data_path = os.getenv('DATA_PATH', './data/credit_risk_combined.csv')
    X, y, feature_names = trainer.load_data(data_path)
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = trainer.prepare_data(X, y)
    
    # Create and train base models
    trainer.create_base_models()
    base_results = trainer.train_base_models(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled)
    
    # Create and train ensemble models
    trainer.create_ensemble_models()
    ensemble_results = trainer.train_ensemble_models(X_train, y_train, X_test, y_test)
    
    # Select best model
    best_name, best_model, best_metrics = trainer.select_best_model()
    
    # Save models
    trainer.save_models()
    
    # Store results in database
    trainer.store_results_in_database()
    
    # Print summary
    print("\n" + "="*80)
    print("ENSEMBLE MODEL TRAINING COMPLETED")
    print("="*80)
    
    print(f"Best Model: {best_name}")
    print(f"Best Metrics: {best_metrics}")
    
    print("\nBase Model Results:")
    for name, metrics in base_results.items():
        accuracy = metrics.get('accuracy', 0)
        auc = metrics.get('auc', 0)
        print(f"  {name:20}: Accuracy={accuracy:.4f}, AUC={auc:.4f}")
    
    print("\nEnsemble Model Results:")
    for name, metrics in ensemble_results.items():
        accuracy = metrics.get('accuracy', 0)
        auc = metrics.get('auc', 0)
        print(f"  {name:20}: Accuracy={accuracy:.4f}, AUC={auc:.4f}")
    
    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main()