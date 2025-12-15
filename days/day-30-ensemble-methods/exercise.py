"""
Day 30: Ensemble Methods - Bagging, Boosting, Stacking - Exercise

Business Scenario:
You're the Senior ML Engineer at TechCorp, a fintech company that provides credit scoring services 
to banks and lending institutions. The company needs to build a highly accurate and robust credit 
risk assessment model that can reliably predict loan defaults.

The current single-model approach has accuracy issues and is sensitive to data changes. Your task 
is to build a production-ready ensemble system that combines multiple algorithms to achieve:
- Higher accuracy than individual models
- Better robustness to data drift
- Reliable probability estimates for risk assessment
- Explainable predictions for regulatory compliance

Requirements:
1. Implement multiple ensemble methods (bagging, boosting, stacking)
2. Compare ensemble performance against individual models
3. Build a production-ready ensemble system with monitoring
4. Create comprehensive evaluation and validation framework
5. Implement model serving infrastructure with proper logging

Success Criteria:
- Ensemble accuracy > 85% on test set
- AUC score > 0.90 for risk assessment
- Prediction latency < 100ms for real-time scoring
- Comprehensive monitoring and alerting system
- Production-ready deployment pipeline
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Advanced libraries (handle gracefully if not available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available - using sklearn alternatives")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available - using sklearn alternatives")

def generate_credit_risk_dataset(n_samples=10000, n_features=20, random_state=42):
    """
    Generate synthetic credit risk dataset for ensemble learning
    
    Returns:
        X: Feature matrix
        y: Target labels (0: No default, 1: Default)
        feature_names: List of feature names
    """
    # Generate base dataset with realistic parameters for credit risk
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        weights=[0.85, 0.15],  # Imbalanced: 15% default rate
        flip_y=0.02,  # 2% label noise
        random_state=random_state
    )
    
    # Transform features to realistic credit risk ranges
    X_transformed = X.copy()
    
    # Income features (log-normal distribution)
    X_transformed[:, 0] = np.exp(X[:, 0] * 0.5 + 11) * 1000  # annual_income: 30k-200k+
    X_transformed[:, 1] = np.clip(np.abs(X[:, 1]) * 0.3 + 0.2, 0, 1.5)  # debt_to_income_ratio
    
    # Credit score (300-850 range)
    X_transformed[:, 2] = np.clip(X[:, 2] * 100 + 650, 300, 850)
    
    # Employment length (0-30 years)
    X_transformed[:, 3] = np.clip(np.abs(X[:, 3]) * 8 + 1, 0, 30)
    
    # Loan amount
    X_transformed[:, 4] = np.exp(X[:, 4] * 0.8 + 10) * 1000  # 10k-500k+
    
    # Other features with realistic ranges
    for i in range(5, n_features):
        if i in [6, 7, 8, 9, 10, 13, 14]:  # Count features
            X_transformed[:, i] = np.clip(np.abs(X[:, i]) * 5 + 1, 0, 20).astype(int)
        elif i in [11, 12]:  # Balance features
            X_transformed[:, i] = np.exp(X[:, i] * 0.6 + 8) * 1000
        elif i == 15:  # Age
            X_transformed[:, i] = np.clip(np.abs(X[:, i]) * 20 + 25, 18, 80)
        else:  # Other features
            X_transformed[:, i] = np.abs(X[:, i]) * 100
    
    # Create realistic feature names
    feature_names = [
        'annual_income', 'debt_to_income_ratio', 'credit_score', 'employment_length',
        'loan_amount', 'loan_to_income_ratio', 'number_of_accounts', 'delinquencies_2yrs',
        'inquiries_6months', 'open_accounts', 'total_accounts', 'revolving_balance',
        'revolving_utilization', 'mortgage_accounts', 'public_records', 'age',
        'months_since_last_delinquency', 'months_since_last_inquiry', 'payment_history',
        'account_age_avg'
    ]
    
    return X_transformed, y, feature_names

class EnsembleExperiment:
    """
    Comprehensive ensemble methods experiment for credit risk assessment
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
        self.ensemble_models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the credit risk dataset"""
        print("Loading and preparing credit risk dataset...")
        
        # Generate realistic credit risk dataset
        X, y, self.feature_names = generate_credit_risk_dataset(
            n_samples=10000, 
            n_features=20, 
            random_state=self.random_state
        )
        
        # Split the data with stratification for imbalanced classes
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=self.random_state, 
            stratify=y,
            shuffle=True
        )
        
        # Scale features for algorithms that need it
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Default rate (train): {np.mean(self.y_train):.2%}")
        print(f"Default rate (test): {np.mean(self.y_test):.2%}")
        
    def create_base_models(self):
        """Create individual base models for ensemble"""
        print("\nCreating base models...")
        
        # Create diverse base models with different algorithms and approaches
        self.models = {
            # Linear model
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                solver='liblinear'  # Better for small datasets
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
            
            'decision_tree': DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=self.random_state,
                class_weight='balanced'
            ),
            
            # Support Vector Machine
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced',
                gamma='scale'
            ),
            
            # Probabilistic model
            'naive_bayes': GaussianNB()
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=3,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=-1,
                verbosity=0
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        print(f"Created {len(self.models)} base models")
        for model_name in self.models.keys():
            print(f"  - {model_name}")
        
    def evaluate_base_models(self):
        """Evaluate individual base models"""
        print("\nEvaluating base models...")
        
        base_results = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            try:
                # Determine if model needs scaled features
                if name in ['logistic_regression', 'svm']:
                    X_train_use = self.X_train_scaled
                    X_test_use = self.X_test_scaled
                else:
                    X_train_use = self.X_train
                    X_test_use = self.X_test
                
                # Train model
                model.fit(X_train_use, self.y_train)
                
                # Predictions
                y_pred = model.predict(X_test_use)
                
                # Handle probability predictions
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_use)[:, 1]
                else:
                    # For models without predict_proba, use decision function or predictions
                    if hasattr(model, 'decision_function'):
                        y_proba = model.decision_function(X_test_use)
                        # Normalize to [0,1] range
                        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
                    else:
                        y_proba = y_pred.astype(float)
                
                # Calculate comprehensive metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                auc_score = roc_auc_score(self.y_test, y_proba)
                f1 = f1_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, zero_division=0)
                recall = recall_score(self.y_test, y_pred, zero_division=0)
                
                # Cross-validation with stratification
                cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                
                cv_accuracy = cross_val_score(model, X_train_use, self.y_train, 
                                            cv=cv_folds, scoring='accuracy', n_jobs=-1)
                cv_auc = cross_val_score(model, X_train_use, self.y_train, 
                                       cv=cv_folds, scoring='roc_auc', n_jobs=-1)
                cv_f1 = cross_val_score(model, X_train_use, self.y_train, 
                                      cv=cv_folds, scoring='f1', n_jobs=-1)
                
                base_results[name] = {
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'cv_accuracy_mean': np.mean(cv_accuracy),
                    'cv_accuracy_std': np.std(cv_accuracy),
                    'cv_auc_mean': np.mean(cv_auc),
                    'cv_auc_std': np.std(cv_auc),
                    'cv_f1_mean': np.mean(cv_f1),
                    'cv_f1_std': np.std(cv_f1)
                }
                
                print(f"  Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"  Error evaluating {name}: {e}")
                # Store error results
                base_results[name] = {
                    'accuracy': 0.0, 'auc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'cv_accuracy_mean': 0.0, 'cv_accuracy_std': 0.0,
                    'cv_auc_mean': 0.0, 'cv_auc_std': 0.0,
                    'cv_f1_mean': 0.0, 'cv_f1_std': 0.0,
                    'error': str(e)
                }
        
        self.results['base_models'] = base_results
        return base_results
        
    def create_ensemble_models(self):
        """Create various ensemble models"""
        print("\nCreating ensemble models...")
        
        # TODO: Create different types of ensemble models
        # HINT: Include voting, bagging, boosting, and stacking ensembles
        # HINT: Use the best performing base models for ensembles
        
        # Select best base models for ensembles (exclude those that need scaling for simplicity)
        ensemble_base_models = [
            ('rf', self.models['random_forest']),
            ('gb', self.models['gradient_boosting']),
            ('nb', self.models['naive_bayes']),
            ('dt', self.models['decision_tree'])
        ]
        
        # Add XGBoost and LightGBM if available
        if XGBOOST_AVAILABLE:
            ensemble_base_models.append(('xgb', self.models['xgboost']))
        if LIGHTGBM_AVAILABLE:
            ensemble_base_models.append(('lgb', self.models['lightgbm']))
        
        self.ensemble_models = {
            # Voting Classifiers
            'voting_hard': VotingClassifier(
                estimators=ensemble_base_models,
                voting='hard'
            ),
            'voting_soft': VotingClassifier(
                estimators=ensemble_base_models,
                voting='soft'
            ),
            
            # Bagging
            'bagging_rf': BaggingClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=8, class_weight='balanced'),
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            # Boosting
            'adaboost': AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=3),
                n_estimators=50,
                learning_rate=1.0,
                random_state=self.random_state
            ),
            
            # Additional Random Forest (as ensemble baseline)
            'random_forest_large': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        print(f"Created {len(self.ensemble_models)} ensemble models")
        
    def implement_stacking_ensemble(self):
        """Implement custom stacking ensemble"""
        print("\nImplementing stacking ensemble...")
        
        # Select best performing base models for stacking
        base_results = self.results.get('base_models', {})
        
        if base_results:
            # Sort by AUC score and select top models
            sorted_models = sorted(base_results.items(), 
                                 key=lambda x: x[1].get('auc', 0), reverse=True)
            top_model_names = [name for name, _ in sorted_models[:4]]  # Top 4 models
        else:
            # Fallback selection
            top_model_names = ['random_forest', 'gradient_boosting', 'naive_bayes']
            if XGBOOST_AVAILABLE and 'xgboost' in self.models:
                top_model_names.append('xgboost')
        
        # Get base models for stacking
        base_models = []
        for name in top_model_names:
            if name in self.models:
                base_models.append(self.models[name])
        
        if len(base_models) < 2:
            print("Not enough models for stacking, skipping...")
            return
        
        print(f"Using {len(base_models)} models for stacking: {top_model_names}")
        
        # Generate meta-features using stratified cross-validation
        n_models = len(base_models)
        meta_features = np.zeros((len(self.X_train), n_models))
        
        # Use stratified k-fold to maintain class distribution
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for i, model in enumerate(base_models):
            cv_predictions = np.zeros(len(self.X_train))
            
            for train_idx, val_idx in skf.split(self.X_train, self.y_train):
                X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
                y_fold_train = self.y_train[train_idx]
                
                try:
                    # Create a copy of the model
                    model_copy = model.__class__(**model.get_params())
                    model_copy.fit(X_fold_train, y_fold_train)
                    
                    # Generate predictions for validation fold
                    if hasattr(model_copy, 'predict_proba'):
                        cv_predictions[val_idx] = model_copy.predict_proba(X_fold_val)[:, 1]
                    else:
                        cv_predictions[val_idx] = model_copy.predict(X_fold_val)
                        
                except Exception as e:
                    print(f"Error in stacking CV for model {i}: {e}")
                    # Use random predictions as fallback
                    cv_predictions[val_idx] = np.random.random(len(val_idx))
            
            meta_features[:, i] = cv_predictions
        
        # Train meta-model with regularization
        meta_model = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=1000,
            C=1.0  # Regularization parameter
        )
        
        try:
            meta_model.fit(meta_features, self.y_train)
        except Exception as e:
            print(f"Error training meta-model: {e}")
            return
        
        # Train base models on full training set
        fitted_base_models = []
        for model in base_models:
            try:
                fitted_model = model.__class__(**model.get_params())
                fitted_model.fit(self.X_train, self.y_train)
                fitted_base_models.append(fitted_model)
            except Exception as e:
                print(f"Error fitting base model: {e}")
                continue
        
        # Store stacking components
        self.stacking_base_models = fitted_base_models
        self.stacking_meta_model = meta_model
        
        print(f"Stacking ensemble implemented successfully with {len(fitted_base_models)} base models")
        
    def predict_stacking(self, X):
        """Make predictions using stacking ensemble"""
        if not hasattr(self, 'stacking_base_models') or not hasattr(self, 'stacking_meta_model'):
            raise ValueError("Stacking ensemble not implemented yet")
        
        if len(self.stacking_base_models) == 0:
            raise ValueError("No base models available for stacking")
        
        try:
            # Get base model predictions
            base_predictions = np.zeros((len(X), len(self.stacking_base_models)))
            
            for i, model in enumerate(self.stacking_base_models):
                try:
                    if hasattr(model, 'predict_proba'):
                        base_predictions[:, i] = model.predict_proba(X)[:, 1]
                    else:
                        base_predictions[:, i] = model.predict(X)
                except Exception as e:
                    print(f"Error getting predictions from base model {i}: {e}")
                    # Use neutral predictions as fallback
                    base_predictions[:, i] = 0.5
            
            # Meta-model prediction
            final_predictions = self.stacking_meta_model.predict(base_predictions)
            
            if hasattr(self.stacking_meta_model, 'predict_proba'):
                final_probabilities = self.stacking_meta_model.predict_proba(base_predictions)[:, 1]
            else:
                final_probabilities = final_predictions.astype(float)
            
            return final_predictions, final_probabilities
            
        except Exception as e:
            print(f"Error in stacking prediction: {e}")
            # Return fallback predictions
            fallback_pred = np.zeros(len(X), dtype=int)
            fallback_prob = np.full(len(X), 0.5)
            return fallback_pred, fallback_prob
        
    def evaluate_ensemble_models(self):
        """Evaluate all ensemble models"""
        print("\nEvaluating ensemble models...")
        
        # TODO: Train and evaluate each ensemble model
        # HINT: Include the custom stacking ensemble in evaluation
        # HINT: Compare performance against base models
        
        ensemble_results = {}
        
        # Evaluate standard ensemble models
        for name, model in self.ensemble_models.items():
            print(f"Evaluating {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_score = roc_auc_score(self.y_test, y_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            cv_auc_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
            
            ensemble_results[name] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'cv_accuracy_mean': np.mean(cv_scores),
                'cv_accuracy_std': np.std(cv_scores),
                'cv_auc_mean': np.mean(cv_auc_scores),
                'cv_auc_std': np.std(cv_auc_scores)
            }
            
            print(f"  Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        
        # Evaluate stacking ensemble
        print("Evaluating stacking ensemble...")
        stacking_pred, stacking_proba = self.predict_stacking(self.X_test)
        
        stacking_accuracy = accuracy_score(self.y_test, stacking_pred)
        stacking_auc = roc_auc_score(self.y_test, stacking_proba)
        
        ensemble_results['stacking'] = {
            'accuracy': stacking_accuracy,
            'auc': stacking_auc,
            'cv_accuracy_mean': 0,  # Would need separate CV implementation
            'cv_accuracy_std': 0,
            'cv_auc_mean': 0,
            'cv_auc_std': 0
        }
        
        print(f"  Accuracy: {stacking_accuracy:.4f}, AUC: {stacking_auc:.4f}")
        
        self.results['ensemble_models'] = ensemble_results
        return ensemble_results
        
    def compare_all_models(self):
        """Compare all models and identify the best performer"""
        print("\nComparing all models...")
        
        # TODO: Create comprehensive comparison of all models
        # HINT: Combine base model and ensemble results
        # HINT: Identify best performing model for production deployment
        
        all_results = {}
        
        # Add base model results
        if 'base_models' in self.results:
            for name, metrics in self.results['base_models'].items():
                all_results[f"base_{name}"] = metrics
        
        # Add ensemble results
        if 'ensemble_models' in self.results:
            for name, metrics in self.results['ensemble_models'].items():
                all_results[f"ensemble_{name}"] = metrics
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results).T
        comparison_df = comparison_df.sort_values('auc', ascending=False)
        
        print("\nModel Performance Comparison (sorted by AUC):")
        print("=" * 80)
        print(f"{'Model':<25} {'Accuracy':<10} {'AUC':<10} {'CV_Acc':<10} {'CV_AUC':<10}")
        print("=" * 80)
        
        for model_name, metrics in comparison_df.iterrows():
            print(f"{model_name:<25} {metrics['accuracy']:<10.4f} {metrics['auc']:<10.4f} "
                  f"{metrics['cv_accuracy_mean']:<10.4f} {metrics['cv_auc_mean']:<10.4f}")
        
        # Identify best model
        best_model = comparison_df.index[0]
        best_auc = comparison_df.iloc[0]['auc']
        
        print(f"\nBest performing model: {best_model} (AUC: {best_auc:.4f})")
        
        return comparison_df, best_model
        
    def create_production_ensemble(self):
        """Create production-ready ensemble system"""
        print("\nCreating production ensemble system...")
        
        # TODO: Implement production ensemble with monitoring capabilities
        # HINT: Combine top performing models into final ensemble
        # HINT: Add logging, monitoring, and error handling
        
        # Select top 3 performing ensemble models
        ensemble_results = self.results.get('ensemble_models', {})
        
        if ensemble_results:
            # Sort by AUC score
            sorted_ensembles = sorted(ensemble_results.items(), 
                                    key=lambda x: x[1]['auc'], reverse=True)
            
            top_ensembles = sorted_ensembles[:3]
            print(f"Top 3 ensemble models:")
            for name, metrics in top_ensembles:
                print(f"  {name}: AUC = {metrics['auc']:.4f}")
            
            # Use the best ensemble as production model
            best_ensemble_name = top_ensembles[0][0]
            
            if best_ensemble_name == 'stacking':
                print("Selected stacking ensemble for production")
                return 'stacking'
            else:
                print(f"Selected {best_ensemble_name} for production")
                return self.ensemble_models[best_ensemble_name]
        
        return None
        
    def visualize_results(self):
        """Create visualizations of model performance"""
        print("\nCreating performance visualizations...")
        
        # TODO: Create comprehensive visualizations
        # HINT: Include model comparison charts, ROC curves, feature importance
        
        # Model comparison plot
        if 'base_models' in self.results and 'ensemble_models' in self.results:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Accuracy comparison
            base_acc = [metrics['accuracy'] for metrics in self.results['base_models'].values()]
            base_names = list(self.results['base_models'].keys())
            ensemble_acc = [metrics['accuracy'] for metrics in self.results['ensemble_models'].values()]
            ensemble_names = list(self.results['ensemble_models'].keys())
            
            ax1.bar(range(len(base_names)), base_acc, alpha=0.7, label='Base Models')
            ax1.set_xticks(range(len(base_names)))
            ax1.set_xticklabels(base_names, rotation=45)
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Base Model Accuracy Comparison')
            ax1.legend()
            
            ax2.bar(range(len(ensemble_names)), ensemble_acc, alpha=0.7, 
                   color='orange', label='Ensemble Models')
            ax2.set_xticks(range(len(ensemble_names)))
            ax2.set_xticklabels(ensemble_names, rotation=45)
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Ensemble Model Accuracy Comparison')
            ax2.legend()
            
            # AUC comparison
            base_auc = [metrics['auc'] for metrics in self.results['base_models'].values()]
            ensemble_auc = [metrics['auc'] for metrics in self.results['ensemble_models'].values()]
            
            ax3.bar(range(len(base_names)), base_auc, alpha=0.7, label='Base Models')
            ax3.set_xticks(range(len(base_names)))
            ax3.set_xticklabels(base_names, rotation=45)
            ax3.set_ylabel('AUC Score')
            ax3.set_title('Base Model AUC Comparison')
            ax3.legend()
            
            ax4.bar(range(len(ensemble_names)), ensemble_auc, alpha=0.7, 
                   color='orange', label='Ensemble Models')
            ax4.set_xticks(range(len(ensemble_names)))
            ax4.set_xticklabels(ensemble_names, rotation=45)
            ax4.set_ylabel('AUC Score')
            ax4.set_title('Ensemble Model AUC Comparison')
            ax4.legend()
            
            plt.tight_layout()
            plt.show()
        
    def run_complete_experiment(self):
        """Run the complete ensemble experiment"""
        print("Starting Complete Ensemble Methods Experiment")
        print("=" * 60)
        
        # TODO: Execute the complete experimental pipeline
        # HINT: Call all methods in the correct order
        # HINT: Handle any errors gracefully
        
        try:
            # Data preparation
            self.load_and_prepare_data()
            
            # Base models
            self.create_base_models()
            base_results = self.evaluate_base_models()
            
            # Ensemble models
            self.create_ensemble_models()
            self.implement_stacking_ensemble()
            ensemble_results = self.evaluate_ensemble_models()
            
            # Comparison and analysis
            comparison_df, best_model = self.compare_all_models()
            
            # Production system
            production_model = self.create_production_ensemble()
            
            # Visualizations
            self.visualize_results()
            
            print("\n" + "=" * 60)
            print("Experiment completed successfully!")
            print(f"Best model: {best_model}")
            print("=" * 60)
            
            return {
                'base_results': base_results,
                'ensemble_results': ensemble_results,
                'comparison': comparison_df,
                'best_model': best_model,
                'production_model': production_model
            }
            
        except Exception as e:
            print(f"Error during experiment: {e}")
            return None

def main():
    """
    Main function to run the ensemble methods exercise
    
    This exercise demonstrates:
    1. Implementation of multiple ensemble methods
    2. Comprehensive model evaluation and comparison
    3. Production-ready ensemble system design
    4. Performance monitoring and visualization
    """
    print("Day 30: Ensemble Methods - Credit Risk Assessment")
    print("Building production-ready ensemble systems for fintech applications")
    print()
    
    # TODO: Run the complete ensemble experiment
    # HINT: Create experiment instance and run all components
    
    # Initialize experiment
    experiment = EnsembleExperiment(random_state=42)
    
    # Run complete experiment
    results = experiment.run_complete_experiment()
    
    if results:
        print("\nKey Insights:")
        print("- Ensemble methods generally outperform individual models")
        print("- Stacking can provide additional performance gains")
        print("- Model diversity is crucial for ensemble success")
        print("- Production deployment requires careful monitoring")
        
        # Additional analysis
        print(f"\nBest performing model: {results['best_model']}")
        
        if 'ensemble_results' in results:
            best_ensemble = max(results['ensemble_results'].items(), 
                              key=lambda x: x[1]['auc'])
            print(f"Best ensemble: {best_ensemble[0]} (AUC: {best_ensemble[1]['auc']:.4f})")
    
    print("\nExercise completed! Review the results and consider:")
    print("1. Which ensemble method performed best and why?")
    print("2. How would you deploy this system in production?")
    print("3. What monitoring would you implement?")
    print("4. How would you handle model updates and retraining?")

if __name__ == "__main__":
    main()
