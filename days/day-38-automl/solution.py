"""
Day 38: AutoML - Solution

Complete implementation of AutoML system with automated feature engineering,
model selection, hyperparameter optimization, and ensemble methods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            VotingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna
from optuna.samplers import TPESampler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedFeatureEngineer:
    """
    Comprehensive automated feature engineering pipeline
    """
    
    def __init__(self, 
                 max_features: int = 1000,
                 feature_selection_ratio: float = 0.8,
                 enable_interactions: bool = True):
        """Initialize automated feature engineering pipeline"""
        self.max_features = max_features
        self.feature_selection_ratio = feature_selection_ratio
        self.enable_interactions = enable_interactions
        
        self.scalers = {}
        self.encoders = {}
        self.selected_features = []
        self.feature_stats = {}
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit feature engineering pipeline and transform data
        """
        logger.info("Starting automated feature engineering...")
        
        # Step 1: Basic preprocessing
        X_processed = self._basic_preprocessing(X)
        
        # Step 2: Generate medical-specific features
        X_medical = self._generate_medical_features(X_processed)
        
        # Step 3: Generate statistical features
        X_generated = self._generate_statistical_features(X_medical)
        
        # Step 4: Feature selection
        if y is not None:
            X_selected = self._select_features(X_generated, y)
        else:
            X_selected = X_generated
        
        # Step 5: Final preprocessing
        X_final = self._final_preprocessing(X_selected)
        
        logger.info(f"Feature engineering complete: {X.shape[1]} -> {X_final.shape[1]} features")
        
        return X_final
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        X_processed = self._apply_basic_preprocessing(X)
        X_medical = self._apply_medical_features(X_processed)
        X_generated = self._apply_statistical_features(X_medical)
        X_selected = self._apply_feature_selection(X_generated)
        X_final = self._apply_final_preprocessing(X_selected)
        
        return X_final
    
    def _basic_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing: handle missing values, outliers"""
        X_clean = X.copy()
        
        # Store original statistics
        self.feature_stats = {}
        
        # Handle missing values
        for column in X_clean.columns:
            if X_clean[column].dtype in ['object', 'category']:
                mode_value = X_clean[column].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else 'unknown'
                X_clean[column].fillna(fill_value, inplace=True)
                self.feature_stats[column] = {'fill_value': fill_value, 'type': 'categorical'}
            else:
                median_value = X_clean[column].median()
                X_clean[column].fillna(median_value, inplace=True)
                
                # Handle outliers using IQR method
                Q1 = X_clean[column].quantile(0.25)
                Q3 = X_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                X_clean[column] = np.clip(X_clean[column], lower_bound, upper_bound)
                
                self.feature_stats[column] = {
                    'fill_value': median_value,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'type': 'numeric'
                }
        
        return X_clean
    
    def _generate_medical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate domain-specific medical features"""
        X_medical = X.copy()
        
        # BMI categories
        if 'bmi' in X_medical.columns:
            X_medical['bmi_category'] = pd.cut(
                X_medical['bmi'],
                bins=[0, 18.5, 25, 30, float('inf')],
                labels=['underweight', 'normal', 'overweight', 'obese']
            )
        
        # Age groups
        if 'age' in X_medical.columns:
            X_medical['age_group'] = pd.cut(
                X_medical['age'],
                bins=[0, 30, 50, 65, float('inf')],
                labels=['young', 'middle_aged', 'senior', 'elderly']
            )
        
        # Blood pressure categories
        if 'systolic_bp' in X_medical.columns and 'diastolic_bp' in X_medical.columns:
            X_medical['bp_category'] = 'normal'
            
            # Hypertension stages
            hypertension_mask = (X_medical['systolic_bp'] >= 140) | (X_medical['diastolic_bp'] >= 90)
            X_medical.loc[hypertension_mask, 'bp_category'] = 'hypertension'
            
            high_normal_mask = ((X_medical['systolic_bp'] >= 130) & (X_medical['systolic_bp'] < 140)) | \
                              ((X_medical['diastolic_bp'] >= 85) & (X_medical['diastolic_bp'] < 90))
            X_medical.loc[high_normal_mask, 'bp_category'] = 'high_normal'
            
            # Pulse pressure
            X_medical['pulse_pressure'] = X_medical['systolic_bp'] - X_medical['diastolic_bp']
        
        # Glucose categories
        if 'glucose' in X_medical.columns:
            X_medical['glucose_category'] = 'normal'
            
            diabetes_mask = X_medical['glucose'] >= 126
            X_medical.loc[diabetes_mask, 'glucose_category'] = 'diabetes'
            
            prediabetes_mask = (X_medical['glucose'] >= 100) & (X_medical['glucose'] < 126)
            X_medical.loc[prediabetes_mask, 'glucose_category'] = 'prediabetes'
        
        # Cholesterol risk
        if 'cholesterol' in X_medical.columns:
            X_medical['cholesterol_risk'] = 'low'
            
            high_risk_mask = X_medical['cholesterol'] >= 240
            X_medical.loc[high_risk_mask, 'cholesterol_risk'] = 'high'
            
            moderate_risk_mask = (X_medical['cholesterol'] >= 200) & (X_medical['cholesterol'] < 240)
            X_medical.loc[moderate_risk_mask, 'cholesterol_risk'] = 'moderate'
        
        # Vital signs ratios
        if 'heart_rate' in X_medical.columns and 'respiratory_rate' in X_medical.columns:
            X_medical['hr_rr_ratio'] = X_medical['heart_rate'] / (X_medical['respiratory_rate'] + 1e-8)
        
        # Symptom count
        symptom_columns = [col for col in X_medical.columns if col.startswith(('chest_pain', 'shortness', 'fatigue', 'dizziness', 'nausea'))]
        if symptom_columns:
            X_medical['symptom_count'] = X_medical[symptom_columns].sum(axis=1)
        
        return X_medical
    
    def _generate_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical aggregation features"""
        X_stats = X.copy()
        
        # Get numeric columns
        numeric_columns = X_stats.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) > 1:
            # Statistical aggregations
            X_stats['numeric_mean'] = X_stats[numeric_columns].mean(axis=1)
            X_stats['numeric_std'] = X_stats[numeric_columns].std(axis=1)
            X_stats['numeric_min'] = X_stats[numeric_columns].min(axis=1)
            X_stats['numeric_max'] = X_stats[numeric_columns].max(axis=1)
            X_stats['numeric_range'] = X_stats['numeric_max'] - X_stats['numeric_min']
            X_stats['numeric_median'] = X_stats[numeric_columns].median(axis=1)
        
        # Polynomial features for key medical indicators
        key_features = ['age', 'bmi', 'systolic_bp', 'glucose', 'cholesterol']
        available_key_features = [f for f in key_features if f in numeric_columns]
        
        if len(available_key_features) >= 2 and self.enable_interactions:
            poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            
            poly_data = poly_features.fit_transform(X_stats[available_key_features])
            poly_feature_names = poly_features.get_feature_names_out(available_key_features)
            
            # Add only interaction terms (not original features)
            for i, name in enumerate(poly_feature_names):
                if name not in available_key_features:  # Skip original features
                    X_stats[f'poly_{name}'] = poly_data[:, i]
        
        # Categorical feature combinations
        categorical_columns = X_stats.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_columns) >= 2 and self.enable_interactions:
            # Create combinations of important categorical features
            important_cats = ['gender', 'smoking_status', 'bmi_category', 'age_group'][:2]
            available_cats = [c for c in important_cats if c in categorical_columns]
            
            if len(available_cats) >= 2:
                for i in range(len(available_cats)):
                    for j in range(i+1, len(available_cats)):
                        col1, col2 = available_cats[i], available_cats[j]
                        X_stats[f'{col1}_{col2}_combo'] = (
                            X_stats[col1].astype(str) + '_' + X_stats[col2].astype(str)
                        )
        
        return X_stats
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select best features using multiple methods"""
        # Encode categorical variables for feature selection
        X_encoded = X.copy()
        
        for column in X_encoded.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_encoded[column] = le.fit_transform(X_encoded[column].astype(str))
            self.encoders[column] = le
        
        # Fill any remaining NaN values
        X_encoded = X_encoded.fillna(0)
        
        # Calculate target number of features
        target_features = min(
            int(X_encoded.shape[1] * self.feature_selection_ratio),
            self.max_features
        )
        
        if target_features >= X_encoded.shape[1]:
            self.selected_features = X.columns.tolist()
            return X
        
        # Univariate feature selection
        selector_f = SelectKBest(score_func=f_classif, k=target_features)
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=target_features)
        
        # Fit selectors
        selector_f.fit(X_encoded, y)
        selector_mi.fit(X_encoded, y)
        
        # Get selected features from both methods
        f_selected = X_encoded.columns[selector_f.get_support()].tolist()
        mi_selected = X_encoded.columns[selector_mi.get_support()].tolist()
        
        # Combine selections (union)
        selected_features = list(set(f_selected + mi_selected))
        
        # If still too many features, take top from f_classif
        if len(selected_features) > target_features:
            selected_features = f_selected[:target_features]
        
        self.selected_features = selected_features
        
        logger.info(f"Selected {len(selected_features)} features from {X.shape[1]}")
        
        return X[selected_features]
    
    def _final_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Final preprocessing: encoding and scaling"""
        X_final = X.copy()
        
        # Encode categorical variables
        categorical_columns = X_final.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_columns:
            if column not in self.encoders:
                le = LabelEncoder()
                X_final[column] = le.fit_transform(X_final[column].astype(str))
                self.encoders[column] = le
        
        # Scale numeric features
        numeric_columns = X_final.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            scaler = StandardScaler()
            X_final[numeric_columns] = scaler.fit_transform(X_final[numeric_columns])
            self.scalers['standard'] = scaler
        
        return X_final
    
    def _apply_basic_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply basic preprocessing to new data"""
        X_clean = X.copy()
        
        for column in X_clean.columns:
            if column in self.feature_stats:
                stats = self.feature_stats[column]
                
                # Fill missing values
                X_clean[column].fillna(stats['fill_value'], inplace=True)
                
                # Apply outlier clipping for numeric features
                if stats['type'] == 'numeric':
                    X_clean[column] = np.clip(
                        X_clean[column], 
                        stats['lower_bound'], 
                        stats['upper_bound']
                    )
        
        return X_clean
    
    def _apply_medical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply medical feature generation to new data"""
        return self._generate_medical_features(X)
    
    def _apply_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply statistical feature generation to new data"""
        return self._generate_statistical_features(X)
    
    def _apply_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection to new data"""
        if self.selected_features:
            available_features = [f for f in self.selected_features if f in X.columns]
            return X[available_features]
        return X
    
    def _apply_final_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply final preprocessing to new data"""
        X_final = X.copy()
        
        # Apply stored encoders
        for column, encoder in self.encoders.items():
            if column in X_final.columns:
                X_final[column] = X_final[column].astype(str)
                known_categories = set(encoder.classes_)
                X_final[column] = X_final[column].apply(
                    lambda x: encoder.transform([x])[0] if x in known_categories else -1
                )
        
        # Apply stored scalers
        numeric_columns = X_final.select_dtypes(include=[np.number]).columns
        
        if 'standard' in self.scalers and len(numeric_columns) > 0:
            X_final[numeric_columns] = self.scalers['standard'].transform(X_final[numeric_columns])
        
        return X_final

class AutomatedModelSelector:
    """
    Automated model selection with hyperparameter optimization
    """
    
    def __init__(self, 
                 task_type: str = 'classification',
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 time_budget: int = 1800):
        """Initialize automated model selector"""
        self.task_type = task_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.time_budget = time_budget
        
        self.best_models = {}
        self.optimization_history = {}
        
        # Configure Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def optimize_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize multiple model types with hyperparameter tuning"""
        logger.info("Starting automated model selection and optimization...")
        
        models_to_optimize = {
            'random_forest': self._optimize_random_forest,
            'gradient_boosting': self._optimize_gradient_boosting,
            'extra_trees': self._optimize_extra_trees,
            'logistic_regression': self._optimize_logistic_regression
        }
        
        results = {}
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for model_name, optimize_func in models_to_optimize.items():
            logger.info(f"Optimizing {model_name}...")
            
            try:
                # Create study
                study = optuna.create_study(
                    direction='maximize',
                    sampler=TPESampler(seed=42)
                )
                
                # Optimize with time budget per model
                time_per_model = self.time_budget // len(models_to_optimize)
                
                study.optimize(
                    lambda trial: optimize_func(trial, X, y, cv),
                    n_trials=self.n_trials,
                    timeout=time_per_model
                )
                
                # Store results
                best_params = study.best_params
                best_score = study.best_value
                
                # Create best model
                best_model = self._create_model_with_params(model_name, best_params)
                best_model.fit(X, y)
                
                results[model_name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'best_score': best_score,
                    'n_trials': len(study.trials)
                }
                
                self.optimization_history[model_name] = study.trials
                
                logger.info(f"{model_name} optimization complete: score = {best_score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to optimize {model_name}: {str(e)}")
                continue
        
        # Find best overall model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['best_score'])
            results['best_overall'] = {
                'model_name': best_model_name,
                'model': results[best_model_name]['model'],
                'score': results[best_model_name]['best_score']
            }
        
        self.best_models = results
        
        logger.info(f"Model optimization complete. Best model: {results.get('best_overall', {}).get('model_name', 'None')}")
        
        return results
    
    def _optimize_random_forest(self, trial, X, y, cv):
        """Optimize Random Forest hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': 42
        }
        
        model = RandomForestClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return np.mean(scores)
    
    def _optimize_gradient_boosting(self, trial, X, y, cv):
        """Optimize Gradient Boosting hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42
        }
        
        model = GradientBoostingClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return np.mean(scores)
    
    def _optimize_extra_trees(self, trial, X, y, cv):
        """Optimize Extra Trees hyperparameters"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': 42
        }
        
        model = ExtraTreesClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return np.mean(scores)
    
    def _optimize_logistic_regression(self, trial, X, y, cv):
        """Optimize Logistic Regression hyperparameters"""
        params = {
            'C': trial.suggest_float('C', 0.001, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'liblinear',
            'max_iter': 1000,
            'random_state': 42
        }
        
        model = LogisticRegression(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return np.mean(scores)
    
    def _create_model_with_params(self, model_name: str, params: Dict[str, Any]):
        """Create model instance with optimized parameters"""
        if model_name == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(**params)
        elif model_name == 'extra_trees':
            return ExtraTreesClassifier(**params)
        elif model_name == 'logistic_regression':
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

class EnsembleModelBuilder:
    """
    Automated ensemble model creation and optimization
    """
    
    def __init__(self, ensemble_size: int = 5, ensemble_method: str = 'voting'):
        """Initialize ensemble builder"""
        self.ensemble_size = ensemble_size
        self.ensemble_method = ensemble_method
        
    def create_ensemble(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
        """Create ensemble model from optimized individual models"""
        logger.info("Creating ensemble model...")
        
        # Select diverse models for ensemble
        selected_models = self._select_diverse_models(models)
        
        if len(selected_models) < 2:
            logger.warning("Not enough models for ensemble, returning best single model")
            return models['best_overall']['model']
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=selected_models,
            voting='soft'  # Use probabilities for voting
        )
        
        # Fit ensemble
        ensemble.fit(X, y)
        
        # Validate ensemble performance
        cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
        ensemble_score = np.mean(cv_scores)
        
        logger.info(f"Ensemble created with {len(selected_models)} models, CV score: {ensemble_score:.4f}")
        
        return ensemble
    
    def _select_diverse_models(self, models: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Select diverse models for ensemble"""
        # Sort models by performance
        model_scores = [(name, result['best_score']) for name, result in models.items() 
                       if name != 'best_overall' and 'model' in result]
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top models up to ensemble_size
        selected = []
        for model_name, score in model_scores[:self.ensemble_size]:
            model = models[model_name]['model']
            selected.append((model_name, model))
        
        return selected
class ComprehensiveAutoMLPipeline:
    """
    Complete AutoML pipeline combining all components
    """
    
    def __init__(self, 
                 task_type: str = 'classification',
                 time_budget: int = 3600,
                 enable_ensemble: bool = True):
        """Initialize comprehensive AutoML pipeline"""
        self.task_type = task_type
        self.time_budget = time_budget
        self.enable_ensemble = enable_ensemble
        
        self.feature_engineer = None
        self.model_selector = None
        self.ensemble_builder = None
        self.final_model = None
        self.training_results = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Fit complete AutoML pipeline"""
        logger.info("Starting Comprehensive AutoML Pipeline...")
        
        start_time = datetime.now()
        
        # Step 1: Automated Feature Engineering (30% of time budget)
        fe_time_budget = int(self.time_budget * 0.3)
        logger.info("Step 1: Automated Feature Engineering...")
        
        self.feature_engineer = AutomatedFeatureEngineer(
            max_features=min(500, X.shape[1] * 5),
            feature_selection_ratio=0.8,
            enable_interactions=True
        )
        
        X_engineered = self.feature_engineer.fit_transform(X, y)
        
        logger.info(f"Feature engineering complete: {X.shape[1]} -> {X_engineered.shape[1]} features")
        
        # Step 2: Automated Model Selection (60% of time budget)
        ms_time_budget = int(self.time_budget * 0.6)
        logger.info("Step 2: Automated Model Selection and Optimization...")
        
        self.model_selector = AutomatedModelSelector(
            task_type=self.task_type,
            n_trials=30,
            cv_folds=5,
            time_budget=ms_time_budget
        )
        
        optimization_results = self.model_selector.optimize_models(X_engineered, y)
        
        # Step 3: Ensemble Creation (10% of time budget)
        if self.enable_ensemble and len(optimization_results) > 2:
            logger.info("Step 3: Creating Ensemble Model...")
            
            self.ensemble_builder = EnsembleModelBuilder(ensemble_size=3)
            ensemble_model = self.ensemble_builder.create_ensemble(
                optimization_results, X_engineered, y
            )
            
            # Compare ensemble vs best single model
            ensemble_score = self._validate_model(ensemble_model, X_engineered, y)
            best_single_score = optimization_results['best_overall']['score']
            
            if ensemble_score > best_single_score:
                self.final_model = ensemble_model
                final_score = ensemble_score
                model_type = 'ensemble'
            else:
                self.final_model = optimization_results['best_overall']['model']
                final_score = best_single_score
                model_type = 'single'
        else:
            self.final_model = optimization_results['best_overall']['model']
            final_score = optimization_results['best_overall']['score']
            model_type = 'single'
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Compile results
        self.training_results = {
            'final_model': self.final_model,
            'model_type': model_type,
            'final_score': final_score,
            'optimization_results': optimization_results,
            'total_training_time': total_time,
            'n_features_original': X.shape[1],
            'n_features_engineered': X_engineered.shape[1],
            'feature_engineering_time': fe_time_budget,
            'model_selection_time': ms_time_budget
        }
        
        logger.info(f"AutoML Pipeline complete. Final model: {model_type}, Score: {final_score:.4f}")
        
        return self.training_results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained pipeline"""
        if self.final_model is None:
            raise ValueError("Pipeline not trained. Call fit() first.")
        
        # Apply feature engineering
        X_engineered = self.feature_engineer.transform(X)
        
        # Make predictions
        predictions = self.final_model.predict(X_engineered)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.final_model is None:
            raise ValueError("Pipeline not trained. Call fit() first.")
        
        # Apply feature engineering
        X_engineered = self.feature_engineer.transform(X)
        
        # Get probabilities
        probabilities = self.final_model.predict_proba(X_engineered)
        
        return probabilities
    
    def get_model_explanation(self) -> Dict[str, Any]:
        """Generate comprehensive model explanation"""
        if self.final_model is None:
            raise ValueError("Pipeline not trained. Call fit() first.")
        
        explanation = {
            'model_type': self.training_results['model_type'],
            'final_score': self.training_results['final_score'],
            'feature_importance': self._get_feature_importance(),
            'model_complexity': self._assess_model_complexity(),
            'training_summary': {
                'total_time': self.training_results['total_training_time'],
                'original_features': self.training_results['n_features_original'],
                'engineered_features': self.training_results['n_features_engineered']
            }
        }
        
        return explanation
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive AutoML report"""
        if self.final_model is None:
            raise ValueError("Pipeline not trained. Call fit() first.")
        
        report = {
            'executive_summary': {
                'model_performance': f"{self.training_results['final_score']:.4f}",
                'model_type': self.training_results['model_type'],
                'training_time': f"{self.training_results['total_training_time']:.1f} seconds",
                'feature_count': self.training_results['n_features_engineered']
            },
            'feature_engineering': {
                'original_features': self.training_results['n_features_original'],
                'engineered_features': self.training_results['n_features_engineered'],
                'feature_expansion_ratio': self.training_results['n_features_engineered'] / self.training_results['n_features_original'],
                'top_features': self._get_top_features(10)
            },
            'model_selection': {
                'models_evaluated': len(self.training_results['optimization_results']) - 1,  # Exclude 'best_overall'
                'best_single_model': self.training_results['optimization_results']['best_overall']['model_name'],
                'ensemble_used': self.training_results['model_type'] == 'ensemble'
            },
            'performance_analysis': {
                'cross_validation_score': self.training_results['final_score'],
                'model_complexity': self._assess_model_complexity(),
                'interpretability': self._assess_interpretability()
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Validate model performance using cross-validation"""
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return np.mean(cv_scores)
    
    def _get_feature_importance(self) -> List[Dict[str, Any]]:
        """Get feature importance from final model"""
        if hasattr(self.final_model, 'feature_importances_'):
            # Single model with feature importance
            importances = self.final_model.feature_importances_
            feature_names = self.feature_engineer.selected_features
            
        elif hasattr(self.final_model, 'estimators'):
            # Ensemble model - aggregate importance
            importances_list = []
            
            for name, estimator in self.final_model.estimators:
                if hasattr(estimator, 'feature_importances_'):
                    importances_list.append(estimator.feature_importances_)
            
            if importances_list:
                importances = np.mean(importances_list, axis=0)
                feature_names = self.feature_engineer.selected_features
            else:
                return []
        else:
            return []
        
        # Create feature importance list
        feature_importance = []
        for i, importance in enumerate(importances):
            if i < len(feature_names):
                feature_importance.append({
                    'feature': feature_names[i],
                    'importance': float(importance)
                })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return feature_importance
    
    def _get_top_features(self, n: int = 10) -> List[str]:
        """Get top N most important features"""
        feature_importance = self._get_feature_importance()
        return [f['feature'] for f in feature_importance[:n]]
    
    def _assess_model_complexity(self) -> str:
        """Assess model complexity"""
        if hasattr(self.final_model, 'estimators'):
            return 'high'  # Ensemble models are complex
        elif hasattr(self.final_model, 'n_estimators'):
            n_estimators = getattr(self.final_model, 'n_estimators', 0)
            if n_estimators > 100:
                return 'high'
            elif n_estimators > 50:
                return 'medium'
            else:
                return 'low'
        else:
            return 'low'  # Linear models are simple
    
    def _assess_interpretability(self) -> str:
        """Assess model interpretability"""
        model_class = self.final_model.__class__.__name__
        
        if 'Logistic' in model_class:
            return 'high'
        elif 'RandomForest' in model_class or 'ExtraTrees' in model_class:
            return 'medium'
        elif 'Voting' in model_class:
            return 'low'
        else:
            return 'medium'
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Performance recommendations
        final_score = self.training_results['final_score']
        
        if final_score >= 0.9:
            recommendations.append("Excellent model performance achieved. Ready for production deployment.")
        elif final_score >= 0.8:
            recommendations.append("Good model performance. Consider additional feature engineering or data collection.")
        else:
            recommendations.append("Model performance below target. Recommend data quality review and additional features.")
        
        # Feature engineering recommendations
        feature_ratio = self.training_results['n_features_engineered'] / self.training_results['n_features_original']
        
        if feature_ratio > 5:
            recommendations.append("High feature expansion detected. Consider feature selection tuning to reduce overfitting.")
        elif feature_ratio < 1.5:
            recommendations.append("Limited feature engineering applied. Consider domain-specific feature creation.")
        
        # Model complexity recommendations
        complexity = self._assess_model_complexity()
        
        if complexity == 'high':
            recommendations.append("High model complexity. Ensure adequate validation and consider simpler alternatives for production.")
        elif complexity == 'low':
            recommendations.append("Simple model selected. Consider ensemble methods if performance needs improvement.")
        
        # Training time recommendations
        training_time = self.training_results['total_training_time']
        
        if training_time > 3600:  # More than 1 hour
            recommendations.append("Long training time detected. Consider reducing search space or using early stopping.")
        
        return recommendations
    
    def save_pipeline(self, filepath: str):
        """Save trained pipeline to disk"""
        if self.final_model is None:
            raise ValueError("Pipeline not trained. Call fit() first.")
        
        pipeline_data = {
            'final_model': self.final_model,
            'feature_engineer': self.feature_engineer,
            'training_results': self.training_results,
            'task_type': self.task_type
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: str):
        """Load trained pipeline from disk"""
        pipeline_data = joblib.load(filepath)
        
        # Create new instance
        pipeline = cls(task_type=pipeline_data['task_type'])
        
        # Restore trained components
        pipeline.final_model = pipeline_data['final_model']
        pipeline.feature_engineer = pipeline_data['feature_engineer']
        pipeline.training_results = pipeline_data['training_results']
        
        logger.info(f"Pipeline loaded from {filepath}")
        
        return pipeline

def create_medical_diagnosis_dataset():
    """Create realistic medical diagnosis dataset for AutoML testing"""
    np.random.seed(42)
    
    n_samples = 5000
    
    # Patient demographics
    data = {
        'age': np.random.randint(18, 90, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'bmi': np.random.normal(25, 5, n_samples),
        'smoking_status': np.random.choice(['never', 'former', 'current'], n_samples, p=[0.5, 0.3, 0.2])
    }
    
    # Vital signs
    data.update({
        'systolic_bp': np.random.normal(120, 20, n_samples),
        'diastolic_bp': np.random.normal(80, 15, n_samples),
        'heart_rate': np.random.normal(70, 15, n_samples),
        'temperature': np.random.normal(98.6, 1.5, n_samples),
        'respiratory_rate': np.random.normal(16, 4, n_samples)
    })
    
    # Lab results
    data.update({
        'glucose': np.random.normal(100, 30, n_samples),
        'cholesterol': np.random.normal(200, 50, n_samples),
        'hemoglobin': np.random.normal(14, 2, n_samples),
        'white_blood_cells': np.random.normal(7000, 2000, n_samples),
        'creatinine': np.random.normal(1.0, 0.3, n_samples)
    })
    
    # Symptoms (binary)
    symptoms = ['chest_pain', 'shortness_of_breath', 'fatigue', 'dizziness', 'nausea']
    for symptom in symptoms:
        data[symptom] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Create complex target variable (diagnosis)
    risk_score = (
        (data['age'] > 60).astype(int) * 0.3 +
        (data['bmi'] > 30).astype(int) * 0.2 +
        (data['systolic_bp'] > 140).astype(int) * 0.25 +
        (data['glucose'] > 126).astype(int) * 0.2 +
        (data['cholesterol'] > 240).astype(int) * 0.15 +
        sum(data[symptom] for symptom in symptoms) * 0.1 +
        (data['smoking_status'] == 'current').astype(int) * 0.15 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Convert to binary diagnosis
    data['diagnosis'] = (risk_score > 0.5).astype(int)
    
    logger.info(f"Created medical dataset with {n_samples} samples, {len(data)-1} features")
    logger.info(f"Diagnosis distribution: {np.mean(data['diagnosis']):.2%} positive cases")
    
    return pd.DataFrame(data)

def evaluate_automl_performance(pipeline, X_test, y_test):
    """Comprehensive evaluation of AutoML pipeline performance"""
    
    # Make predictions
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    
    # Classification report
    class_report = classification_report(y_test, predictions, output_dict=True)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    
    # Feature importance
    feature_importance = pipeline._get_feature_importance()
    
    evaluation_results = {
        'accuracy': accuracy,
        'precision': class_report['1']['precision'],
        'recall': class_report['1']['recall'],
        'f1_score': class_report['1']['f1-score'],
        'confusion_matrix': conf_matrix.tolist(),
        'top_features': [f['feature'] for f in feature_importance[:10]],
        'model_explanation': pipeline.get_model_explanation()
    }
    
    return evaluation_results

def main():
    """
    Main solution implementation demonstrating comprehensive AutoML system
    """
    print("=== Day 38: AutoML - Solution ===")
    
    # Create medical diagnosis dataset
    print("\n1. Creating Medical Diagnosis Dataset...")
    data = create_medical_diagnosis_dataset()
    
    # Split features and target
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    # Scenario 1: Automated Feature Engineering
    print("\n2. Testing Automated Feature Engineering...")
    feature_engineer = AutomatedFeatureEngineer(
        max_features=200,
        feature_selection_ratio=0.8,
        enable_interactions=True
    )
    
    X_train_engineered = feature_engineer.fit_transform(X_train, y_train)
    X_test_engineered = feature_engineer.transform(X_test)
    
    print(f"   Original features: {X_train.shape[1]}")
    print(f"   Engineered features: {X_train_engineered.shape[1]}")
    print(f"   Feature expansion ratio: {X_train_engineered.shape[1] / X_train.shape[1]:.2f}x")
    
    # Scenario 2: Automated Model Selection
    print("\n3. Testing Automated Model Selection...")
    model_selector = AutomatedModelSelector(
        task_type='classification',
        n_trials=20,  # Reduced for demo
        cv_folds=3,
        time_budget=300  # 5 minutes for demo
    )
    
    optimization_results = model_selector.optimize_models(X_train_engineered, y_train)
    
    print(f"   Models optimized: {len(optimization_results) - 1}")  # Exclude 'best_overall'
    print(f"   Best model: {optimization_results['best_overall']['model_name']}")
    print(f"   Best CV score: {optimization_results['best_overall']['score']:.4f}")
    
    # Show all model performances
    for model_name, result in optimization_results.items():
        if model_name != 'best_overall' and 'best_score' in result:
            print(f"     {model_name}: {result['best_score']:.4f}")
    
    # Scenario 3: Ensemble Model Creation
    print("\n4. Testing Ensemble Model Creation...")
    ensemble_builder = EnsembleModelBuilder(ensemble_size=3)
    
    if len(optimization_results) > 2:
        ensemble_model = ensemble_builder.create_ensemble(
            optimization_results, X_train_engineered, y_train
        )
        
        # Evaluate ensemble
        ensemble_predictions = ensemble_model.predict(X_test_engineered)
        ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
        
        print(f"   Ensemble accuracy: {ensemble_accuracy:.4f}")
        print(f"   Ensemble models: {len(ensemble_model.estimators)}")
    else:
        print("   Not enough models for ensemble creation")
    
    # Scenario 4: Comprehensive AutoML Pipeline
    print("\n5. Testing Comprehensive AutoML Pipeline...")
    automl_pipeline = ComprehensiveAutoMLPipeline(
        task_type='classification',
        time_budget=600,  # 10 minutes for demo
        enable_ensemble=True
    )
    
    # Fit pipeline
    training_results = automl_pipeline.fit(X_train, y_train)
    
    print(f"   Final model type: {training_results['model_type']}")
    print(f"   Final CV score: {training_results['final_score']:.4f}")
    print(f"   Training time: {training_results['total_training_time']:.1f} seconds")
    print(f"   Feature engineering: {training_results['n_features_original']} -> {training_results['n_features_engineered']} features")
    
    # Scenario 5: Model Evaluation and Explanation
    print("\n6. Evaluating AutoML Pipeline Performance...")
    
    # Make predictions on test set
    test_predictions = automl_pipeline.predict(X_test)
    test_probabilities = automl_pipeline.predict_proba(X_test)
    
    # Evaluate performance
    evaluation_results = evaluate_automl_performance(automl_pipeline, X_test, y_test)
    
    print(f"   Test Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"   Test Precision: {evaluation_results['precision']:.4f}")
    print(f"   Test Recall: {evaluation_results['recall']:.4f}")
    print(f"   Test F1-Score: {evaluation_results['f1_score']:.4f}")
    
    # Show top features
    print(f"\n   Top 5 Important Features:")
    for i, feature in enumerate(evaluation_results['top_features'][:5]):
        print(f"     {i+1}. {feature}")
    
    # Scenario 6: Generate Comprehensive Report
    print("\n7. Generating Comprehensive AutoML Report...")
    
    report = automl_pipeline.generate_report()
    
    print(f"\n   === AutoML Pipeline Report ===")
    print(f"   Model Performance: {report['executive_summary']['model_performance']}")
    print(f"   Model Type: {report['executive_summary']['model_type']}")
    print(f"   Training Time: {report['executive_summary']['training_time']}")
    print(f"   Feature Count: {report['executive_summary']['feature_count']}")
    
    print(f"\n   Feature Engineering:")
    print(f"     Original Features: {report['feature_engineering']['original_features']}")
    print(f"     Engineered Features: {report['feature_engineering']['engineered_features']}")
    print(f"     Expansion Ratio: {report['feature_engineering']['feature_expansion_ratio']:.2f}x")
    
    print(f"\n   Model Selection:")
    print(f"     Models Evaluated: {report['model_selection']['models_evaluated']}")
    print(f"     Best Single Model: {report['model_selection']['best_single_model']}")
    print(f"     Ensemble Used: {report['model_selection']['ensemble_used']}")
    
    print(f"\n   Recommendations:")
    for i, recommendation in enumerate(report['recommendations'], 1):
        print(f"     {i}. {recommendation}")
    
    # Scenario 7: Save and Load Pipeline
    print("\n8. Testing Pipeline Persistence...")
    
    # Save pipeline
    pipeline_path = "automl_pipeline.pkl"
    automl_pipeline.save_pipeline(pipeline_path)
    
    # Load pipeline
    loaded_pipeline = ComprehensiveAutoMLPipeline.load_pipeline(pipeline_path)
    
    # Test loaded pipeline
    loaded_predictions = loaded_pipeline.predict(X_test[:10])
    original_predictions = automl_pipeline.predict(X_test[:10])
    
    predictions_match = np.array_equal(loaded_predictions, original_predictions)
    print(f"   Pipeline saved and loaded successfully: {predictions_match}")
    
    # Summary
    print("\n=== AutoML System Summary ===")
    
    summary_stats = {
        'Dataset Size': f"{len(data)} samples",
        'Original Features': X_train.shape[1],
        'Engineered Features': training_results['n_features_engineered'],
        'Models Evaluated': len(optimization_results) - 1,
        'Final Model Type': training_results['model_type'],
        'Training Time': f"{training_results['total_training_time']:.1f}s",
        'Test Accuracy': f"{evaluation_results['accuracy']:.4f}",
        'Model Complexity': report['performance_analysis']['model_complexity'],
        'Interpretability': report['performance_analysis']['interpretability']
    }
    
    for metric, value in summary_stats.items():
        print(f"  {metric}: {value}")
    
    print(f"\n🎯 AutoML system successfully implemented!")
    print("   • Automated feature engineering with domain knowledge")
    print("   • Multi-algorithm optimization with hyperparameter tuning")
    print("   • Intelligent ensemble creation and selection")
    print("   • Comprehensive model evaluation and explanation")
    print("   • Production-ready pipeline with persistence")
    
    # Performance assessment
    if evaluation_results['accuracy'] >= 0.9:
        print("\n🚀 Excellent performance achieved! Ready for production deployment.")
    elif evaluation_results['accuracy'] >= 0.8:
        print("\n✅ Good performance achieved. Consider additional optimization.")
    else:
        print("\n⚠️  Performance below target. Recommend data quality review.")
    
    print("\n=== Solution Complete ===")
    print("Review the implementation to understand AutoML best practices!")

if __name__ == "__main__":
    main()