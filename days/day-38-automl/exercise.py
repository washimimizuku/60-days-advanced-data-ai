"""
Day 38: AutoML - Exercise

Business Scenario:
You're the Lead ML Engineer at HealthTech Diagnostics Inc. The company processes 
thousands of medical diagnostic cases daily and needs to build predictive models 
quickly and efficiently for various medical conditions. Manual model development 
is too slow and resource-intensive for the volume of different prediction tasks.

Your task is to build a comprehensive AutoML system that can automatically 
engineer features, select optimal algorithms, tune hyperparameters, and create 
ensemble models while maintaining the interpretability and reliability required 
for medical applications.

Requirements:
1. Implement automated feature engineering with domain-specific transformations
2. Build automated model selection with hyperparameter optimization
3. Create ensemble methods for improved reliability
4. Integrate popular AutoML frameworks (H2O.ai, AutoGluon)
5. Develop production pipeline with monitoring and explainability

Success Criteria:
- Achieve >90% accuracy on medical diagnosis tasks
- Complete full AutoML pipeline in <2 hours
- Generate interpretable models with feature importance
- Provide confidence intervals and uncertainty quantification
- Create comprehensive model reports and recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class AutomatedFeatureEngineer:
    """
    Comprehensive automated feature engineering pipeline
    """
    
    def __init__(self, 
                 max_features: int = 1000,
                 feature_selection_ratio: float = 0.8,
                 enable_interactions: bool = True):
        # TODO: Initialize automated feature engineering pipeline
        # HINT: Set up configuration parameters and storage for transformations
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit feature engineering pipeline and transform data
        
        TODO: Implement comprehensive feature engineering including:
        - Basic preprocessing (missing values, outliers)
        - Feature generation (polynomial, statistical aggregations)
        - Temporal features (if datetime columns exist)
        - Categorical combinations and frequency encoding
        - Feature selection using multiple methods
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            Transformed feature matrix
        """
        # TODO: Implement automated feature engineering pipeline
        # HINT: Use multiple feature generation techniques and selection methods
        pass
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        TODO: Transform new data using fitted pipeline
        - Apply same transformations as during fit
        - Handle unseen categories gracefully
        """
        pass
    
    def _generate_medical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        TODO: Generate domain-specific medical features
        - Age groups and risk categories
        - BMI calculations and categories
        - Vital sign ratios and combinations
        - Lab result normalizations
        """
        pass

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
        
        # Configure Optuna to reduce verbosity
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def optimize_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize multiple model types with hyperparameter tuning"""
        import optuna
        from optuna.samplers import TPESampler
        
        logger = logging.getLogger(__name__)
        logger.info("Starting automated model selection and optimization...")
        
        models_to_optimize = {
            'random_forest': self._optimize_random_forest,
            'gradient_boosting': self._optimize_gradient_boosting,
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
        from sklearn.ensemble import GradientBoostingClassifier
        
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
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**params)
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
        logger = logging.getLogger(__name__)
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

class H2OAutoMLWrapper:
    """
    Wrapper for H2O AutoML integration (simplified for demo)
    """
    
    def __init__(self, max_runtime_secs: int = 1800, max_models: int = 20):
        """Initialize H2O AutoML wrapper"""
        self.max_runtime_secs = max_runtime_secs
        self.max_models = max_models
        self.automl = None
        
        # Note: H2O requires Java and additional setup
        # This is a simplified version for demonstration
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Fit H2O AutoML pipeline (simplified)"""
        # Simplified implementation - would normally use H2O
        # For demo purposes, use sklearn RandomForest
        from sklearn.ensemble import RandomForestClassifier
        
        self.automl = RandomForestClassifier(n_estimators=100, random_state=42)
        self.automl.fit(X, y)
        
        # Simulate H2O results
        cv_scores = cross_val_score(self.automl, X, y, cv=5)
        
        return {
            'best_model': self.automl,
            'best_score': np.mean(cv_scores),
            'n_models': 1,  # Simplified
            'framework': 'H2O (simulated)'
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using H2O best model"""
        if self.automl is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.automl.predict(X)

class AutoGluonWrapper:
    """
    Wrapper for AutoGluon integration (simplified for demo)
    """
    
    def __init__(self, time_limit: int = 1800, quality: str = 'good_quality'):
        """Initialize AutoGluon wrapper"""
        self.time_limit = time_limit
        self.quality = quality
        self.predictor = None
        
        # Note: AutoGluon requires additional setup
        # This is a simplified version for demonstration
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Fit AutoGluon pipeline (simplified)"""
        # Simplified implementation - would normally use AutoGluon
        # For demo purposes, use sklearn GradientBoosting
        from sklearn.ensemble import GradientBoostingClassifier
        
        self.predictor = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.predictor.fit(X, y)
        
        # Simulate AutoGluon results
        cv_scores = cross_val_score(self.predictor, X, y, cv=5)
        
        return {
            'best_model': self.predictor,
            'best_score': np.mean(cv_scores),
            'n_models': 1,  # Simplified
            'framework': 'AutoGluon (simulated)'
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using AutoGluon predictor"""
        if self.predictor is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.predictor.predict(X)

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
        logger = logging.getLogger(__name__)
        logger.info("Starting Comprehensive AutoML Pipeline...")
        
        start_time = datetime.now()
        
        # Step 1: Automated Feature Engineering (30% of time budget)
        fe_time_budget = int(self.time_budget * 0.3)
        logger.info("Step 1: Automated Feature Engineering...")
        
        self.feature_engineer = AutomatedFeatureEngineer(
            max_features=min(200, X.shape[1] * 3),
            feature_selection_ratio=0.8,
            enable_interactions=True
        )
        
        X_engineered = self.feature_engineer.fit_transform(X, y)
        
        # Step 2: Automated Model Selection (60% of time budget)
        ms_time_budget = int(self.time_budget * 0.6)
        logger.info("Step 2: Automated Model Selection and Optimization...")
        
        self.model_selector = AutomatedModelSelector(
            task_type=self.task_type,
            n_trials=20,  # Reduced for demo
            cv_folds=3,
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
            'n_features_engineered': X_engineered.shape[1]
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
                'feature_expansion_ratio': self.training_results['n_features_engineered'] / self.training_results['n_features_original']
            },
            'model_selection': {
                'models_evaluated': len(self.training_results['optimization_results']) - 1,
                'best_single_model': self.training_results['optimization_results']['best_overall']['model_name'],
                'ensemble_used': self.training_results['model_type'] == 'ensemble'
            },
            'performance_analysis': {
                'cross_validation_score': self.training_results['final_score'],
                'model_complexity': 'medium',
                'interpretability': 'medium'
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Validate model performance using cross-validation"""
        cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return np.mean(cv_scores)
    
    def _get_feature_importance(self) -> List[Dict[str, Any]]:
        """Get feature importance from final model"""
        if hasattr(self.final_model, 'feature_importances_'):
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
        
        if feature_ratio > 3:
            recommendations.append("High feature expansion detected. Consider feature selection tuning to reduce overfitting.")
        
        return recommendations

def create_medical_diagnosis_dataset():
    """
    Create realistic medical diagnosis dataset for AutoML testing
    """
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
    
    return pd.DataFrame(data)

def evaluate_automl_performance(pipeline, X_test, y_test):
    """
    Comprehensive evaluation of AutoML pipeline performance
    """
    # Make predictions
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    
    # Classification report
    class_report = classification_report(y_test, predictions, output_dict=True)
    
    # Feature importance
    feature_importance = pipeline._get_feature_importance()
    
    evaluation_results = {
        'accuracy': accuracy,
        'precision': class_report['1']['precision'],
        'recall': class_report['1']['recall'],
        'f1_score': class_report['1']['f1-score'],
        'top_features': [f['feature'] for f in feature_importance[:10]],
        'model_explanation': pipeline.get_model_explanation()
    }
    
    return evaluation_results

def main():
    """
    Main exercise implementation
    """
    print("=== Day 38: AutoML - Exercise ===")
    
    # Create medical diagnosis dataset
    print("\n1. Creating Medical Diagnosis Dataset...")
    data = create_medical_diagnosis_dataset()
    print(f"   Created dataset with {len(data)} samples and {len(data.columns)-1} features")
    
    # Split features and target
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Scenario 1: Automated Feature Engineering
    print("\n2. Implementing Automated Feature Engineering...")
    feature_engineer = AutomatedFeatureEngineer(
        max_features=100,  # Reduced for demo
        feature_selection_ratio=0.8,
        enable_interactions=True
    )
    
    X_train_engineered = feature_engineer.fit_transform(X_train, y_train)
    X_test_engineered = feature_engineer.transform(X_test)
    
    print(f"   Original features: {X_train.shape[1]}")
    print(f"   Engineered features: {X_train_engineered.shape[1]}")
    print(f"   Feature expansion ratio: {X_train_engineered.shape[1] / X_train.shape[1]:.2f}x")
    
    # Scenario 2: Automated Model Selection
    print("\n3. Implementing Automated Model Selection...")
    model_selector = AutomatedModelSelector(
        task_type='classification',
        n_trials=10,  # Reduced for demo
        cv_folds=3,
        time_budget=180  # 3 minutes for demo
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
    print("\n4. Creating Ensemble Models...")
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
    print("\n5. Building Comprehensive AutoML Pipeline...")
    automl_pipeline = ComprehensiveAutoMLPipeline(
        task_type='classification',
        time_budget=300,  # 5 minutes for demo
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
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_report = classification_report(y_test, test_predictions, output_dict=True)
    
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Test Precision: {test_report['1']['precision']:.4f}")
    print(f"   Test Recall: {test_report['1']['recall']:.4f}")
    print(f"   Test F1-Score: {test_report['1']['f1-score']:.4f}")
    
    # Get feature importance
    feature_importance = automl_pipeline._get_feature_importance()
    
    if feature_importance:
        print(f"\n   Top 5 Important Features:")
        for i, feature_info in enumerate(feature_importance[:5]):
            print(f"     {i+1}. {feature_info['feature']}: {feature_info['importance']:.4f}")
    
    # Scenario 6: Generate Report
    print("\n7. Generating AutoML Report...")
    
    report = automl_pipeline.generate_report()
    
    print(f"\n   === AutoML Pipeline Report ===")
    print(f"   Model Performance: {report['executive_summary']['model_performance']}")
    print(f"   Model Type: {report['executive_summary']['model_type']}")
    print(f"   Training Time: {report['executive_summary']['training_time']}")
    
    print(f"\n   Recommendations:")
    for i, recommendation in enumerate(report['recommendations'][:3], 1):
        print(f"     {i}. {recommendation}")
    
    print("\n=== Exercise Complete ===")
    print("\n🎯 Successfully implemented:")
    print("   • Automated feature engineering with medical domain knowledge")
    print("   • Multi-algorithm hyperparameter optimization")
    print("   • Intelligent ensemble model creation")
    print("   • Comprehensive AutoML pipeline with reporting")
    print("   • Model evaluation and explanation")
    
    # Summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   Dataset Size: {len(data)} samples")
    print(f"   Original Features: {X_train.shape[1]}")
    print(f"   Engineered Features: {training_results['n_features_engineered']}")
    print(f"   Models Evaluated: {len(optimization_results) - 1}")
    print(f"   Final Model Type: {training_results['model_type']}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    
    # Performance assessment
    if test_accuracy >= 0.9:
        print("\n🚀 Excellent performance achieved! Ready for production.")
    elif test_accuracy >= 0.8:
        print("\n✅ Good performance achieved. Consider additional optimization.")
    else:
        print("\n⚠️  Performance below target. Recommend data quality review.")
    
    print("\nNext: Review solution.py for complete implementation and take the quiz!")

if __name__ == "__main__":
    main()