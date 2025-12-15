# Day 30: Ensemble Methods - Bagging, Boosting, Stacking

## 📖 Learning Objectives
By the end of today, you will be able to:
- **Implement** bagging techniques including Random Forest and Extra Trees
- **Build** boosting algorithms like AdaBoost, Gradient Boosting, and XGBoost
- **Design** stacking and blending ensemble systems for optimal performance
- **Apply** voting classifiers and regressors for robust predictions
- **Deploy** production-ready ensemble systems with proper validation and monitoring

⭐ **Difficulty Level**: Advanced  
🕒 **Estimated Time**: 60 minutes  
🛠️ **Prerequisites**: Machine learning fundamentals, scikit-learn, model evaluation

---

## 🎯 What are Ensemble Methods?

Ensemble methods combine multiple machine learning models to create a stronger predictor than any individual model alone. The key principle is that diverse models can compensate for each other's weaknesses, leading to improved accuracy, robustness, and generalization.

### The Wisdom of Crowds

Just as a group of experts can make better decisions than any single expert, ensemble methods leverage the collective intelligence of multiple models:

- **Reduced Overfitting**: Individual models may overfit to different aspects of the data
- **Improved Accuracy**: Combining predictions often yields better results than single models
- **Increased Robustness**: Less sensitive to outliers and noise in the data
- **Better Generalization**: More stable performance across different datasets

### Business Impact

Ensemble methods are widely used in production systems:
- **Netflix Prize**: The winning solution used ensemble methods combining 100+ algorithms
- **Kaggle Competitions**: Top solutions almost always use ensemble techniques
- **Financial Trading**: Hedge funds use ensemble models for risk assessment and trading decisions
- **Medical Diagnosis**: Combining multiple diagnostic models improves accuracy and reduces false positives
- **Fraud Detection**: Banks use ensemble methods to detect fraudulent transactions

---

## 🌳 Bagging (Bootstrap Aggregating)

Bagging trains multiple models on different subsets of the training data and combines their predictions through averaging or voting.

### Random Forest

Random Forest is one of the most popular ensemble methods, combining bagging with random feature selection.

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Classification example
rf_classifier = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=10,           # Maximum depth of trees
    min_samples_split=5,    # Minimum samples to split
    min_samples_leaf=2,     # Minimum samples in leaf
    max_features='sqrt',    # Features to consider for splits
    bootstrap=True,         # Use bootstrap sampling
    random_state=42,
    n_jobs=-1              # Use all available cores
)

# Regression example
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
```

### Extra Trees (Extremely Randomized Trees)

Extra Trees introduces additional randomness by selecting split thresholds randomly.

```python
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

# Extra Trees with additional randomization
extra_trees = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=None,         # Grow trees until pure leaves
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=False,        # Use entire dataset for each tree
    random_state=42,
    n_jobs=-1
)
```

### Custom Bagging Implementation

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from collections import Counter

class CustomBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=10, 
                 max_samples=1.0, random_state=None):
        self.base_estimator = base_estimator or DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_ = []
        
    def fit(self, X, y):
        """Fit bagging ensemble"""
        np.random.seed(self.random_state)
        self.estimators_ = []
        
        n_samples = int(self.max_samples * len(X))
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            X_bootstrap, y_bootstrap = resample(
                X, y, 
                n_samples=n_samples,
                random_state=self.random_state + i if self.random_state else None
            )
            
            # Train base estimator
            estimator = self.base_estimator.__class__(**self.base_estimator.get_params())
            estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(estimator)
        
        return self
    
    def predict(self, X):
        """Predict using majority voting"""
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        
        # Majority voting
        final_predictions = []
        for i in range(len(X)):
            votes = predictions[:, i]
            majority_vote = Counter(votes).most_common(1)[0][0]
            final_predictions.append(majority_vote)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not hasattr(self.estimators_[0], 'predict_proba'):
            raise AttributeError("Base estimator doesn't support probability prediction")
        
        probabilities = np.array([estimator.predict_proba(X) for estimator in self.estimators_])
        return np.mean(probabilities, axis=0)
```

---

## 🚀 Boosting Methods

Boosting trains models sequentially, where each model learns from the mistakes of previous models.

### AdaBoost (Adaptive Boosting)

AdaBoost adjusts the weights of training examples based on previous model errors.

```python
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier

# AdaBoost Classifier
ada_classifier = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Weak learners
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME.R',  # Real AdaBoost algorithm
    random_state=42
)

# AdaBoost Regressor
ada_regressor = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=50,
    learning_rate=1.0,
    loss='linear',  # 'linear', 'square', 'exponential'
    random_state=42
)
```

### Gradient Boosting

Gradient Boosting builds models sequentially to minimize a loss function.

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.8,          # Stochastic gradient boosting
    max_features='sqrt',
    random_state=42
)

# Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.8,
    max_features='sqrt',
    loss='huber',           # Robust to outliers
    alpha=0.9,              # Quantile for Huber loss
    random_state=42
)
```

### XGBoost (Extreme Gradient Boosting)

XGBoost is an optimized gradient boosting framework designed for speed and performance.

```python
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

if XGBOOST_AVAILABLE:
    # XGBoost Classifier
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        gamma=0,                # Minimum loss reduction for split
        subsample=0.8,
        colsample_bytree=0.8,   # Feature sampling
        reg_alpha=0,            # L1 regularization
        reg_lambda=1,           # L2 regularization
        random_state=42,
        n_jobs=-1
    )
    
    # XGBoost Regressor
    xgb_regressor = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1
    )
```

### LightGBM

LightGBM is another high-performance gradient boosting framework.

```python
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

if LIGHTGBM_AVAILABLE:
    # LightGBM Classifier
    lgb_classifier = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,          # Maximum number of leaves
        min_child_samples=20,   # Minimum samples in leaf
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=0,
        random_state=42,
        n_jobs=-1
    )
```

---

## 🏗️ Stacking and Blending

Stacking uses a meta-model to learn how to best combine predictions from multiple base models.

### Basic Stacking Implementation

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class StackingClassifier:
    def __init__(self, base_models, meta_model=None, cv=5):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression()
        self.cv = cv
        self.fitted_base_models = []
        
    def fit(self, X, y):
        """Fit stacking ensemble"""
        # Generate cross-validation predictions for meta-model training
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        for i, model in enumerate(self.base_models):
            cv_predictions = np.zeros(len(X))
            
            for train_idx, val_idx in kfold.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                # Train model on fold
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train, y_train)
                
                # Predict on validation set
                if hasattr(model_copy, 'predict_proba'):
                    cv_predictions[val_idx] = model_copy.predict_proba(X_val)[:, 1]
                else:
                    cv_predictions[val_idx] = model_copy.predict(X_val)
            
            meta_features[:, i] = cv_predictions
        
        # Train base models on full dataset
        self.fitted_base_models = []
        for model in self.base_models:
            fitted_model = model.__class__(**model.get_params())
            fitted_model.fit(X, y)
            self.fitted_base_models.append(fitted_model)
        
        # Train meta-model
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """Predict using stacked ensemble"""
        # Get base model predictions
        base_predictions = np.zeros((len(X), len(self.fitted_base_models)))
        
        for i, model in enumerate(self.fitted_base_models):
            if hasattr(model, 'predict_proba'):
                base_predictions[:, i] = model.predict_proba(X)[:, 1]
            else:
                base_predictions[:, i] = model.predict(X)
        
        # Meta-model prediction
        return self.meta_model.predict(base_predictions)
    
    def predict_proba(self, X):
        """Predict probabilities using stacked ensemble"""
        # Get base model predictions
        base_predictions = np.zeros((len(X), len(self.fitted_base_models)))
        
        for i, model in enumerate(self.fitted_base_models):
            if hasattr(model, 'predict_proba'):
                base_predictions[:, i] = model.predict_proba(X)[:, 1]
            else:
                base_predictions[:, i] = model.predict(X)
        
        # Meta-model probability prediction
        return self.meta_model.predict_proba(base_predictions)
```

### Advanced Stacking with Multiple Layers

```python
class MultiLevelStackingClassifier:
    def __init__(self, level1_models, level2_models, meta_model, cv=5):
        self.level1_models = level1_models
        self.level2_models = level2_models
        self.meta_model = meta_model
        self.cv = cv
        self.fitted_level1_models = []
        self.fitted_level2_models = []
        
    def fit(self, X, y):
        """Fit multi-level stacking ensemble"""
        # Level 1: Generate meta-features from base models
        level1_meta_features = self._generate_meta_features(X, y, self.level1_models)
        
        # Train Level 1 models on full dataset
        self.fitted_level1_models = self._fit_models(self.level1_models, X, y)
        
        # Level 2: Use Level 1 meta-features as input
        level2_meta_features = self._generate_meta_features(
            level1_meta_features, y, self.level2_models
        )
        
        # Train Level 2 models
        self.fitted_level2_models = self._fit_models(
            self.level2_models, level1_meta_features, y
        )
        
        # Train final meta-model
        self.meta_model.fit(level2_meta_features, y)
        
        return self
    
    def _generate_meta_features(self, X, y, models):
        """Generate meta-features using cross-validation"""
        meta_features = np.zeros((len(X), len(models)))
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        for i, model in enumerate(models):
            cv_predictions = np.zeros(len(X))
            
            for train_idx, val_idx in kfold.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train, y_train)
                
                if hasattr(model_copy, 'predict_proba'):
                    cv_predictions[val_idx] = model_copy.predict_proba(X_val)[:, 1]
                else:
                    cv_predictions[val_idx] = model_copy.predict(X_val)
            
            meta_features[:, i] = cv_predictions
        
        return meta_features
    
    def _fit_models(self, models, X, y):
        """Fit models on full dataset"""
        fitted_models = []
        for model in models:
            fitted_model = model.__class__(**model.get_params())
            fitted_model.fit(X, y)
            fitted_models.append(fitted_model)
        return fitted_models
    
    def predict(self, X):
        """Predict using multi-level stacking"""
        # Level 1 predictions
        level1_predictions = self._get_model_predictions(self.fitted_level1_models, X)
        
        # Level 2 predictions
        level2_predictions = self._get_model_predictions(
            self.fitted_level2_models, level1_predictions
        )
        
        # Final meta-model prediction
        return self.meta_model.predict(level2_predictions)
    
    def _get_model_predictions(self, models, X):
        """Get predictions from a list of models"""
        predictions = np.zeros((len(X), len(models)))
        
        for i, model in enumerate(models):
            if hasattr(model, 'predict_proba'):
                predictions[:, i] = model.predict_proba(X)[:, 1]
            else:
                predictions[:, i] = model.predict(X)
        
        return predictions
```

---

## 🗳️ Voting Methods

Voting ensembles combine predictions through simple voting or averaging mechanisms.

### Hard and Soft Voting

```python
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Hard Voting (majority vote)
hard_voting_classifier = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('svc', SVC(random_state=42)),
        ('nb', GaussianNB())
    ],
    voting='hard'  # Use predicted class labels
)

# Soft Voting (average probabilities)
soft_voting_classifier = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),  # Enable probability
        ('nb', GaussianNB())
    ],
    voting='soft'  # Use predicted probabilities
)

# Voting Regressor (average predictions)
voting_regressor = VotingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42)),
        ('lr', LinearRegression())
    ]
)
```

### Weighted Voting

```python
class WeightedVotingClassifier:
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights or [1.0] * len(estimators)
        self.fitted_estimators = []
        
    def fit(self, X, y):
        """Fit all estimators"""
        self.fitted_estimators = []
        
        for name, estimator in self.estimators:
            fitted_estimator = estimator.__class__(**estimator.get_params())
            fitted_estimator.fit(X, y)
            self.fitted_estimators.append((name, fitted_estimator))
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities using weighted voting"""
        weighted_probas = None
        total_weight = sum(self.weights)
        
        for i, (name, estimator) in enumerate(self.fitted_estimators):
            if hasattr(estimator, 'predict_proba'):
                probas = estimator.predict_proba(X)
                weight = self.weights[i] / total_weight
                
                if weighted_probas is None:
                    weighted_probas = weight * probas
                else:
                    weighted_probas += weight * probas
        
        return weighted_probas
    
    def predict(self, X):
        """Predict using weighted voting"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
```

---

## 📊 Model Selection and Hyperparameter Tuning

### Automated Ensemble Selection

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

class AutoEnsembleSelector:
    def __init__(self, base_models, ensemble_methods, scoring='accuracy', cv=5):
        self.base_models = base_models
        self.ensemble_methods = ensemble_methods
        self.scoring = scoring
        self.cv = cv
        self.best_ensemble = None
        self.results = {}
        
    def fit(self, X, y):
        """Find best ensemble configuration"""
        best_score = -np.inf
        
        for ensemble_name, ensemble_class in self.ensemble_methods.items():
            print(f"Testing {ensemble_name}...")
            
            try:
                # Create ensemble with base models
                if ensemble_name == 'voting':
                    ensemble = ensemble_class(
                        estimators=[(f'model_{i}', model) for i, model in enumerate(self.base_models)]
                    )
                elif ensemble_name == 'stacking':
                    ensemble = ensemble_class(
                        base_models=self.base_models
                    )
                else:
                    continue
                
                # Cross-validation
                scores = cross_val_score(ensemble, X, y, cv=self.cv, scoring=self.scoring)
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                self.results[ensemble_name] = {
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'scores': scores
                }
                
                print(f"{ensemble_name}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    self.best_ensemble = ensemble
                    
            except Exception as e:
                print(f"Error with {ensemble_name}: {e}")
        
        # Fit best ensemble on full data
        if self.best_ensemble:
            self.best_ensemble.fit(X, y)
        
        return self
    
    def predict(self, X):
        """Predict using best ensemble"""
        if self.best_ensemble is None:
            raise ValueError("No ensemble has been fitted")
        return self.best_ensemble.predict(X)
    
    def get_results(self):
        """Get detailed results for all tested ensembles"""
        return self.results
```

### Hyperparameter Optimization for Ensembles

```python
def optimize_random_forest(X, y, cv=5):
    """Optimize Random Forest hyperparameters"""
    
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Use RandomizedSearchCV for efficiency
    random_search = RandomizedSearchCV(
        rf, param_grid, 
        n_iter=50,  # Number of parameter settings sampled
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X, y)
    
    return random_search.best_estimator_, random_search.best_params_

def optimize_xgboost(X, y, cv=5):
    """Optimize XGBoost hyperparameters"""
    
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        xgb_model, param_grid,
        n_iter=30,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X, y)
    
    return random_search.best_estimator_, random_search.best_params_
```

---

## 📈 Evaluation and Validation

### Comprehensive Ensemble Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class EnsembleEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_ensemble(self, ensemble, X_test, y_test, ensemble_name):
        """Comprehensive evaluation of ensemble model"""
        
        # Predictions
        y_pred = ensemble.predict(X_test)
        
        # Probabilities (if available)
        if hasattr(ensemble, 'predict_proba'):
            y_proba = ensemble.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = None
        else:
            y_proba_pos = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # ROC AUC for binary classification
        if y_proba_pos is not None and len(np.unique(y_test)) == 2:
            auc_score = roc_auc_score(y_test, y_proba_pos)
            results['auc_score'] = auc_score
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_proba_pos)
            results['roc_curve'] = (fpr, tpr)
        
        self.results[ensemble_name] = results
        
        return results
    
    def compare_ensembles(self, ensembles, X_test, y_test):
        """Compare multiple ensemble methods"""
        
        comparison_results = {}
        
        for name, ensemble in ensembles.items():
            print(f"Evaluating {name}...")
            results = self.evaluate_ensemble(ensemble, X_test, y_test, name)
            comparison_results[name] = results
        
        # Create comparison summary
        summary = pd.DataFrame({
            name: {
                'Accuracy': results['accuracy'],
                'AUC': results.get('auc_score', 'N/A')
            }
            for name, results in comparison_results.items()
        }).T
        
        return summary, comparison_results
    
    def plot_roc_curves(self, figsize=(10, 8)):
        """Plot ROC curves for all evaluated ensembles"""
        
        plt.figure(figsize=figsize)
        
        for name, results in self.results.items():
            if 'roc_curve' in results:
                fpr, tpr = results['roc_curve']
                auc_score = results['auc_score']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
```

### Cross-Validation for Ensemble Methods

```python
def ensemble_cross_validation(ensemble, X, y, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1']):
    """Perform comprehensive cross-validation for ensemble"""
    
    results = {}
    
    for score in scoring:
        scores = cross_val_score(ensemble, X, y, cv=cv, scoring=score)
        results[score] = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
    
    return results

def nested_cross_validation(ensemble, param_grid, X, y, inner_cv=3, outer_cv=5):
    """Nested cross-validation for unbiased performance estimation"""
    
    outer_scores = []
    
    outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    for train_idx, test_idx in outer_kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner cross-validation for hyperparameter tuning
        grid_search = GridSearchCV(
            ensemble, param_grid, 
            cv=inner_cv, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate best model on outer test set
        best_model = grid_search.best_estimator_
        score = best_model.score(X_test, y_test)
        outer_scores.append(score)
    
    return {
        'scores': outer_scores,
        'mean': np.mean(outer_scores),
        'std': np.std(outer_scores)
    }
```

---

## 🏭 Production Deployment

### Ensemble Model Serving

```python
import joblib
import json
from datetime import datetime

class EnsembleModelServer:
    def __init__(self, model_path=None):
        self.ensemble = None
        self.metadata = {}
        self.prediction_history = []
        
        if model_path:
            self.load_model(model_path)
    
    def save_model(self, ensemble, model_path, metadata=None):
        """Save ensemble model with metadata"""
        
        # Save model
        joblib.dump(ensemble, f"{model_path}.pkl")
        
        # Save metadata
        model_metadata = {
            'model_type': type(ensemble).__name__,
            'created_at': datetime.now().isoformat(),
            'sklearn_version': sklearn.__version__,
            'metadata': metadata or {}
        }
        
        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load ensemble model with metadata"""
        
        # Load model
        self.ensemble = joblib.load(f"{model_path}.pkl")
        
        # Load metadata
        try:
            with open(f"{model_path}_metadata.json", 'r') as f:
                self.metadata = json.load(f)
            print(f"Model loaded: {self.metadata.get('model_type', 'Unknown')}")
        except FileNotFoundError:
            print("Metadata file not found")
    
    def predict(self, X, return_probabilities=False, log_prediction=True):
        """Make predictions with logging"""
        
        if self.ensemble is None:
            raise ValueError("No model loaded")
        
        # Make predictions
        predictions = self.ensemble.predict(X)
        
        result = {'predictions': predictions.tolist()}
        
        # Add probabilities if requested and available
        if return_probabilities and hasattr(self.ensemble, 'predict_proba'):
            probabilities = self.ensemble.predict_proba(X)
            result['probabilities'] = probabilities.tolist()
        
        # Log prediction
        if log_prediction:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(X),
                'predictions': predictions.tolist()
            }
            self.prediction_history.append(log_entry)
        
        return result
    
    def get_model_info(self):
        """Get model information"""
        
        info = {
            'model_loaded': self.ensemble is not None,
            'metadata': self.metadata,
            'prediction_count': len(self.prediction_history)
        }
        
        if self.ensemble is not None:
            info['model_type'] = type(self.ensemble).__name__
            
            # Add ensemble-specific information
            if hasattr(self.ensemble, 'estimators_'):
                info['n_estimators'] = len(self.ensemble.estimators_)
            elif hasattr(self.ensemble, 'base_models'):
                info['n_base_models'] = len(self.ensemble.base_models)
        
        return info
    
    def health_check(self):
        """Perform health check on the model"""
        
        if self.ensemble is None:
            return {'status': 'error', 'message': 'No model loaded'}
        
        try:
            # Test prediction with dummy data
            if hasattr(self.ensemble, 'n_features_in_'):
                n_features = self.ensemble.n_features_in_
            else:
                n_features = 10  # Default
            
            dummy_data = np.random.random((1, n_features))
            _ = self.ensemble.predict(dummy_data)
            
            return {'status': 'healthy', 'message': 'Model is working correctly'}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Model error: {str(e)}'}
```

### Monitoring and Alerting

```python
class EnsembleMonitor:
    def __init__(self, alert_thresholds=None):
        self.alert_thresholds = alert_thresholds or {
            'accuracy_drop': 0.05,
            'prediction_drift': 0.1,
            'latency_increase': 2.0
        }
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.alerts = []
        
    def set_baseline(self, accuracy, avg_latency, prediction_distribution):
        """Set baseline metrics for monitoring"""
        
        self.baseline_metrics = {
            'accuracy': accuracy,
            'avg_latency': avg_latency,
            'prediction_distribution': prediction_distribution
        }
    
    def update_metrics(self, accuracy, avg_latency, prediction_distribution):
        """Update current metrics and check for alerts"""
        
        self.current_metrics = {
            'accuracy': accuracy,
            'avg_latency': avg_latency,
            'prediction_distribution': prediction_distribution,
            'timestamp': datetime.now().isoformat()
        }
        
        self._check_alerts()
    
    def _check_alerts(self):
        """Check for alert conditions"""
        
        if not self.baseline_metrics:
            return
        
        # Check accuracy drop
        accuracy_drop = self.baseline_metrics['accuracy'] - self.current_metrics['accuracy']
        if accuracy_drop > self.alert_thresholds['accuracy_drop']:
            self._create_alert('accuracy_drop', f"Accuracy dropped by {accuracy_drop:.3f}")
        
        # Check latency increase
        latency_ratio = self.current_metrics['avg_latency'] / self.baseline_metrics['avg_latency']
        if latency_ratio > self.alert_thresholds['latency_increase']:
            self._create_alert('latency_increase', f"Latency increased by {latency_ratio:.2f}x")
        
        # Check prediction drift
        baseline_dist = np.array(self.baseline_metrics['prediction_distribution'])
        current_dist = np.array(self.current_metrics['prediction_distribution'])
        
        # Calculate KL divergence as drift measure
        kl_div = self._calculate_kl_divergence(baseline_dist, current_dist)
        if kl_div > self.alert_thresholds['prediction_drift']:
            self._create_alert('prediction_drift', f"Prediction drift detected: KL={kl_div:.3f}")
    
    def _calculate_kl_divergence(self, p, q):
        """Calculate KL divergence between two distributions"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log(p / q))
    
    def _create_alert(self, alert_type, message):
        """Create alert"""
        
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.current_metrics.copy()
        }
        
        self.alerts.append(alert)
        print(f"ALERT [{alert_type}]: {message}")
    
    def get_alerts(self, alert_type=None):
        """Get alerts, optionally filtered by type"""
        
        if alert_type:
            return [alert for alert in self.alerts if alert['type'] == alert_type]
        return self.alerts
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
```

---

## 📚 Key Takeaways

- **Bagging methods** like Random Forest reduce overfitting by training on different data subsets
- **Boosting methods** like XGBoost sequentially improve predictions by learning from previous errors
- **Stacking** uses meta-models to optimally combine base model predictions
- **Voting ensembles** provide simple but effective ways to combine multiple models
- **Hyperparameter tuning** is crucial for optimal ensemble performance
- **Cross-validation** provides unbiased estimates of ensemble performance
- **Production deployment** requires careful model versioning, monitoring, and alerting
- **Ensemble diversity** is key - combining different types of models often works better than similar models
- **Computational cost** increases with ensemble complexity - balance performance vs. efficiency
- **Interpretability** decreases with ensemble complexity - consider explainability requirements

### Business Considerations

- **Performance vs. Complexity**: More complex ensembles may not always justify the additional computational cost
- **Maintenance Overhead**: Multiple models require more maintenance and monitoring
- **Training Time**: Ensemble methods typically take longer to train than single models
- **Feature Importance**: Ensemble methods can provide robust feature importance estimates
- **Model Stability**: Ensembles are generally more stable and less prone to overfitting
- **Scalability**: Consider computational requirements for real-time predictions

---

## 🔄 What's Next?

Tomorrow, we'll explore **Model Explainability** where you'll learn techniques like SHAP, LIME, and other interpretability methods. We'll cover how to make complex models (including the ensemble methods you learned today) more transparent and trustworthy for business stakeholders.

The ensemble techniques you've mastered today will be valuable for understanding how to explain complex model predictions and build trust in AI systems through interpretability frameworks.
