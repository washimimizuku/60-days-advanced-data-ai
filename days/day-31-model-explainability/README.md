# Day 31: Model Explainability - SHAP, LIME, Interpretability

## 📖 Learning Objectives
By the end of today, you will be able to:
- **Implement** SHAP (SHapley Additive exPlanations) for model interpretability
- **Apply** LIME (Local Interpretable Model-agnostic Explanations) for local explanations
- **Build** comprehensive model interpretability frameworks for production systems
- **Design** explainable AI pipelines that meet regulatory and business requirements
- **Deploy** scalable explanation services for real-time and batch inference

⭐ **Difficulty Level**: Advanced  
🕒 **Estimated Time**: 60 minutes  
🛠️ **Prerequisites**: Machine learning models, ensemble methods, statistical concepts

---

## 🎯 What is Model Explainability?

Model explainability (also called interpretability or explainable AI/XAI) refers to the ability to understand and interpret how machine learning models make predictions. As models become more complex and are deployed in critical applications, the need for transparency and trust becomes paramount.

### The Explainability Spectrum

**Intrinsically Interpretable Models**
- Linear regression, logistic regression
- Decision trees (small ones)
- Rule-based systems
- Simple enough to understand directly

**Post-hoc Explanation Methods**
- SHAP, LIME, permutation importance
- Applied to complex "black box" models
- Provide insights after model training

### Business and Regulatory Drivers

**Regulatory Requirements**
- **GDPR**: Right to explanation for automated decision-making
- **Fair Credit Reporting Act**: Adverse action notices must include reasons
- **Model Risk Management**: Banking regulations require model interpretability
- **Healthcare**: FDA guidance on AI/ML-based medical devices

**Business Benefits**
- **Trust Building**: Stakeholders understand model decisions
- **Debugging**: Identify model biases and errors
- **Feature Engineering**: Understand which features matter most
- **Compliance**: Meet regulatory and audit requirements
- **Risk Management**: Detect unexpected model behavior

---

## 🔍 SHAP (SHapley Additive exPlanations)

SHAP is based on cooperative game theory and provides a unified framework for interpreting model predictions by computing the contribution of each feature to the prediction.

### Core SHAP Concepts

**Shapley Values**
- Originated from cooperative game theory
- Fair allocation of "payout" (prediction) among "players" (features)
- Satisfies efficiency, symmetry, dummy, and additivity axioms

**SHAP Equation**
```
f(x) = E[f(X)] + Σ φᵢ
```
Where:
- `f(x)` is the model prediction
- `E[f(X)]` is the expected model output
- `φᵢ` is the SHAP value for feature i

### SHAP Implementation

```python
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=7, 
                          n_redundant=2, random_state=42)
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, use class 1 SHAP values
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Waterfall plot for single prediction
shap.waterfall_plot(explainer.expected_value[1], shap_values[0], X_test.iloc[0])

# Force plot for single prediction
shap.force_plot(explainer.expected_value[1], shap_values[0], X_test.iloc[0])
```

### Different SHAP Explainers

**TreeExplainer** - For tree-based models
```python
# For Random Forest, XGBoost, LightGBM
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

**LinearExplainer** - For linear models
```python
# For linear regression, logistic regression
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)
```

**KernelExplainer** - Model-agnostic (slower)
```python
# For any model type (black box)
explainer = shap.KernelExplainer(model.predict_proba, X_train.sample(100))
shap_values = explainer.shap_values(X_test.sample(10))
```

**DeepExplainer** - For neural networks
```python
# For TensorFlow/Keras models
explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])
```

### Advanced SHAP Techniques

**Partial Dependence with SHAP**
```python
def shap_partial_dependence(model, X, feature_name, feature_values=None):
    """
    Calculate SHAP-based partial dependence for a feature
    """
    explainer = shap.TreeExplainer(model)
    
    if feature_values is None:
        feature_values = np.linspace(X[feature_name].min(), X[feature_name].max(), 50)
    
    partial_dependence = []
    
    for value in feature_values:
        X_modified = X.copy()
        X_modified[feature_name] = value
        
        shap_values = explainer.shap_values(X_modified)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        
        # Average SHAP contribution for this feature value
        avg_contribution = np.mean(shap_values[:, X.columns.get_loc(feature_name)])
        partial_dependence.append(avg_contribution)
    
    return feature_values, partial_dependence

# Usage
feature_values, pd_values = shap_partial_dependence(model, X_test, 'feature_0')
plt.plot(feature_values, pd_values)
plt.xlabel('Feature Value')
plt.ylabel('SHAP Contribution')
plt.title('SHAP Partial Dependence')
plt.show()
```

**SHAP Interaction Values**
```python
# Calculate feature interactions
interaction_values = explainer.shap_interaction_values(X_test[:100])

# Plot interaction between two features
shap.dependence_plot(
    (0, 1),  # Feature indices for interaction
    interaction_values, 
    X_test[:100],
    feature_names=feature_names
)
```

---

## 🔬 LIME (Local Interpretable Model-agnostic Explanations)

LIME explains individual predictions by learning an interpretable model locally around the prediction.

### LIME Core Concepts

**Local Fidelity**
- Explains individual predictions (not global model behavior)
- Creates interpretable model that approximates the complex model locally
- Perturbs input and observes output changes

**Model Agnostic**
- Works with any machine learning model
- Treats model as black box
- Only requires prediction function

### LIME Implementation

```python
import lime
import lime.lime_tabular
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline with preprocessing
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['Class 0', 'Class 1'],
    mode='classification',
    discretize_continuous=True
)

# Explain a single prediction
instance_idx = 0
explanation = explainer.explain_instance(
    X_test.iloc[instance_idx].values,
    pipeline.predict_proba,
    num_features=len(feature_names),
    top_labels=2
)

# Show explanation
explanation.show_in_notebook(show_table=True)

# Get explanation as list
explanation.as_list()
```

### LIME for Different Data Types

**Text Data**
```python
from lime.lime_text import LimeTextExplainer

# For text classification
text_explainer = LimeTextExplainer(class_names=['negative', 'positive'])

# Explain text prediction
explanation = text_explainer.explain_instance(
    text_instance, 
    classifier_fn, 
    num_features=10
)
```

**Image Data**
```python
from lime import lime_image
from skimage.segmentation import mark_boundaries

# For image classification
image_explainer = lime_image.LimeImageExplainer()

# Explain image prediction
explanation = image_explainer.explain_instance(
    image, 
    classifier_fn, 
    top_labels=5, 
    hide_color=0, 
    num_samples=1000
)

# Visualize explanation
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only=True, 
    num_features=5, 
    hide_rest=True
)
```

---

## 📊 Global Interpretability Methods

### Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(
    model, X_test, y_test, 
    n_repeats=10, 
    random_state=42,
    scoring='accuracy'
)

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

# Plot importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance_mean'])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance via Permutation')
plt.show()
```

### Partial Dependence Plots

```python
from sklearn.inspection import PartialDependenceDisplay

# Create partial dependence plots
features = [0, 1, (0, 1)]  # Individual features and interaction
PartialDependenceDisplay.from_estimator(
    model, X_test, features, 
    feature_names=feature_names,
    target=1  # For binary classification
)
plt.show()
```

### Accumulated Local Effects (ALE)

```python
def ale_plot(model, X, feature_idx, bins=20):
    """
    Calculate and plot Accumulated Local Effects
    """
    feature_name = X.columns[feature_idx]
    feature_values = X.iloc[:, feature_idx].values
    
    # Create bins
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(feature_values, quantiles)
    
    ale_values = []
    bin_centers = []
    
    for i in range(len(bin_edges) - 1):
        # Find instances in this bin
        mask = (feature_values >= bin_edges[i]) & (feature_values < bin_edges[i + 1])
        if not np.any(mask):
            continue
            
        X_bin = X[mask].copy()
        
        # Calculate local effects
        X_lower = X_bin.copy()
        X_upper = X_bin.copy()
        
        X_lower.iloc[:, feature_idx] = bin_edges[i]
        X_upper.iloc[:, feature_idx] = bin_edges[i + 1]
        
        # Predict with modified values
        pred_lower = model.predict_proba(X_lower)[:, 1]
        pred_upper = model.predict_proba(X_upper)[:, 1]
        
        # Local effect
        local_effect = np.mean(pred_upper - pred_lower)
        ale_values.append(local_effect)
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
    
    # Accumulate effects
    ale_accumulated = np.cumsum(ale_values)
    
    # Center around zero
    ale_accumulated = ale_accumulated - np.mean(ale_accumulated)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, ale_accumulated, marker='o')
    plt.xlabel(feature_name)
    plt.ylabel('Accumulated Local Effect')
    plt.title(f'ALE Plot for {feature_name}')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return bin_centers, ale_accumulated

# Usage
ale_plot(model, X_test, 0)
```

---

## 🏭 Production Explainability Systems

### Explainability Service Architecture

```python
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ExplanationRequest:
    """Request for model explanation"""
    model_id: str
    instance_data: Dict[str, Any]
    explanation_type: str  # 'shap', 'lime', 'permutation'
    num_features: int = 10
    return_probabilities: bool = True

@dataclass
class ExplanationResponse:
    """Response containing explanation"""
    request_id: str
    model_id: str
    prediction: float
    probability: Optional[float]
    explanations: List[Dict[str, Any]]
    explanation_type: str
    computation_time_ms: float
    timestamp: str

class ExplainerInterface(ABC):
    """Abstract interface for explainers"""
    
    @abstractmethod
    def explain(self, model, instance, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_global_importance(self, model, X, **kwargs) -> Dict[str, Any]:
        pass

class SHAPExplainer(ExplainerInterface):
    """SHAP-based explainer"""
    
    def __init__(self, explainer_type='tree'):
        self.explainer_type = explainer_type
        self.explainer = None
        
    def fit(self, model, X_background):
        """Initialize explainer with background data"""
        if self.explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        elif self.explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(
                model.predict_proba, 
                X_background.sample(min(100, len(X_background)))
            )
        elif self.explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(model, X_background)
    
    def explain(self, model, instance, **kwargs) -> Dict[str, Any]:
        """Generate SHAP explanation for single instance"""
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        # Ensure instance is 2D
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Binary classification, class 1
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]  # Single instance
        
        # Create explanation
        feature_names = kwargs.get('feature_names', [f'feature_{i}' for i in range(len(shap_values))])
        
        explanations = []
        for i, (feature, importance) in enumerate(zip(feature_names, shap_values)):
            explanations.append({
                'feature': feature,
                'importance': float(importance),
                'feature_value': float(instance[0, i]) if hasattr(instance, 'shape') else float(instance[i])
            })
        
        # Sort by absolute importance
        explanations.sort(key=lambda x: abs(x['importance']), reverse=True)
        
        return {
            'explanations': explanations,
            'base_value': float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) 
                              else self.explainer.expected_value),
            'method': 'SHAP'
        }
    
    def get_global_importance(self, model, X, **kwargs) -> Dict[str, Any]:
        """Calculate global feature importance using SHAP"""
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        feature_names = kwargs.get('feature_names', [f'feature_{i}' for i in range(len(mean_abs_shap))])
        
        importance_list = []
        for feature, importance in zip(feature_names, mean_abs_shap):
            importance_list.append({
                'feature': feature,
                'importance': float(importance)
            })
        
        importance_list.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'global_importance': importance_list,
            'method': 'SHAP_global'
        }

class LIMEExplainer(ExplainerInterface):
    """LIME-based explainer"""
    
    def __init__(self):
        self.explainer = None
        
    def fit(self, model, X_background):
        """Initialize LIME explainer"""
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_background.values,
            mode='classification',
            discretize_continuous=True
        )
    
    def explain(self, model, instance, **kwargs) -> Dict[str, Any]:
        """Generate LIME explanation for single instance"""
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        # Ensure instance is 1D for LIME
        if len(instance.shape) > 1:
            instance = instance.flatten()
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=kwargs.get('num_features', 10)
        )
        
        # Extract explanations
        explanations = []
        for feature_idx, importance in explanation.as_list():
            explanations.append({
                'feature': feature_idx,
                'importance': float(importance),
                'feature_value': float(instance[int(feature_idx.split('=')[0]) if '=' in feature_idx else 0])
            })
        
        return {
            'explanations': explanations,
            'method': 'LIME',
            'local_prediction': explanation.local_pred[1] if hasattr(explanation, 'local_pred') else None
        }
    
    def get_global_importance(self, model, X, **kwargs) -> Dict[str, Any]:
        """LIME doesn't provide global importance directly"""
        return {
            'message': 'LIME provides local explanations only',
            'method': 'LIME_global_not_available'
        }

class ProductionExplainabilityService:
    """Production service for model explanations"""
    
    def __init__(self):
        self.explainers = {}
        self.models = {}
        self.explanation_cache = {}
        self.request_history = []
        
    def register_model(self, model_id: str, model: Any, explainer_type: str = 'shap'):
        """Register a model with explainer"""
        self.models[model_id] = model
        
        if explainer_type == 'shap':
            self.explainers[model_id] = SHAPExplainer()
        elif explainer_type == 'lime':
            self.explainers[model_id] = LIMEExplainer()
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
    
    def fit_explainer(self, model_id: str, X_background: pd.DataFrame):
        """Fit explainer for a model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")
        
        model = self.models[model_id]
        explainer = self.explainers[model_id]
        explainer.fit(model, X_background)
    
    def explain_prediction(self, request: ExplanationRequest) -> ExplanationResponse:
        """Generate explanation for a prediction"""
        start_time = time.time()
        request_id = f"req_{int(time.time() * 1000)}"
        
        try:
            # Get model and explainer
            model = self.models[request.model_id]
            explainer = self.explainers[request.model_id]
            
            # Prepare instance data
            instance_df = pd.DataFrame([request.instance_data])
            
            # Make prediction
            prediction = model.predict(instance_df)[0]
            probability = None
            if request.return_probabilities and hasattr(model, 'predict_proba'):
                probability = model.predict_proba(instance_df)[0, 1]
            
            # Generate explanation
            explanation_result = explainer.explain(
                model, 
                instance_df.values,
                feature_names=list(request.instance_data.keys()),
                num_features=request.num_features
            )
            
            computation_time = (time.time() - start_time) * 1000
            
            response = ExplanationResponse(
                request_id=request_id,
                model_id=request.model_id,
                prediction=float(prediction),
                probability=float(probability) if probability is not None else None,
                explanations=explanation_result['explanations'][:request.num_features],
                explanation_type=request.explanation_type,
                computation_time_ms=computation_time,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Log request
            self.request_history.append({
                'request_id': request_id,
                'model_id': request.model_id,
                'explanation_type': request.explanation_type,
                'computation_time_ms': computation_time,
                'timestamp': response.timestamp
            })
            
            return response
            
        except Exception as e:
            computation_time = (time.time() - start_time) * 1000
            
            return ExplanationResponse(
                request_id=request_id,
                model_id=request.model_id,
                prediction=0.0,
                probability=None,
                explanations=[{'error': str(e)}],
                explanation_type=request.explanation_type,
                computation_time_ms=computation_time,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
    
    def get_global_explanations(self, model_id: str, X_sample: pd.DataFrame) -> Dict[str, Any]:
        """Get global model explanations"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")
        
        model = self.models[model_id]
        explainer = self.explainers[model_id]
        
        return explainer.get_global_importance(
            model, 
            X_sample,
            feature_names=X_sample.columns.tolist()
        )
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        if not self.request_history:
            return {'message': 'No requests processed yet'}
        
        total_requests = len(self.request_history)
        avg_computation_time = np.mean([req['computation_time_ms'] for req in self.request_history])
        
        explanation_types = {}
        for req in self.request_history:
            exp_type = req['explanation_type']
            explanation_types[exp_type] = explanation_types.get(exp_type, 0) + 1
        
        return {
            'total_requests': total_requests,
            'avg_computation_time_ms': avg_computation_time,
            'explanation_types': explanation_types,
            'registered_models': list(self.models.keys())
        }
```

---

## 🔧 Advanced Explainability Techniques

### Counterfactual Explanations

```python
class CounterfactualExplainer:
    """Generate counterfactual explanations"""
    
    def __init__(self, model, feature_ranges):
        self.model = model
        self.feature_ranges = feature_ranges
        
    def generate_counterfactual(self, instance, target_class, max_iterations=1000):
        """
        Generate counterfactual explanation using optimization
        """
        from scipy.optimize import minimize
        
        def objective(x):
            # Distance from original instance
            distance = np.sum((x - instance) ** 2)
            
            # Prediction constraint
            pred_proba = self.model.predict_proba([x])[0]
            pred_class = np.argmax(pred_proba)
            
            # Penalty if not target class
            class_penalty = 0 if pred_class == target_class else 1000
            
            return distance + class_penalty
        
        # Constraints to keep features in valid ranges
        bounds = [(self.feature_ranges[i][0], self.feature_ranges[i][1]) 
                 for i in range(len(instance))]
        
        # Optimize
        result = minimize(
            objective, 
            instance, 
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if result.success:
            counterfactual = result.x
            changes = []
            
            for i, (original, new) in enumerate(zip(instance, counterfactual)):
                if abs(original - new) > 1e-6:
                    changes.append({
                        'feature_index': i,
                        'original_value': original,
                        'counterfactual_value': new,
                        'change': new - original
                    })
            
            return {
                'counterfactual': counterfactual,
                'changes': changes,
                'success': True
            }
        else:
            return {
                'counterfactual': None,
                'changes': [],
                'success': False,
                'message': 'Optimization failed'
            }

# Usage
feature_ranges = [(X_train.iloc[:, i].min(), X_train.iloc[:, i].max()) 
                 for i in range(X_train.shape[1])]

cf_explainer = CounterfactualExplainer(model, feature_ranges)
counterfactual = cf_explainer.generate_counterfactual(
    X_test.iloc[0].values, 
    target_class=1
)
```

### Anchors Explanations

```python
try:
    import anchor
    ANCHOR_AVAILABLE = True
except ImportError:
    ANCHOR_AVAILABLE = False

if ANCHOR_AVAILABLE:
    class AnchorExplainer:
        """Generate anchor explanations"""
        
        def __init__(self):
            self.explainer = None
            
        def fit(self, X_train, feature_names, categorical_features=None):
            """Initialize anchor explainer"""
            self.explainer = anchor.anchor_tabular.AnchorTabularExplainer(
                class_names=['Class 0', 'Class 1'],
                feature_names=feature_names,
                train_data=X_train.values,
                categorical_names=categorical_features or {}
            )
        
        def explain(self, model, instance, **kwargs):
            """Generate anchor explanation"""
            explanation = self.explainer.explain_instance(
                instance.values if hasattr(instance, 'values') else instance,
                model.predict,
                threshold=kwargs.get('threshold', 0.95)
            )
            
            return {
                'anchor': explanation.names(),
                'precision': explanation.precision(),
                'coverage': explanation.coverage(),
                'examples': explanation.examples()
            }
```

---

## 📈 Explainability Evaluation and Validation

### Explanation Quality Metrics

```python
class ExplanationEvaluator:
    """Evaluate quality of explanations"""
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        
    def faithfulness_test(self, explainer, num_samples=100):
        """
        Test if explanations are faithful to model behavior
        """
        faithfulness_scores = []
        
        for i in range(min(num_samples, len(self.X_test))):
            instance = self.X_test.iloc[i:i+1]
            
            # Get explanation
            explanation = explainer.explain(self.model, instance.values)
            
            # Get top features
            top_features = [exp['feature'] for exp in explanation['explanations'][:5]]
            
            # Remove top features and measure prediction change
            instance_modified = instance.copy()
            for feature in top_features:
                if feature in instance_modified.columns:
                    # Replace with mean value
                    instance_modified[feature] = self.X_test[feature].mean()
            
            # Compare predictions
            original_pred = self.model.predict_proba(instance)[0, 1]
            modified_pred = self.model.predict_proba(instance_modified)[0, 1]
            
            # Faithfulness score (higher change = more faithful)
            faithfulness = abs(original_pred - modified_pred)
            faithfulness_scores.append(faithfulness)
        
        return {
            'mean_faithfulness': np.mean(faithfulness_scores),
            'std_faithfulness': np.std(faithfulness_scores),
            'scores': faithfulness_scores
        }
    
    def stability_test(self, explainer, num_samples=50, noise_level=0.01):
        """
        Test stability of explanations to small input perturbations
        """
        stability_scores = []
        
        for i in range(min(num_samples, len(self.X_test))):
            instance = self.X_test.iloc[i:i+1]
            
            # Original explanation
            original_exp = explainer.explain(self.model, instance.values)
            original_features = [exp['feature'] for exp in original_exp['explanations'][:10]]
            
            # Add small noise
            noise = np.random.normal(0, noise_level, instance.shape)
            noisy_instance = instance + noise
            
            # Noisy explanation
            noisy_exp = explainer.explain(self.model, noisy_instance.values)
            noisy_features = [exp['feature'] for exp in noisy_exp['explanations'][:10]]
            
            # Calculate overlap
            overlap = len(set(original_features) & set(noisy_features)) / len(original_features)
            stability_scores.append(overlap)
        
        return {
            'mean_stability': np.mean(stability_scores),
            'std_stability': np.std(stability_scores),
            'scores': stability_scores
        }
    
    def comprehensiveness_test(self, explainer, num_samples=50):
        """
        Test if removing important features significantly changes prediction
        """
        comprehensiveness_scores = []
        
        for i in range(min(num_samples, len(self.X_test))):
            instance = self.X_test.iloc[i:i+1]
            
            # Get explanation
            explanation = explainer.explain(self.model, instance.values)
            
            # Original prediction
            original_pred = self.model.predict_proba(instance)[0, 1]
            
            # Remove features one by one (starting with most important)
            pred_changes = []
            instance_modified = instance.copy()
            
            for j, exp in enumerate(explanation['explanations'][:5]):
                feature = exp['feature']
                if feature in instance_modified.columns:
                    # Replace with mean
                    instance_modified[feature] = self.X_test[feature].mean()
                    
                    # New prediction
                    new_pred = self.model.predict_proba(instance_modified)[0, 1]
                    pred_changes.append(abs(original_pred - new_pred))
            
            # Comprehensiveness is cumulative change
            comprehensiveness = sum(pred_changes)
            comprehensiveness_scores.append(comprehensiveness)
        
        return {
            'mean_comprehensiveness': np.mean(comprehensiveness_scores),
            'std_comprehensiveness': np.std(comprehensiveness_scores),
            'scores': comprehensiveness_scores
        }
```

---

## 📚 Key Takeaways

- **SHAP provides theoretically grounded explanations** based on cooperative game theory with consistency guarantees
- **LIME offers model-agnostic local explanations** that are intuitive but may lack global consistency
- **Global methods like permutation importance** help understand overall model behavior and feature relationships
- **Production explainability requires careful architecture** with caching, monitoring, and performance optimization
- **Explanation quality should be evaluated** using faithfulness, stability, and comprehensiveness metrics
- **Regulatory compliance** often requires explainable models, especially in finance and healthcare
- **Different stakeholders need different explanations** - technical vs. business vs. regulatory audiences
- **Counterfactual explanations** answer "what if" questions and provide actionable insights
- **Explanation overhead** must be balanced against business value and regulatory requirements

### Business Considerations

- **Regulatory Requirements**: Understand specific explainability requirements for your industry
- **Stakeholder Needs**: Different audiences require different types of explanations
- **Performance Trade-offs**: Balance explanation quality with computational cost
- **Trust Building**: Use explanations to build confidence in model decisions
- **Bias Detection**: Explanations can reveal unfair or biased model behavior
- **Model Debugging**: Use explanations to identify and fix model issues

---

## 🔄 What's Next?

Tomorrow, we'll dive into **Project - ML Model with Feature Store** where you'll integrate everything learned so far. You'll build a complete ML system that combines feature stores, advanced ML techniques, ensemble methods, and explainability into a production-ready solution.

The explainability techniques you've mastered today will be crucial for making your ML models trustworthy and compliant with business and regulatory requirements in tomorrow's comprehensive project.
