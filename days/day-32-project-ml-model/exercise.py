#!/usr/bin/env python3
"""
Day 32: Project - ML Model with Feature Store - Exercise

Build a simplified version of the production ML platform to understand
the core concepts before implementing the full solution.

Learning Objectives:
1. Implement a basic feature store
2. Train multiple ML models
3. Create a simple API for predictions
4. Add basic explainability
5. Set up monitoring metrics

This exercise provides hands-on experience with each component
before integrating them into the complete production system.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# EXERCISE 1: BASIC FEATURE STORE IMPLEMENTATION
# =============================================================================

class SimpleFeatureStore:
    """
    TODO: Implement a basic feature store with the following methods:
    - store_features(): Store features in memory
    - get_features(): Retrieve features by key
    - list_features(): List all available features
    """
    
    def __init__(self):
        # TODO: Initialize storage for features
        pass
    
    def store_features(self, feature_group: str, features: Dict[str, Any]):
        """
        TODO: Store features for a given feature group
        
        Args:
            feature_group: Name of the feature group (e.g., 'user_features')
            features: Dictionary of features to store
        """
        pass
    
    def get_features(self, feature_group: str, keys: List[str] = None) -> Dict[str, Any]:
        """
        TODO: Retrieve features from a feature group
        
        Args:
            feature_group: Name of the feature group
            keys: Specific feature keys to retrieve (if None, return all)
            
        Returns:
            Dictionary of requested features
        """
        pass
    
    def list_features(self) -> List[str]:
        """
        TODO: List all available feature groups
        
        Returns:
            List of feature group names
        """
        pass

# =============================================================================
# EXERCISE 2: SIMPLE ML MODELS
# =============================================================================

class SimpleCreditRiskModel:
    """
    TODO: Implement a basic credit risk model with the following methods:
    - train(): Train the model on credit data
    - predict(): Make predictions
    - predict_proba(): Get prediction probabilities
    """
    
    def __init__(self):
        # TODO: Initialize model and preprocessing components
        pass
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        TODO: Train the credit risk model
        
        Args:
            X: Feature matrix
            y: Target variable (0=no default, 1=default)
            
        Returns:
            AUC score on test set
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        TODO: Make binary predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions (0 or 1)
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        TODO: Get prediction probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability of default (class 1)
        """
        pass

class SimpleFraudDetectionModel:
    """
    TODO: Implement a basic fraud detection model using anomaly detection
    """
    
    def __init__(self):
        # TODO: Initialize anomaly detection model
        pass
    
    def train(self, X: pd.DataFrame, y: pd.Series = None) -> float:
        """
        TODO: Train the fraud detection model (unsupervised)
        
        Args:
            X: Feature matrix
            y: Labels (optional, for evaluation only)
            
        Returns:
            F1 score if labels provided, else 0
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        TODO: Predict fraud (1) or normal (0)
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        TODO: Get fraud probability
        """
        pass

# =============================================================================
# EXERCISE 3: SIMPLE API SERVICE
# =============================================================================

class SimpleMLAPI:
    """
    TODO: Implement a basic ML API service with the following methods:
    - load_models(): Load trained models
    - predict_credit_risk(): Credit risk prediction endpoint
    - predict_fraud(): Fraud detection endpoint
    - get_health(): Health check endpoint
    """
    
    def __init__(self):
        # TODO: Initialize API service
        pass
    
    def load_models(self, models: Dict[str, Any]):
        """
        TODO: Load trained models into the API service
        
        Args:
            models: Dictionary of trained models
        """
        pass
    
    def predict_credit_risk(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Credit risk prediction endpoint
        
        Args:
            features: Input features for prediction
            
        Returns:
            Prediction response with probability and explanation
        """
        pass
    
    def predict_fraud(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Fraud detection endpoint
        
        Args:
            features: Transaction features
            
        Returns:
            Fraud prediction response
        """
        pass
    
    def get_health(self) -> Dict[str, Any]:
        """
        TODO: Health check endpoint
        
        Returns:
            API health status
        """
        pass

# =============================================================================
# EXERCISE 4: DATA GENERATION
# =============================================================================

def generate_sample_credit_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    TODO: Generate synthetic credit application data
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with credit features and target
        
    Features to include:
    - age: Age of applicant (18-80)
    - income: Annual income (20k-200k)
    - credit_score: Credit score (300-850)
    - loan_amount: Requested loan amount (1k-100k)
    - employment_length: Years of employment (0-40)
    - debt_to_income: Debt to income ratio (0-1)
    - default: Target variable (0/1)
    """
    # TODO: Implement data generation
    # Hint: Use numpy.random for generating realistic distributions
    # Make sure default probability correlates with risk factors
    pass

def generate_sample_transaction_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    TODO: Generate synthetic transaction data for fraud detection
    
    Args:
        n_samples: Number of transactions to generate
        
    Returns:
        DataFrame with transaction features and fraud labels
        
    Features to include:
    - amount: Transaction amount
    - hour: Hour of day (0-23)
    - day_of_week: Day of week (0-6)
    - merchant_category: Type of merchant
    - location_type: domestic/international
    - user_age: Age of user
    - is_fraud: Target variable (0/1)
    """
    # TODO: Implement transaction data generation
    # Make fraud probability depend on suspicious patterns
    pass

# =============================================================================
# EXERCISE 5: BASIC EXPLAINABILITY
# =============================================================================

class SimpleExplainer:
    """
    TODO: Implement basic model explainability without SHAP
    Use feature importance and simple rules-based explanations
    """
    
    def __init__(self):
        pass
    
    def explain_credit_prediction(self, model, features: Dict[str, Any], 
                                prediction: float) -> Dict[str, Any]:
        """
        TODO: Generate explanation for credit risk prediction
        
        Args:
            model: Trained credit risk model
            features: Input features
            prediction: Model prediction probability
            
        Returns:
            Explanation with key factors and recommendations
        """
        # TODO: Implement rule-based explanations
        # Consider factors like credit_score, debt_to_income, etc.
        pass
    
    def explain_fraud_prediction(self, features: Dict[str, Any], 
                               prediction: float) -> Dict[str, Any]:
        """
        TODO: Generate explanation for fraud prediction
        
        Args:
            features: Transaction features
            prediction: Fraud probability
            
        Returns:
            Explanation with suspicious factors
        """
        # TODO: Implement fraud explanation logic
        # Consider unusual amounts, times, locations, etc.
        pass

# =============================================================================
# EXERCISE 6: INTEGRATION AND TESTING
# =============================================================================

def run_exercise():
    """
    TODO: Complete integration test of all components
    
    Steps:
    1. Generate sample data
    2. Set up feature store
    3. Train models
    4. Initialize API
    5. Test predictions
    6. Generate explanations
    7. Print results
    """
    
    print("=" * 60)
    print("DAY 32 EXERCISE: BASIC ML PLATFORM")
    print("=" * 60)
    
    # Step 1: Generate Data
    print("\n1. Generating sample data...")
    # TODO: Generate credit and transaction data
    
    # Step 2: Feature Store
    print("\n2. Setting up feature store...")
    # TODO: Initialize and populate feature store
    
    # Step 3: Train Models
    print("\n3. Training ML models...")
    # TODO: Train credit risk and fraud detection models
    
    # Step 4: API Setup
    print("\n4. Initializing API service...")
    # TODO: Set up API with trained models
    
    # Step 5: Test Predictions
    print("\n5. Testing predictions...")
    # TODO: Test both credit risk and fraud detection
    
    # Step 6: Explanations
    print("\n6. Generating explanations...")
    # TODO: Generate and display explanations
    
    # Step 7: Results Summary
    print("\n7. EXERCISE RESULTS:")
    print("-" * 40)
    # TODO: Print performance metrics and summary
    
    print("\n✅ Exercise completed! Ready for full implementation.")

# =============================================================================
# HINTS AND GUIDANCE
# =============================================================================

"""
IMPLEMENTATION HINTS:

1. Feature Store:
   - Use simple dictionary storage: {feature_group: {key: value}}
   - Store features like: {'user_123': {'age': 35, 'income': 75000}}

2. Credit Risk Model:
   - Use RandomForestClassifier from sklearn
   - Features: age, income, credit_score, loan_amount, employment_length, debt_to_income
   - Target: binary (0=no default, 1=default)
   - Aim for AUC > 0.8

3. Fraud Detection:
   - Use IsolationForest for anomaly detection
   - Features: amount, hour, day_of_week, merchant_category, location_type, user_age
   - Train on normal transactions only
   - Aim for F1 > 0.8

4. API Service:
   - Simple class with methods for each endpoint
   - Return dictionaries with prediction, probability, timestamp
   - Include basic error handling

5. Data Generation:
   - Use numpy.random with realistic distributions
   - Make target variables correlate with risk factors
   - Credit: higher default risk for low credit scores, high debt ratios
   - Fraud: higher risk for large amounts, unusual times, international

6. Explainability:
   - Use feature importance from RandomForest
   - Create rule-based explanations
   - Credit: "Low credit score increases risk"
   - Fraud: "Large amount at unusual hour"

TESTING YOUR IMPLEMENTATION:
- Generate 1000 credit applications and 1000 transactions
- Train models and check performance metrics
- Test API endpoints with sample data
- Verify explanations make sense
- Ensure all components work together

SUCCESS CRITERIA:
✅ Feature store stores and retrieves features correctly
✅ Credit model achieves AUC > 0.8
✅ Fraud model achieves F1 > 0.8
✅ API responds with valid predictions
✅ Explanations provide meaningful insights
✅ All components integrate successfully
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*60)
    print("READY TO START EXERCISE")
    print("="*60)
    print("\nImplement the TODO sections above, then run:")
    print("python exercise.py")
    print("\nGood luck! 🚀")