"""
Test suite for AutoML system components
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

# Import classes from solution
from solution import (
    AutomatedFeatureEngineer,
    AutomatedModelSelector,
    EnsembleModelBuilder,
    ComprehensiveAutoMLPipeline,
    create_medical_diagnosis_dataset
)

class TestAutomatedFeatureEngineer:
    """Test automated feature engineering"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'bmi': np.random.normal(25, 5, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'systolic_bp': np.random.normal(120, 20, 100),
            'diastolic_bp': np.random.normal(80, 15, 100),
            'glucose': np.random.normal(100, 30, 100)
        })
        
        self.target = np.random.choice([0, 1], 100)
        
        self.feature_engineer = AutomatedFeatureEngineer(
            max_features=50,
            feature_selection_ratio=0.8,
            enable_interactions=True
        )
    
    def test_initialization(self):
        """Test feature engineer initialization"""
        assert self.feature_engineer.max_features == 50
        assert self.feature_engineer.feature_selection_ratio == 0.8
        assert self.feature_engineer.enable_interactions == True
    
    def test_fit_transform(self):
        """Test feature engineering fit and transform"""
        X_transformed = self.feature_engineer.fit_transform(self.data, self.target)
        
        # Should have more features than original
        assert X_transformed.shape[1] >= self.data.shape[1]
        assert X_transformed.shape[0] == self.data.shape[0]
        
        # Should have selected features stored
        assert hasattr(self.feature_engineer, 'selected_features')
        assert len(self.feature_engineer.selected_features) > 0
    
    def test_medical_feature_generation(self):
        """Test medical-specific feature generation"""
        X_medical = self.feature_engineer._generate_medical_features(self.data)
        
        # Should have BMI categories
        if 'bmi' in self.data.columns:
            assert 'bmi_category' in X_medical.columns
        
        # Should have age groups
        if 'age' in self.data.columns:
            assert 'age_group' in X_medical.columns
        
        # Should have BP categories
        if 'systolic_bp' in self.data.columns and 'diastolic_bp' in self.data.columns:
            assert 'bp_category' in X_medical.columns
            assert 'pulse_pressure' in X_medical.columns
    
    def test_transform_new_data(self):
        """Test transforming new data"""
        # Fit on original data
        self.feature_engineer.fit_transform(self.data, self.target)
        
        # Create new data
        new_data = pd.DataFrame({
            'age': np.random.randint(18, 80, 20),
            'bmi': np.random.normal(25, 5, 20),
            'gender': np.random.choice(['M', 'F'], 20),
            'systolic_bp': np.random.normal(120, 20, 20),
            'diastolic_bp': np.random.normal(80, 15, 20),
            'glucose': np.random.normal(100, 30, 20)
        })
        
        # Transform new data
        X_new_transformed = self.feature_engineer.transform(new_data)
        
        # Should have same number of features as training
        expected_features = len(self.feature_engineer.selected_features)
        assert X_new_transformed.shape[1] == expected_features
        assert X_new_transformed.shape[0] == new_data.shape[0]

class TestAutomatedModelSelector:
    """Test automated model selection"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(200, 10))
        self.y = np.random.choice([0, 1], 200)
        
        self.model_selector = AutomatedModelSelector(
            task_type='classification',
            n_trials=5,  # Small for testing
            cv_folds=3,
            time_budget=60  # 1 minute for testing
        )
    
    def test_initialization(self):
        """Test model selector initialization"""
        assert self.model_selector.task_type == 'classification'
        assert self.model_selector.n_trials == 5
        assert self.model_selector.cv_folds == 3
    
    def test_optimize_models(self):
        """Test model optimization"""
        results = self.model_selector.optimize_models(self.X, self.y)
        
        # Should have results for multiple models
        assert len(results) >= 2  # At least some models + best_overall
        assert 'best_overall' in results
        
        # Best overall should have required fields
        best = results['best_overall']
        assert 'model_name' in best
        assert 'model' in best
        assert 'score' in best
        
        # Individual model results should have required fields
        for model_name, result in results.items():
            if model_name != 'best_overall':
                assert 'model' in result
                assert 'best_params' in result
                assert 'best_score' in result
    
    def test_model_creation(self):
        """Test model creation with parameters"""
        # Test Random Forest
        params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
        model = self.model_selector._create_model_with_params('random_forest', params)
        
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert model.n_estimators == 100
        assert model.max_depth == 10

class TestEnsembleModelBuilder:
    """Test ensemble model building"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(200, 10))
        self.y = np.random.choice([0, 1], 200)
        
        # Create mock optimization results
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        self.mock_results = {
            'random_forest': {
                'model': RandomForestClassifier(n_estimators=50, random_state=42),
                'best_score': 0.85,
                'best_params': {'n_estimators': 50}
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42),
                'best_score': 0.82,
                'best_params': {'C': 1.0}
            },
            'best_overall': {
                'model_name': 'random_forest',
                'model': RandomForestClassifier(n_estimators=50, random_state=42),
                'score': 0.85
            }
        }
        
        # Fit models
        for name, result in self.mock_results.items():
            if name != 'best_overall':
                result['model'].fit(self.X, self.y)
        
        self.ensemble_builder = EnsembleModelBuilder(ensemble_size=2)
    
    def test_initialization(self):
        """Test ensemble builder initialization"""
        assert self.ensemble_builder.ensemble_size == 2
        assert self.ensemble_builder.ensemble_method == 'voting'
    
    def test_create_ensemble(self):
        """Test ensemble creation"""
        ensemble = self.ensemble_builder.create_ensemble(
            self.mock_results, self.X, self.y
        )
        
        # Should be a voting classifier
        assert hasattr(ensemble, 'estimators')
        assert hasattr(ensemble, 'predict')
        assert hasattr(ensemble, 'predict_proba')
        
        # Should have the expected number of estimators
        assert len(ensemble.estimators) <= self.ensemble_builder.ensemble_size
    
    def test_select_diverse_models(self):
        """Test diverse model selection"""
        selected = self.ensemble_builder._select_diverse_models(self.mock_results)
        
        # Should return list of tuples
        assert isinstance(selected, list)
        assert all(isinstance(item, tuple) for item in selected)
        assert all(len(item) == 2 for item in selected)
        
        # Should be sorted by performance
        scores = [self.mock_results[name]['best_score'] for name, _ in selected]
        assert scores == sorted(scores, reverse=True)

class TestComprehensiveAutoMLPipeline:
    """Test comprehensive AutoML pipeline"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.data = create_medical_diagnosis_dataset()
        self.X = self.data.drop('diagnosis', axis=1)
        self.y = self.data['diagnosis']
        
        self.pipeline = ComprehensiveAutoMLPipeline(
            task_type='classification',
            time_budget=120,  # 2 minutes for testing
            enable_ensemble=True
        )
    
    def test_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline.task_type == 'classification'
        assert self.pipeline.time_budget == 120
        assert self.pipeline.enable_ensemble == True
    
    def test_fit_pipeline(self):
        """Test complete pipeline fitting"""
        # Use smaller dataset for faster testing
        X_small = self.X.head(500)
        y_small = self.y.head(500)
        
        results = self.pipeline.fit(X_small, y_small)
        
        # Should have training results
        assert 'final_model' in results
        assert 'model_type' in results
        assert 'final_score' in results
        assert 'total_training_time' in results
        
        # Should have fitted components
        assert self.pipeline.feature_engineer is not None
        assert self.pipeline.model_selector is not None
        assert self.pipeline.final_model is not None
    
    def test_predict(self):
        """Test pipeline predictions"""
        # Fit pipeline first
        X_small = self.X.head(200)
        y_small = self.y.head(200)
        
        self.pipeline.fit(X_small, y_small)
        
        # Test predictions
        X_test = self.X.tail(50)
        predictions = self.pipeline.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba(self):
        """Test pipeline probability predictions"""
        # Fit pipeline first
        X_small = self.X.head(200)
        y_small = self.y.head(200)
        
        self.pipeline.fit(X_small, y_small)
        
        # Test probability predictions
        X_test = self.X.tail(50)
        probabilities = self.pipeline.predict_proba(X_test)
        
        assert probabilities.shape == (len(X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_model_explanation(self):
        """Test model explanation generation"""
        # Fit pipeline first
        X_small = self.X.head(200)
        y_small = self.y.head(200)
        
        self.pipeline.fit(X_small, y_small)
        
        # Get explanation
        explanation = self.pipeline.get_model_explanation()
        
        assert 'model_type' in explanation
        assert 'final_score' in explanation
        assert 'feature_importance' in explanation
        assert 'training_summary' in explanation
    
    def test_generate_report(self):
        """Test comprehensive report generation"""
        # Fit pipeline first
        X_small = self.X.head(200)
        y_small = self.y.head(200)
        
        self.pipeline.fit(X_small, y_small)
        
        # Generate report
        report = self.pipeline.generate_report()
        
        assert 'executive_summary' in report
        assert 'feature_engineering' in report
        assert 'model_selection' in report
        assert 'performance_analysis' in report
        assert 'recommendations' in report
    
    def test_save_load_pipeline(self):
        """Test pipeline persistence"""
        # Fit pipeline first
        X_small = self.X.head(100)
        y_small = self.y.head(100)
        
        self.pipeline.fit(X_small, y_small)
        
        # Save pipeline
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name
        
        try:
            self.pipeline.save_pipeline(filepath)
            
            # Load pipeline
            loaded_pipeline = ComprehensiveAutoMLPipeline.load_pipeline(filepath)
            
            # Test loaded pipeline
            X_test = self.X.tail(10)
            original_predictions = self.pipeline.predict(X_test)
            loaded_predictions = loaded_pipeline.predict(X_test)
            
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)

class TestDataGeneration:
    """Test data generation functions"""
    
    def test_create_medical_diagnosis_dataset(self):
        """Test medical dataset creation"""
        data = create_medical_diagnosis_dataset()
        
        # Check data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5000
        
        # Check required columns
        required_columns = [
            'age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp',
            'glucose', 'cholesterol', 'diagnosis'
        ]
        
        for col in required_columns:
            assert col in data.columns
        
        # Check data ranges
        assert data['age'].min() >= 18
        assert data['age'].max() <= 90
        assert data['diagnosis'].isin([0, 1]).all()
        
        # Check target distribution
        positive_rate = data['diagnosis'].mean()
        assert 0.1 <= positive_rate <= 0.9  # Reasonable class balance

# Integration tests
class TestIntegration:
    """Integration tests for complete AutoML workflow"""
    
    def test_end_to_end_automl(self):
        """Test complete AutoML workflow"""
        # Create small dataset for fast testing
        np.random.seed(42)
        data = create_medical_diagnosis_dataset()
        
        # Use subset for speed
        data_small = data.head(300)
        X = data_small.drop('diagnosis', axis=1)
        y = data_small['diagnosis']
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Create and fit pipeline
        pipeline = ComprehensiveAutoMLPipeline(
            task_type='classification',
            time_budget=60,  # 1 minute for testing
            enable_ensemble=False  # Disable for speed
        )
        
        results = pipeline.fit(X_train, y_train)
        
        # Test predictions
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        
        # Validate results
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)
        
        # Check performance
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, predictions)
        assert accuracy > 0.5  # Should be better than random
        
        # Generate report
        report = pipeline.generate_report()
        assert isinstance(report, dict)
        assert len(report) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])