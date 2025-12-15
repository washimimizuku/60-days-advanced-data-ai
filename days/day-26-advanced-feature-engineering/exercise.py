"""
Day 26: Advanced Feature Engineering - Time Series, NLP & Automated Selection - Exercise

Business Scenario:
You're the Senior ML Engineer at "FinTech Insights", a financial technology company. 
You need to build advanced feature engineering pipelines for multiple ML models including:
- Credit risk assessment using transaction history and customer data
- Fraud detection with real-time transaction patterns
- Customer lifetime value prediction using behavioral and demographic data
- Sentiment analysis of customer feedback for product improvement

Your mission is to implement comprehensive feature engineering pipelines that handle 
time series financial data, NLP customer feedback, and automated feature selection 
for production ML systems.

Requirements:
1. Create sophisticated temporal and lag features for financial time series
2. Extract advanced linguistic features from customer feedback and support tickets
3. Implement ensemble feature selection with multiple algorithms
4. Build automated feature generation for interaction and derived features
5. Create scalable, production-ready feature engineering pipeline
6. Add feature validation and monitoring capabilities

Success Criteria:
- Feature engineering pipeline processes 100K+ records efficiently
- Time series features capture seasonal and trend patterns
- NLP features extract meaningful sentiment and linguistic patterns
- Feature selection reduces dimensionality while maintaining predictive power
- Production pipeline handles missing data and outliers robustly
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, 
    mutual_info_regression, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# NLP libraries
import re
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK not available. Install with: pip install nltk")
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    print("spaCy not available. Install with: pip install spacy")
    SPACY_AVAILABLE = False

# =============================================================================
# EXERCISE 1: TIME SERIES FEATURE ENGINEERING
# =============================================================================

class TimeSeriesFeatureEngineer:
    """Advanced time series feature engineering for financial data"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_temporal_features(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Create comprehensive temporal features from timestamp"""
        
        # TODO: Implement temporal feature creation
        # HINT: Extract year, month, day, hour, dayofweek, quarter
        # Add cyclical encoding for periodic features (hour, dayofweek, month)
        # Include business calendar features (is_weekend, is_month_start, etc.)
        
        df_features = df.copy()
        df_features[timestamp_col] = pd.to_datetime(df_features[timestamp_col])
        
        # Basic temporal features
        df_features['year'] = df_features[timestamp_col].dt.year
        df_features['month'] = df_features[timestamp_col].dt.month
        df_features['day'] = df_features[timestamp_col].dt.day
        df_features['hour'] = df_features[timestamp_col].dt.hour
        df_features['dayofweek'] = df_features[timestamp_col].dt.dayofweek
        df_features['quarter'] = df_features[timestamp_col].dt.quarter
        
        # Cyclical encoding for periodic features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['dayofweek_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7)
        df_features['dayofweek_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        # Business calendar features
        df_features['is_weekend'] = df_features['dayofweek'].isin([5, 6]).astype(int)
        df_features['is_month_start'] = df_features[timestamp_col].dt.is_month_start.astype(int)
        df_features['is_quarter_end'] = df_features[timestamp_col].dt.is_quarter_end.astype(int)
        
        return df_features
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                           entity_col: str, lags: List[int]) -> pd.DataFrame:
        """Create lag features for time series data"""
        
        # TODO: Implement lag feature creation
        # HINT: Use groupby with entity_col and shift for each lag
        # Handle missing values appropriately
        
        df_features = df.copy()
        
        for lag in lags:
            # Create lag features grouped by entity
            lag_feature_name = f'{target_col}_lag_{lag}'
            df_features[lag_feature_name] = df_features.groupby(entity_col)[target_col].shift(lag)
        
        return df_features
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str, 
                              entity_col: str, windows: List[int]) -> pd.DataFrame:
        """Create rolling window statistical features"""
        
        # TODO: Implement rolling window features
        # HINT: Use groupby with entity_col and rolling for each window
        # Calculate mean, std, min, max, median for each window
        # Add trend features (current value vs rolling mean)
        
        df_features = df.copy()
        
        for window in windows:
            # Create rolling statistics
            rolling_group = df_features.groupby(entity_col)[target_col].rolling(window, min_periods=1)
            
            df_features[f'{target_col}_rolling_mean_{window}'] = rolling_group.mean().values
            df_features[f'{target_col}_rolling_std_{window}'] = rolling_group.std().values
            df_features[f'{target_col}_rolling_min_{window}'] = rolling_group.min().values
            df_features[f'{target_col}_rolling_max_{window}'] = rolling_group.max().values
            df_features[f'{target_col}_rolling_median_{window}'] = rolling_group.median().values
            
            # Add trend features (deviation from rolling mean)
            rolling_mean = df_features[f'{target_col}_rolling_mean_{window}']
            rolling_std = df_features[f'{target_col}_rolling_std_{window}']
            df_features[f'{target_col}_rolling_zscore_{window}'] = (
                (df_features[target_col] - rolling_mean) / (rolling_std + 1e-8)
            )
        
        return df_features
    
    def create_seasonality_features(self, df: pd.DataFrame, target_col: str, 
                                  seasonal_periods: List[int]) -> pd.DataFrame:
        """Create seasonality features using Fourier transforms"""
        
        # TODO: Implement seasonality features
        # HINT: Use Fourier series (sin/cos) for different seasonal periods
        # Common periods: 24 (daily), 168 (weekly), 8760 (yearly)
        
        df_features = df.copy()
        
        for period in seasonal_periods:
            for k in range(1, min(period//2, 5)):  # Limit harmonics
                # TODO: Create Fourier features
                # df_features[f'{target_col}_fourier_sin_{period}_{k}'] = ...
                # df_features[f'{target_col}_fourier_cos_{period}_{k}'] = ...
                pass
        
        return df_features

# TODO: Test the TimeSeriesFeatureEngineer
def test_time_series_features():
    """Test time series feature engineering"""
    
    print("🧪 Testing Time Series Feature Engineering...")
    
    # TODO: Create sample financial time series data
    # HINT: Create data with customer_id, timestamp, transaction_amount
    # Include multiple customers and time periods
    
    # Sample data structure:
    # dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    # customers = np.random.choice(range(1, 101), 1000)  # 100 customers
    # amounts = np.random.lognormal(3, 1, 1000)  # Transaction amounts
    
    # TODO: Initialize feature engineer and create features
    # engineer = TimeSeriesFeatureEngineer()
    # df_with_features = engineer.create_temporal_features(sample_data, 'timestamp')
    # df_with_features = engineer.create_lag_features(df_with_features, 'amount', 'customer_id', [1, 6, 24])
    # df_with_features = engineer.create_rolling_features(df_with_features, 'amount', 'customer_id', [6, 24, 168])
    
    print("✅ Time series features created successfully")

# =============================================================================
# EXERCISE 2: NLP FEATURE ENGINEERING
# =============================================================================

class NLPFeatureEngineer:
    """Advanced NLP feature engineering for customer feedback"""
    
    def __init__(self):
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        else:
            self.lemmatizer = None
            self.stop_words = set()
        
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                import spacy
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning and preprocessing"""
        
        # TODO: Implement comprehensive text cleaning
        # HINT: Handle missing values, convert to lowercase
        # Remove URLs, email addresses, special characters
        # Remove extra whitespace
        
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Text cleaning steps
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'\S+@\S+', '', text)  # Remove emails
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        
        return text
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive linguistic features from text"""
        
        # TODO: Implement linguistic feature extraction
        # HINT: Calculate basic length features (char_count, word_count, sentence_count)
        # Add advanced features using spaCy (POS tags, named entities)
        # Include sentiment and readability scores
        
        if not text or pd.isna(text):
            return self._empty_linguistic_features()
        
        features = {}
        
        # Basic length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # TODO: Advanced linguistic features with spaCy
        if self.nlp:
            # doc = self.nlp(text)
            # TODO: POS tag ratios, named entity counts, etc.
            pass
        
        # TODO: Sentiment and readability
        # features['sentiment_polarity'] = self._calculate_sentiment(text)
        # features['readability_score'] = self._calculate_readability(text)
        
        return features
    
    def _empty_linguistic_features(self) -> Dict[str, float]:
        """Return empty feature dictionary for null text"""
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'noun_ratio': 0, 'verb_ratio': 0,
            'adj_ratio': 0, 'entity_count': 0, 'sentiment_polarity': 0,
            'readability_score': 0
        }
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment polarity score"""
        
        # TODO: Implement sentiment calculation
        # HINT: Use simple word lists or integrate with sentiment libraries
        # Return score between -1 (negative) and 1 (positive)
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing']
        
        # Calculate sentiment based on word counts
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
        
        return 0.0
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (Flesch Reading Ease approximation)"""
        
        # TODO: Implement readability calculation
        # HINT: Use sentence count, word count, and syllable count
        # Return score between 0-100 (higher = more readable)
        
        return 50.0  # Placeholder
    
    def create_tfidf_features(self, texts: List[str], max_features: int = 100) -> np.ndarray:
        """Create TF-IDF features with dimensionality reduction"""
        
        # TODO: Implement TF-IDF feature creation
        # HINT: Use TfidfVectorizer with appropriate parameters
        # Apply dimensionality reduction with TruncatedSVD
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create and fit TF-IDF vectorizer
        tfidf = TfidfVectorizer(max_features=max_features*10, ngram_range=(1, 2), stop_words='english')
        tfidf_matrix = tfidf.fit_transform(texts)
        
        # Apply dimensionality reduction
        svd = TruncatedSVD(n_components=max_features)
        reduced_features = svd.fit_transform(tfidf_matrix)
        
        return reduced_features

# TODO: Test the NLPFeatureEngineer
def test_nlp_features():
    """Test NLP feature engineering"""
    
    print("🧪 Testing NLP Feature Engineering...")
    
    # TODO: Create sample customer feedback data
    sample_feedback = [
        "This product is absolutely amazing! I love everything about it.",
        "Terrible quality. Would not recommend to anyone. Waste of money.",
        "Average product. Nothing special but does the job adequately.",
        "Excellent customer service. Very helpful and responsive team.",
        "The worst experience ever. Product broke after one day."
    ]
    
    # TODO: Initialize NLP engineer and extract features
    # nlp_engineer = NLPFeatureEngineer()
    # for feedback in sample_feedback:
    #     cleaned = nlp_engineer.clean_text(feedback)
    #     features = nlp_engineer.extract_linguistic_features(cleaned)
    #     print(f"Feedback: {feedback[:50]}...")
    #     print(f"Features: {features}")
    
    print("✅ NLP features extracted successfully")

# =============================================================================
# EXERCISE 3: AUTOMATED FEATURE SELECTION
# =============================================================================

class AdvancedFeatureSelector:
    """Ensemble feature selection with multiple algorithms"""
    
    def __init__(self, task_type: str = 'classification'):
        self.task_type = task_type
        self.selected_features = {}
        self.feature_scores = {}
    
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> List[str]:
        """Select features using univariate statistical tests"""
        
        # TODO: Implement univariate feature selection
        # HINT: Use SelectKBest with appropriate score function
        # For classification: f_classif, for regression: f_regression
        
        if self.task_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))
        
        # Handle missing values
        X_filled = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        # Fit selector and get selected features
        X_selected = selector.fit_transform(X_filled, y)
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Store feature scores
        self.feature_scores['univariate'] = dict(zip(X.columns, selector.scores_))
        
        return selected_features
    
    def mutual_information_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> List[str]:
        """Select features using mutual information"""
        
        # Implement mutual information selection
        if self.task_type == 'classification':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(X.columns)))
        else:
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(X.columns)))
        
        # Handle missing values and ensure numeric data
        X_filled = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        X_numeric = X_filled.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            return X.columns.tolist()[:k]
        
        X_selected = selector.fit_transform(X_numeric, y)
        selected_mask = selector.get_support()
        selected_features = X_numeric.columns[selected_mask].tolist()
        
        self.feature_scores['mutual_info'] = dict(zip(X_numeric.columns, selector.scores_))
        
        return selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, n_features: int = 50) -> List[str]:
        """Select features using recursive feature elimination"""
        
        # TODO: Implement RFE with RandomForest estimator
        # HINT: Use RFE with appropriate estimator for task type
        
        return []  # Placeholder
    
    def lasso_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using Lasso regularization"""
        
        # TODO: Implement Lasso feature selection
        # HINT: Use LassoCV for automatic alpha selection
        # Select features with non-zero coefficients
        
        return []  # Placeholder
    
    def ensemble_selection(self, X: pd.DataFrame, y: pd.Series, 
                          voting_threshold: float = 0.5) -> List[str]:
        """Combine multiple selection methods using ensemble voting"""
        
        # Implement ensemble feature selection
        methods = ['univariate', 'mutual_info', 'rfe', 'lasso']
        all_selections = {}
        
        # Run each selection method
        try:
            all_selections['univariate'] = set(self.univariate_selection(X, y))
        except:
            pass
        
        try:
            all_selections['mutual_info'] = set(self.mutual_information_selection(X, y))
        except:
            pass
        
        try:
            all_selections['rfe'] = set(self.recursive_feature_elimination(X, y))
        except:
            pass
        
        try:
            all_selections['lasso'] = set(self.lasso_selection(X, y))
        except:
            pass
        
        if not all_selections:
            return X.columns.tolist()[:50]
        
        # Count votes for each feature
        feature_votes = {}
        for feature in X.columns:
            votes = sum(1 for selection in all_selections.values() if feature in selection)
            feature_votes[feature] = votes / len(all_selections)
        
        # Select features meeting voting threshold
        selected_features = [f for f, vote in feature_votes.items() if vote >= voting_threshold]
        
        self.feature_scores['ensemble_votes'] = feature_votes
        
        return selected_features

# TODO: Test the AdvancedFeatureSelector
def test_feature_selection():
    """Test automated feature selection"""
    
    print("🧪 Testing Automated Feature Selection...")
    
    # TODO: Create sample dataset with relevant and irrelevant features
    # HINT: Create synthetic data where only some features are predictive
    
    # np.random.seed(42)
    # n_samples, n_features = 1000, 100
    # X = pd.DataFrame(np.random.randn(n_samples, n_features), 
    #                 columns=[f'feature_{i}' for i in range(n_features)])
    
    # TODO: Create target with some features being relevant
    # relevant_features = [0, 5, 10, 15, 20]
    # y = X.iloc[:, relevant_features].sum(axis=1) + np.random.randn(n_samples) * 0.1
    
    # TODO: Apply feature selection
    # selector = AdvancedFeatureSelector(task_type='regression')
    # selected_features = selector.ensemble_selection(X, y, voting_threshold=0.3)
    
    print("✅ Feature selection completed successfully")

# =============================================================================
# EXERCISE 4: AUTOMATED FEATURE GENERATION
# =============================================================================

class AutomatedFeatureGenerator:
    """Automatically generate new features from existing ones"""
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
    
    def generate_polynomial_features(self, X: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Generate polynomial and interaction features"""
        
        # TODO: Implement polynomial feature generation
        # HINT: Use PolynomialFeatures from sklearn
        # Limit number of features to avoid explosion
        
        from sklearn.preprocessing import PolynomialFeatures
        
        # TODO: Create polynomial features
        # poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
        # X_poly = poly.fit_transform(X)
        # feature_names = poly.get_feature_names_out(X.columns)
        
        return X  # Placeholder
    
    def generate_ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate ratio features between numeric columns"""
        
        # TODO: Implement ratio feature generation
        # HINT: Create ratios for all pairs of numeric columns
        # Handle division by zero appropriately
        
        from itertools import combinations
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_ratios = X.copy()
        
        # TODO: Generate ratios for column pairs
        # for col1, col2 in combinations(numeric_cols, 2):
        #     ratio_name = f'{col1}_div_{col2}'
        #     X_ratios[ratio_name] = X[col1] / (X[col2] + 1e-8)  # Avoid division by zero
        
        return X_ratios
    
    def generate_binning_features(self, X: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
        """Generate binning features for numeric columns"""
        
        # TODO: Implement binning feature generation
        # HINT: Use pd.cut for equal-width and pd.qcut for equal-frequency binning
        # Create binary features for each bin
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_binned = X.copy()
        
        # TODO: Create binning features
        # for col in numeric_cols:
        #     X_binned[f'{col}_bin'] = pd.cut(X[col], bins=n_bins, labels=False)
        #     X_binned[f'{col}_qbin'] = pd.qcut(X[col], q=n_bins, labels=False, duplicates='drop')
        
        return X_binned

# TODO: Test the AutomatedFeatureGenerator
def test_feature_generation():
    """Test automated feature generation"""
    
    print("🧪 Testing Automated Feature Generation...")
    
    # TODO: Create sample dataset
    # sample_data = pd.DataFrame({
    #     'age': np.random.randint(18, 80, 1000),
    #     'income': np.random.lognormal(10, 1, 1000),
    #     'education_years': np.random.randint(8, 20, 1000)
    # })
    
    # TODO: Generate features
    # generator = AutomatedFeatureGenerator()
    # enhanced_data = generator.generate_ratio_features(sample_data)
    # enhanced_data = generator.generate_binning_features(enhanced_data)
    
    print("✅ Feature generation completed successfully")

# =============================================================================
# EXERCISE 5: PRODUCTION FEATURE ENGINEERING PIPELINE
# =============================================================================

class ProductionFeaturePipeline:
    """Production-ready feature engineering pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = None
        self.feature_names = None
        self.is_fitted = False
    
    def build_pipeline(self, X: pd.DataFrame, y: pd.Series = None) -> Pipeline:
        """Build comprehensive feature engineering pipeline"""
        
        # TODO: Implement production pipeline
        # HINT: Identify different column types (numeric, categorical, datetime)
        # Create separate preprocessing pipelines for each type
        # Combine with ColumnTransformer
        
        # TODO: Identify column types
        # numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        # categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # TODO: Create preprocessing pipelines
        # numeric_pipeline = Pipeline([
        #     ('imputer', KNNImputer(n_neighbors=5)),
        #     ('scaler', RobustScaler())
        # ])
        
        # categorical_pipeline = Pipeline([
        #     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        #     ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ])
        
        # TODO: Combine pipelines
        # preprocessor = ColumnTransformer([
        #     ('numeric', numeric_pipeline, numeric_features),
        #     ('categorical', categorical_pipeline, categorical_features)
        # ])
        
        # TODO: Create full pipeline
        # self.pipeline = Pipeline([
        #     ('preprocessor', preprocessor),
        #     ('feature_selector', SelectKBest(k=min(100, len(X.columns))))
        # ])
        
        return self.pipeline
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the feature engineering pipeline"""
        
        # TODO: Implement pipeline fitting
        # HINT: Build pipeline if not exists, then fit
        
        if self.pipeline is None:
            self.build_pipeline(X, y)
        
        # TODO: Fit pipeline
        # self.pipeline.fit(X, y)
        # self.is_fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted pipeline"""
        
        # TODO: Implement pipeline transformation
        # HINT: Check if pipeline is fitted, then transform
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        # TODO: Transform data
        # X_transformed = self.pipeline.transform(X)
        # return X_transformed
        
        return X.values  # Placeholder
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fit pipeline and transform data"""
        
        return self.fit(X, y).transform(X)

# TODO: Test the ProductionFeaturePipeline
def test_production_pipeline():
    """Test production feature engineering pipeline"""
    
    print("🧪 Testing Production Feature Engineering Pipeline...")
    
    # TODO: Create comprehensive test dataset
    # HINT: Include numeric, categorical, and text features
    # Add missing values and outliers to test robustness
    
    # TODO: Initialize and test pipeline
    # config = {'max_features': 50}
    # pipeline = ProductionFeaturePipeline(config)
    
    # TODO: Fit and transform data
    # X_transformed = pipeline.fit_transform(test_data, y_test)
    
    print("✅ Production pipeline tested successfully")

# =============================================================================
# EXERCISE 6: FEATURE VALIDATION AND MONITORING
# =============================================================================

class FeatureValidator:
    """Validate feature quality and detect issues"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_feature_quality(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive feature quality validation"""
        
        # TODO: Implement feature quality validation
        # HINT: Check for missing values, outliers, data types
        # Calculate quality scores and identify issues
        
        results = {
            'total_features': len(X.columns),
            'missing_value_features': 0,
            'high_cardinality_features': 0,
            'constant_features': 0,
            'quality_score': 0.0,
            'issues': []
        }
        
        # TODO: Check for various quality issues
        # for col in X.columns:
        #     # Check missing values
        #     missing_pct = X[col].isnull().sum() / len(X)
        #     if missing_pct > 0.5:
        #         results['issues'].append(f'{col}: High missing values ({missing_pct:.2%})')
        
        #     # Check constant features
        #     if X[col].nunique() == 1:
        #         results['constant_features'] += 1
        #         results['issues'].append(f'{col}: Constant feature')
        
        return results
    
    def detect_feature_drift(self, X_reference: pd.DataFrame, 
                           X_current: pd.DataFrame) -> Dict[str, Any]:
        """Detect feature drift between reference and current data"""
        
        # TODO: Implement feature drift detection
        # HINT: Compare distributions using statistical tests
        # Use Kolmogorov-Smirnov test for numeric features
        
        from scipy import stats
        
        drift_results = {
            'features_with_drift': [],
            'drift_scores': {},
            'overall_drift_score': 0.0
        }
        
        # TODO: Compare feature distributions
        # common_features = set(X_reference.columns) & set(X_current.columns)
        # for feature in common_features:
        #     if X_reference[feature].dtype in ['int64', 'float64']:
        #         # Use KS test for numeric features
        #         ks_stat, p_value = stats.ks_2samp(X_reference[feature], X_current[feature])
        #         drift_results['drift_scores'][feature] = ks_stat
        #         if p_value < 0.05:  # Significant drift
        #             drift_results['features_with_drift'].append(feature)
        
        return drift_results

# TODO: Test the FeatureValidator
def test_feature_validation():
    """Test feature validation and monitoring"""
    
    print("🧪 Testing Feature Validation and Monitoring...")
    
    # TODO: Create test datasets with quality issues
    # HINT: Include missing values, constant features, outliers
    
    # TODO: Test validation
    # validator = FeatureValidator()
    # quality_results = validator.validate_feature_quality(test_data)
    # print(f"Quality results: {quality_results}")
    
    print("✅ Feature validation completed successfully")

# =============================================================================
# EXERCISE 7: INTEGRATION TEST
# =============================================================================

def run_comprehensive_feature_engineering_test():
    """Run comprehensive test of all feature engineering components"""
    
    print("🚀 Running Comprehensive Feature Engineering Test...")
    print("=" * 60)
    
    # TODO: Create comprehensive test dataset
    print("📊 Creating comprehensive test dataset...")
    
    # TODO: Generate synthetic financial dataset
    # HINT: Include time series data, customer feedback, and mixed data types
    
    # Sample structure:
    # - customer_id: int
    # - timestamp: datetime
    # - transaction_amount: float
    # - feedback_text: str
    # - customer_age: int
    # - account_type: str
    # - credit_score: float
    
    print("✅ Test dataset created")
    
    # TODO: Test each component
    print("\n🔧 Testing individual components...")
    
    # Test 1: Time Series Features
    test_time_series_features()
    
    # Test 2: NLP Features
    test_nlp_features()
    
    # Test 3: Feature Selection
    test_feature_selection()
    
    # Test 4: Feature Generation
    test_feature_generation()
    
    # Test 5: Production Pipeline
    test_production_pipeline()
    
    # Test 6: Feature Validation
    test_feature_validation()
    
    print("\n🎯 Integration Test Summary:")
    print("✅ Time Series Feature Engineering: Implemented")
    print("✅ NLP Feature Engineering: Implemented")
    print("✅ Automated Feature Selection: Implemented")
    print("✅ Automated Feature Generation: Implemented")
    print("✅ Production Pipeline: Implemented")
    print("✅ Feature Validation: Implemented")
    
    print("\n🚀 All feature engineering components ready for production!")

if __name__ == "__main__":
    print("🎯 Day 26: Advanced Feature Engineering - Exercise")
    print("Building comprehensive feature engineering pipelines for FinTech ML systems")
    print()
    
    # TODO: Run the comprehensive test
    # run_comprehensive_feature_engineering_test()
    
    print("💡 Complete all TODO items to build production-ready feature engineering!")
    print("🔧 Focus on time series, NLP, and automated feature engineering techniques!")
    print("📊 Remember to handle missing data, outliers, and feature validation!")
