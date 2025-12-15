"""
Day 26: Advanced Feature Engineering - Time Series, NLP & Automated Selection - Complete Solution

Production-ready feature engineering implementation for FinTech ML systems.
Demonstrates enterprise-grade feature engineering across multiple domains.

This solution showcases:
- Sophisticated time series feature engineering with lag, rolling, and seasonality features
- Advanced NLP feature extraction with linguistic analysis and embeddings
- Ensemble feature selection combining multiple algorithms
- Automated feature generation with polynomial, ratio, and binning techniques
- Production-ready pipelines with validation and monitoring
- Comprehensive feature quality assessment and drift detection
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
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats
from itertools import combinations
import re
import json
import time
import logging

# NLP libraries
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
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
# =============================================================================
# PRODUCTION TIME SERIES FEATURE ENGINEERING
# =============================================================================

class AdvancedTimeSeriesFeatureEngineer:
    """Production-grade time series feature engineering for financial data"""
    
    def __init__(self):
        self.feature_names = []
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for feature engineering"""
        logger = logging.getLogger('time_series_features')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_temporal_features(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Create comprehensive temporal features from timestamp"""
        
        self.logger.info("Creating temporal features...")
        
        df_features = df.copy()
        df_features[timestamp_col] = pd.to_datetime(df_features[timestamp_col])
        
        # Basic temporal features
        df_features['year'] = df_features[timestamp_col].dt.year
        df_features['month'] = df_features[timestamp_col].dt.month
        df_features['day'] = df_features[timestamp_col].dt.day
        df_features['hour'] = df_features[timestamp_col].dt.hour
        df_features['minute'] = df_features[timestamp_col].dt.minute
        df_features['dayofweek'] = df_features[timestamp_col].dt.dayofweek
        df_features['dayofyear'] = df_features[timestamp_col].dt.dayofyear
        df_features['week'] = df_features[timestamp_col].dt.isocalendar().week
        df_features['quarter'] = df_features[timestamp_col].dt.quarter
        
        # Cyclical encoding for periodic features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['dayofweek_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7)
        df_features['dayofweek_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['dayofyear_sin'] = np.sin(2 * np.pi * df_features['dayofyear'] / 365)
        df_features['dayofyear_cos'] = np.cos(2 * np.pi * df_features['dayofyear'] / 365)
        
        # Business calendar features
        df_features['is_weekend'] = df_features['dayofweek'].isin([5, 6]).astype(int)
        df_features['is_month_start'] = df_features[timestamp_col].dt.is_month_start.astype(int)
        df_features['is_month_end'] = df_features[timestamp_col].dt.is_month_end.astype(int)
        df_features['is_quarter_start'] = df_features[timestamp_col].dt.is_quarter_start.astype(int)
        df_features['is_quarter_end'] = df_features[timestamp_col].dt.is_quarter_end.astype(int)
        df_features['is_year_start'] = df_features[timestamp_col].dt.is_year_start.astype(int)
        df_features['is_year_end'] = df_features[timestamp_col].dt.is_year_end.astype(int)
        
        # Business hours and special periods
        df_features['is_business_hour'] = (
            (df_features['hour'] >= 9) & (df_features['hour'] <= 17) & 
            (df_features['dayofweek'] < 5)
        ).astype(int)
        
        df_features['is_lunch_hour'] = (
            (df_features['hour'] >= 12) & (df_features['hour'] <= 13)
        ).astype(int)
        
        df_features['is_early_morning'] = (df_features['hour'] < 6).astype(int)
        df_features['is_late_night'] = (df_features['hour'] >= 22).astype(int)
        
        self.logger.info(f"Created {len(df_features.columns) - len(df.columns)} temporal features")
        
        return df_features
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                           entity_col: str, lags: List[int]) -> pd.DataFrame:
        """Create lag features for time series data"""
        
        self.logger.info(f"Creating lag features for {target_col} with lags: {lags}")
        
        df_features = df.copy()
        
        for lag in lags:
            lag_feature_name = f'{target_col}_lag_{lag}'
            df_features[lag_feature_name] = df_features.groupby(entity_col)[target_col].shift(lag)
            
            # Create lag difference features
            if lag > 1:
                diff_feature_name = f'{target_col}_lag_diff_{lag}'
                df_features[diff_feature_name] = (
                    df_features[target_col] - df_features[lag_feature_name]
                )
                
                # Create lag ratio features
                ratio_feature_name = f'{target_col}_lag_ratio_{lag}'
                df_features[ratio_feature_name] = (
                    df_features[target_col] / (df_features[lag_feature_name] + 1e-8)
                )
        
        self.logger.info(f"Created {len(lags) * 3} lag-based features")
        
        return df_features
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str, 
                              entity_col: str, windows: List[int]) -> pd.DataFrame:
        """Create rolling window statistical features"""
        
        self.logger.info(f"Creating rolling features for {target_col} with windows: {windows}")
        
        df_features = df.copy()
        
        for window in windows:
            # Basic rolling statistics
            rolling_group = df_features.groupby(entity_col)[target_col].rolling(window, min_periods=1)
            
            df_features[f'{target_col}_rolling_mean_{window}'] = rolling_group.mean().values
            df_features[f'{target_col}_rolling_std_{window}'] = rolling_group.std().values
            df_features[f'{target_col}_rolling_min_{window}'] = rolling_group.min().values
            df_features[f'{target_col}_rolling_max_{window}'] = rolling_group.max().values
            df_features[f'{target_col}_rolling_median_{window}'] = rolling_group.median().values
            
            # Rolling percentiles
            df_features[f'{target_col}_rolling_q25_{window}'] = rolling_group.quantile(0.25).values
            df_features[f'{target_col}_rolling_q75_{window}'] = rolling_group.quantile(0.75).values
            
            # Rolling range and IQR
            df_features[f'{target_col}_rolling_range_{window}'] = (
                df_features[f'{target_col}_rolling_max_{window}'] - 
                df_features[f'{target_col}_rolling_min_{window}']
            )
            
            df_features[f'{target_col}_rolling_iqr_{window}'] = (
                df_features[f'{target_col}_rolling_q75_{window}'] - 
                df_features[f'{target_col}_rolling_q25_{window}']
            )
            
            # Rolling trend features
            rolling_mean = df_features[f'{target_col}_rolling_mean_{window}']
            rolling_std = df_features[f'{target_col}_rolling_std_{window}']
            
            df_features[f'{target_col}_rolling_zscore_{window}'] = (
                (df_features[target_col] - rolling_mean) / (rolling_std + 1e-8)
            )
            
            # Rolling momentum (rate of change)
            df_features[f'{target_col}_rolling_momentum_{window}'] = (
                df_features[target_col] / (rolling_mean + 1e-8) - 1
            )
            
            # Exponential moving averages
            for alpha in [0.1, 0.3, 0.5]:
                ema_col = f'{target_col}_ema_{alpha}_{window}'
                df_features[ema_col] = (
                    df_features.groupby(entity_col)[target_col]
                    .ewm(alpha=alpha, min_periods=1)
                    .mean()
                    .values
                )
        
        self.logger.info(f"Created {len(windows) * 13} rolling window features")
        
        return df_features
    
    def create_seasonality_features(self, df: pd.DataFrame, target_col: str, 
                                  seasonal_periods: List[int]) -> pd.DataFrame:
        """Create seasonality features using Fourier transforms"""
        
        self.logger.info(f"Creating seasonality features for periods: {seasonal_periods}")
        
        df_features = df.copy()
        
        for period in seasonal_periods:
            # Fourier features for seasonality
            for k in range(1, min(period//2, 10)):  # Limit to 10 harmonics
                df_features[f'{target_col}_fourier_sin_{period}_{k}'] = np.sin(
                    2 * np.pi * k * np.arange(len(df_features)) / period
                )
                df_features[f'{target_col}_fourier_cos_{period}_{k}'] = np.cos(
                    2 * np.pi * k * np.arange(len(df_features)) / period
                )
        
        # Trend features using polynomial fitting
        x = np.arange(len(df_features))
        for degree in [1, 2, 3]:
            try:
                coeffs = np.polyfit(x, df_features[target_col].fillna(method='ffill'), degree)
                trend = np.polyval(coeffs, x)
                df_features[f'{target_col}_trend_poly_{degree}'] = trend
                df_features[f'{target_col}_detrended_{degree}'] = df_features[target_col] - trend
            except:
                # Handle cases where polynomial fitting fails
                df_features[f'{target_col}_trend_poly_{degree}'] = 0
                df_features[f'{target_col}_detrended_{degree}'] = df_features[target_col]
        
        fourier_features = sum(min(p//2, 10) * 2 for p in seasonal_periods)
        self.logger.info(f"Created {fourier_features + 6} seasonality and trend features")
        
        return df_features
# =============================================================================
# ADVANCED NLP FEATURE ENGINEERING
# =============================================================================

class ProductionNLPFeatureEngineer:
    """Production-grade NLP feature engineering for customer feedback"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        else:
            self.lemmatizer = None
            self.stop_words = set()
        
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                self.logger.warning("spaCy model not found. Some features will be unavailable.")
        
        # Initialize TF-IDF and clustering components
        self.tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words='english')
        self.svd = TruncatedSVD(n_components=100)
        self.kmeans = KMeans(n_clusters=20, random_state=42)
        
        # Sentiment word lists (in production, use proper sentiment libraries)
        self.positive_words = {
            'excellent', 'amazing', 'outstanding', 'fantastic', 'great', 'good', 'wonderful',
            'perfect', 'love', 'best', 'awesome', 'brilliant', 'superb', 'magnificent',
            'exceptional', 'remarkable', 'impressive', 'delightful', 'satisfying', 'pleased'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'disappointing',
            'frustrating', 'annoying', 'useless', 'pathetic', 'disgusting', 'appalling',
            'dreadful', 'atrocious', 'abysmal', 'deplorable', 'inadequate', 'unsatisfactory'
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for NLP feature engineering"""
        logger = logging.getLogger('nlp_features')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning and preprocessing"""
        
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive linguistic features from text"""
        
        if not text or pd.isna(text):
            return self._empty_linguistic_features()
        
        features = {}
        
        # Basic length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['comma_count'] = text.count(',')
        features['period_count'] = text.count('.')
        features['punctuation_ratio'] = sum([features['exclamation_count'], 
                                           features['question_count'], 
                                           features['comma_count'], 
                                           features['period_count']]) / max(features['char_count'], 1)
        
        # Capitalization features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(features['char_count'], 1)
        features['title_case_words'] = sum(1 for word in text.split() if word.istitle())
        
        # Advanced linguistic features using spaCy
        if self.nlp:
            doc = self.nlp(text)
            
            # Part-of-speech features
            pos_counts = {}
            for token in doc:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
            total_tokens = len(doc)
            if total_tokens > 0:
                features['noun_ratio'] = pos_counts.get('NOUN', 0) / total_tokens
                features['verb_ratio'] = pos_counts.get('VERB', 0) / total_tokens
                features['adj_ratio'] = pos_counts.get('ADJ', 0) / total_tokens
                features['adv_ratio'] = pos_counts.get('ADV', 0) / total_tokens
                features['pronoun_ratio'] = pos_counts.get('PRON', 0) / total_tokens
            else:
                features.update({
                    'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0, 
                    'adv_ratio': 0, 'pronoun_ratio': 0
                })
            
            # Named entity features
            features['entity_count'] = len(doc.ents)
            features['person_count'] = len([ent for ent in doc.ents if ent.label_ == 'PERSON'])
            features['org_count'] = len([ent for ent in doc.ents if ent.label_ == 'ORG'])
            features['location_count'] = len([ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC']])
            features['money_count'] = len([ent for ent in doc.ents if ent.label_ == 'MONEY'])
            features['date_count'] = len([ent for ent in doc.ents if ent.label_ == 'DATE'])
            
            # Dependency features
            features['dependency_depth'] = max([len(list(token.ancestors)) for token in doc], default=0)
            
        else:
            # Fallback features when spaCy is not available
            features.update({
                'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0, 'adv_ratio': 0,
                'pronoun_ratio': 0, 'entity_count': 0, 'person_count': 0,
                'org_count': 0, 'location_count': 0, 'money_count': 0,
                'date_count': 0, 'dependency_depth': 0
            })
        
        # Sentiment and readability
        features['sentiment_polarity'] = self._calculate_sentiment(text)
        features['sentiment_intensity'] = abs(features['sentiment_polarity'])
        features['readability_score'] = self._calculate_readability(text)
        
        # Lexical diversity
        unique_words = len(set(text.split()))
        total_words = len(text.split())
        features['lexical_diversity'] = unique_words / max(total_words, 1)
        
        # Stop word ratio
        if self.stop_words:
            stop_word_count = sum(1 for word in text.split() if word.lower() in self.stop_words)
            features['stop_word_ratio'] = stop_word_count / max(total_words, 1)
        else:
            features['stop_word_ratio'] = 0
        
        return features
    
    def _empty_linguistic_features(self) -> Dict[str, float]:
        """Return empty feature dict for null text"""
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0, 'avg_word_length': 0,
            'exclamation_count': 0, 'question_count': 0, 'comma_count': 0, 'period_count': 0,
            'punctuation_ratio': 0, 'uppercase_ratio': 0, 'title_case_words': 0,
            'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0, 'adv_ratio': 0, 'pronoun_ratio': 0,
            'entity_count': 0, 'person_count': 0, 'org_count': 0, 'location_count': 0,
            'money_count': 0, 'date_count': 0, 'dependency_depth': 0,
            'sentiment_polarity': 0, 'sentiment_intensity': 0, 'readability_score': 0,
            'lexical_diversity': 0, 'stop_word_ratio': 0
        }
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment polarity score"""
        
        words = set(text.lower().split())
        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (Flesch Reading Ease approximation)"""
        
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * words / sentences) - (84.6 * syllables / words)
        return max(0, min(100, score))  # Clamp between 0-100
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)  # Every word has at least 1 syllable
    
    def create_tfidf_features(self, texts: List[str], max_features: int = 100) -> np.ndarray:
        """Create TF-IDF features with dimensionality reduction"""
        
        self.logger.info(f"Creating TF-IDF features with max_features={max_features}")
        
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Fit TF-IDF
        tfidf_matrix = self.tfidf.fit_transform(cleaned_texts)
        
        # Apply SVD for dimensionality reduction
        reduced_features = self.svd.fit_transform(tfidf_matrix)
        
        # Ensure we have the right number of features
        if reduced_features.shape[1] > max_features:
            reduced_features = reduced_features[:, :max_features]
        elif reduced_features.shape[1] < max_features:
            # Pad with zeros if we have fewer features
            padding = np.zeros((reduced_features.shape[0], max_features - reduced_features.shape[1]))
            reduced_features = np.hstack([reduced_features, padding])
        
        self.logger.info(f"Created TF-IDF features with shape: {reduced_features.shape}")
        
        return reduced_features
    
    def create_topic_features(self, texts: List[str]) -> np.ndarray:
        """Create topic-based features using clustering"""
        
        self.logger.info("Creating topic-based features...")
        
        # Get TF-IDF features
        tfidf_features = self.create_tfidf_features(texts, max_features=100)
        
        # Cluster documents
        cluster_labels = self.kmeans.fit_predict(tfidf_features)
        
        # Create one-hot encoded cluster features
        n_clusters = self.kmeans.n_clusters
        topic_features = np.zeros((len(texts), n_clusters))
        
        for i, label in enumerate(cluster_labels):
            topic_features[i, label] = 1
        
        # Add cluster distances as features
        cluster_distances = self.kmeans.transform(tfidf_features)
        
        combined_features = np.concatenate([topic_features, cluster_distances], axis=1)
        
        self.logger.info(f"Created topic features with shape: {combined_features.shape}")
        
        return combined_features
# =============================================================================
# ENSEMBLE FEATURE SELECTION
# =============================================================================

class EnsembleFeatureSelector:
    """Advanced ensemble feature selection with multiple algorithms"""
    
    def __init__(self, task_type: str = 'classification'):
        self.task_type = task_type
        self.selected_features = {}
        self.feature_scores = {}
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for feature selection"""
        logger = logging.getLogger('feature_selection')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> List[str]:
        """Select features using univariate statistical tests"""
        
        self.logger.info(f"Running univariate selection with k={k}")
        
        if self.task_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))
        
        # Handle missing values
        X_filled = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        try:
            X_selected = selector.fit_transform(X_filled, y)
            selected_mask = selector.get_support()
            selected_features = X.columns[selected_mask].tolist()
            
            # Store feature scores
            self.feature_scores['univariate'] = dict(zip(X.columns, selector.scores_))
            
            self.logger.info(f"Univariate selection: {len(selected_features)} features selected")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Univariate selection failed: {e}")
            return X.columns.tolist()[:k]
    
    def mutual_information_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> List[str]:
        """Select features using mutual information"""
        
        self.logger.info(f"Running mutual information selection with k={k}")
        
        if self.task_type == 'classification':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(X.columns)))
        else:
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(X.columns)))
        
        # Handle missing values and ensure numeric data
        X_filled = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        X_numeric = X_filled.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            self.logger.warning("No numeric features for mutual information selection")
            return X.columns.tolist()[:k]
        
        try:
            X_selected = selector.fit_transform(X_numeric, y)
            selected_mask = selector.get_support()
            selected_features = X_numeric.columns[selected_mask].tolist()
            
            # Store feature scores
            self.feature_scores['mutual_info'] = dict(zip(X_numeric.columns, selector.scores_))
            
            self.logger.info(f"Mutual information selection: {len(selected_features)} features selected")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Mutual information selection failed: {e}")
            return X_numeric.columns.tolist()[:k]
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, n_features: int = 50) -> List[str]:
        """Select features using recursive feature elimination"""
        
        self.logger.info(f"Running RFE with n_features={n_features}")
        
        if self.task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Handle missing values and ensure numeric data
        X_filled = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        X_numeric = X_filled.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            self.logger.warning("No numeric features for RFE")
            return X.columns.tolist()[:n_features]
        
        try:
            selector = RFE(estimator, n_features_to_select=min(n_features, len(X_numeric.columns)))
            X_selected = selector.fit_transform(X_numeric, y)
            
            selected_mask = selector.get_support()
            selected_features = X_numeric.columns[selected_mask].tolist()
            
            # Store feature rankings
            self.feature_scores['rfe_ranking'] = dict(zip(X_numeric.columns, selector.ranking_))
            
            self.logger.info(f"RFE selection: {len(selected_features)} features selected")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"RFE selection failed: {e}")
            return X_numeric.columns.tolist()[:n_features]
    
    def lasso_selection(self, X: pd.DataFrame, y: pd.Series, alpha: float = None) -> List[str]:
        """Select features using Lasso regularization"""
        
        self.logger.info("Running Lasso feature selection")
        
        # Handle missing values and ensure numeric data
        X_filled = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        X_numeric = X_filled.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            self.logger.warning("No numeric features for Lasso selection")
            return []
        
        try:
            if alpha is None:
                # Use cross-validation to find optimal alpha
                lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
            else:
                lasso = Lasso(alpha=alpha, random_state=42, max_iter=1000)
            
            # Scale features for Lasso
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_numeric)
            
            lasso.fit(X_scaled, y)
            
            # Select features with non-zero coefficients
            selected_mask = np.abs(lasso.coef_) > 1e-5
            selected_features = X_numeric.columns[selected_mask].tolist()
            
            self.feature_scores['lasso_coef'] = dict(zip(X_numeric.columns, np.abs(lasso.coef_)))
            
            self.logger.info(f"Lasso selection: {len(selected_features)} features selected")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Lasso selection failed: {e}")
            return X_numeric.columns.tolist()[:50]
    
    def tree_based_selection(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.01) -> List[str]:
        """Select features using tree-based feature importance"""
        
        self.logger.info(f"Running tree-based selection with threshold={threshold}")
        
        if self.task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Handle missing values and ensure numeric data
        X_filled = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        X_numeric = X_filled.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            self.logger.warning("No numeric features for tree-based selection")
            return []
        
        try:
            estimator.fit(X_numeric, y)
            
            # Get feature importances
            importances = estimator.feature_importances_
            
            # Select features above threshold
            selected_mask = importances > threshold
            selected_features = X_numeric.columns[selected_mask].tolist()
            
            self.feature_scores['tree_importance'] = dict(zip(X_numeric.columns, importances))
            
            self.logger.info(f"Tree-based selection: {len(selected_features)} features selected")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Tree-based selection failed: {e}")
            return X_numeric.columns.tolist()[:50]
    
    def ensemble_selection(self, X: pd.DataFrame, y: pd.Series, 
                          methods: List[str] = None, 
                          voting_threshold: float = 0.4) -> List[str]:
        """Combine multiple selection methods using ensemble voting"""
        
        self.logger.info(f"Running ensemble selection with threshold={voting_threshold}")
        
        if methods is None:
            methods = ['univariate', 'mutual_info', 'rfe', 'lasso', 'tree_based']
        
        all_selections = {}
        
        # Run each selection method
        if 'univariate' in methods:
            try:
                all_selections['univariate'] = set(self.univariate_selection(X, y))
            except Exception as e:
                self.logger.error(f"Univariate selection failed in ensemble: {e}")
        
        if 'mutual_info' in methods:
            try:
                all_selections['mutual_info'] = set(self.mutual_information_selection(X, y))
            except Exception as e:
                self.logger.error(f"Mutual info selection failed in ensemble: {e}")
        
        if 'rfe' in methods:
            try:
                all_selections['rfe'] = set(self.recursive_feature_elimination(X, y))
            except Exception as e:
                self.logger.error(f"RFE selection failed in ensemble: {e}")
        
        if 'lasso' in methods:
            try:
                all_selections['lasso'] = set(self.lasso_selection(X, y))
            except Exception as e:
                self.logger.error(f"Lasso selection failed in ensemble: {e}")
        
        if 'tree_based' in methods:
            try:
                all_selections['tree_based'] = set(self.tree_based_selection(X, y))
            except Exception as e:
                self.logger.error(f"Tree-based selection failed in ensemble: {e}")
        
        if not all_selections:
            self.logger.warning("No selection methods succeeded")
            return X.columns.tolist()[:50]
        
        # Count votes for each feature
        feature_votes = {}
        for feature in X.columns:
            votes = sum(1 for selection in all_selections.values() if feature in selection)
            feature_votes[feature] = votes / len(all_selections)
        
        # Select features that meet voting threshold
        selected_features = [
            feature for feature, vote_ratio in feature_votes.items()
            if vote_ratio >= voting_threshold
        ]
        
        self.feature_scores['ensemble_votes'] = feature_votes
        
        self.logger.info(f"Ensemble selection: {len(selected_features)} features selected from {len(X.columns)} total")
        
        return selected_features
# =============================================================================
# AUTOMATED FEATURE GENERATION
# =============================================================================

class AdvancedFeatureGenerator:
    """Automatically generate sophisticated features from existing ones"""
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.generated_features = []
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for feature generation"""
        logger = logging.getLogger('feature_generation')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_polynomial_features(self, X: pd.DataFrame, degree: int = 2, 
                                   interaction_only: bool = False) -> pd.DataFrame:
        """Generate polynomial and interaction features"""
        
        self.logger.info(f"Generating polynomial features with degree={degree}")
        
        # Only use numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            self.logger.warning("No numeric columns for polynomial features")
            return X
        
        # Limit number of input features to avoid explosion
        if len(numeric_cols) > 10:
            self.logger.info(f"Limiting to first 10 numeric columns to avoid feature explosion")
            numeric_cols = numeric_cols[:10]
        
        X_numeric = X[numeric_cols]
        
        poly = PolynomialFeatures(
            degree=degree, 
            interaction_only=interaction_only,
            include_bias=False
        )
        
        X_poly = poly.fit_transform(X_numeric)
        feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Limit number of features
        if len(feature_names) > self.max_features:
            n_original = len(numeric_cols)
            n_new = min(self.max_features - n_original, len(feature_names) - n_original)
            
            X_poly = X_poly[:, :n_original + n_new]
            feature_names = feature_names[:n_original + n_new]
        
        # Create result DataFrame
        X_result = X.copy()
        
        # Add new polynomial features (skip original features)
        for i, name in enumerate(feature_names):
            if name not in X.columns:  # Only add new features
                X_result[name] = X_poly[:, i]
        
        self.logger.info(f"Generated {len(X_result.columns) - len(X.columns)} polynomial features")
        
        return X_result
    
    def generate_ratio_features(self, X: pd.DataFrame, max_ratios: int = 100) -> pd.DataFrame:
        """Generate ratio features between numeric columns"""
        
        self.logger.info("Generating ratio features...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            self.logger.warning("Need at least 2 numeric columns for ratio features")
            return X
        
        X_ratios = X.copy()
        ratio_count = 0
        
        # Generate ratios for all pairs of numeric columns
        for col1, col2 in combinations(numeric_cols, 2):
            if ratio_count >= max_ratios:
                break
            
            # Avoid division by zero
            denominator = X[col2].replace(0, np.nan)
            
            # Forward ratio
            ratio_name = f'{col1}_div_{col2}'
            X_ratios[ratio_name] = X[col1] / denominator
            ratio_count += 1
            
            if ratio_count >= max_ratios:
                break
            
            # Inverse ratio
            numerator = X[col1].replace(0, np.nan)
            inverse_ratio_name = f'{col2}_div_{col1}'
            X_ratios[inverse_ratio_name] = X[col2] / numerator
            ratio_count += 1
        
        self.logger.info(f"Generated {ratio_count} ratio features")
        
        return X_ratios
    
    def generate_aggregation_features(self, X: pd.DataFrame, 
                                    group_cols: List[str],
                                    agg_cols: List[str],
                                    agg_funcs: List[str] = None) -> pd.DataFrame:
        """Generate aggregation features based on grouping columns"""
        
        if agg_funcs is None:
            agg_funcs = ['mean', 'std', 'min', 'max', 'count', 'median']
        
        self.logger.info(f"Generating aggregation features for {len(group_cols)} group columns")
        
        X_agg = X.copy()
        feature_count = 0
        
        for group_col in group_cols:
            if group_col not in X.columns:
                continue
                
            for agg_col in agg_cols:
                if agg_col not in X.columns or group_col == agg_col:
                    continue
                
                for func in agg_funcs:
                    try:
                        # Calculate group statistics
                        group_stats = X.groupby(group_col)[agg_col].agg(func)
                        
                        # Map back to original dataframe
                        feature_name = f'{agg_col}_{func}_by_{group_col}'
                        X_agg[feature_name] = X[group_col].map(group_stats)
                        feature_count += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to create aggregation feature {feature_name}: {e}")
        
        self.logger.info(f"Generated {feature_count} aggregation features")
        
        return X_agg
    
    def generate_binning_features(self, X: pd.DataFrame, 
                                numeric_cols: List[str] = None,
                                n_bins: int = 5) -> pd.DataFrame:
        """Generate binning features for numeric columns"""
        
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.logger.info(f"Generating binning features for {len(numeric_cols)} columns")
        
        X_binned = X.copy()
        feature_count = 0
        
        for col in numeric_cols:
            if col not in X.columns:
                continue
            
            try:
                # Equal-width binning
                X_binned[f'{col}_bin_equal'] = pd.cut(X[col], bins=n_bins, labels=False)
                feature_count += 1
                
                # Equal-frequency binning (quantiles)
                X_binned[f'{col}_bin_quantile'] = pd.qcut(
                    X[col], q=n_bins, labels=False, duplicates='drop'
                )
                feature_count += 1
                
                # Binary features for each bin (only for equal-width)
                for bin_val in range(n_bins):
                    bin_feature_name = f'{col}_is_bin_{bin_val}'
                    X_binned[bin_feature_name] = (
                        X_binned[f'{col}_bin_equal'] == bin_val
                    ).astype(int)
                    feature_count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to create binning features for {col}: {e}")
        
        self.logger.info(f"Generated {feature_count} binning features")
        
        return X_binned
    
    def generate_statistical_features(self, X: pd.DataFrame, 
                                    numeric_cols: List[str] = None) -> pd.DataFrame:
        """Generate statistical transformation features"""
        
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.logger.info(f"Generating statistical features for {len(numeric_cols)} columns")
        
        X_stats = X.copy()
        feature_count = 0
        
        for col in numeric_cols:
            if col not in X.columns:
                continue
            
            try:
                # Log transformation (handle negative values)
                X_positive = X[col] - X[col].min() + 1
                X_stats[f'{col}_log'] = np.log(X_positive)
                feature_count += 1
                
                # Square root transformation
                X_stats[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
                feature_count += 1
                
                # Square transformation
                X_stats[f'{col}_square'] = X[col] ** 2
                feature_count += 1
                
                # Z-score normalization
                X_stats[f'{col}_zscore'] = (X[col] - X[col].mean()) / (X[col].std() + 1e-8)
                feature_count += 1
                
                # Rank transformation
                X_stats[f'{col}_rank'] = X[col].rank()
                feature_count += 1
                
                # Percentile transformation
                X_stats[f'{col}_percentile'] = X[col].rank(pct=True)
                feature_count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to create statistical features for {col}: {e}")
        
        self.logger.info(f"Generated {feature_count} statistical features")
        
        return X_stats
    
    def generate_all_features(self, X: pd.DataFrame, y: pd.Series = None,
                            include_polynomial: bool = True,
                            include_ratios: bool = True,
                            include_binning: bool = True,
                            include_statistical: bool = True) -> pd.DataFrame:
        """Generate all types of features with intelligent limits"""
        
        self.logger.info("Generating comprehensive feature set...")
        
        X_enhanced = X.copy()
        original_feature_count = len(X.columns)
        
        # Get numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if include_polynomial and len(numeric_cols) > 0:
            # Generate polynomial features (limited to avoid explosion)
            if len(numeric_cols) <= 8:  # Only for small number of features
                X_enhanced = self.generate_polynomial_features(
                    X_enhanced, degree=2, interaction_only=True
                )
        
        if include_ratios and len(numeric_cols) > 1:
            X_enhanced = self.generate_ratio_features(X_enhanced, max_ratios=50)
        
        if include_binning and len(numeric_cols) > 0:
            X_enhanced = self.generate_binning_features(X_enhanced, numeric_cols[:10])
        
        if include_statistical and len(numeric_cols) > 0:
            X_enhanced = self.generate_statistical_features(X_enhanced, numeric_cols[:10])
        
        final_feature_count = len(X_enhanced.columns)
        new_features = final_feature_count - original_feature_count
        
        self.logger.info(f"Generated {new_features} new features (total: {final_feature_count})")
        
        return X_enhanced
# =============================================================================
# PRODUCTION FEATURE ENGINEERING PIPELINE
# =============================================================================

class ProductionFeatureEngineeringPipeline:
    """Production-ready feature engineering pipeline with comprehensive capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = None
        self.feature_names = None
        self.is_fitted = False
        self.logger = self._setup_logging()
        
        # Initialize components
        self.time_series_engineer = AdvancedTimeSeriesFeatureEngineer()
        self.nlp_engineer = ProductionNLPFeatureEngineer()
        self.feature_selector = EnsembleFeatureSelector(config.get('task_type', 'classification'))
        self.feature_generator = AdvancedFeatureGenerator(config.get('max_features', 1000))
        self.validator = FeatureQualityValidator()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        logger = logging.getLogger('feature_pipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def build_pipeline(self, X: pd.DataFrame, y: pd.Series = None) -> Pipeline:
        """Build comprehensive feature engineering pipeline"""
        
        self.logger.info("Building production feature engineering pipeline...")
        
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_features = X.select_dtypes(include=['datetime64']).columns.tolist()
        
        self.logger.info(f"Identified {len(numeric_features)} numeric, {len(categorical_features)} categorical, "
                        f"{len(datetime_features)} datetime features")
        
        # Numeric pipeline with advanced preprocessing
        numeric_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('outlier_clipper', OutlierClipper(quantile_range=(0.01, 0.99))),
            ('scaler', RobustScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False, max_categories=20))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('numeric', numeric_pipeline, numeric_features),
            ('categorical', categorical_pipeline, categorical_features)
        ], remainder='drop')
        
        # Full pipeline with feature generation and selection
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_generator', FeatureGenerationTransformer(self.feature_generator)),
            ('feature_selector', FeatureSelectionTransformer(
                self.feature_selector, 
                max_features=self.config.get('max_features', 100)
            ))
        ])
        
        return self.pipeline
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the feature engineering pipeline"""
        
        self.logger.info("Fitting feature engineering pipeline...")
        
        if self.pipeline is None:
            self.build_pipeline(X, y)
        
        # Validate input data
        validation_results = self.validator.validate_feature_quality(X)
        if validation_results['quality_score'] < 0.7:
            self.logger.warning(f"Input data quality score: {validation_results['quality_score']:.3f}")
        
        # Fit pipeline
        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        # Store feature names after fitting
        try:
            self.feature_names = self._get_feature_names()
        except:
            self.feature_names = None
        
        self.logger.info("Pipeline fitting completed")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted pipeline"""
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        self.logger.info("Transforming data...")
        
        X_transformed = self.pipeline.transform(X)
        
        self.logger.info(f"Transformed data shape: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fit pipeline and transform data"""
        
        return self.fit(X, y).transform(X)
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names from fitted pipeline"""
        
        try:
            # This is a simplified version - in practice, you'd need to track
            # feature names through each transformation step
            if hasattr(self.pipeline, 'feature_names_in_'):
                return list(self.pipeline.feature_names_in_)
            else:
                return [f'feature_{i}' for i in range(self.pipeline.transform(pd.DataFrame()).shape[1])]
        except:
            return None


# =============================================================================
# CUSTOM TRANSFORMERS FOR PIPELINE
# =============================================================================

class OutlierClipper:
    """Custom transformer for outlier clipping"""
    
    def __init__(self, quantile_range=(0.01, 0.99)):
        self.quantile_range = quantile_range
        self.clip_values = {}
    
    def fit(self, X, y=None):
        for i in range(X.shape[1]):
            lower = np.quantile(X[:, i], self.quantile_range[0])
            upper = np.quantile(X[:, i], self.quantile_range[1])
            self.clip_values[i] = (lower, upper)
        return self
    
    def transform(self, X):
        X_clipped = X.copy()
        for i, (lower, upper) in self.clip_values.items():
            X_clipped[:, i] = np.clip(X_clipped[:, i], lower, upper)
        return X_clipped


class FeatureGenerationTransformer:
    """Custom transformer for feature generation"""
    
    def __init__(self, feature_generator: AdvancedFeatureGenerator):
        self.feature_generator = feature_generator
        self.feature_names = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        
        # Generate features
        X_enhanced = self.feature_generator.generate_all_features(
            X_df, 
            include_polynomial=False,  # Skip polynomial to avoid explosion
            include_ratios=True,
            include_binning=True,
            include_statistical=True
        )
        
        return X_enhanced.values


class FeatureSelectionTransformer:
    """Custom transformer for feature selection"""
    
    def __init__(self, feature_selector: EnsembleFeatureSelector, max_features: int = 100):
        self.feature_selector = feature_selector
        self.max_features = max_features
        self.selected_indices = None
    
    def fit(self, X, y=None):
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        
        if y is not None:
            # Select features using ensemble method
            selected_features = self.feature_selector.ensemble_selection(
                X_df, y, voting_threshold=0.3
            )
            
            # Limit to max_features
            if len(selected_features) > self.max_features:
                selected_features = selected_features[:self.max_features]
            
            # Get indices of selected features
            self.selected_indices = [X_df.columns.get_loc(feat) for feat in selected_features 
                                   if feat in X_df.columns]
        else:
            # If no target, select first max_features
            self.selected_indices = list(range(min(self.max_features, X.shape[1])))
        
        return self
    
    def transform(self, X):
        if self.selected_indices is None:
            return X
        
        if isinstance(X, np.ndarray):
            return X[:, self.selected_indices]
        else:
            return X.iloc[:, self.selected_indices].values


# =============================================================================
# FEATURE QUALITY VALIDATION AND MONITORING
# =============================================================================

class FeatureQualityValidator:
    """Comprehensive feature quality validation and monitoring"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.validation_history = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation"""
        logger = logging.getLogger('feature_validator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_feature_quality(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive feature quality validation"""
        
        self.logger.info("Running comprehensive feature quality validation...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(X.columns),
            'total_samples': len(X),
            'missing_value_features': 0,
            'high_cardinality_features': 0,
            'constant_features': 0,
            'duplicate_features': 0,
            'quality_score': 0.0,
            'issues': [],
            'feature_stats': {}
        }
        
        # Check for various quality issues
        for col in X.columns:
            feature_stats = {}
            
            # Missing values
            missing_count = X[col].isnull().sum()
            missing_pct = missing_count / len(X)
            feature_stats['missing_pct'] = missing_pct
            
            if missing_pct > 0.5:
                results['missing_value_features'] += 1
                results['issues'].append(f'{col}: High missing values ({missing_pct:.2%})')
            
            # Constant features
            unique_count = X[col].nunique()
            feature_stats['unique_count'] = unique_count
            
            if unique_count == 1:
                results['constant_features'] += 1
                results['issues'].append(f'{col}: Constant feature')
            
            # High cardinality for categorical features
            if X[col].dtype == 'object' and unique_count > 100:
                results['high_cardinality_features'] += 1
                results['issues'].append(f'{col}: High cardinality ({unique_count} unique values)')
            
            # Data type consistency
            if X[col].dtype == 'object':
                try:
                    pd.to_numeric(X[col], errors='raise')
                    results['issues'].append(f'{col}: Numeric data stored as object')
                except:
                    pass
            
            results['feature_stats'][col] = feature_stats
        
        # Check for duplicate features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = X[numeric_cols].corr().abs()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > 0.95:
                        col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, correlation_matrix.iloc[i, j]))
            
            results['duplicate_features'] = len(high_corr_pairs)
            for col1, col2, corr in high_corr_pairs:
                results['issues'].append(f'{col1} and {col2}: High correlation ({corr:.3f})')
        
        # Calculate overall quality score
        total_issues = (results['missing_value_features'] + 
                       results['constant_features'] + 
                       results['high_cardinality_features'] + 
                       results['duplicate_features'])
        
        results['quality_score'] = max(0, 1 - (total_issues / max(results['total_features'], 1)))
        
        # Store validation history
        self.validation_history.append(results)
        
        self.logger.info(f"Validation completed. Quality score: {results['quality_score']:.3f}")
        
        return results
    
    def detect_feature_drift(self, X_reference: pd.DataFrame, 
                           X_current: pd.DataFrame,
                           significance_level: float = 0.05) -> Dict[str, Any]:
        """Detect feature drift between reference and current data"""
        
        self.logger.info("Detecting feature drift...")
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'features_with_drift': [],
            'drift_scores': {},
            'p_values': {},
            'overall_drift_score': 0.0,
            'significance_level': significance_level
        }
        
        # Find common features
        common_features = set(X_reference.columns) & set(X_current.columns)
        
        drift_scores = []
        
        for feature in common_features:
            try:
                ref_data = X_reference[feature].dropna()
                curr_data = X_current[feature].dropna()
                
                if len(ref_data) == 0 or len(curr_data) == 0:
                    continue
                
                if X_reference[feature].dtype in ['int64', 'float64']:
                    # Use Kolmogorov-Smirnov test for numeric features
                    ks_stat, p_value = stats.ks_2samp(ref_data, curr_data)
                    drift_score = ks_stat
                    
                else:
                    # Use Chi-square test for categorical features
                    ref_counts = ref_data.value_counts()
                    curr_counts = curr_data.value_counts()
                    
                    # Align categories
                    all_categories = set(ref_counts.index) | set(curr_counts.index)
                    ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                    curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                    
                    if sum(ref_aligned) > 0 and sum(curr_aligned) > 0:
                        chi2_stat, p_value = stats.chisquare(curr_aligned, ref_aligned)
                        drift_score = chi2_stat / (sum(ref_aligned) + sum(curr_aligned))
                    else:
                        drift_score, p_value = 0.0, 1.0
                
                drift_results['drift_scores'][feature] = drift_score
                drift_results['p_values'][feature] = p_value
                drift_scores.append(drift_score)
                
                if p_value < significance_level:
                    drift_results['features_with_drift'].append(feature)
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate drift for feature {feature}: {e}")
        
        # Calculate overall drift score
        if drift_scores:
            drift_results['overall_drift_score'] = np.mean(drift_scores)
        
        self.logger.info(f"Drift detection completed. {len(drift_results['features_with_drift'])} features with significant drift")
        
        return drift_results
# =============================================================================
# COMPREHENSIVE INTEGRATION AND TESTING
# =============================================================================

def create_comprehensive_fintech_dataset(n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.Series]:
    """Create comprehensive synthetic FinTech dataset for testing"""
    
    np.random.seed(42)
    
    # Customer demographics
    customer_ids = np.random.randint(1, 1001, n_samples)
    ages = np.random.normal(35, 12, n_samples).clip(18, 80).astype(int)
    incomes = np.random.lognormal(10.5, 0.8, n_samples)
    credit_scores = np.random.normal(650, 100, n_samples).clip(300, 850).astype(int)
    
    # Account information
    account_types = np.random.choice(['checking', 'savings', 'premium', 'business'], n_samples, 
                                   p=[0.4, 0.3, 0.2, 0.1])
    account_ages = np.random.exponential(2, n_samples).clip(0, 20)
    
    # Transaction data with time series
    base_date = datetime(2023, 1, 1)
    timestamps = [base_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Transaction amounts with patterns
    hour_of_day = np.array([ts.hour for ts in timestamps])
    day_of_week = np.array([ts.weekday() for ts in timestamps])
    
    # Add seasonality and trends to transaction amounts
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24)
    weekly_factor = 1 + 0.2 * np.sin(2 * np.pi * day_of_week / 7)
    
    base_amounts = np.random.lognormal(4, 1.5, n_samples)
    transaction_amounts = base_amounts * seasonal_factor * weekly_factor
    
    # Customer feedback text
    feedback_templates = [
        "The service is {} and the app is {}. {} experience overall.",
        "I {} this bank. The {} is {} and staff is {}.",
        "Transaction was {} and the process was {}. {} recommend.",
        "Account management is {} and fees are {}. {} satisfied.",
        "Mobile banking is {} and customer service is {}. {} experience."
    ]
    
    positive_words = ['excellent', 'great', 'amazing', 'wonderful', 'fantastic']
    negative_words = ['terrible', 'awful', 'horrible', 'disappointing', 'frustrating']
    neutral_words = ['okay', 'average', 'decent', 'acceptable', 'standard']
    
    feedback_texts = []
    for i in range(n_samples):
        template = np.random.choice(feedback_templates)
        
        # Create sentiment-consistent feedback
        sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])
        
        if sentiment == 'positive':
            words = np.random.choice(positive_words, 3)
            overall = np.random.choice(['Great', 'Excellent', 'Highly'])
        elif sentiment == 'negative':
            words = np.random.choice(negative_words, 3)
            overall = np.random.choice(['Poor', 'Terrible', 'Would not'])
        else:
            words = np.random.choice(neutral_words, 3)
            overall = np.random.choice(['Average', 'Okay', 'Might'])
        
        feedback = template.format(words[0], words[1], overall)
        feedback_texts.append(feedback)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'timestamp': timestamps,
        'transaction_amount': transaction_amounts,
        'age': ages,
        'income': incomes,
        'credit_score': credit_scores,
        'account_type': account_types,
        'account_age_years': account_ages,
        'feedback_text': feedback_texts
    })
    
    # Create target variable (fraud detection)
    # Higher risk for: unusual amounts, off-hours, low credit scores
    risk_score = (
        (transaction_amounts > np.percentile(transaction_amounts, 95)).astype(int) * 0.3 +
        ((hour_of_day < 6) | (hour_of_day > 22)).astype(int) * 0.2 +
        (credit_scores < 500).astype(int) * 0.4 +
        np.random.random(n_samples) * 0.1
    )
    
    y = (risk_score > 0.5).astype(int)
    
    return df, y


def run_comprehensive_feature_engineering_test():
    """Run comprehensive test of all feature engineering components"""
    
    print("🚀 Running Comprehensive Feature Engineering Test")
    print("=" * 60)
    
    # Create test dataset
    print("📊 Creating comprehensive FinTech dataset...")
    df, y = create_comprehensive_fintech_dataset(n_samples=5000)
    print(f"✅ Created dataset with {len(df)} samples and {len(df.columns)} features")
    
    # Test 1: Time Series Feature Engineering
    print("\n🕒 Testing Time Series Feature Engineering...")
    ts_engineer = AdvancedTimeSeriesFeatureEngineer()
    
    # Create temporal features
    df_with_temporal = ts_engineer.create_temporal_features(df, 'timestamp')
    
    # Create lag features
    df_with_lags = ts_engineer.create_lag_features(
        df_with_temporal, 'transaction_amount', 'customer_id', [1, 6, 24]
    )
    
    # Create rolling features
    df_with_rolling = ts_engineer.create_rolling_features(
        df_with_lags, 'transaction_amount', 'customer_id', [6, 24, 168]
    )
    
    # Create seasonality features
    df_with_seasonality = ts_engineer.create_seasonality_features(
        df_with_rolling, 'transaction_amount', [24, 168]
    )
    
    print(f"✅ Time series features: {len(df_with_seasonality.columns)} total features")
    
    # Test 2: NLP Feature Engineering
    print("\n📝 Testing NLP Feature Engineering...")
    nlp_engineer = ProductionNLPFeatureEngineer()
    
    # Extract linguistic features
    linguistic_features = []
    for text in df['feedback_text'].head(100):  # Test on subset for speed
        features = nlp_engineer.extract_linguistic_features(text)
        linguistic_features.append(features)
    
    linguistic_df = pd.DataFrame(linguistic_features)
    
    # Create TF-IDF features
    tfidf_features = nlp_engineer.create_tfidf_features(
        df['feedback_text'].head(100).tolist(), max_features=50
    )
    
    print(f"✅ NLP features: {len(linguistic_df.columns)} linguistic + {tfidf_features.shape[1]} TF-IDF")
    
    # Test 3: Feature Selection
    print("\n🎯 Testing Ensemble Feature Selection...")
    
    # Prepare numeric dataset for selection
    numeric_df = df_with_seasonality.select_dtypes(include=[np.number]).fillna(0)
    
    selector = EnsembleFeatureSelector(task_type='classification')
    selected_features = selector.ensemble_selection(
        numeric_df.head(1000), y.head(1000), voting_threshold=0.3
    )
    
    print(f"✅ Feature selection: {len(selected_features)} selected from {len(numeric_df.columns)}")
    
    # Test 4: Feature Generation
    print("\n🔧 Testing Automated Feature Generation...")
    
    # Use subset of features for generation
    base_features = df[['age', 'income', 'credit_score', 'account_age_years']].head(1000)
    
    generator = AdvancedFeatureGenerator(max_features=200)
    enhanced_features = generator.generate_all_features(
        base_features,
        include_polynomial=False,  # Skip to avoid explosion
        include_ratios=True,
        include_binning=True,
        include_statistical=True
    )
    
    print(f"✅ Feature generation: {len(enhanced_features.columns)} total features")
    
    # Test 5: Production Pipeline
    print("\n🏭 Testing Production Pipeline...")
    
    config = {
        'task_type': 'classification',
        'max_features': 100
    }
    
    pipeline = ProductionFeatureEngineeringPipeline(config)
    
    # Use subset for pipeline testing
    test_df = df[['age', 'income', 'credit_score', 'account_type', 'account_age_years']].head(1000)
    test_y = y.head(1000)
    
    X_transformed = pipeline.fit_transform(test_df, test_y)
    
    print(f"✅ Production pipeline: {X_transformed.shape} output shape")
    
    # Test 6: Feature Validation
    print("\n✅ Testing Feature Validation...")
    
    validator = FeatureQualityValidator()
    
    # Test quality validation
    quality_results = validator.validate_feature_quality(test_df)
    print(f"✅ Quality validation: {quality_results['quality_score']:.3f} quality score")
    
    # Test drift detection (simulate drift)
    test_df_drift = test_df.copy()
    test_df_drift['income'] = test_df_drift['income'] * 1.2  # Simulate income inflation
    
    drift_results = validator.detect_feature_drift(test_df, test_df_drift)
    print(f"✅ Drift detection: {len(drift_results['features_with_drift'])} features with drift")
    
    # Summary
    print("\n🎯 Comprehensive Test Summary:")
    print("=" * 40)
    print(f"✅ Time Series Features: {len(df_with_seasonality.columns)} features created")
    print(f"✅ NLP Features: {len(linguistic_df.columns)} + {tfidf_features.shape[1]} features")
    print(f"✅ Feature Selection: {len(selected_features)} features selected")
    print(f"✅ Feature Generation: {len(enhanced_features.columns)} features generated")
    print(f"✅ Production Pipeline: {X_transformed.shape} final output")
    print(f"✅ Feature Validation: {quality_results['quality_score']:.3f} quality score")
    print(f"✅ Drift Detection: {len(drift_results['features_with_drift'])} drifted features")
    
    print("\n🚀 All advanced feature engineering components working successfully!")
    print("🎉 Ready for production FinTech ML systems!")
    
    return {
        'time_series_features': len(df_with_seasonality.columns),
        'nlp_features': len(linguistic_df.columns) + tfidf_features.shape[1],
        'selected_features': len(selected_features),
        'generated_features': len(enhanced_features.columns),
        'pipeline_output_shape': X_transformed.shape,
        'quality_score': quality_results['quality_score'],
        'drift_features': len(drift_results['features_with_drift'])
    }


if __name__ == "__main__":
    print("🎯 Day 26: Advanced Feature Engineering - Complete Solution")
    print("Production-ready feature engineering for FinTech ML systems")
    print()
    
    # Run comprehensive test
    results = run_comprehensive_feature_engineering_test()
    
    print(f"\n📊 Final Results Summary:")
    print(f"   • Time Series Features: {results['time_series_features']}")
    print(f"   • NLP Features: {results['nlp_features']}")
    print(f"   • Selected Features: {results['selected_features']}")
    print(f"   • Generated Features: {results['generated_features']}")
    print(f"   • Pipeline Output: {results['pipeline_output_shape']}")
    print(f"   • Quality Score: {results['quality_score']:.3f}")
    print(f"   • Drift Features: {results['drift_features']}")
    
    print("\n🏆 Advanced feature engineering implementation complete!")
    print("🚀 Ready for production ML systems with sophisticated feature engineering!")