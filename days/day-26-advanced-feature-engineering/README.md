# Day 26: Advanced Feature Engineering - Time Series, NLP & Automated Selection

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- **Master** advanced feature engineering techniques for time series, NLP, and categorical data
- **Implement** automated feature selection and generation algorithms for production ML systems
- **Build** scalable feature engineering pipelines that handle complex data transformations
- **Deploy** feature validation and monitoring systems to ensure feature quality in production
- **Apply** advanced statistical and domain-specific feature engineering patterns

⭐ **Difficulty**: Advanced ML Engineering (1 hour)

---

## Theory

### Advanced Feature Engineering: Beyond Basic Transformations

Feature engineering is often the difference between a mediocre ML model and a production-ready system that delivers real business value. Advanced feature engineering goes beyond simple transformations to create sophisticated, domain-aware features that capture complex patterns in data.

**Why Advanced Feature Engineering Matters**:
- **Model Performance**: Well-engineered features can improve model accuracy by 20-50%
- **Interpretability**: Domain-aware features make models more explainable to stakeholders
- **Generalization**: Robust features help models perform well on unseen data
- **Efficiency**: Good features can reduce model complexity and training time
- **Production Stability**: Engineered features are more stable than raw data features

### Time Series Feature Engineering

Time series data requires specialized feature engineering techniques that capture temporal patterns, seasonality, and trends.

#### 1. Temporal Features
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_temporal_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Create comprehensive temporal features from timestamp"""
    
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Basic temporal features
    df['year'] = df[timestamp_col].dt.year
    df['month'] = df[timestamp_col].dt.month
    df['day'] = df[timestamp_col].dt.day
    df['hour'] = df[timestamp_col].dt.hour
    df['minute'] = df[timestamp_col].dt.minute
    df['dayofweek'] = df[timestamp_col].dt.dayofweek
    df['dayofyear'] = df[timestamp_col].dt.dayofyear
    df['week'] = df[timestamp_col].dt.isocalendar().week
    df['quarter'] = df[timestamp_col].dt.quarter
    
    # Cyclical encoding for periodic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Business calendar features
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[timestamp_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[timestamp_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[timestamp_col].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[timestamp_col].dt.is_quarter_end.astype(int)
    
    return df

# Example usage for ride-sharing demand prediction
ride_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
    'ride_count': np.random.poisson(50, 1000)
})

ride_features = create_temporal_features(ride_data, 'timestamp')
print("Temporal features created:", ride_features.columns.tolist())
```

#### 2. Lag and Rolling Window Features
```python
def create_lag_features(df: pd.DataFrame, target_col: str, 
                       lags: List[int], windows: List[int]) -> pd.DataFrame:
    """Create lag and rolling window features for time series"""
    
    df = df.copy()
    
    # Lag features
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling window statistics
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
        df[f'{target_col}_rolling_median_{window}'] = df[target_col].rolling(window).median()
        
        # Rolling percentiles
        df[f'{target_col}_rolling_q25_{window}'] = df[target_col].rolling(window).quantile(0.25)
        df[f'{target_col}_rolling_q75_{window}'] = df[target_col].rolling(window).quantile(0.75)
        
        # Rolling trend features
        df[f'{target_col}_rolling_trend_{window}'] = (
            df[target_col] - df[f'{target_col}_rolling_mean_{window}']
        ) / df[f'{target_col}_rolling_std_{window}']
    
    # Exponential moving averages
    for alpha in [0.1, 0.3, 0.5]:
        df[f'{target_col}_ema_{alpha}'] = df[target_col].ewm(alpha=alpha).mean()
    
    return df

# Create lag and rolling features
ride_features = create_lag_features(
    ride_features, 
    'ride_count', 
    lags=[1, 2, 3, 6, 12, 24, 168],  # 1h, 2h, 3h, 6h, 12h, 1d, 1w
    windows=[3, 6, 12, 24, 168]       # 3h, 6h, 12h, 1d, 1w
)
```

#### 3. Seasonality and Trend Decomposition
```python
from scipy import signal
from sklearn.preprocessing import StandardScaler

def create_seasonality_features(df: pd.DataFrame, target_col: str, 
                              seasonal_periods: List[int]) -> pd.DataFrame:
    """Extract seasonality and trend components"""
    
    df = df.copy()
    
    for period in seasonal_periods:
        # Fourier features for seasonality
        for k in range(1, min(period//2, 10)):  # Limit to 10 harmonics
            df[f'{target_col}_fourier_sin_{period}_{k}'] = np.sin(
                2 * np.pi * k * np.arange(len(df)) / period
            )
            df[f'{target_col}_fourier_cos_{period}_{k}'] = np.cos(
                2 * np.pi * k * np.arange(len(df)) / period
            )
    
    # Trend features using polynomial fitting
    x = np.arange(len(df))
    for degree in [1, 2, 3]:
        coeffs = np.polyfit(x, df[target_col].fillna(method='ffill'), degree)
        trend = np.polyval(coeffs, x)
        df[f'{target_col}_trend_poly_{degree}'] = trend
        df[f'{target_col}_detrended_{degree}'] = df[target_col] - trend
    
    return df

# Add seasonality features
ride_features = create_seasonality_features(
    ride_features, 
    'ride_count',
    seasonal_periods=[24, 168, 8760]  # Daily, weekly, yearly patterns
)
```

### NLP Feature Engineering

Natural Language Processing requires specialized techniques to convert text into meaningful numerical features.

#### 1. Advanced Text Preprocessing
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy

class AdvancedTextPreprocessor:
    """Advanced text preprocessing for feature engineering"""
    
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model for advanced processing
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features from text"""
        
        if not text or pd.isna(text):
            return self._empty_features()
        
        features = {}
        
        # Basic length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
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
            
            # Named entity features
            features['entity_count'] = len(doc.ents)
            features['person_count'] = len([ent for ent in doc.ents if ent.label_ == 'PERSON'])
            features['org_count'] = len([ent for ent in doc.ents if ent.label_ == 'ORG'])
            features['location_count'] = len([ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC']])
            
            # Sentiment and readability
            features['sentiment_polarity'] = self._calculate_sentiment(text)
            features['readability_score'] = self._calculate_readability(text)
        
        return features
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dict for null text"""
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'noun_ratio': 0, 'verb_ratio': 0,
            'adj_ratio': 0, 'adv_ratio': 0, 'entity_count': 0,
            'person_count': 0, 'org_count': 0, 'location_count': 0,
            'sentiment_polarity': 0, 'readability_score': 0
        }
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment polarity (simplified)"""
        # In production, use proper sentiment analysis libraries
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
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

# Example usage
preprocessor = AdvancedTextPreprocessor()

sample_reviews = [
    "This product is absolutely amazing! I love everything about it.",
    "Terrible quality. Would not recommend to anyone. Waste of money.",
    "Average product. Nothing special but does the job adequately."
]

for review in sample_reviews:
    features = preprocessor.extract_linguistic_features(review)
    print(f"Review: {review[:50]}...")
    print(f"Features: {features}")
    print()
```

#### 2. Advanced Embedding Features
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import gensim.downloader as api

class AdvancedEmbeddingFeatures:
    """Create advanced embedding-based features"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        self.svd = TruncatedSVD(n_components=100)
        self.kmeans = KMeans(n_clusters=20, random_state=42)
        
        # Load pre-trained word embeddings
        try:
            self.word_vectors = api.load('word2vec-google-news-300')
        except:
            print("Word2Vec model not available. Using random embeddings.")
            self.word_vectors = None
    
    def create_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Create TF-IDF features with dimensionality reduction"""
        
        # Fit TF-IDF
        tfidf_matrix = self.tfidf.fit_transform(texts)
        
        # Apply SVD for dimensionality reduction
        reduced_features = self.svd.fit_transform(tfidf_matrix)
        
        return reduced_features
    
    def create_word_embedding_features(self, texts: List[str]) -> np.ndarray:
        """Create features from word embeddings"""
        
        if not self.word_vectors:
            return np.zeros((len(texts), 300))
        
        features = []
        
        for text in texts:
            words = text.split()
            word_embeddings = []
            
            for word in words:
                if word in self.word_vectors:
                    word_embeddings.append(self.word_vectors[word])
            
            if word_embeddings:
                # Aggregate word embeddings
                embeddings_array = np.array(word_embeddings)
                
                # Multiple aggregation strategies
                mean_embedding = np.mean(embeddings_array, axis=0)
                max_embedding = np.max(embeddings_array, axis=0)
                min_embedding = np.min(embeddings_array, axis=0)
                std_embedding = np.std(embeddings_array, axis=0)
                
                # Combine aggregations
                combined = np.concatenate([
                    mean_embedding, max_embedding, min_embedding, std_embedding
                ])
                features.append(combined)
            else:
                # No valid words found
                features.append(np.zeros(300 * 4))
        
        return np.array(features)
    
    def create_topic_features(self, texts: List[str]) -> np.ndarray:
        """Create topic-based features using clustering"""
        
        # Get TF-IDF features
        tfidf_features = self.create_tfidf_features(texts)
        
        # Cluster documents
        cluster_labels = self.kmeans.fit_predict(tfidf_features)
        
        # Create one-hot encoded cluster features
        n_clusters = self.kmeans.n_clusters
        topic_features = np.zeros((len(texts), n_clusters))
        
        for i, label in enumerate(cluster_labels):
            topic_features[i, label] = 1
        
        # Add cluster distances as features
        cluster_distances = self.kmeans.transform(tfidf_features)
        
        return np.concatenate([topic_features, cluster_distances], axis=1)

# Example usage
embedding_extractor = AdvancedEmbeddingFeatures()

sample_texts = [
    "Machine learning is transforming the technology industry",
    "Data science requires strong analytical and programming skills",
    "Artificial intelligence will revolutionize healthcare and finance"
]

tfidf_features = embedding_extractor.create_tfidf_features(sample_texts)
topic_features = embedding_extractor.create_topic_features(sample_texts)

print(f"TF-IDF features shape: {tfidf_features.shape}")
print(f"Topic features shape: {topic_features.shape}")
```

### Automated Feature Selection and Generation

Automated feature engineering helps scale feature creation and selection for large datasets and complex problems.

#### 1. Statistical Feature Selection
```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, 
    mutual_info_regression, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV
import scipy.stats as stats

class AdvancedFeatureSelector:
    """Advanced feature selection with multiple methods"""
    
    def __init__(self, task_type='classification'):
        self.task_type = task_type
        self.selected_features = {}
        self.feature_scores = {}
    
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, 
                           k: int = 50) -> List[str]:
        """Select features using univariate statistical tests"""
        
        if self.task_type == 'classification':
            # Use F-test for classification
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            # Use F-test for regression
            selector = SelectKBest(score_func=f_regression, k=k)
        
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Store feature scores
        self.feature_scores['univariate'] = dict(zip(
            X.columns, selector.scores_
        ))
        
        return selected_features
    
    def mutual_information_selection(self, X: pd.DataFrame, y: pd.Series, 
                                   k: int = 50) -> List[str]:
        """Select features using mutual information"""
        
        if self.task_type == 'classification':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        self.feature_scores['mutual_info'] = dict(zip(
            X.columns, selector.scores_
        ))
        
        return selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                    n_features: int = 50) -> List[str]:
        """Select features using recursive feature elimination"""
        
        if self.task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Store feature rankings
        self.feature_scores['rfe_ranking'] = dict(zip(
            X.columns, selector.ranking_
        ))
        
        return selected_features
    
    def lasso_selection(self, X: pd.DataFrame, y: pd.Series, 
                       alpha: float = None) -> List[str]:
        """Select features using Lasso regularization"""
        
        if alpha is None:
            # Use cross-validation to find optimal alpha
            lasso = LassoCV(cv=5, random_state=42)
        else:
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=alpha, random_state=42)
        
        lasso.fit(X, y)
        
        # Select features with non-zero coefficients
        selected_mask = lasso.coef_ != 0
        selected_features = X.columns[selected_mask].tolist()
        
        self.feature_scores['lasso_coef'] = dict(zip(
            X.columns, np.abs(lasso.coef_)
        ))
        
        return selected_features
    
    def ensemble_selection(self, X: pd.DataFrame, y: pd.Series, 
                          methods: List[str] = None, 
                          voting_threshold: float = 0.5) -> List[str]:
        """Combine multiple selection methods using ensemble voting"""
        
        if methods is None:
            methods = ['univariate', 'mutual_info', 'rfe', 'lasso']
        
        all_selections = {}
        
        # Run each selection method
        if 'univariate' in methods:
            all_selections['univariate'] = set(self.univariate_selection(X, y))
        
        if 'mutual_info' in methods:
            all_selections['mutual_info'] = set(self.mutual_information_selection(X, y))
        
        if 'rfe' in methods:
            all_selections['rfe'] = set(self.recursive_feature_elimination(X, y))
        
        if 'lasso' in methods:
            all_selections['lasso'] = set(self.lasso_selection(X, y))
        
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
        
        return selected_features

# Example usage
np.random.seed(42)
n_samples, n_features = 1000, 200

# Create synthetic dataset
X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                columns=[f'feature_{i}' for i in range(n_features)])

# Create target with some features being relevant
relevant_features = [0, 5, 10, 15, 20]
y = (X.iloc[:, relevant_features].sum(axis=1) + 
     np.random.randn(n_samples) * 0.1)

# Apply feature selection
selector = AdvancedFeatureSelector(task_type='regression')
selected_features = selector.ensemble_selection(X, y, voting_threshold=0.3)

print(f"Selected {len(selected_features)} features from {n_features}")
print(f"Relevant features captured: {len(set(selected_features) & set([f'feature_{i}' for i in relevant_features]))}")
```

#### 2. Automated Feature Generation
```python
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

class AutomatedFeatureGenerator:
    """Automatically generate new features from existing ones"""
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.generated_features = []
    
    def generate_polynomial_features(self, X: pd.DataFrame, 
                                   degree: int = 2, 
                                   interaction_only: bool = False) -> pd.DataFrame:
        """Generate polynomial and interaction features"""
        
        poly = PolynomialFeatures(
            degree=degree, 
            interaction_only=interaction_only,
            include_bias=False
        )
        
        X_poly = poly.fit_transform(X)
        
        # Create feature names
        feature_names = poly.get_feature_names_out(X.columns)
        
        # Limit number of features
        if len(feature_names) > self.max_features:
            # Keep original features + top polynomial features
            n_original = len(X.columns)
            n_new = self.max_features - n_original
            
            X_poly = X_poly[:, :n_original + n_new]
            feature_names = feature_names[:n_original + n_new]
        
        return pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    def generate_ratio_features(self, X: pd.DataFrame, 
                              numeric_cols: List[str] = None) -> pd.DataFrame:
        """Generate ratio features between numeric columns"""
        
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        X_ratios = X.copy()
        
        # Generate ratios for all pairs of numeric columns
        for col1, col2 in combinations(numeric_cols, 2):
            # Avoid division by zero
            denominator = X[col2].replace(0, np.nan)
            
            ratio_name = f'{col1}_div_{col2}'
            X_ratios[ratio_name] = X[col1] / denominator
            
            # Also create the inverse ratio
            numerator = X[col1].replace(0, np.nan)
            inverse_ratio_name = f'{col2}_div_{col1}'
            X_ratios[inverse_ratio_name] = X[col2] / numerator
        
        return X_ratios
    
    def generate_aggregation_features(self, X: pd.DataFrame, 
                                    group_cols: List[str],
                                    agg_cols: List[str],
                                    agg_funcs: List[str] = None) -> pd.DataFrame:
        """Generate aggregation features based on grouping columns"""
        
        if agg_funcs is None:
            agg_funcs = ['mean', 'std', 'min', 'max', 'count']
        
        X_agg = X.copy()
        
        for group_col in group_cols:
            for agg_col in agg_cols:
                if group_col != agg_col:
                    for func in agg_funcs:
                        # Calculate group statistics
                        group_stats = X.groupby(group_col)[agg_col].agg(func)
                        
                        # Map back to original dataframe
                        feature_name = f'{agg_col}_{func}_by_{group_col}'
                        X_agg[feature_name] = X[group_col].map(group_stats)
        
        return X_agg
    
    def generate_binning_features(self, X: pd.DataFrame, 
                                numeric_cols: List[str] = None,
                                n_bins: int = 5) -> pd.DataFrame:
        """Generate binning features for numeric columns"""
        
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        X_binned = X.copy()
        
        for col in numeric_cols:
            # Equal-width binning
            X_binned[f'{col}_bin_equal'] = pd.cut(X[col], bins=n_bins, labels=False)
            
            # Equal-frequency binning (quantiles)
            X_binned[f'{col}_bin_quantile'] = pd.qcut(
                X[col], q=n_bins, labels=False, duplicates='drop'
            )
            
            # Binary features for each bin
            for bin_val in range(n_bins):
                X_binned[f'{col}_is_bin_{bin_val}'] = (
                    X_binned[f'{col}_bin_equal'] == bin_val
                ).astype(int)
        
        return X_binned
    
    def generate_all_features(self, X: pd.DataFrame, y: pd.Series = None,
                            include_polynomial: bool = True,
                            include_ratios: bool = True,
                            include_binning: bool = True) -> pd.DataFrame:
        """Generate all types of features"""
        
        X_enhanced = X.copy()
        
        # Get numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if include_polynomial and len(numeric_cols) > 0:
            # Generate polynomial features (limited to avoid explosion)
            if len(numeric_cols) <= 10:  # Only for small number of features
                X_poly = self.generate_polynomial_features(
                    X[numeric_cols], degree=2, interaction_only=True
                )
                # Add only new features (exclude original ones)
                new_cols = [col for col in X_poly.columns if col not in X.columns]
                for col in new_cols[:50]:  # Limit to 50 new features
                    X_enhanced[col] = X_poly[col]
        
        if include_ratios and len(numeric_cols) > 1:
            X_ratios = self.generate_ratio_features(X, numeric_cols)
            # Add ratio features
            ratio_cols = [col for col in X_ratios.columns if '_div_' in col]
            for col in ratio_cols[:50]:  # Limit to 50 ratio features
                X_enhanced[col] = X_ratios[col]
        
        if include_binning and len(numeric_cols) > 0:
            X_binned = self.generate_binning_features(X, numeric_cols)
            # Add binning features
            bin_cols = [col for col in X_binned.columns if '_bin_' in col or '_is_bin_' in col]
            for col in bin_cols:
                X_enhanced[col] = X_binned[col]
        
        return X_enhanced

# Example usage
# Create sample dataset
np.random.seed(42)
sample_data = pd.DataFrame({
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.lognormal(10, 1, 1000),
    'education_years': np.random.randint(8, 20, 1000),
    'experience_years': np.random.randint(0, 40, 1000),
    'city_id': np.random.randint(1, 10, 1000)
})

# Generate features
feature_generator = AutomatedFeatureGenerator(max_features=200)
enhanced_data = feature_generator.generate_all_features(sample_data)

print(f"Original features: {len(sample_data.columns)}")
print(f"Enhanced features: {len(enhanced_data.columns)}")
print(f"New features created: {len(enhanced_data.columns) - len(sample_data.columns)}")
```

### Production Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

class ProductionFeatureEngineeringPipeline:
    """Production-ready feature engineering pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = None
        self.feature_names = None
        
    def build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Build comprehensive feature engineering pipeline"""
        
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_features = X.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Numeric pipeline
        numeric_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', RobustScaler()),
            ('outlier_clipper', self._create_outlier_clipper())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('numeric', numeric_pipeline, numeric_features),
            ('categorical', categorical_pipeline, categorical_features)
        ])
        
        # Full pipeline with feature generation
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_generator', self._create_feature_generator()),
            ('feature_selector', self._create_feature_selector())
        ])
        
        return self.pipeline
    
    def _create_outlier_clipper(self):
        """Create custom outlier clipping transformer"""
        
        class OutlierClipper:
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
        
        return OutlierClipper()
    
    def _create_feature_generator(self):
        """Create custom feature generation transformer"""
        
        class FeatureGenerator:
            def __init__(self):
                pass
            
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                # Add polynomial features for first few columns
                if X.shape[1] >= 2:
                    # Add interaction between first two features
                    interaction = X[:, 0] * X[:, 1]
                    X = np.column_stack([X, interaction])
                
                # Add squared features for first few numeric features
                n_squared = min(5, X.shape[1])
                squared_features = X[:, :n_squared] ** 2
                X = np.column_stack([X, squared_features])
                
                return X
        
        return FeatureGenerator()
    
    def _create_feature_selector(self):
        """Create feature selection transformer"""
        
        from sklearn.feature_selection import SelectKBest, f_regression
        
        return SelectKBest(
            score_func=f_regression, 
            k=min(100, self.config.get('max_features', 100))
        )

# Example usage
config = {'max_features': 50}
pipeline_builder = ProductionFeatureEngineeringPipeline(config)

# Build and fit pipeline
pipeline = pipeline_builder.build_pipeline(sample_data)
y_sample = np.random.randn(len(sample_data))

pipeline.fit(sample_data, y_sample)
X_transformed = pipeline.transform(sample_data)

print(f"Original shape: {sample_data.shape}")
print(f"Transformed shape: {X_transformed.shape}")
```

---

## 🚀 Quick Start

### Option 1: Full Infrastructure Setup (Recommended)

1. **Prerequisites**:
   - Docker and Docker Compose installed
   - 8GB+ RAM available
   - Ports 3000, 5432, 6379, 8000, 8888, 9090 available

2. **One-command setup**:
   ```bash
   ./setup.sh
   ```

3. **Access services**:
   - Feature Engineering API: http://localhost:8000
   - Jupyter Lab: http://localhost:8888
   - Grafana Dashboard: http://localhost:3000 (admin/admin123)
   - Prometheus: http://localhost:9090

4. **Run interactive demo**:
   ```bash
   python demo.py
   ```

### Option 2: Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Run exercises**:
   ```bash
   python exercise.py
   ```

3. **Check solution**:
   ```bash
   python solution.py
   ```

## 📁 Project Structure

```
day-26-advanced-feature-engineering/
├── README.md                    # Complete guide and documentation
├── exercise.py                  # Hands-on exercises with TODOs
├── solution.py                  # Complete production implementation
├── quiz.md                     # Knowledge assessment
│
├── setup.sh                    # Automated environment setup
├── demo.py                     # Interactive demonstration
├── data_generator.py           # FinTech sample data generation
├── api.py                      # Feature engineering API server
├── docker-compose.yml          # Complete infrastructure
├── requirements.txt            # Python dependencies
├── .env                       # Environment configuration
│
├── Dockerfile.api             # API server container
├── Dockerfile.jupyter         # Jupyter development environment
├── Dockerfile.datagen         # Data generation container
│
├── data/                      # Generated sample data
│   ├── raw/                   # Original datasets
│   ├── processed/             # Processed features
│   ├── init/                  # Database initialization
│   └── models/                # Trained models
│
├── notebooks/                 # Interactive Jupyter notebooks
│   └── feature_engineering_demo.ipynb
│
├── monitoring/                # Observability configuration
│   ├── prometheus.yml         # Metrics collection
│   └── grafana/              # Dashboard configuration
│       ├── dashboards/
│       └── datasources/
│
└── logs/                     # Application logs
```

## 🎯 Learning Objectives

By completing this day, you will:

✅ **Master** advanced feature engineering techniques for time series, NLP, and categorical data  
✅ **Implement** automated feature selection and generation algorithms for production ML systems  
✅ **Build** scalable feature engineering pipelines that handle complex data transformations  
✅ **Deploy** feature validation and monitoring systems to ensure feature quality in production  
✅ **Apply** advanced statistical and domain-specific feature engineering patterns  
✅ **Create** production APIs for feature engineering services with monitoring  
✅ **Integrate** feature engineering with complete ML infrastructure

---

## 💻 Hands-On Exercise (40 min)

### Exercise Overview

**Business Scenario**: You're the Senior ML Engineer at "FinTech Insights", a financial technology company. You need to build advanced feature engineering pipelines for multiple ML models including credit risk assessment, fraud detection, and customer lifetime value prediction.

**Your Mission**: Implement comprehensive feature engineering pipelines that handle time series financial data, NLP customer feedback, and automated feature selection for production ML systems.

### Requirements

1. **Time Series Features**: Create sophisticated temporal and lag features for financial time series
2. **NLP Features**: Extract advanced linguistic features from customer feedback and support tickets
3. **Automated Selection**: Implement ensemble feature selection with multiple algorithms
4. **Feature Generation**: Build automated feature generation for interaction and derived features
5. **Production Pipeline**: Create scalable, production-ready feature engineering pipeline

### Exercise Steps

1. **Setup Environment**:
   ```bash
   ./setup.sh  # Full infrastructure
   # OR
   pip install -r requirements.txt  # Local only
   ```

2. **Complete TODOs in exercise.py**:
   - Time series feature engineering (temporal, lag, rolling)
   - NLP feature extraction (linguistic, TF-IDF, sentiment)
   - Automated feature selection (ensemble methods)
   - Feature generation (ratios, binning, statistical)
   - Production pipeline with validation

3. **Test Your Implementation**:
   ```bash
   python exercise.py  # Run your implementation
   python demo.py      # Interactive demonstration
   ```

4. **Verify with Infrastructure**:
   - Check feature engineering API: http://localhost:8000/health
   - Explore Jupyter notebooks: http://localhost:8888
   - Monitor with Grafana: http://localhost:3000

### Success Criteria

- ✅ Time series features capture seasonal and trend patterns
- ✅ NLP features extract meaningful sentiment and linguistic patterns
- ✅ Feature selection reduces dimensionality while maintaining predictive power
- ✅ Production pipeline handles missing data and outliers robustly
- ✅ Complete infrastructure runs successfully
- ✅ All integration tests pass

---

## 🛠️ Infrastructure Components

### Services Included

| Service | Port | Purpose | Access |
|---------|------|---------|--------|
| **Feature Engineering API** | 8000 | Production feature engineering services | http://localhost:8000 |
| **PostgreSQL** | 5432 | Feature storage and analysis | Internal |
| **Redis** | 6379 | Fast feature caching | Internal |
| **Jupyter Lab** | 8888 | Interactive development | http://localhost:8888 |
| **Grafana** | 3000 | Monitoring dashboards | http://localhost:3000 |
| **Prometheus** | 9090 | Metrics collection | http://localhost:9090 |

### Key Features

- **🚀 Complete Infrastructure**: Docker Compose with all services
- **📊 Sample Data**: 10K customers, 100K transactions, 5K feedback records
- **⚡ Production API**: FastAPI with feature engineering endpoints
- **📈 Monitoring**: Prometheus metrics + Grafana dashboards
- **🔍 Interactive Demo**: Rich CLI demo with performance benchmarks
- **📓 Jupyter Notebooks**: Interactive development environment
- **🧪 Integration Tests**: Comprehensive test suite

### API Endpoints

- **POST /features/time-series**: Create time series features
- **POST /features/nlp**: Extract NLP features from text
- **POST /features/select**: Automated feature selection
- **POST /features/generate**: Generate new features
- **POST /features/validate**: Feature quality validation
- **GET /features/stats**: Feature engineering statistics
- **GET /health**: Service health check
- **GET /metrics**: Prometheus metrics

### Monitoring & Observability

- **Feature Engineering Latency**: Processing time per feature type
- **Feature Quality Metrics**: Missing values, outliers, drift scores
- **API Performance**: Request rates, error rates, throughput
- **Data Quality**: Feature validation results and alerts
- **System Health**: Database and cache connection status

## 📚 Resources

- **Feature Engineering Guide**: [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) - Comprehensive O'Reilly book
- **Time Series Features**: [tsfresh Documentation](https://tsfresh.readthedocs.io/) - Automated time series feature extraction
- **NLP Feature Engineering**: [spaCy Documentation](https://spacy.io/) - Advanced NLP processing
- **Scikit-learn Feature Selection**: [Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html) - Official documentation
- **Feature Tools**: [Featuretools Documentation](https://docs.featuretools.com/) - Automated feature engineering framework

---

## 🎯 Key Takeaways

- **Advanced feature engineering significantly improves model performance** through domain-aware transformations
- **Time series features require specialized techniques** including lag features, rolling statistics, and seasonality decomposition
- **NLP feature engineering goes beyond bag-of-words** to include linguistic, semantic, and syntactic features
- **Automated feature selection prevents overfitting** and improves model interpretability in high-dimensional spaces
- **Feature generation techniques create new informative features** through mathematical transformations and interactions
- **Production pipelines must handle missing data, outliers, and scaling** consistently across training and serving
- **Ensemble feature selection methods are more robust** than single selection algorithms
- **Feature engineering is iterative and domain-specific** requiring deep understanding of the business problem

---

## 🚀 What's Next?

Tomorrow (Day 27), you'll learn **Time Series Forecasting** with ARIMA, Prophet, and Neural Networks, building on the time series feature engineering techniques you've mastered today.

**Preview**: You'll explore advanced forecasting models, seasonal decomposition, trend analysis, and production forecasting systems that leverage the sophisticated features you can now engineer!

---

## ✅ Before Moving On

- [ ] Understand advanced feature engineering techniques for different data types
- [ ] Can create sophisticated time series features with lag and rolling statistics
- [ ] Know how to extract meaningful features from text data using NLP techniques
- [ ] Understand automated feature selection and generation methods
- [ ] Can build production-ready feature engineering pipelines
- [ ] Complete the comprehensive feature engineering implementation exercise
- [ ] Review feature validation and monitoring best practices

**Time spent**: ~1 hour  
**Difficulty**: ⭐⭐⭐⭐ (Advanced ML Engineering)

**Phase 3 Progress**: Building the advanced ML infrastructure foundation! 🔧
