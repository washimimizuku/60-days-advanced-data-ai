"""
Day 29: Recommendation Systems - Collaborative & Content-based Methods - Complete Solution

Production-ready recommendation system for StreamFlix's personalized content discovery.
Demonstrates comprehensive recommendation approaches and production deployment patterns.

This solution showcases:
- Collaborative filtering (user-based, item-based, matrix factorization)
- Content-based filtering with TF-IDF and metadata
- Hybrid recommendation systems with intelligent weighting
- Cold start solutions for new users and items
- Real-time serving with caching and performance optimization
- Comprehensive evaluation framework with multiple metrics
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Advanced libraries (handle gracefully if not available)
try:
    import implicit
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

import logging
import time
import json
from collections import defaultdict, Counter
import scipy.sparse as sp

# =============================================================================
# PRODUCTION COLLABORATIVE FILTERING
# =============================================================================

class ProductionCollaborativeFiltering:
    """Production-grade collaborative filtering with multiple algorithms"""
    
    def __init__(self, method='user_based', n_neighbors=50, min_interactions=5):
        self.method = method
        self.n_neighbors = n_neighbors
        self.min_interactions = min_interactions
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_means = None
        self.item_means = None
        self.global_mean = None
        self.fitted = False
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for collaborative filtering"""
        logger = logging.getLogger('collaborative_filtering')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fit(self, user_item_matrix: np.ndarray) -> 'ProductionCollaborativeFiltering':
        """
        Fit collaborative filtering model with comprehensive preprocessing
        
        Args:
            user_item_matrix: Matrix where rows are users, columns are items
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting {self.method} collaborative filtering model")
        
        # Store original matrix
        self.user_item_matrix = user_item_matrix.copy()
        
        # Calculate baseline statistics
        self.global_mean = np.mean(user_item_matrix[user_item_matrix > 0])
        
        # Calculate user and item means for bias correction
        user_sums = np.sum(user_item_matrix, axis=1)
        user_counts = np.sum(user_item_matrix > 0, axis=1)
        self.user_means = np.divide(user_sums, user_counts, 
                                   out=np.full_like(user_sums, self.global_mean), 
                                   where=user_counts > 0)
        
        item_sums = np.sum(user_item_matrix, axis=0)
        item_counts = np.sum(user_item_matrix > 0, axis=0)
        self.item_means = np.divide(item_sums, item_counts,
                                   out=np.full_like(item_sums, self.global_mean),
                                   where=item_counts > 0)
        
        # Calculate similarity matrix
        if self.method == 'user_based':
            self._fit_user_based()
        elif self.method == 'item_based':
            self._fit_item_based()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.fitted = True
        self.logger.info("Collaborative filtering model fitted successfully")
        
        return self
    
    def _fit_user_based(self):
        """Fit user-based collaborative filtering"""
        self.logger.info("Computing user-user similarities")
        
        # Normalize ratings by subtracting user means
        normalized_matrix = self.user_item_matrix.copy().astype(float)
        for i in range(len(normalized_matrix)):
            user_ratings = normalized_matrix[i]
            rated_items = user_ratings > 0
            if np.any(rated_items):
                normalized_matrix[i, rated_items] -= self.user_means[i]
        
        # Calculate cosine similarity
        self.similarity_matrix = cosine_similarity(normalized_matrix)
        
        # Set self-similarity to 0 to avoid recommending based on self
        np.fill_diagonal(self.similarity_matrix, 0)
        
        self.logger.info(f"Computed similarities for {len(self.similarity_matrix)} users")
    
    def _fit_item_based(self):
        """Fit item-based collaborative filtering"""
        self.logger.info("Computing item-item similarities")
        
        # Normalize ratings by subtracting item means
        normalized_matrix = self.user_item_matrix.copy().astype(float)
        for j in range(normalized_matrix.shape[1]):
            item_ratings = normalized_matrix[:, j]
            rated_users = item_ratings > 0
            if np.any(rated_users):
                normalized_matrix[rated_users, j] -= self.item_means[j]
        
        # Calculate cosine similarity between items
        self.similarity_matrix = cosine_similarity(normalized_matrix.T)
        
        # Set self-similarity to 0
        np.fill_diagonal(self.similarity_matrix, 0)
        
        self.logger.info(f"Computed similarities for {self.similarity_matrix.shape[0]} items")
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a specific user-item pair with bias correction
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            
        Returns:
            Predicted rating
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.method == 'user_based':
            return self._predict_user_based(user_id, item_id)
        elif self.method == 'item_based':
            return self._predict_item_based(user_id, item_id)
    
    def _predict_user_based(self, user_id: int, item_id: int) -> float:
        """Predict rating using user-based collaborative filtering"""
        
        # Get similar users who rated this item
        user_similarities = self.similarity_matrix[user_id]
        item_ratings = self.user_item_matrix[:, item_id]
        
        # Find users who rated this item
        rated_users = np.where(item_ratings > 0)[0]
        
        if len(rated_users) == 0:
            return self.global_mean
        
        # Get similarities and ratings for users who rated this item
        relevant_similarities = user_similarities[rated_users]
        relevant_ratings = item_ratings[rated_users]
        relevant_user_means = self.user_means[rated_users]
        
        # Filter by minimum similarity threshold
        similarity_threshold = 0.1
        valid_mask = np.abs(relevant_similarities) > similarity_threshold
        
        if not np.any(valid_mask):
            return self.user_means[user_id]
        
        valid_similarities = relevant_similarities[valid_mask]
        valid_ratings = relevant_ratings[valid_mask]
        valid_user_means = relevant_user_means[valid_mask]
        
        # Calculate weighted average with bias correction
        numerator = np.sum(valid_similarities * (valid_ratings - valid_user_means))
        denominator = np.sum(np.abs(valid_similarities))
        
        if denominator == 0:
            return self.user_means[user_id]
        
        prediction = self.user_means[user_id] + (numerator / denominator)
        
        # Clip to valid rating range (assuming 1-5 scale)
        return np.clip(prediction, 1.0, 5.0)
    
    def _predict_item_based(self, user_id: int, item_id: int) -> float:
        """Predict rating using item-based collaborative filtering"""
        
        # Get items rated by this user
        user_ratings = self.user_item_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return self.global_mean
        
        # Get similarities between target item and rated items
        item_similarities = self.similarity_matrix[item_id]
        relevant_similarities = item_similarities[rated_items]
        relevant_ratings = user_ratings[rated_items]
        relevant_item_means = self.item_means[rated_items]
        
        # Filter by minimum similarity threshold
        similarity_threshold = 0.1
        valid_mask = np.abs(relevant_similarities) > similarity_threshold
        
        if not np.any(valid_mask):
            return self.item_means[item_id]
        
        valid_similarities = relevant_similarities[valid_mask]
        valid_ratings = relevant_ratings[valid_mask]
        valid_item_means = relevant_item_means[valid_mask]
        
        # Calculate weighted average with bias correction
        numerator = np.sum(valid_similarities * (valid_ratings - valid_item_means))
        denominator = np.sum(np.abs(valid_similarities))
        
        if denominator == 0:
            return self.item_means[item_id]
        
        prediction = self.item_means[item_id] + (numerator / denominator)
        
        # Clip to valid rating range
        return np.clip(prediction, 1.0, 5.0)
    
    def recommend_items(self, user_id: int, n_recommendations: int = 10,
                       exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Generate item recommendations for a user
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        user_ratings = self.user_item_matrix[user_id]
        
        # Find unrated items
        if exclude_rated:
            candidate_items = np.where(user_ratings == 0)[0]
        else:
            candidate_items = np.arange(self.user_item_matrix.shape[1])
        
        if len(candidate_items) == 0:
            return []
        
        # Predict ratings for candidate items
        predictions = []
        for item_id in candidate_items:
            predicted_rating = self.predict_rating(user_id, item_id)
            predictions.append((item_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_similar_users(self, user_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find users similar to the given user
        
        Args:
            user_id: ID of the user
            n_similar: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if not self.fitted or self.method != 'user_based':
            raise ValueError("Model must be fitted with user_based method")
        
        similarities = self.similarity_matrix[user_id]
        
        # Get indices of most similar users (excluding self)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Filter out the user themselves and return top N
        similar_users = []
        for idx in similar_indices:
            if idx != user_id and len(similar_users) < n_similar:
                similar_users.append((idx, similarities[idx]))
        
        return similar_users
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find items similar to the given item
        
        Args:
            item_id: ID of the item
            n_similar: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self.fitted or self.method != 'item_based':
            raise ValueError("Model must be fitted with item_based method")
        
        similarities = self.similarity_matrix[item_id]
        
        # Get indices of most similar items (excluding self)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Filter out the item itself and return top N
        similar_items = []
        for idx in similar_indices:
            if idx != item_id and len(similar_items) < n_similar:
                similar_items.append((idx, similarities[idx]))
        
        return similar_items
# =============================================================================
# PRODUCTION MATRIX FACTORIZATION
# =============================================================================

class ProductionMatrixFactorization:
    """Production-grade matrix factorization with advanced techniques"""
    
    def __init__(self, n_factors=50, learning_rate=0.01, regularization=0.1,
                 n_iterations=100, random_state=42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_iterations = n_iterations
        self.random_state = random_state
        
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = None
        self.training_history = []
        self.fitted = False
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for matrix factorization"""
        logger = logging.getLogger('matrix_factorization')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fit(self, user_item_matrix: np.ndarray, validation_split: float = 0.1) -> 'ProductionMatrixFactorization':
        """
        Fit matrix factorization model using SGD with comprehensive features
        
        Args:
            user_item_matrix: User-item interaction matrix
            validation_split: Fraction of data to use for validation
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Starting matrix factorization training")
        
        np.random.seed(self.random_state)
        
        n_users, n_items = user_item_matrix.shape
        
        # Initialize parameters
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_bias = np.mean(user_item_matrix[user_item_matrix > 0])
        
        # Prepare training data
        train_data, val_data = self._prepare_training_data(user_item_matrix, validation_split)
        
        # Training loop
        for iteration in range(self.n_iterations):
            start_time = time.time()
            
            # Shuffle training data
            np.random.shuffle(train_data)
            
            # SGD updates
            train_loss = self._sgd_step(train_data)
            
            # Validation
            val_loss = self._calculate_loss(val_data) if len(val_data) > 0 else 0
            
            iteration_time = time.time() - start_time
            
            # Store training history
            self.training_history.append({
                'iteration': iteration + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'time': iteration_time
            })
            
            if (iteration + 1) % 10 == 0:
                self.logger.info(f"Iteration {iteration + 1}/{self.n_iterations}: "
                               f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        self.fitted = True
        self.logger.info("Matrix factorization training completed")
        
        return self
    
    def _prepare_training_data(self, user_item_matrix: np.ndarray, 
                             validation_split: float) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training and validation data"""
        
        # Get all non-zero entries
        user_ids, item_ids = np.where(user_item_matrix > 0)
        ratings = user_item_matrix[user_ids, item_ids]
        
        # Create training data array
        data = np.column_stack([user_ids, item_ids, ratings])
        
        # Split into train and validation
        if validation_split > 0:
            n_val = int(len(data) * validation_split)
            indices = np.random.permutation(len(data))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            train_data = data[train_indices]
            val_data = data[val_indices]
        else:
            train_data = data
            val_data = np.array([])
        
        return train_data, val_data
    
    def _sgd_step(self, train_data: np.ndarray) -> float:
        """Perform one SGD step and return training loss"""
        
        total_loss = 0
        
        for user_id, item_id, rating in train_data:
            user_id, item_id = int(user_id), int(item_id)
            
            # Predict rating
            prediction = self._predict_single(user_id, item_id)
            
            # Calculate error
            error = rating - prediction
            
            # Store current values for update
            user_factor = self.user_factors[user_id].copy()
            item_factor = self.item_factors[item_id].copy()
            
            # Update biases
            self.user_biases[user_id] += self.learning_rate * (error - self.regularization * self.user_biases[user_id])
            self.item_biases[item_id] += self.learning_rate * (error - self.regularization * self.item_biases[item_id])
            
            # Update factors
            self.user_factors[user_id] += self.learning_rate * (error * item_factor - self.regularization * user_factor)
            self.item_factors[item_id] += self.learning_rate * (error * user_factor - self.regularization * item_factor)
            
            # Accumulate loss
            total_loss += error ** 2
        
        # Add regularization to loss
        reg_loss = self.regularization * (
            np.sum(self.user_factors ** 2) + 
            np.sum(self.item_factors ** 2) +
            np.sum(self.user_biases ** 2) +
            np.sum(self.item_biases ** 2)
        )
        
        return (total_loss + reg_loss) / len(train_data)
    
    def _calculate_loss(self, data: np.ndarray) -> float:
        """Calculate loss on given data"""
        if len(data) == 0:
            return 0
        
        total_loss = 0
        for user_id, item_id, rating in data:
            user_id, item_id = int(user_id), int(item_id)
            prediction = self._predict_single(user_id, item_id)
            total_loss += (rating - prediction) ** 2
        
        return total_loss / len(data)
    
    def _predict_single(self, user_id: int, item_id: int) -> float:
        """Predict rating for single user-item pair"""
        prediction = (self.global_bias + 
                     self.user_biases[user_id] + 
                     self.item_biases[item_id] +
                     np.dot(self.user_factors[user_id], self.item_factors[item_id]))
        
        return np.clip(prediction, 1.0, 5.0)
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for user-item pair
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            
        Returns:
            Predicted rating
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self._predict_single(user_id, item_id)
    
    def recommend_items(self, user_id: int, user_item_matrix: np.ndarray,
                       n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Generate recommendations using matrix factorization
        
        Args:
            user_id: ID of the user
            user_item_matrix: Original user-item matrix
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        user_ratings = user_item_matrix[user_id]
        unrated_items = np.where(user_ratings == 0)[0]
        
        if len(unrated_items) == 0:
            return []
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items:
            predicted_rating = self.predict_rating(user_id, item_id)
            predictions.append((item_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_user_factors(self) -> np.ndarray:
        """Return user factor matrix"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.user_factors
    
    def get_item_factors(self) -> np.ndarray:
        """Return item factor matrix"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.item_factors
    
    def find_similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Find items similar to given item using learned factors"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        item_factor = self.item_factors[item_id]
        similarities = cosine_similarity([item_factor], self.item_factors)[0]
        
        # Get most similar items (excluding self)
        similar_indices = np.argsort(similarities)[::-1]
        
        similar_items = []
        for idx in similar_indices:
            if idx != item_id and len(similar_items) < n_similar:
                similar_items.append((idx, similarities[idx]))
        
        return similar_items
# =============================================================================
# PRODUCTION CONTENT-BASED FILTERING
# =============================================================================

class ProductionContentBasedRecommender:
    """Production-grade content-based filtering with advanced features"""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        
        self.tfidf_vectorizer = None
        self.item_features_matrix = None
        self.item_similarity_matrix = None
        self.item_metadata = None
        self.user_profiles = {}
        self.fitted = False
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for content-based recommender"""
        logger = logging.getLogger('content_based')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fit(self, item_descriptions: List[str], 
            item_metadata: pd.DataFrame = None) -> 'ProductionContentBasedRecommender':
        """
        Fit content-based model using item descriptions and metadata
        
        Args:
            item_descriptions: List of text descriptions for each item
            item_metadata: DataFrame with additional item features
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting content-based recommendation model")
        
        # Initialize and fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Transform item descriptions
        tfidf_features = self.tfidf_vectorizer.fit_transform(item_descriptions)
        
        # Combine with metadata if provided
        if item_metadata is not None:
            self.item_metadata = item_metadata
            metadata_features = self._process_metadata(item_metadata)
            
            # Combine TF-IDF and metadata features
            self.item_features_matrix = sp.hstack([tfidf_features, metadata_features])
        else:
            self.item_features_matrix = tfidf_features
        
        # Calculate item similarity matrix
        self.logger.info("Computing item similarity matrix")
        self.item_similarity_matrix = cosine_similarity(self.item_features_matrix)
        
        # Set diagonal to 0 to avoid self-similarity
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        self.fitted = True
        self.logger.info(f"Content-based model fitted with {self.item_features_matrix.shape[1]} features")
        
        return self
    
    def _process_metadata(self, metadata: pd.DataFrame) -> sp.csr_matrix:
        """Process item metadata into feature matrix"""
        
        features = []
        
        for column in metadata.columns:
            if metadata[column].dtype == 'object':
                # Categorical features - one-hot encode
                unique_values = metadata[column].unique()
                for value in unique_values:
                    feature_vector = (metadata[column] == value).astype(int)
                    features.append(feature_vector)
            else:
                # Numerical features - normalize
                normalized_feature = (metadata[column] - metadata[column].mean()) / metadata[column].std()
                features.append(normalized_feature.fillna(0))
        
        if features:
            feature_matrix = np.column_stack(features)
            return sp.csr_matrix(feature_matrix)
        else:
            return sp.csr_matrix((len(metadata), 0))
    
    def find_similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find items similar to the given item
        
        Args:
            item_id: ID of the item
            n_similar: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Get similarity scores for the item
        similarities = self.item_similarity_matrix[item_id]
        
        # Get indices of most similar items (excluding self)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Return top N similar items
        similar_items = []
        for idx in similar_indices:
            if idx != item_id and len(similar_items) < n_similar:
                similar_items.append((idx, similarities[idx]))
        
        return similar_items
    
    def create_user_profile(self, user_item_matrix: np.ndarray, user_id: int,
                          min_rating: float = 3.0) -> np.ndarray:
        """
        Create user profile based on liked items' features
        
        Args:
            user_item_matrix: User-item interaction matrix
            user_id: ID of the user
            min_rating: Minimum rating to consider as "liked"
            
        Returns:
            User profile vector
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        user_ratings = user_item_matrix[user_id]
        liked_items = np.where(user_ratings >= min_rating)[0]
        
        if len(liked_items) == 0:
            # Return zero profile for users with no liked items
            return np.zeros(self.item_features_matrix.shape[1])
        
        # Weight item features by user ratings
        weighted_features = []
        total_weight = 0
        
        for item_id in liked_items:
            weight = user_ratings[item_id]
            item_features = self.item_features_matrix[item_id].toarray().flatten()
            weighted_features.append(weight * item_features)
            total_weight += weight
        
        # Aggregate weighted features
        if total_weight > 0:
            user_profile = np.sum(weighted_features, axis=0) / total_weight
        else:
            user_profile = np.mean(weighted_features, axis=0)
        
        # Normalize profile
        profile_norm = np.linalg.norm(user_profile)
        if profile_norm > 0:
            user_profile = user_profile / profile_norm
        
        # Cache user profile
        self.user_profiles[user_id] = user_profile
        
        return user_profile
    
    def recommend_for_user(self, user_item_matrix: np.ndarray, user_id: int,
                          n_recommendations: int = 10, min_rating: float = 3.0) -> List[Tuple[int, float]]:
        """
        Recommend items based on user's content preferences
        
        Args:
            user_item_matrix: User-item interaction matrix
            user_id: ID of the user
            n_recommendations: Number of recommendations to return
            min_rating: Minimum rating to consider as "liked"
            
        Returns:
            List of (item_id, score) tuples
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Create or get cached user profile
        if user_id not in self.user_profiles:
            user_profile = self.create_user_profile(user_item_matrix, user_id, min_rating)
        else:
            user_profile = self.user_profiles[user_id]
        
        # Find unrated items
        user_ratings = user_item_matrix[user_id]
        unrated_items = np.where(user_ratings == 0)[0]
        
        if len(unrated_items) == 0:
            return []
        
        # Calculate similarity between user profile and unrated items
        item_scores = []
        for item_id in unrated_items:
            item_features = self.item_features_matrix[item_id].toarray().flatten()
            
            # Calculate cosine similarity
            similarity = np.dot(user_profile, item_features)
            item_scores.append((item_id, similarity))
        
        # Sort by similarity score and return top N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_recommendations]
    
    def recommend_similar_to_items(self, item_ids: List[int], 
                                 n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommend items similar to a list of given items
        
        Args:
            item_ids: List of item IDs to find similar items for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, aggregated_similarity) tuples
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Aggregate similarity scores across all input items
        aggregated_scores = defaultdict(float)
        
        for item_id in item_ids:
            similar_items = self.find_similar_items(item_id, n_recommendations * 2)
            
            for similar_item_id, similarity in similar_items:
                if similar_item_id not in item_ids:  # Exclude input items
                    aggregated_scores[similar_item_id] += similarity
        
        # Normalize by number of input items
        for item_id in aggregated_scores:
            aggregated_scores[item_id] /= len(item_ids)
        
        # Sort and return top recommendations
        sorted_items = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]
    
    def explain_recommendation(self, user_id: int, item_id: int,
                             user_item_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Explain why an item was recommended to a user
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            user_item_matrix: User-item interaction matrix
            
        Returns:
            Dictionary with explanation details
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Get user profile
        if user_id not in self.user_profiles:
            user_profile = self.create_user_profile(user_item_matrix, user_id)
        else:
            user_profile = self.user_profiles[user_id]
        
        # Get item features
        item_features = self.item_features_matrix[item_id].toarray().flatten()
        
        # Calculate feature contributions
        feature_contributions = user_profile * item_features
        
        # Get top contributing features
        if hasattr(self.tfidf_vectorizer, 'get_feature_names_out'):
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(len(feature_contributions))]
        
        # Get top positive and negative contributions
        top_indices = np.argsort(np.abs(feature_contributions))[::-1][:10]
        
        explanations = []
        for idx in top_indices:
            if idx < len(feature_names):
                explanations.append({
                    'feature': feature_names[idx],
                    'contribution': feature_contributions[idx],
                    'user_weight': user_profile[idx],
                    'item_weight': item_features[idx]
                })
        
        # Find similar items that user liked
        user_ratings = user_item_matrix[user_id]
        liked_items = np.where(user_ratings >= 3.0)[0]
        
        similar_liked_items = []
        if len(liked_items) > 0:
            item_similarities = self.item_similarity_matrix[item_id]
            for liked_item in liked_items:
                similarity = item_similarities[liked_item]
                if similarity > 0.1:  # Threshold for meaningful similarity
                    similar_liked_items.append({
                        'item_id': liked_item,
                        'similarity': similarity,
                        'user_rating': user_ratings[liked_item]
                    })
            
            # Sort by similarity
            similar_liked_items.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'overall_score': np.dot(user_profile, item_features),
            'top_feature_contributions': explanations,
            'similar_liked_items': similar_liked_items[:5],
            'explanation_text': self._generate_explanation_text(explanations, similar_liked_items)
        }
    
    def _generate_explanation_text(self, feature_contributions: List[Dict],
                                 similar_items: List[Dict]) -> str:
        """Generate human-readable explanation text"""
        
        explanation_parts = []
        
        # Feature-based explanation
        if feature_contributions:
            top_features = [f['feature'] for f in feature_contributions[:3] if f['contribution'] > 0]
            if top_features:
                explanation_parts.append(f"Based on your interest in: {', '.join(top_features)}")
        
        # Similar items explanation
        if similar_items:
            explanation_parts.append(f"Similar to {len(similar_items)} items you've enjoyed")
        
        if explanation_parts:
            return ". ".join(explanation_parts) + "."
        else:
            return "Recommended based on your viewing history."
# =============================================================================
# PRODUCTION HYBRID RECOMMENDATION SYSTEM
# =============================================================================

class ProductionHybridRecommender:
    """Production hybrid recommendation system combining multiple approaches"""
    
    def __init__(self, collaborative_weight=0.4, content_weight=0.3, 
                 matrix_factorization_weight=0.3, popularity_weight=0.0):
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.matrix_factorization_weight = matrix_factorization_weight
        self.popularity_weight = popularity_weight
        
        # Normalize weights
        total_weight = (collaborative_weight + content_weight + 
                       matrix_factorization_weight + popularity_weight)
        if total_weight > 0:
            self.collaborative_weight /= total_weight
            self.content_weight /= total_weight
            self.matrix_factorization_weight /= total_weight
            self.popularity_weight /= total_weight
        
        self.collaborative_recommender = None
        self.content_recommender = None
        self.matrix_factorization = None
        self.popularity_baseline = None
        self.fitted = False
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for hybrid recommender"""
        logger = logging.getLogger('hybrid_recommender')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fit(self, user_item_matrix: np.ndarray, 
            item_descriptions: List[str],
            item_metadata: pd.DataFrame = None) -> 'ProductionHybridRecommender':
        """
        Fit all component recommenders
        
        Args:
            user_item_matrix: User-item interaction matrix
            item_descriptions: Item text descriptions
            item_metadata: Additional item metadata
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting hybrid recommendation system")
        
        # Fit collaborative filtering
        if self.collaborative_weight > 0:
            self.logger.info("Fitting collaborative filtering component")
            self.collaborative_recommender = ProductionCollaborativeFiltering(method='user_based')
            self.collaborative_recommender.fit(user_item_matrix)
        
        # Fit content-based filtering
        if self.content_weight > 0:
            self.logger.info("Fitting content-based component")
            self.content_recommender = ProductionContentBasedRecommender()
            self.content_recommender.fit(item_descriptions, item_metadata)
        
        # Fit matrix factorization
        if self.matrix_factorization_weight > 0:
            self.logger.info("Fitting matrix factorization component")
            self.matrix_factorization = ProductionMatrixFactorization(n_iterations=50)
            self.matrix_factorization.fit(user_item_matrix)
        
        # Calculate popularity baseline
        if self.popularity_weight > 0:
            self.logger.info("Computing popularity baseline")
            self.popularity_baseline = self._calculate_popularity_baseline(user_item_matrix)
        
        self.fitted = True
        self.logger.info("Hybrid recommendation system fitted successfully")
        
        return self
    
    def _calculate_popularity_baseline(self, user_item_matrix: np.ndarray) -> np.ndarray:
        """Calculate popularity-based scores for items"""
        
        # Calculate popularity as combination of rating count and average rating
        item_counts = np.sum(user_item_matrix > 0, axis=0)
        
        # Calculate average ratings
        item_sums = np.sum(user_item_matrix, axis=0)
        item_averages = np.divide(item_sums, item_counts, 
                                 out=np.zeros_like(item_sums), 
                                 where=item_counts > 0)
        
        # Combine count and average (weighted by count to handle items with few ratings)
        min_ratings = 10  # Minimum ratings for reliable average
        popularity_scores = np.where(
            item_counts >= min_ratings,
            item_averages * np.log(item_counts + 1),  # Log to dampen effect of very popular items
            item_counts * 0.1  # Lower score for items with few ratings
        )
        
        return popularity_scores
    
    def recommend_items(self, user_id: int, user_item_matrix: np.ndarray,
                       n_recommendations: int = 10, 
                       diversify: bool = True) -> List[Tuple[int, float, Dict[str, float]]]:
        """
        Generate hybrid recommendations combining all methods
        
        Args:
            user_id: ID of the user
            user_item_matrix: User-item interaction matrix
            n_recommendations: Number of recommendations to return
            diversify: Whether to apply diversification
            
        Returns:
            List of (item_id, combined_score, method_scores) tuples
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        self.logger.info(f"Generating hybrid recommendations for user {user_id}")
        
        # Get candidate items (unrated by user)
        user_ratings = user_item_matrix[user_id]
        candidate_items = np.where(user_ratings == 0)[0]
        
        if len(candidate_items) == 0:
            return []
        
        # Get recommendations from each method
        method_scores = {}
        
        # Collaborative filtering
        if self.collaborative_recommender and self.collaborative_weight > 0:
            collab_recs = self.collaborative_recommender.recommend_items(
                user_id, n_recommendations * 3
            )
            method_scores['collaborative'] = {item_id: score for item_id, score in collab_recs}
        
        # Content-based filtering
        if self.content_recommender and self.content_weight > 0:
            content_recs = self.content_recommender.recommend_for_user(
                user_item_matrix, user_id, n_recommendations * 3
            )
            method_scores['content'] = {item_id: score for item_id, score in content_recs}
        
        # Matrix factorization
        if self.matrix_factorization and self.matrix_factorization_weight > 0:
            mf_recs = self.matrix_factorization.recommend_items(
                user_id, user_item_matrix, n_recommendations * 3
            )
            method_scores['matrix_factorization'] = {item_id: score for item_id, score in mf_recs}
        
        # Popularity baseline
        if self.popularity_baseline is not None and self.popularity_weight > 0:
            popularity_scores = {}
            for item_id in candidate_items:
                popularity_scores[item_id] = self.popularity_baseline[item_id]
            method_scores['popularity'] = popularity_scores
        
        # Combine scores
        combined_scores = self._combine_method_scores(method_scores, candidate_items)
        
        # Apply diversification if requested
        if diversify and hasattr(self, 'content_recommender') and self.content_recommender:
            combined_scores = self._apply_diversification(combined_scores, n_recommendations)
        
        # Sort and return top recommendations
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        
        recommendations = []
        for item_id, scores in sorted_items[:n_recommendations]:
            recommendations.append((item_id, scores['combined_score'], scores['method_scores']))
        
        return recommendations
    
    def _combine_method_scores(self, method_scores: Dict[str, Dict[int, float]], 
                             candidate_items: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Combine scores from different methods"""
        
        combined_scores = {}
        
        # Normalize scores within each method
        normalized_method_scores = {}
        for method, scores in method_scores.items():
            if scores:
                score_values = list(scores.values())
                min_score = min(score_values)
                max_score = max(score_values)
                
                if max_score > min_score:
                    normalized_scores = {
                        item_id: (score - min_score) / (max_score - min_score)
                        for item_id, score in scores.items()
                    }
                else:
                    normalized_scores = {item_id: 0.5 for item_id in scores}
                
                normalized_method_scores[method] = normalized_scores
        
        # Combine normalized scores
        for item_id in candidate_items:
            method_contributions = {}
            total_score = 0
            
            # Collaborative filtering
            if 'collaborative' in normalized_method_scores:
                collab_score = normalized_method_scores['collaborative'].get(item_id, 0)
                method_contributions['collaborative'] = collab_score * self.collaborative_weight
                total_score += method_contributions['collaborative']
            
            # Content-based filtering
            if 'content' in normalized_method_scores:
                content_score = normalized_method_scores['content'].get(item_id, 0)
                method_contributions['content'] = content_score * self.content_weight
                total_score += method_contributions['content']
            
            # Matrix factorization
            if 'matrix_factorization' in normalized_method_scores:
                mf_score = normalized_method_scores['matrix_factorization'].get(item_id, 0)
                method_contributions['matrix_factorization'] = mf_score * self.matrix_factorization_weight
                total_score += method_contributions['matrix_factorization']
            
            # Popularity
            if 'popularity' in normalized_method_scores:
                pop_score = normalized_method_scores['popularity'].get(item_id, 0)
                method_contributions['popularity'] = pop_score * self.popularity_weight
                total_score += method_contributions['popularity']
            
            combined_scores[item_id] = {
                'combined_score': total_score,
                'method_scores': method_contributions
            }
        
        return combined_scores
    
    def _apply_diversification(self, combined_scores: Dict[int, Dict[str, Any]], 
                             n_recommendations: int) -> Dict[int, Dict[str, Any]]:
        """Apply diversification to recommendations using content similarity"""
        
        if not self.content_recommender or len(combined_scores) <= n_recommendations:
            return combined_scores
        
        # Get sorted items by score
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        
        # Apply MMR (Maximal Marginal Relevance) for diversification
        selected_items = []
        remaining_items = [item_id for item_id, _ in sorted_items]
        
        lambda_param = 0.7  # Balance between relevance and diversity
        
        # Select first item (highest score)
        if remaining_items:
            first_item = remaining_items[0]
            selected_items.append(first_item)
            remaining_items.remove(first_item)
        
        # Select remaining items using MMR
        while len(selected_items) < n_recommendations and remaining_items:
            best_item = None
            best_mmr_score = -1
            
            for item_id in remaining_items:
                relevance_score = combined_scores[item_id]['combined_score']
                
                # Calculate maximum similarity to already selected items
                max_similarity = 0
                for selected_item in selected_items:
                    similarity = self.content_recommender.item_similarity_matrix[item_id, selected_item]
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_item = item_id
            
            if best_item is not None:
                selected_items.append(best_item)
                remaining_items.remove(best_item)
        
        # Return only selected items
        diversified_scores = {item_id: combined_scores[item_id] for item_id in selected_items}
        return diversified_scores
    
    def explain_recommendation(self, user_id: int, item_id: int, 
                             user_item_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Provide comprehensive explanation for recommendation
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            user_item_matrix: User-item interaction matrix
            
        Returns:
            Dictionary with explanation components
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before explaining recommendations")
        
        explanations = {
            'item_id': item_id,
            'user_id': user_id,
            'method_contributions': {},
            'detailed_explanations': {},
            'overall_confidence': 0.0
        }
        
        # Get method contributions
        recommendations = self.recommend_items(user_id, user_item_matrix, n_recommendations=100)
        
        # Find the specific item in recommendations
        item_explanation = None
        for rec_item_id, combined_score, method_scores in recommendations:
            if rec_item_id == item_id:
                explanations['method_contributions'] = method_scores
                explanations['overall_confidence'] = combined_score
                break
        
        # Get detailed explanations from each method
        if self.content_recommender and self.content_weight > 0:
            content_explanation = self.content_recommender.explain_recommendation(
                user_id, item_id, user_item_matrix
            )
            explanations['detailed_explanations']['content'] = content_explanation
        
        if self.collaborative_recommender and self.collaborative_weight > 0:
            # Get similar users for collaborative explanation
            try:
                similar_users = self.collaborative_recommender.get_similar_users(user_id, 5)
                explanations['detailed_explanations']['collaborative'] = {
                    'similar_users': similar_users,
                    'explanation_text': f"Users similar to you also enjoyed this content"
                }
            except:
                pass
        
        # Generate overall explanation text
        explanations['explanation_text'] = self._generate_hybrid_explanation(explanations)
        
        return explanations
    
    def _generate_hybrid_explanation(self, explanations: Dict[str, Any]) -> str:
        """Generate human-readable explanation for hybrid recommendation"""
        
        method_contributions = explanations.get('method_contributions', {})
        
        explanation_parts = []
        
        # Find dominant method
        if method_contributions:
            dominant_method = max(method_contributions.items(), key=lambda x: x[1])
            
            if dominant_method[0] == 'content' and dominant_method[1] > 0.3:
                explanation_parts.append("Based on the content you typically enjoy")
            elif dominant_method[0] == 'collaborative' and dominant_method[1] > 0.3:
                explanation_parts.append("Recommended by users with similar tastes")
            elif dominant_method[0] == 'matrix_factorization' and dominant_method[1] > 0.3:
                explanation_parts.append("Matches your viewing patterns")
        
        # Add confidence indicator
        confidence = explanations.get('overall_confidence', 0)
        if confidence > 0.8:
            explanation_parts.append("(High confidence)")
        elif confidence > 0.5:
            explanation_parts.append("(Medium confidence)")
        
        if explanation_parts:
            return " ".join(explanation_parts)
        else:
            return "Recommended based on your viewing history"
# =============================================================================
# COLD START PROBLEM SOLUTIONS
# =============================================================================

class ProductionColdStartSolver:
    """Production-grade solutions for cold start problems"""
    
    def __init__(self):
        self.popular_items = None
        self.item_categories = None
        self.demographic_profiles = None
        self.category_popularity = None
        self.onboarding_items = None
        self.fitted = False
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for cold start solver"""
        logger = logging.getLogger('cold_start_solver')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fit(self, user_item_matrix: np.ndarray, 
            item_metadata: pd.DataFrame,
            user_demographics: pd.DataFrame = None) -> 'ProductionColdStartSolver':
        """
        Fit cold start solutions using available data
        
        Args:
            user_item_matrix: User-item interaction matrix
            item_metadata: Item metadata including categories
            user_demographics: User demographic information
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting cold start solutions")
        
        # Calculate item popularity scores
        self._calculate_item_popularity(user_item_matrix)
        
        # Extract item categories
        if 'category' in item_metadata.columns:
            self.item_categories = item_metadata['category'].to_dict()
            self._calculate_category_popularity(user_item_matrix, item_metadata)
        
        # Build demographic profiles if available
        if user_demographics is not None:
            self._build_demographic_profiles(user_item_matrix, user_demographics)
        
        # Select onboarding items
        self._select_onboarding_items(user_item_matrix, item_metadata)
        
        self.fitted = True
        self.logger.info("Cold start solutions fitted successfully")
        
        return self
    
    def _calculate_item_popularity(self, user_item_matrix: np.ndarray):
        """Calculate comprehensive item popularity scores"""
        
        # Number of ratings
        rating_counts = np.sum(user_item_matrix > 0, axis=0)
        
        # Average ratings
        rating_sums = np.sum(user_item_matrix, axis=0)
        avg_ratings = np.divide(rating_sums, rating_counts,
                               out=np.zeros_like(rating_sums),
                               where=rating_counts > 0)
        
        # Recency weight (assuming more recent items have higher indices)
        n_items = len(rating_counts)
        recency_weights = np.linspace(0.8, 1.2, n_items)  # Recent items get higher weight
        
        # Combined popularity score
        # Use Wilson score for better handling of items with few ratings
        popularity_scores = []
        for i in range(n_items):
            count = rating_counts[i]
            avg_rating = avg_ratings[i]
            
            if count > 0:
                # Wilson score confidence interval
                z = 1.96  # 95% confidence
                p = (avg_rating - 1) / 4  # Normalize to 0-1 (assuming 1-5 rating scale)
                
                wilson_score = (p + z*z/(2*count) - z * np.sqrt((p*(1-p)+z*z/(4*count))/count)) / (1 + z*z/count)
                
                # Combine with count and recency
                popularity_score = wilson_score * np.log(count + 1) * recency_weights[i]
            else:
                popularity_score = 0
            
            popularity_scores.append(popularity_score)
        
        # Sort items by popularity
        popularity_indices = np.argsort(popularity_scores)[::-1]
        self.popular_items = popularity_indices
        
        self.logger.info(f"Calculated popularity for {n_items} items")
    
    def _calculate_category_popularity(self, user_item_matrix: np.ndarray, 
                                     item_metadata: pd.DataFrame):
        """Calculate popularity within each category"""
        
        self.category_popularity = {}
        
        for category in item_metadata['category'].unique():
            category_items = item_metadata[item_metadata['category'] == category].index.tolist()
            
            if category_items:
                # Calculate popularity within category
                category_matrix = user_item_matrix[:, category_items]
                category_counts = np.sum(category_matrix > 0, axis=0)
                category_sums = np.sum(category_matrix, axis=0)
                
                # Sort by popularity within category
                category_popularity = []
                for i, item_idx in enumerate(category_items):
                    count = category_counts[i]
                    avg_rating = category_sums[i] / count if count > 0 else 0
                    popularity = count * avg_rating
                    category_popularity.append((item_idx, popularity))
                
                # Sort and store
                category_popularity.sort(key=lambda x: x[1], reverse=True)
                self.category_popularity[category] = [item_id for item_id, _ in category_popularity]
        
        self.logger.info(f"Calculated category popularity for {len(self.category_popularity)} categories")
    
    def _build_demographic_profiles(self, user_item_matrix: np.ndarray,
                                  user_demographics: pd.DataFrame):
        """Build preference profiles for different demographic groups"""
        
        self.demographic_profiles = {}
        
        # Group by demographic features
        demographic_columns = ['age_group', 'gender', 'location']
        available_columns = [col for col in demographic_columns if col in user_demographics.columns]
        
        for column in available_columns:
            self.demographic_profiles[column] = {}
            
            for group in user_demographics[column].unique():
                group_users = user_demographics[user_demographics[column] == group].index.tolist()
                
                if group_users:
                    # Calculate average preferences for this demographic group
                    group_matrix = user_item_matrix[group_users]
                    group_preferences = np.mean(group_matrix, axis=0)
                    
                    # Get top items for this group
                    top_items = np.argsort(group_preferences)[::-1][:50]
                    self.demographic_profiles[column][group] = top_items.tolist()
        
        self.logger.info(f"Built demographic profiles for {len(available_columns)} features")
    
    def _select_onboarding_items(self, user_item_matrix: np.ndarray,
                               item_metadata: pd.DataFrame):
        """Select diverse, high-quality items for user onboarding"""
        
        # Criteria for onboarding items:
        # 1. High quality (good ratings)
        # 2. Popular enough to have reliable ratings
        # 3. Diverse across categories
        # 4. Representative of different content types
        
        min_ratings = 20  # Minimum number of ratings
        rating_counts = np.sum(user_item_matrix > 0, axis=0)
        qualified_items = np.where(rating_counts >= min_ratings)[0]
        
        if len(qualified_items) == 0:
            # Fallback to most popular items
            self.onboarding_items = self.popular_items[:20].tolist()
            return
        
        # Calculate quality scores for qualified items
        quality_scores = {}
        for item_id in qualified_items:
            item_ratings = user_item_matrix[:, item_id]
            rated_users = item_ratings > 0
            
            if np.any(rated_users):
                avg_rating = np.mean(item_ratings[rated_users])
                rating_count = np.sum(rated_users)
                
                # Quality score combines average rating and confidence
                quality_score = avg_rating * np.log(rating_count + 1)
                quality_scores[item_id] = quality_score
        
        # Select diverse items across categories
        selected_items = []
        items_per_category = 3
        
        if self.item_categories and 'category' in item_metadata.columns:
            # Select top items from each category
            for category in item_metadata['category'].unique():
                category_items = item_metadata[item_metadata['category'] == category].index.tolist()
                category_qualified = [item for item in category_items if item in quality_scores]
                
                if category_qualified:
                    # Sort by quality score
                    category_qualified.sort(key=lambda x: quality_scores[x], reverse=True)
                    selected_items.extend(category_qualified[:items_per_category])
        
        # Fill remaining slots with top quality items
        remaining_slots = 20 - len(selected_items)
        if remaining_slots > 0:
            all_qualified = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
            for item_id, score in all_qualified:
                if item_id not in selected_items and len(selected_items) < 20:
                    selected_items.append(item_id)
        
        self.onboarding_items = selected_items[:20]
        self.logger.info(f"Selected {len(self.onboarding_items)} onboarding items")
    
    def recommend_for_new_user(self, user_preferences: Dict[str, Any] = None,
                              demographic_info: Dict[str, Any] = None,
                              n_recommendations: int = 10) -> List[int]:
        """
        Recommend items for new users with no interaction history
        
        Args:
            user_preferences: User's stated preferences
            demographic_info: User's demographic information
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended item IDs
        """
        if not self.fitted:
            raise ValueError("Cold start solver must be fitted first")
        
        recommendations = []
        
        # Use demographic information if available
        if demographic_info and self.demographic_profiles:
            for demo_feature, demo_value in demographic_info.items():
                if (demo_feature in self.demographic_profiles and 
                    demo_value in self.demographic_profiles[demo_feature]):
                    
                    demo_recs = self.demographic_profiles[demo_feature][demo_value]
                    recommendations.extend(demo_recs[:n_recommendations//2])
        
        # Use category preferences if provided
        if user_preferences and 'preferred_categories' in user_preferences:
            preferred_categories = user_preferences['preferred_categories']
            
            if self.category_popularity:
                for category in preferred_categories:
                    if category in self.category_popularity:
                        category_recs = self.category_popularity[category]
                        recommendations.extend(category_recs[:3])
        
        # Fill remaining slots with popular items
        for item_id in self.popular_items:
            if item_id not in recommendations and len(recommendations) < n_recommendations:
                recommendations.append(item_id)
        
        return recommendations[:n_recommendations]
    
    def recommend_new_item_to_users(self, new_item_features: Dict[str, Any],
                                   user_item_matrix: np.ndarray,
                                   content_recommender: ProductionContentBasedRecommender = None,
                                   n_target_users: int = 100) -> List[int]:
        """
        Recommend new item to users most likely to be interested
        
        Args:
            new_item_features: Features of the new item
            user_item_matrix: Historical user-item interactions
            content_recommender: Fitted content-based recommender
            n_target_users: Number of users to target
            
        Returns:
            List of user IDs most likely to be interested
        """
        if not self.fitted:
            raise ValueError("Cold start solver must be fitted first")
        
        target_users = []
        
        # Method 1: Use category information
        if 'category' in new_item_features and self.category_popularity:
            new_item_category = new_item_features['category']
            
            if new_item_category in self.category_popularity:
                # Find users who like items in this category
                category_items = self.category_popularity[new_item_category][:10]
                
                user_category_scores = {}
                for user_id in range(user_item_matrix.shape[0]):
                    user_ratings = user_item_matrix[user_id]
                    
                    # Calculate user's affinity for this category
                    category_ratings = [user_ratings[item_id] for item_id in category_items 
                                      if item_id < len(user_ratings) and user_ratings[item_id] > 0]
                    
                    if category_ratings:
                        avg_category_rating = np.mean(category_ratings)
                        num_category_items = len(category_ratings)
                        
                        # Score based on average rating and engagement with category
                        category_score = avg_category_rating * np.log(num_category_items + 1)
                        user_category_scores[user_id] = category_score
                
                # Sort users by category affinity
                sorted_users = sorted(user_category_scores.items(), 
                                    key=lambda x: x[1], reverse=True)
                target_users.extend([user_id for user_id, score in sorted_users[:n_target_users//2]])
        
        # Method 2: Use content similarity (if content recommender available)
        if content_recommender and hasattr(content_recommender, 'item_similarity_matrix'):
            # Find existing items similar to new item (would need to be implemented)
            # This is a simplified version
            pass
        
        # Method 3: Target active users (users with many ratings)
        user_activity = np.sum(user_item_matrix > 0, axis=1)
        active_users = np.argsort(user_activity)[::-1]
        
        # Add active users not already in target list
        for user_id in active_users:
            if user_id not in target_users and len(target_users) < n_target_users:
                target_users.append(user_id)
        
        return target_users[:n_target_users]
    
    def onboarding_recommendations(self, user_responses: Dict[str, Any],
                                 n_recommendations: int = 20) -> List[int]:
        """
        Generate recommendations based on onboarding questionnaire
        
        Args:
            user_responses: User's responses to onboarding questions
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended item IDs for onboarding
        """
        if not self.fitted:
            raise ValueError("Cold start solver must be fitted first")
        
        if self.onboarding_items:
            # Filter onboarding items based on user responses
            filtered_items = []
            
            # Apply category filter if provided
            if 'preferred_categories' in user_responses and self.item_categories:
                preferred_categories = user_responses['preferred_categories']
                
                for item_id in self.onboarding_items:
                    item_category = self.item_categories.get(item_id)
                    if item_category in preferred_categories:
                        filtered_items.append(item_id)
            
            # If not enough filtered items, add popular items
            if len(filtered_items) < n_recommendations:
                for item_id in self.onboarding_items:
                    if item_id not in filtered_items and len(filtered_items) < n_recommendations:
                        filtered_items.append(item_id)
            
            return filtered_items[:n_recommendations]
        else:
            # Fallback to popular items
            return self.popular_items[:n_recommendations].tolist()
# =============================================================================
# COMPREHENSIVE EVALUATION FRAMEWORK
# =============================================================================

class ProductionRecommendationEvaluator:
    """Comprehensive evaluation framework for recommendation systems"""
    
    def __init__(self):
        self.metrics_history = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for evaluator"""
        logger = logging.getLogger('recommendation_evaluator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate_accuracy_metrics(self, predictions: np.ndarray, 
                                 actuals: np.ndarray) -> Dict[str, float]:
        """
        Calculate accuracy metrics for rating predictions
        
        Args:
            predictions: Predicted ratings
            actuals: Actual ratings
            
        Returns:
            Dictionary with accuracy metrics
        """
        if len(predictions) == 0 or len(actuals) == 0:
            return {'rmse': np.inf, 'mae': np.inf, 'correlation': 0.0}
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
        if not np.any(valid_mask):
            return {'rmse': np.inf, 'mae': np.inf, 'correlation': 0.0}
        
        pred_clean = predictions[valid_mask]
        actual_clean = actuals[valid_mask]
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(actual_clean, pred_clean))
        
        # MAE
        mae = mean_absolute_error(actual_clean, pred_clean)
        
        # Correlation
        correlation = np.corrcoef(pred_clean, actual_clean)[0, 1] if len(pred_clean) > 1 else 0.0
        if np.isnan(correlation):
            correlation = 0.0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation
        }
    
    def calculate_ranking_metrics(self, recommended_items: List[int],
                                relevant_items: List[int],
                                k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[int, float]]:
        """
        Calculate ranking metrics for recommendation lists
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant (ground truth) item IDs
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with ranking metrics for each k
        """
        metrics = {
            'precision_at_k': {},
            'recall_at_k': {},
            'f1_at_k': {},
            'ndcg_at_k': {}
        }
        
        relevant_set = set(relevant_items)
        
        for k in k_values:
            # Precision@K
            recommended_k = recommended_items[:k]
            relevant_recommended = len(set(recommended_k) & relevant_set)
            
            precision_k = relevant_recommended / k if k > 0 else 0
            metrics['precision_at_k'][k] = precision_k
            
            # Recall@K
            recall_k = relevant_recommended / len(relevant_items) if len(relevant_items) > 0 else 0
            metrics['recall_at_k'][k] = recall_k
            
            # F1@K
            if precision_k + recall_k > 0:
                f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                f1_k = 0
            metrics['f1_at_k'][k] = f1_k
            
            # NDCG@K
            ndcg_k = self._calculate_ndcg_at_k(recommended_k, relevant_items, k)
            metrics['ndcg_at_k'][k] = ndcg_k
        
        return metrics
    
    def _calculate_ndcg_at_k(self, recommended_items: List[int], 
                           relevant_items: List[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K"""
        
        def dcg_at_k(relevance_scores: List[float], k: int) -> float:
            relevance_scores = relevance_scores[:k]
            if not relevance_scores:
                return 0.0
            
            dcg = relevance_scores[0]
            for i in range(1, len(relevance_scores)):
                dcg += relevance_scores[i] / np.log2(i + 1)
            
            return dcg
        
        # Create relevance scores (1 if relevant, 0 if not)
        relevance_scores = [1.0 if item in relevant_items else 0.0 
                           for item in recommended_items[:k]]
        
        # Calculate DCG
        dcg = dcg_at_k(relevance_scores, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = [1.0] * min(len(relevant_items), k)
        idcg = dcg_at_k(ideal_relevance, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_diversity_metrics(self, all_recommendations: List[List[int]],
                                  item_features: np.ndarray = None,
                                  total_items: int = None) -> Dict[str, float]:
        """
        Calculate diversity and coverage metrics
        
        Args:
            all_recommendations: List of recommendation lists for all users
            item_features: Feature matrix for items (for diversity calculation)
            total_items: Total number of items in catalog
            
        Returns:
            Dictionary with diversity metrics
        """
        metrics = {}
        
        # Catalog coverage
        if total_items:
            unique_recommended = set()
            for rec_list in all_recommendations:
                unique_recommended.update(rec_list)
            
            coverage = len(unique_recommended) / total_items
            metrics['catalog_coverage'] = coverage
        
        # Intra-list diversity (average pairwise distance within recommendation lists)
        if item_features is not None:
            intra_list_diversities = []
            
            for rec_list in all_recommendations:
                if len(rec_list) < 2:
                    continue
                
                # Calculate pairwise distances
                distances = []
                for i in range(len(rec_list)):
                    for j in range(i + 1, len(rec_list)):
                        item1_features = item_features[rec_list[i]]
                        item2_features = item_features[rec_list[j]]
                        
                        # Cosine distance
                        similarity = cosine_similarity([item1_features], [item2_features])[0, 0]
                        distance = 1 - similarity
                        distances.append(distance)
                
                if distances:
                    avg_distance = np.mean(distances)
                    intra_list_diversities.append(avg_distance)
            
            if intra_list_diversities:
                metrics['intra_list_diversity'] = np.mean(intra_list_diversities)
        
        # Gini coefficient for recommendation distribution
        item_recommendation_counts = Counter()
        for rec_list in all_recommendations:
            for item_id in rec_list:
                item_recommendation_counts[item_id] += 1
        
        if item_recommendation_counts:
            counts = list(item_recommendation_counts.values())
            gini = self._calculate_gini_coefficient(counts)
            metrics['gini_coefficient'] = gini
        
        return metrics
    
    def _calculate_gini_coefficient(self, values: List[int]) -> float:
        """Calculate Gini coefficient for measuring inequality"""
        
        if not values:
            return 0.0
        
        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_values)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return gini
    
    def cross_validate_recommender(self, recommender: Any, 
                                 user_item_matrix: np.ndarray,
                                 item_descriptions: List[str] = None,
                                 n_folds: int = 5,
                                 test_ratio: float = 0.2) -> Dict[str, List[float]]:
        """
        Perform cross-validation for recommendation system
        
        Args:
            recommender: Recommendation system to evaluate
            user_item_matrix: User-item interaction matrix
            item_descriptions: Item descriptions (for content-based methods)
            n_folds: Number of cross-validation folds
            test_ratio: Ratio of ratings to use for testing
            
        Returns:
            Dictionary with metrics for each fold
        """
        self.logger.info(f"Starting {n_folds}-fold cross-validation")
        
        results = {
            'rmse': [],
            'mae': [],
            'precision_at_10': [],
            'recall_at_10': [],
            'ndcg_at_10': []
        }
        
        # Get all non-zero entries
        user_ids, item_ids = np.where(user_item_matrix > 0)
        ratings = user_item_matrix[user_ids, item_ids]
        
        # Create data array
        data = np.column_stack([user_ids, item_ids, ratings])
        
        # Perform k-fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            self.logger.info(f"Processing fold {fold + 1}/{n_folds}")
            
            # Split data
            train_data = data[train_idx]
            test_data = data[test_idx]
            
            # Create training matrix
            train_matrix = np.zeros_like(user_item_matrix)
            for user_id, item_id, rating in train_data:
                train_matrix[int(user_id), int(item_id)] = rating
            
            try:
                # Fit recommender on training data
                if hasattr(recommender, 'fit'):
                    if item_descriptions is not None:
                        recommender.fit(train_matrix, item_descriptions)
                    else:
                        recommender.fit(train_matrix)
                
                # Evaluate on test data
                fold_rmse = []
                fold_mae = []
                fold_precision = []
                fold_recall = []
                fold_ndcg = []
                
                # Group test data by user
                test_by_user = defaultdict(list)
                for user_id, item_id, rating in test_data:
                    test_by_user[int(user_id)].append((int(item_id), rating))
                
                for user_id, user_test_items in test_by_user.items():
                    if len(user_test_items) < 2:  # Need at least 2 items for meaningful evaluation
                        continue
                    
                    # Predict ratings for test items
                    predictions = []
                    actuals = []
                    
                    for item_id, actual_rating in user_test_items:
                        try:
                            if hasattr(recommender, 'predict_rating'):
                                pred_rating = recommender.predict_rating(user_id, item_id)
                            else:
                                pred_rating = 3.0  # Default prediction
                            
                            predictions.append(pred_rating)
                            actuals.append(actual_rating)
                        except:
                            continue
                    
                    if predictions and actuals:
                        # Calculate accuracy metrics
                        accuracy_metrics = self.calculate_accuracy_metrics(
                            np.array(predictions), np.array(actuals)
                        )
                        fold_rmse.append(accuracy_metrics['rmse'])
                        fold_mae.append(accuracy_metrics['mae'])
                    
                    # Generate recommendations for ranking metrics
                    try:
                        if hasattr(recommender, 'recommend_items'):
                            recommendations = recommender.recommend_items(user_id, train_matrix, 10)
                            if isinstance(recommendations[0], tuple):
                                recommended_items = [item_id for item_id, score in recommendations]
                            else:
                                recommended_items = recommendations
                        else:
                            recommended_items = []
                        
                        # Get relevant items (items with high ratings in test set)
                        relevant_items = [item_id for item_id, rating in user_test_items if rating >= 4.0]
                        
                        if recommended_items and relevant_items:
                            ranking_metrics = self.calculate_ranking_metrics(
                                recommended_items, relevant_items, [10]
                            )
                            fold_precision.append(ranking_metrics['precision_at_k'][10])
                            fold_recall.append(ranking_metrics['recall_at_k'][10])
                            fold_ndcg.append(ranking_metrics['ndcg_at_k'][10])
                    except:
                        continue
                
                # Store fold results
                if fold_rmse:
                    results['rmse'].append(np.mean(fold_rmse))
                if fold_mae:
                    results['mae'].append(np.mean(fold_mae))
                if fold_precision:
                    results['precision_at_10'].append(np.mean(fold_precision))
                if fold_recall:
                    results['recall_at_10'].append(np.mean(fold_recall))
                if fold_ndcg:
                    results['ndcg_at_10'].append(np.mean(fold_ndcg))
                
            except Exception as e:
                self.logger.error(f"Error in fold {fold + 1}: {e}")
                continue
        
        # Calculate summary statistics
        summary_results = {}
        for metric, values in results.items():
            if values:
                summary_results[f'{metric}_mean'] = np.mean(values)
                summary_results[f'{metric}_std'] = np.std(values)
                summary_results[f'{metric}_values'] = values
        
        self.logger.info("Cross-validation completed")
        return summary_results
    
    def evaluate_business_metrics(self, recommendations: Dict[int, List[int]],
                                user_interactions: Dict[int, List[int]],
                                time_window_days: int = 30) -> Dict[str, float]:
        """
        Evaluate business-relevant metrics
        
        Args:
            recommendations: Dictionary mapping user_id to recommended items
            user_interactions: Dictionary mapping user_id to items they interacted with
            time_window_days: Time window for measuring interactions
            
        Returns:
            Dictionary with business metrics
        """
        metrics = {}
        
        # Click-through rate (CTR)
        total_recommendations = 0
        total_clicks = 0
        
        for user_id, rec_items in recommendations.items():
            user_clicks = user_interactions.get(user_id, [])
            
            total_recommendations += len(rec_items)
            total_clicks += len(set(rec_items) & set(user_clicks))
        
        ctr = total_clicks / total_recommendations if total_recommendations > 0 else 0
        metrics['click_through_rate'] = ctr
        
        # Conversion rate (assuming interactions represent conversions)
        total_users = len(recommendations)
        converted_users = len([user_id for user_id in recommendations 
                             if user_id in user_interactions and user_interactions[user_id]])
        
        conversion_rate = converted_users / total_users if total_users > 0 else 0
        metrics['conversion_rate'] = conversion_rate
        
        # Average recommendations per user
        avg_recs_per_user = np.mean([len(recs) for recs in recommendations.values()])
        metrics['avg_recommendations_per_user'] = avg_recs_per_user
        
        return metrics
# =============================================================================
# SYNTHETIC DATA GENERATION FOR TESTING
# =============================================================================

def generate_synthetic_recommendation_data(n_users=1000, n_items=500, n_ratings=50000,
                                         rating_scale=(1, 5), sparsity=0.95,
                                         random_state=42):
    """
    Generate synthetic recommendation data for testing
    
    Args:
        n_users: Number of users
        n_items: Number of items
        n_ratings: Number of ratings to generate
        rating_scale: Tuple of (min_rating, max_rating)
        sparsity: Sparsity level (fraction of missing ratings)
        random_state: Random seed
        
    Returns:
        Tuple of (user_item_matrix, item_descriptions, item_metadata, user_demographics)
    """
    np.random.seed(random_state)
    
    # Generate user-item matrix
    user_item_matrix = np.zeros((n_users, n_items))
    
    # Generate ratings with some structure
    for _ in range(n_ratings):
        user_id = np.random.randint(0, n_users)
        item_id = np.random.randint(0, n_items)
        
        # Add some user and item biases
        user_bias = np.random.normal(0, 0.5)
        item_bias = np.random.normal(0, 0.5)
        
        # Generate rating
        base_rating = 3.0  # Average rating
        rating = base_rating + user_bias + item_bias + np.random.normal(0, 0.5)
        rating = np.clip(rating, rating_scale[0], rating_scale[1])
        
        user_item_matrix[user_id, item_id] = rating
    
    # Generate item descriptions
    genres = ['action', 'comedy', 'drama', 'horror', 'romance', 'sci-fi', 'thriller', 'documentary']
    adjectives = ['amazing', 'thrilling', 'heartwarming', 'suspenseful', 'hilarious', 'touching', 'exciting']
    
    item_descriptions = []
    for i in range(n_items):
        genre = np.random.choice(genres)
        adjective = np.random.choice(adjectives)
        description = f"An {adjective} {genre} movie with great storytelling and memorable characters."
        item_descriptions.append(description)
    
    # Generate item metadata
    item_metadata = pd.DataFrame({
        'item_id': range(n_items),
        'category': [np.random.choice(genres) for _ in range(n_items)],
        'year': np.random.randint(1990, 2024, n_items),
        'duration': np.random.randint(80, 180, n_items),
        'budget': np.random.lognormal(15, 1, n_items)  # Log-normal distribution for budget
    })
    
    # Generate user demographics
    age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
    genders = ['M', 'F', 'Other']
    locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'Other']
    
    user_demographics = pd.DataFrame({
        'user_id': range(n_users),
        'age_group': [np.random.choice(age_groups) for _ in range(n_users)],
        'gender': [np.random.choice(genders) for _ in range(n_users)],
        'location': [np.random.choice(locations) for _ in range(n_users)]
    })
    
    return user_item_matrix, item_descriptions, item_metadata, user_demographics

# =============================================================================
# COMPREHENSIVE INTEGRATION TEST
# =============================================================================

def comprehensive_recommendation_test():
    """Comprehensive test of all recommendation system components"""
    
    print("=" * 80)
    print("COMPREHENSIVE RECOMMENDATION SYSTEMS TEST")
    print("=" * 80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic StreamFlix data...")
    user_item_matrix, item_descriptions, item_metadata, user_demographics = generate_synthetic_recommendation_data(
        n_users=500, n_items=200, n_ratings=10000
    )
    
    print(f"Generated data: {user_item_matrix.shape[0]} users, {user_item_matrix.shape[1]} items")
    print(f"Sparsity: {(np.sum(user_item_matrix == 0) / user_item_matrix.size) * 100:.1f}%")
    
    # Split data for evaluation
    train_matrix, test_matrix = train_test_split_matrix(user_item_matrix, test_ratio=0.2)
    
    # Test 1: Collaborative Filtering
    print("\n2. Testing Collaborative Filtering...")
    print("-" * 50)
    
    # User-based collaborative filtering
    print("Testing User-based Collaborative Filtering...")
    user_cf = ProductionCollaborativeFiltering(method='user_based')
    user_cf.fit(train_matrix)
    
    # Test recommendations
    test_user = 0
    user_recs = user_cf.recommend_items(test_user, n_recommendations=10)
    print(f"User-based CF recommendations for user {test_user}: {len(user_recs)} items")
    
    # Item-based collaborative filtering
    print("Testing Item-based Collaborative Filtering...")
    item_cf = ProductionCollaborativeFiltering(method='item_based')
    item_cf.fit(train_matrix)
    
    item_recs = item_cf.recommend_items(test_user, n_recommendations=10)
    print(f"Item-based CF recommendations for user {test_user}: {len(item_recs)} items")
    
    # Test 2: Matrix Factorization
    print("\n3. Testing Matrix Factorization...")
    print("-" * 50)
    
    mf = ProductionMatrixFactorization(n_factors=20, n_iterations=30)
    mf.fit(train_matrix)
    
    mf_recs = mf.recommend_items(test_user, train_matrix, n_recommendations=10)
    print(f"Matrix Factorization recommendations: {len(mf_recs)} items")
    print(f"Training history: {len(mf.training_history)} iterations")
    
    # Test 3: Content-Based Filtering
    print("\n4. Testing Content-Based Filtering...")
    print("-" * 50)
    
    content_recommender = ProductionContentBasedRecommender()
    content_recommender.fit(item_descriptions, item_metadata)
    
    content_recs = content_recommender.recommend_for_user(train_matrix, test_user, n_recommendations=10)
    print(f"Content-based recommendations: {len(content_recs)} items")
    
    # Test similar items
    similar_items = content_recommender.find_similar_items(0, n_similar=5)
    print(f"Items similar to item 0: {len(similar_items)} items")
    
    # Test 4: Hybrid System
    print("\n5. Testing Hybrid Recommendation System...")
    print("-" * 50)
    
    hybrid = ProductionHybridRecommender(
        collaborative_weight=0.4,
        content_weight=0.3,
        matrix_factorization_weight=0.3
    )
    hybrid.fit(train_matrix, item_descriptions, item_metadata)
    
    hybrid_recs = hybrid.recommend_items(test_user, train_matrix, n_recommendations=10)
    print(f"Hybrid recommendations: {len(hybrid_recs)} items")
    
    # Test explanation
    if hybrid_recs:
        explanation = hybrid.explain_recommendation(test_user, hybrid_recs[0][0], train_matrix)
        print(f"Explanation for top recommendation: {explanation['explanation_text']}")
    
    # Test 5: Cold Start Solutions
    print("\n6. Testing Cold Start Solutions...")
    print("-" * 50)
    
    cold_start = ProductionColdStartSolver()
    cold_start.fit(train_matrix, item_metadata, user_demographics)
    
    # New user recommendations
    new_user_prefs = {'preferred_categories': ['action', 'sci-fi']}
    new_user_recs = cold_start.recommend_for_new_user(new_user_prefs, n_recommendations=10)
    print(f"New user recommendations: {len(new_user_recs)} items")
    
    # Onboarding recommendations
    onboarding_responses = {'preferred_categories': ['comedy', 'drama']}
    onboarding_recs = cold_start.onboarding_recommendations(onboarding_responses, n_recommendations=15)
    print(f"Onboarding recommendations: {len(onboarding_recs)} items")
    
    # Test 6: Comprehensive Evaluation
    print("\n7. Testing Evaluation Framework...")
    print("-" * 50)
    
    evaluator = ProductionRecommendationEvaluator()
    
    # Test accuracy metrics
    predictions = np.array([3.5, 4.2, 2.8, 4.0, 3.1])
    actuals = np.array([4.0, 4.5, 3.0, 3.8, 3.2])
    accuracy_metrics = evaluator.calculate_accuracy_metrics(predictions, actuals)
    print(f"Accuracy metrics - RMSE: {accuracy_metrics['rmse']:.3f}, MAE: {accuracy_metrics['mae']:.3f}")
    
    # Test ranking metrics
    recommended = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    relevant = [1, 3, 4, 8, 12, 15]
    ranking_metrics = evaluator.calculate_ranking_metrics(recommended, relevant, [5, 10])
    print(f"Ranking metrics - Precision@10: {ranking_metrics['precision_at_k'][10]:.3f}")
    print(f"                  Recall@10: {ranking_metrics['recall_at_k'][10]:.3f}")
    
    # Test cross-validation (simplified)
    print("Running simplified cross-validation...")
    cv_results = evaluator.cross_validate_recommender(
        user_cf, train_matrix, n_folds=3
    )
    if 'rmse_mean' in cv_results:
        print(f"Cross-validation RMSE: {cv_results['rmse_mean']:.3f} ± {cv_results['rmse_std']:.3f}")
    
    # Performance Comparison
    print("\n8. Performance Comparison...")
    print("-" * 50)
    
    methods = [
        ('User-based CF', user_cf),
        ('Item-based CF', item_cf),
        ('Matrix Factorization', mf),
        ('Hybrid System', hybrid)
    ]
    
    print("Method Performance Summary:")
    for method_name, method in methods:
        try:
            # Generate recommendations for a sample of users
            sample_users = range(min(10, train_matrix.shape[0]))
            total_recs = 0
            
            for user_id in sample_users:
                if hasattr(method, 'recommend_items'):
                    if method_name == 'Hybrid System':
                        recs = method.recommend_items(user_id, train_matrix, 5)
                    else:
                        recs = method.recommend_items(user_id, 5)
                    total_recs += len(recs)
            
            avg_recs = total_recs / len(sample_users)
            print(f"  {method_name}: {avg_recs:.1f} avg recommendations per user")
            
        except Exception as e:
            print(f"  {method_name}: Error - {e}")
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RECOMMENDATION SYSTEMS TEST COMPLETED!")
    print("=" * 80)
    
    return True

def train_test_split_matrix(user_item_matrix, test_ratio=0.2, random_state=42):
    """Split user-item matrix into train and test sets"""
    
    np.random.seed(random_state)
    
    train_matrix = user_item_matrix.copy()
    test_matrix = np.zeros_like(user_item_matrix)
    
    # For each user, randomly select some ratings for test set
    for user_id in range(user_item_matrix.shape[0]):
        user_ratings = np.where(user_item_matrix[user_id] > 0)[0]
        
        if len(user_ratings) > 1:
            n_test = max(1, int(len(user_ratings) * test_ratio))
            test_items = np.random.choice(user_ratings, n_test, replace=False)
            
            for item_id in test_items:
                test_matrix[user_id, item_id] = user_item_matrix[user_id, item_id]
                train_matrix[user_id, item_id] = 0
    
    return train_matrix, test_matrix

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Main execution demonstrating comprehensive recommendation systems
    """
    
    print("Recommendation Systems - Production Implementation")
    print("=" * 60)
    
    # Check library availability
    print("\nLibrary Availability Check:")
    print(f"📊 Scikit-learn: ✅ Available")
    print(f"📈 Pandas/Numpy: ✅ Available")
    print(f"🤖 Implicit (ALS): {'✅ Available' if IMPLICIT_AVAILABLE else '❌ Not Available'}")
    print(f"🧠 TensorFlow (Neural CF): {'✅ Available' if TF_AVAILABLE else '❌ Not Available'}")
    print(f"💾 Redis (Caching): {'✅ Available' if REDIS_AVAILABLE else '❌ Not Available'}")
    
    # Run comprehensive test
    try:
        success = comprehensive_recommendation_test()
        
        if success:
            print("\n🎉 All tests completed successfully!")
            print("\nKey Takeaways:")
            print("• Collaborative filtering leverages user behavior but suffers from sparsity")
            print("• Content-based filtering handles new items but may lack diversity")
            print("• Matrix factorization captures latent factors and scales well")
            print("• Hybrid approaches combine strengths and improve robustness")
            print("• Cold start solutions are essential for practical systems")
            
            print("\nProduction Considerations:")
            print("• Implement caching for real-time serving performance")
            print("• Use approximate algorithms for scalability")
            print("• Monitor recommendation quality and user engagement")
            print("• Handle concept drift and changing user preferences")
            print("• Ensure diversity and avoid filter bubbles")
            
            print("\nNext Steps:")
            print("• Deploy recommendation API with A/B testing")
            print("• Implement real-time model updates")
            print("• Set up recommendation quality monitoring")
            print("• Build explanation and transparency features")
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)