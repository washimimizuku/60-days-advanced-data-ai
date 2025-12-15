"""
Day 29: Recommendation Systems - Collaborative & Content-based Methods - Exercise

Business Scenario:
You're the Lead ML Engineer at StreamFlix, a video streaming platform with 10 million users
and 50,000 movies/TV shows. The product team has tasked you with building a comprehensive
recommendation system to increase user engagement and reduce churn.

Current challenges:
1. 70% of users watch <5 shows per month (sparse interaction data)
2. 1000+ new titles added weekly (cold start for new content)
3. 50,000+ new users sign up daily (cold start for new users)
4. Current system shows same popular content to everyone (no personalization)
5. Users complain about lack of diversity in recommendations

Your mission: Build a production-ready recommendation system that handles:
- Collaborative filtering for personalized recommendations
- Content-based filtering for new content and user preferences
- Hybrid approaches combining multiple methods
- Cold start solutions for new users and items
- Real-time serving with sub-100ms latency
- A/B testing framework for continuous optimization

Success Criteria:
- Increase average watch time by 25%
- Improve user retention by 15%
- Achieve 95th percentile response time <100ms
- Handle 10,000+ recommendation requests per second
- Maintain recommendation diversity score >0.7
"""

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Advanced libraries (handle gracefully if not available)
try:
    import implicit
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False
    print("Implicit library not available. Install with: pip install implicit")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Neural collaborative filtering will be skipped.")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis not available. Caching features will be simulated.")

import logging
import time
import json
from collections import defaultdict
# =============================================================================
# TASK 1: COLLABORATIVE FILTERING IMPLEMENTATION
# =============================================================================

class CollaborativeFilteringRecommender:
    """
    Implement collaborative filtering algorithms for StreamFlix
    """
    
    def __init__(self, method='user_based', n_neighbors=50):
        self.method = method
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.fitted = False
        
    def fit(self, user_item_matrix: np.ndarray) -> 'CollaborativeFilteringRecommender':
        """
        Fit collaborative filtering model
        """
        self.user_item_matrix = user_item_matrix
        
        if self.method == 'user_based':
            # Calculate user-user similarity
            self.similarity_matrix = cosine_similarity(user_item_matrix)
        elif self.method == 'item_based':
            # Calculate item-item similarity
            self.similarity_matrix = cosine_similarity(user_item_matrix.T)
        else:
            raise ValueError("Method must be 'user_based' or 'item_based'")
        
        self.fitted = True
        return self
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a specific user-item pair
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.method == 'user_based':
            # Find similar users who rated this item
            item_ratings = self.user_item_matrix[:, item_id]
            user_similarities = self.similarity_matrix[user_id]
            
            # Get users who rated this item (non-zero ratings)
            rated_mask = item_ratings != 0
            
            if not np.any(rated_mask):
                return 0.0  # No one rated this item
            
            # Calculate weighted average
            similarities = user_similarities[rated_mask]
            ratings = item_ratings[rated_mask]
            
            if np.sum(np.abs(similarities)) == 0:
                return np.mean(ratings)
            
            return np.sum(similarities * ratings) / np.sum(np.abs(similarities))
        
        else:  # item_based
            # Find similar items that this user rated
            user_ratings = self.user_item_matrix[user_id, :]
            item_similarities = self.similarity_matrix[item_id]
            
            # Get items this user rated (non-zero ratings)
            rated_mask = user_ratings != 0
            
            if not np.any(rated_mask):
                return 0.0  # User hasn't rated anything
            
            # Calculate weighted average
            similarities = item_similarities[rated_mask]
            ratings = user_ratings[rated_mask]
            
            if np.sum(np.abs(similarities)) == 0:
                return np.mean(ratings)
            
            return np.sum(similarities * ratings) / np.sum(np.abs(similarities))
    
    def recommend_items(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """
        Generate item recommendations for a user
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        user_ratings = self.user_item_matrix[user_id, :]
        unrated_items = np.where(user_ratings == 0)[0]
        
        if len(unrated_items) == 0:
            return []  # User has rated all items
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict_rating(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in predictions[:n_recommendations]]
    
    def get_similar_users(self, user_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        TODO: Find users similar to the given user
        
        Args:
            user_id: ID of the user
            n_similar: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
            
        Hints:
        - Use the similarity matrix calculated during fit
        - Sort by similarity score (excluding the user themselves)
        - Return top N similar users with their similarity scores
        """
        # TODO: Get similarity scores for the user
        # TODO: Sort and return top similar users
        pass

# =============================================================================
# TASK 2: MATRIX FACTORIZATION IMPLEMENTATION
# =============================================================================

class MatrixFactorizationRecommender:
    """
    Implement matrix factorization using SVD for scalable recommendations
    """
    
    def __init__(self, n_factors: int = 50, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.fitted = False
        
    def fit(self, user_item_matrix: np.ndarray) -> 'MatrixFactorizationRecommender':
        """
        TODO: Fit SVD model to user-item matrix
        
        Args:
            user_item_matrix: Matrix where rows are users, columns are items
            
        Returns:
            Self for method chaining
            
        Hints:
        - Handle missing values (zeros) by replacing with global mean
        - Use TruncatedSVD for dimensionality reduction
        - Store user and item factor matrices
        - Calculate global mean for baseline predictions
        """
        # TODO: Calculate global mean of non-zero ratings
        # TODO: Handle missing values appropriately
        # TODO: Fit TruncatedSVD model
        # TODO: Extract user and item factors
        pass
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        TODO: Predict rating using matrix factorization
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            
        Returns:
            Predicted rating
            
        Hints:
        - Use dot product of user and item factors
        - Add global mean as baseline
        - Clip predictions to valid rating range
        """
        # TODO: Check if model is fitted
        # TODO: Calculate prediction using dot product
        # TODO: Add baseline and clip to valid range
        pass
    
    def recommend_items(self, user_id: int, user_item_matrix: np.ndarray, 
                       n_recommendations: int = 10) -> List[int]:
        """
        TODO: Generate recommendations using matrix factorization
        
        Args:
            user_id: ID of the user
            user_item_matrix: Original user-item matrix
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended item IDs
            
        Hints:
        - Find items the user hasn't rated
        - Predict ratings for unrated items
        - Sort by predicted rating and return top N
        """
        # TODO: Find unrated items
        # TODO: Predict ratings for unrated items
        # TODO: Sort and return top recommendations
        pass
    
    def get_item_factors(self) -> np.ndarray:
        """
        TODO: Return item factor matrix for similarity calculations
        
        Returns:
            Item factor matrix
        """
        # TODO: Return item factors if model is fitted
        pass

# =============================================================================
# TASK 3: CONTENT-BASED FILTERING IMPLEMENTATION
# =============================================================================

class ContentBasedRecommender:
    """
    Implement content-based filtering using item features
    """
    
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.item_features_matrix = None
        self.item_similarity_matrix = None
        self.fitted = False
        
    def fit(self, item_descriptions: List[str], item_metadata: pd.DataFrame = None) -> 'ContentBasedRecommender':
        """
        TODO: Fit content-based model using item descriptions and metadata
        
        Args:
            item_descriptions: List of text descriptions for each item
            item_metadata: DataFrame with additional item features (optional)
            
        Returns:
            Self for method chaining
            
        Hints:
        - Use TF-IDF vectorization for text descriptions
        - Combine with numerical metadata if provided
        - Calculate item-item similarity matrix
        - Store all components for later use
        """
        # TODO: Initialize and fit TF-IDF vectorizer
        # TODO: Transform item descriptions to feature matrix
        # TODO: Combine with metadata if provided
        # TODO: Calculate item similarity matrix
        pass
    
    def find_similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        TODO: Find items similar to the given item
        
        Args:
            item_id: ID of the item
            n_similar: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
            
        Hints:
        - Use the item similarity matrix
        - Sort by similarity score (excluding the item itself)
        - Return top N similar items with scores
        """
        # TODO: Get similarity scores for the item
        # TODO: Sort and return top similar items
        pass
    
    def recommend_for_user(self, user_item_matrix: np.ndarray, user_id: int, 
                          n_recommendations: int = 10, min_rating: float = 3.0) -> List[int]:
        """
        TODO: Recommend items based on user's content preferences
        
        Args:
            user_item_matrix: User-item interaction matrix
            user_id: ID of the user
            n_recommendations: Number of recommendations to return
            min_rating: Minimum rating to consider as "liked"
            
        Returns:
            List of recommended item IDs
            
        Hints:
        - Find items the user liked (rating >= min_rating)
        - For each liked item, find similar items
        - Aggregate similarity scores weighted by user ratings
        - Remove items the user has already rated
        - Return top N items by aggregated score
        """
        # TODO: Find items user liked
        # TODO: Calculate aggregated similarity scores
        # TODO: Remove already rated items
        # TODO: Return top recommendations
        pass
    
    def create_user_profile(self, user_item_matrix: np.ndarray, user_id: int) -> np.ndarray:
        """
        TODO: Create user profile based on liked items' features
        
        Args:
            user_item_matrix: User-item interaction matrix
            user_id: ID of the user
            
        Returns:
            User profile vector
            
        Hints:
        - Find items the user rated positively
        - Weight item features by user ratings
        - Aggregate to create user profile vector
        - Normalize the profile vector
        """
        # TODO: Find positively rated items
        # TODO: Weight and aggregate item features
        # TODO: Normalize and return user profile
        pass

# =============================================================================
# TASK 4: HYBRID RECOMMENDATION SYSTEM
# =============================================================================

class HybridRecommendationSystem:
    """
    Combine multiple recommendation approaches for robust performance
    """
    
    def __init__(self, collaborative_weight: float = 0.5, 
                 content_weight: float = 0.3, 
                 popularity_weight: float = 0.2):
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight
        
        self.collaborative_recommender = None
        self.content_recommender = None
        self.matrix_factorization = None
        self.popularity_baseline = None
        self.fitted = False
        
    def fit(self, user_item_matrix: np.ndarray, 
            item_descriptions: List[str],
            item_metadata: pd.DataFrame = None) -> 'HybridRecommendationSystem':
        """
        TODO: Fit all component recommenders
        
        Args:
            user_item_matrix: User-item interaction matrix
            item_descriptions: Item text descriptions
            item_metadata: Additional item metadata
            
        Returns:
            Self for method chaining
            
        Hints:
        - Initialize and fit collaborative filtering recommender
        - Initialize and fit content-based recommender
        - Initialize and fit matrix factorization model
        - Calculate popularity baseline (most popular items)
        - Set fitted flag
        """
        # TODO: Initialize and fit collaborative filtering
        # TODO: Initialize and fit content-based filtering
        # TODO: Initialize and fit matrix factorization
        # TODO: Calculate popularity baseline
        pass
    
    def recommend_items(self, user_id: int, user_item_matrix: np.ndarray,
                       n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        TODO: Generate hybrid recommendations combining all methods
        
        Args:
            user_id: ID of the user
            user_item_matrix: User-item interaction matrix
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (item_id, combined_score) tuples
            
        Hints:
        - Get recommendations from each component method
        - Combine scores using weighted average
        - Handle cases where methods return different items
        - Normalize scores before combining
        - Return top N items by combined score
        """
        # TODO: Get recommendations from each method
        # TODO: Combine scores using weights
        # TODO: Sort and return top recommendations
        pass
    
    def explain_recommendation(self, user_id: int, item_id: int, 
                             user_item_matrix: np.ndarray) -> Dict[str, Any]:
        """
        TODO: Provide explanation for why an item was recommended
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            user_item_matrix: User-item interaction matrix
            
        Returns:
            Dictionary with explanation components
            
        Hints:
        - Calculate contribution from each recommendation method
        - Find similar users/items that influenced the recommendation
        - Provide human-readable explanations
        - Include confidence scores
        """
        # TODO: Calculate method contributions
        # TODO: Find influential similar users/items
        # TODO: Create explanation dictionary
        pass

# =============================================================================
# TASK 5: COLD START PROBLEM SOLUTIONS
# =============================================================================

class ColdStartSolver:
    """
    Handle cold start problems for new users and items
    """
    
    def __init__(self):
        self.popular_items = None
        self.item_categories = None
        self.demographic_profiles = None
        self.fitted = False
        
    def fit(self, user_item_matrix: np.ndarray, 
            item_metadata: pd.DataFrame,
            user_demographics: pd.DataFrame = None) -> 'ColdStartSolver':
        """
        TODO: Fit cold start solutions using available data
        
        Args:
            user_item_matrix: User-item interaction matrix
            item_metadata: Item metadata including categories
            user_demographics: User demographic information
            
        Returns:
            Self for method chaining
            
        Hints:
        - Calculate item popularity scores
        - Extract item categories for category-based recommendations
        - Build demographic profiles if user data available
        - Store all components for cold start scenarios
        """
        # TODO: Calculate item popularity
        # TODO: Extract item categories
        # TODO: Build demographic profiles
        pass
    
    def recommend_for_new_user(self, user_preferences: Dict[str, Any] = None,
                              n_recommendations: int = 10) -> List[int]:
        """
        TODO: Recommend items for new users with no interaction history
        
        Args:
            user_preferences: User's stated preferences (categories, genres, etc.)
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended item IDs
            
        Hints:
        - If preferences provided, filter by preferred categories
        - Otherwise, return most popular items
        - Consider diversity in recommendations
        - Handle edge cases gracefully
        """
        # TODO: Check if preferences are provided
        # TODO: Filter by preferences or use popularity
        # TODO: Ensure diversity in recommendations
        pass
    
    def recommend_new_item_to_users(self, new_item_features: Dict[str, Any],
                                   user_item_matrix: np.ndarray,
                                   n_target_users: int = 100) -> List[int]:
        """
        TODO: Recommend new item to users most likely to be interested
        
        Args:
            new_item_features: Features of the new item
            user_item_matrix: Historical user-item interactions
            n_target_users: Number of users to target
            
        Returns:
            List of user IDs most likely to be interested
            
        Hints:
        - Find existing items similar to new item
        - Identify users who liked similar items
        - Score users based on their affinity for similar content
        - Return top users by affinity score
        """
        # TODO: Find similar existing items
        # TODO: Identify users who liked similar items
        # TODO: Calculate user affinity scores
        # TODO: Return top target users
        pass
    
    def onboarding_recommendations(self, user_responses: Dict[str, Any],
                                 n_recommendations: int = 20) -> List[int]:
        """
        TODO: Generate recommendations based on onboarding questionnaire
        
        Args:
            user_responses: User's responses to onboarding questions
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended item IDs for onboarding
            
        Hints:
        - Map user responses to item features
        - Find items matching user preferences
        - Include popular items from preferred categories
        - Ensure good coverage of different types of content
        """
        # TODO: Map responses to preferences
        # TODO: Find matching items
        # TODO: Include popular items from preferred categories
        pass

# =============================================================================
# TASK 6: EVALUATION AND METRICS
# =============================================================================

class RecommendationEvaluator:
    """
    Comprehensive evaluation framework for recommendation systems
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def calculate_accuracy_metrics(self, predictions: np.ndarray, 
                                 actuals: np.ndarray) -> Dict[str, float]:
        """
        TODO: Calculate accuracy metrics for rating predictions
        
        Args:
            predictions: Predicted ratings
            actuals: Actual ratings
            
        Returns:
            Dictionary with accuracy metrics
            
        Hints:
        - Calculate RMSE (Root Mean Square Error)
        - Calculate MAE (Mean Absolute Error)
        - Calculate R-squared correlation
        - Handle edge cases (empty arrays, etc.)
        """
        # TODO: Calculate RMSE
        # TODO: Calculate MAE
        # TODO: Calculate correlation
        pass
    
    def calculate_ranking_metrics(self, recommended_items: List[int],
                                relevant_items: List[int],
                                k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[int, float]]:
        """
        TODO: Calculate ranking metrics for recommendation lists
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant (ground truth) item IDs
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with ranking metrics for each k
            
        Hints:
        - Calculate Precision@K for each k value
        - Calculate Recall@K for each k value
        - Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        - Calculate F1@K score
        """
        # TODO: Calculate Precision@K
        # TODO: Calculate Recall@K
        # TODO: Calculate NDCG@K
        # TODO: Calculate F1@K
        pass
    
    def calculate_diversity_metrics(self, all_recommendations: List[List[int]],
                                  item_features: np.ndarray) -> Dict[str, float]:
        """
        TODO: Calculate diversity and coverage metrics
        
        Args:
            all_recommendations: List of recommendation lists for all users
            item_features: Feature matrix for items
            
        Returns:
            Dictionary with diversity metrics
            
        Hints:
        - Calculate intra-list diversity (diversity within each recommendation list)
        - Calculate catalog coverage (fraction of items being recommended)
        - Calculate Gini coefficient for recommendation distribution
        - Calculate novelty scores
        """
        # TODO: Calculate intra-list diversity
        # TODO: Calculate catalog coverage
        # TODO: Calculate distribution metrics
        pass
    
    def cross_validate_recommender(self, recommender: Any, 
                                 user_item_matrix: np.ndarray,
                                 n_folds: int = 5) -> Dict[str, List[float]]:
        """
        TODO: Perform cross-validation for recommendation system
        
        Args:
            recommender: Recommendation system to evaluate
            user_item_matrix: User-item interaction matrix
            n_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with metrics for each fold
            
        Hints:
        - Split data into train/test folds
        - For each fold, fit recommender on training data
        - Generate recommendations for test users
        - Calculate metrics for each fold
        - Return aggregated results
        """
        # TODO: Create cross-validation folds
        # TODO: For each fold, train and evaluate
        # TODO: Aggregate results across folds
        pass

# =============================================================================
# MAIN EXERCISE EXECUTION
# =============================================================================

def main():
    """
    Main function to execute the comprehensive recommendation systems exercise
    """
    print("=" * 80)
    print("DAY 29: RECOMMENDATION SYSTEMS - COMPREHENSIVE EXERCISE")
    print("=" * 80)
    
    print("\n🎬 StreamFlix Recommendation System Development")
    print("Building next-generation personalized content discovery...")
    
    # TODO: Task 1 - Implement and Test Collaborative Filtering
    print("\n" + "="*50)
    print("TASK 1: COLLABORATIVE FILTERING")
    print("="*50)
    
    # TODO: Generate synthetic user-item interaction data
    # TODO: Test user-based collaborative filtering
    # TODO: Test item-based collaborative filtering
    # TODO: Compare performance and analyze results
    
    # TODO: Task 2 - Implement and Test Matrix Factorization
    print("\n" + "="*50)
    print("TASK 2: MATRIX FACTORIZATION")
    print("="*50)
    
    # TODO: Test SVD-based matrix factorization
    # TODO: Compare with collaborative filtering
    # TODO: Analyze latent factors and interpretability
    
    # TODO: Task 3 - Implement and Test Content-Based Filtering
    print("\n" + "="*50)
    print("TASK 3: CONTENT-BASED FILTERING")
    print("="*50)
    
    # TODO: Generate item descriptions and metadata
    # TODO: Test content-based recommendations
    # TODO: Analyze content similarity and user profiles
    
    # TODO: Task 4 - Implement and Test Hybrid System
    print("\n" + "="*50)
    print("TASK 4: HYBRID RECOMMENDATION SYSTEM")
    print("="*50)
    
    # TODO: Combine all recommendation methods
    # TODO: Test hybrid recommendations
    # TODO: Analyze method contributions and explanations
    
    # TODO: Task 5 - Test Cold Start Solutions
    print("\n" + "="*50)
    print("TASK 5: COLD START PROBLEM SOLUTIONS")
    print("="*50)
    
    # TODO: Test new user recommendations
    # TODO: Test new item targeting
    # TODO: Test onboarding recommendations
    
    # TODO: Task 6 - Comprehensive Evaluation
    print("\n" + "="*50)
    print("TASK 6: EVALUATION AND METRICS")
    print("="*50)
    
    # TODO: Evaluate all recommendation methods
    # TODO: Compare accuracy, diversity, and coverage
    # TODO: Perform cross-validation
    # TODO: Generate performance report
    
    print("\n" + "="*80)
    print("EXERCISE COMPLETED!")
    print("="*80)
    
    print("\n📊 Key Insights:")
    print("• Collaborative filtering works well with sufficient interaction data")
    print("• Content-based filtering handles new items and provides explainability")
    print("• Matrix factorization scales well and captures latent factors")
    print("• Hybrid approaches combine strengths and improve robustness")
    print("• Cold start solutions are essential for practical systems")
    
    print("\n🎯 Next Steps:")
    print("• Implement the solution with production-grade code")
    print("• Deploy real-time recommendation API")
    print("• Set up A/B testing framework")
    print("• Monitor recommendation performance and user engagement")

if __name__ == "__main__":
    main()