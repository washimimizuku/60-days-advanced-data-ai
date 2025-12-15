# Day 29: Recommendation Systems - Collaborative & Content-based Methods

## 📖 Learning Objectives
By the end of today, you will be able to:
- **Implement** collaborative filtering algorithms (user-based, item-based, matrix factorization)
- **Build** content-based recommendation systems using feature similarity
- **Design** hybrid recommendation systems that combine multiple approaches
- **Handle** the cold start problem for new users and items
- **Deploy** scalable recommendation systems for production environments

⭐ **Difficulty Level**: Advanced  
🕒 **Estimated Time**: 60 minutes  
🛠️ **Prerequisites**: Machine learning fundamentals, linear algebra, feature engineering

---

## 🎯 What are Recommendation Systems?

Recommendation systems are algorithms designed to suggest relevant items to users based on their preferences, behavior, and characteristics. They power the personalized experiences we see across digital platforms:

- **E-commerce**: Product recommendations on Amazon, eBay
- **Entertainment**: Movie/TV show suggestions on Netflix, Spotify playlists
- **Social Media**: Friend suggestions on Facebook, content feeds on Instagram
- **News & Content**: Article recommendations on Medium, YouTube video suggestions
- **Professional**: Job recommendations on LinkedIn, course suggestions on educational platforms

### Business Impact

Recommendation systems drive significant business value:
- **Increased Revenue**: Amazon reports 35% of revenue comes from recommendations
- **User Engagement**: Netflix saves $1B annually by reducing churn through recommendations
- **Discovery**: Help users find relevant content in vast catalogs
- **Personalization**: Create unique experiences for each user

---

## 🔍 Types of Recommendation Systems

### 1. Collaborative Filtering

Collaborative filtering makes recommendations based on user behavior patterns and similarities between users or items.

#### User-Based Collaborative Filtering

Finds users with similar preferences and recommends items they liked.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def user_based_recommendations(user_item_matrix, user_id, n_recommendations=5):
    """
    Generate recommendations using user-based collaborative filtering
    
    Args:
        user_item_matrix: Matrix where rows are users, columns are items
        user_id: ID of user to generate recommendations for
        n_recommendations: Number of recommendations to return
    
    Returns:
        List of recommended item indices
    """
    # Calculate user similarities
    user_similarities = cosine_similarity(user_item_matrix)
    
    # Get similarity scores for target user
    user_sim_scores = user_similarities[user_id]
    
    # Find most similar users (excluding self)
    similar_users = np.argsort(user_sim_scores)[::-1][1:]
    
    # Get items rated by target user
    user_ratings = user_item_matrix[user_id]
    unrated_items = np.where(user_ratings == 0)[0]
    
    # Calculate weighted ratings for unrated items
    item_scores = {}
    for item in unrated_items:
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user in similar_users[:50]:  # Top 50 similar users
            if user_item_matrix[similar_user, item] > 0:
                weighted_sum += user_sim_scores[similar_user] * user_item_matrix[similar_user, item]
                similarity_sum += abs(user_sim_scores[similar_user])
        
        if similarity_sum > 0:
            item_scores[item] = weighted_sum / similarity_sum
    
    # Sort and return top recommendations
    recommended_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, score in recommended_items[:n_recommendations]]
```

#### Item-Based Collaborative Filtering

Finds items similar to those the user has liked before.

```python
def item_based_recommendations(user_item_matrix, user_id, n_recommendations=5):
    """
    Generate recommendations using item-based collaborative filtering
    """
    # Calculate item similarities
    item_similarities = cosine_similarity(user_item_matrix.T)
    
    # Get items rated by user
    user_ratings = user_item_matrix[user_id]
    rated_items = np.where(user_ratings > 0)[0]
    unrated_items = np.where(user_ratings == 0)[0]
    
    # Calculate scores for unrated items
    item_scores = {}
    for item in unrated_items:
        weighted_sum = 0
        similarity_sum = 0
        
        for rated_item in rated_items:
            similarity = item_similarities[item, rated_item]
            if similarity > 0:
                weighted_sum += similarity * user_ratings[rated_item]
                similarity_sum += abs(similarity)
        
        if similarity_sum > 0:
            item_scores[item] = weighted_sum / similarity_sum
    
    # Sort and return recommendations
    recommended_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, score in recommended_items[:n_recommendations]]
```

### 2. Matrix Factorization

Advanced collaborative filtering using dimensionality reduction to discover latent factors.

#### Singular Value Decomposition (SVD)

```python
from sklearn.decomposition import TruncatedSVD
import pandas as pd

class SVDRecommender:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        
    def fit(self, user_item_matrix):
        """Fit SVD model to user-item matrix"""
        # Handle missing values by replacing with global mean
        self.global_mean = np.mean(user_item_matrix[user_item_matrix > 0])
        matrix_filled = user_item_matrix.copy()
        matrix_filled[matrix_filled == 0] = self.global_mean
        
        # Fit SVD
        self.user_factors = self.svd.fit_transform(matrix_filled)
        self.item_factors = self.svd.components_.T
        
        return self
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if self.user_factors is None:
            raise ValueError("Model must be fitted first")
        
        prediction = np.dot(self.user_factors[user_id], self.item_factors[item_id])
        return prediction
    
    def recommend(self, user_id, user_item_matrix, n_recommendations=5):
        """Generate recommendations for user"""
        user_ratings = user_item_matrix[user_id]
        unrated_items = np.where(user_ratings == 0)[0]
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, rating in predictions[:n_recommendations]]
```

### 3. Content-Based Filtering

Makes recommendations based on item features and user preferences.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.item_features = None
        self.similarity_matrix = None
        
    def fit(self, item_descriptions):
        """
        Fit content-based model using item descriptions
        
        Args:
            item_descriptions: List of text descriptions for each item
        """
        # Create TF-IDF matrix
        self.item_features = self.tfidf.fit_transform(item_descriptions)
        
        # Calculate item similarity matrix
        self.similarity_matrix = linear_kernel(self.item_features, self.item_features)
        
        return self
    
    def recommend_similar_items(self, item_id, n_recommendations=5):
        """Find items similar to given item"""
        # Get similarity scores for the item
        sim_scores = list(enumerate(self.similarity_matrix[item_id]))
        
        # Sort by similarity (excluding the item itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
        
        # Get top similar items
        similar_items = [item[0] for item in sim_scores[:n_recommendations]]
        return similar_items
    
    def recommend_for_user(self, user_item_matrix, user_id, n_recommendations=5):
        """Recommend items based on user's past preferences"""
        user_ratings = user_item_matrix[user_id]
        liked_items = np.where(user_ratings > 3)[0]  # Items rated > 3
        
        # Aggregate similarity scores for liked items
        item_scores = np.zeros(len(user_ratings))
        
        for liked_item in liked_items:
            # Weight by user's rating
            weight = user_ratings[liked_item]
            item_scores += self.similarity_matrix[liked_item] * weight
        
        # Remove already rated items
        item_scores[user_ratings > 0] = 0
        
        # Get top recommendations
        recommended_items = np.argsort(item_scores)[::-1][:n_recommendations]
        return recommended_items.tolist()
```

---

## 🔄 Hybrid Recommendation Systems

Hybrid systems combine multiple recommendation approaches to leverage their strengths and mitigate weaknesses.

### Weighted Hybrid

```python
class WeightedHybridRecommender:
    def __init__(self, collaborative_weight=0.6, content_weight=0.4):
        self.collaborative_recommender = None
        self.content_recommender = None
        self.collab_weight = collaborative_weight
        self.content_weight = content_weight
        
    def fit(self, user_item_matrix, item_descriptions):
        """Fit both collaborative and content-based models"""
        # Fit collaborative filtering (using SVD)
        self.collaborative_recommender = SVDRecommender()
        self.collaborative_recommender.fit(user_item_matrix)
        
        # Fit content-based filtering
        self.content_recommender = ContentBasedRecommender()
        self.content_recommender.fit(item_descriptions)
        
        return self
    
    def predict(self, user_id, item_id, user_item_matrix):
        """Predict rating using weighted combination"""
        # Get collaborative prediction
        collab_pred = self.collaborative_recommender.predict(user_id, item_id)
        
        # Get content-based prediction (simplified)
        # In practice, you'd need user profile for content-based prediction
        content_pred = 3.0  # Placeholder - would use user profile similarity
        
        # Weighted combination
        hybrid_pred = (self.collab_weight * collab_pred + 
                      self.content_weight * content_pred)
        
        return hybrid_pred
    
    def recommend(self, user_id, user_item_matrix, n_recommendations=5):
        """Generate hybrid recommendations"""
        # Get recommendations from both systems
        collab_recs = self.collaborative_recommender.recommend(
            user_id, user_item_matrix, n_recommendations * 2
        )
        content_recs = self.content_recommender.recommend_for_user(
            user_item_matrix, user_id, n_recommendations * 2
        )
        
        # Combine and score recommendations
        all_items = set(collab_recs + content_recs)
        item_scores = {}
        
        for item in all_items:
            score = 0
            if item in collab_recs:
                # Higher score for higher position in collaborative list
                collab_score = (len(collab_recs) - collab_recs.index(item)) / len(collab_recs)
                score += self.collab_weight * collab_score
            
            if item in content_recs:
                # Higher score for higher position in content list
                content_score = (len(content_recs) - content_recs.index(item)) / len(content_recs)
                score += self.content_weight * content_score
            
            item_scores[item] = score
        
        # Sort and return top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_items[:n_recommendations]]
```

---

## 🆕 Handling the Cold Start Problem

The cold start problem occurs when there's insufficient data for new users or items.

### New User Cold Start

```python
class ColdStartHandler:
    def __init__(self):
        self.popular_items = None
        self.item_categories = None
        
    def fit_popularity_baseline(self, user_item_matrix):
        """Fit popularity-based baseline for new users"""
        # Calculate item popularity (number of ratings)
        item_popularity = np.sum(user_item_matrix > 0, axis=0)
        
        # Calculate average ratings
        item_ratings = user_item_matrix.copy()
        item_ratings[item_ratings == 0] = np.nan
        avg_ratings = np.nanmean(item_ratings, axis=0)
        
        # Combine popularity and quality
        popularity_scores = item_popularity * avg_ratings
        
        # Get top popular items
        self.popular_items = np.argsort(popularity_scores)[::-1]
        
        return self
    
    def recommend_for_new_user(self, n_recommendations=5, category_filter=None):
        """Recommend popular items for new users"""
        if category_filter:
            # Filter by category if provided
            filtered_items = [item for item in self.popular_items 
                            if self.item_categories.get(item) == category_filter]
            return filtered_items[:n_recommendations]
        else:
            return self.popular_items[:n_recommendations].tolist()
    
    def recommend_with_onboarding(self, user_preferences, item_features):
        """
        Recommend items based on user's stated preferences during onboarding
        
        Args:
            user_preferences: Dict of user's stated preferences
            item_features: DataFrame with item features
        """
        # Create user profile from preferences
        user_profile = self._create_user_profile(user_preferences)
        
        # Calculate similarity between user profile and items
        similarities = []
        for idx, item in item_features.iterrows():
            similarity = self._calculate_profile_similarity(user_profile, item)
            similarities.append((idx, similarity))
        
        # Sort by similarity and return top items
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, sim in similarities[:5]]
    
    def _create_user_profile(self, preferences):
        """Create user profile vector from preferences"""
        # Implementation depends on preference structure
        # This is a simplified example
        profile = {}
        for key, value in preferences.items():
            profile[key] = value
        return profile
    
    def _calculate_profile_similarity(self, user_profile, item_features):
        """Calculate similarity between user profile and item"""
        # Simplified similarity calculation
        similarity = 0
        for feature, value in user_profile.items():
            if feature in item_features:
                similarity += abs(value - item_features[feature])
        return similarity
```

### New Item Cold Start

```python
def recommend_new_item(new_item_features, existing_items_features, 
                      user_item_matrix, n_target_users=10):
    """
    Recommend new item to users most likely to be interested
    
    Args:
        new_item_features: Features of the new item
        existing_items_features: Features of existing items
        user_item_matrix: Historical user-item interactions
        n_target_users: Number of users to target
    
    Returns:
        List of user IDs most likely to be interested
    """
    # Find existing items similar to new item
    similarities = cosine_similarity([new_item_features], existing_items_features)[0]
    similar_items = np.argsort(similarities)[::-1][:10]  # Top 10 similar items
    
    # Find users who liked similar items
    user_scores = {}
    for user_id in range(user_item_matrix.shape[0]):
        user_ratings = user_item_matrix[user_id]
        
        # Calculate user's affinity for similar items
        affinity_score = 0
        for item_id in similar_items:
            if user_ratings[item_id] > 0:
                # Weight by similarity and rating
                affinity_score += similarities[item_id] * user_ratings[item_id]
        
        if affinity_score > 0:
            user_scores[user_id] = affinity_score
    
    # Return users with highest affinity scores
    sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
    return [user_id for user_id, score in sorted_users[:n_target_users]]
```

---

## 📊 Evaluation Metrics

### Accuracy Metrics

```python
def calculate_rmse(predictions, actuals):
    """Root Mean Square Error for rating predictions"""
    return np.sqrt(np.mean((predictions - actuals) ** 2))

def calculate_mae(predictions, actuals):
    """Mean Absolute Error for rating predictions"""
    return np.mean(np.abs(predictions - actuals))

def calculate_precision_at_k(recommended_items, relevant_items, k):
    """Precision@K for recommendation lists"""
    recommended_k = recommended_items[:k]
    relevant_recommended = len(set(recommended_k) & set(relevant_items))
    return relevant_recommended / k if k > 0 else 0

def calculate_recall_at_k(recommended_items, relevant_items, k):
    """Recall@K for recommendation lists"""
    recommended_k = recommended_items[:k]
    relevant_recommended = len(set(recommended_k) & set(relevant_items))
    return relevant_recommended / len(relevant_items) if len(relevant_items) > 0 else 0

def calculate_ndcg_at_k(recommended_items, relevant_items, k):
    """Normalized Discounted Cumulative Gain@K"""
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0
    
    # Create relevance scores (1 if relevant, 0 if not)
    relevance_scores = [1 if item in relevant_items else 0 
                       for item in recommended_items[:k]]
    
    # Calculate DCG
    dcg = dcg_at_k(relevance_scores, k)
    
    # Calculate IDCG (ideal DCG)
    ideal_relevance = [1] * min(len(relevant_items), k)
    idcg = dcg_at_k(ideal_relevance, k)
    
    return dcg / idcg if idcg > 0 else 0
```

### Diversity and Coverage Metrics

```python
def calculate_diversity(recommended_items, item_features):
    """Calculate diversity of recommended items"""
    if len(recommended_items) < 2:
        return 0
    
    total_distance = 0
    count = 0
    
    for i in range(len(recommended_items)):
        for j in range(i + 1, len(recommended_items)):
            item1_features = item_features[recommended_items[i]]
            item2_features = item_features[recommended_items[j]]
            
            # Calculate distance between items
            distance = np.linalg.norm(item1_features - item2_features)
            total_distance += distance
            count += 1
    
    return total_distance / count if count > 0 else 0

def calculate_catalog_coverage(all_recommendations, total_items):
    """Calculate what fraction of catalog is being recommended"""
    unique_recommended = set()
    for rec_list in all_recommendations:
        unique_recommended.update(rec_list)
    
    return len(unique_recommended) / total_items
```

---

## 🏭 Production Considerations

### Scalability Strategies

#### 1. Approximate Nearest Neighbors

```python
from sklearn.neighbors import NearestNeighbors
import faiss  # Facebook AI Similarity Search (optional)

class ScalableCollaborativeFiltering:
    def __init__(self, n_neighbors=50, algorithm='auto'):
        self.n_neighbors = n_neighbors
        self.nn_model = NearestNeighbors(
            n_neighbors=n_neighbors, 
            algorithm=algorithm,
            metric='cosine'
        )
        self.user_item_matrix = None
        
    def fit(self, user_item_matrix):
        """Fit approximate nearest neighbors model"""
        self.user_item_matrix = user_item_matrix
        
        # Fit on user vectors
        self.nn_model.fit(user_item_matrix)
        
        return self
    
    def find_similar_users(self, user_id):
        """Find similar users using approximate search"""
        user_vector = self.user_item_matrix[user_id].reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(user_vector)
        
        # Convert distances to similarities
        similarities = 1 - distances[0]
        similar_users = indices[0][1:]  # Exclude self
        
        return similar_users, similarities[1:]
    
    def recommend(self, user_id, n_recommendations=5):
        """Generate recommendations using approximate neighbors"""
        similar_users, similarities = self.find_similar_users(user_id)
        
        user_ratings = self.user_item_matrix[user_id]
        unrated_items = np.where(user_ratings == 0)[0]
        
        # Calculate weighted ratings
        item_scores = {}
        for item in unrated_items:
            weighted_sum = 0
            similarity_sum = 0
            
            for i, similar_user in enumerate(similar_users):
                if self.user_item_matrix[similar_user, item] > 0:
                    weighted_sum += similarities[i] * self.user_item_matrix[similar_user, item]
                    similarity_sum += similarities[i]
            
            if similarity_sum > 0:
                item_scores[item] = weighted_sum / similarity_sum
        
        # Return top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, score in sorted_items[:n_recommendations]]
```

#### 2. Matrix Factorization with Implicit Feedback

```python
try:
    import implicit
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False

class ImplicitRecommender:
    def __init__(self, factors=50, regularization=0.01, iterations=15):
        if not IMPLICIT_AVAILABLE:
            raise ImportError("implicit library not available")
        
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations
        )
        self.user_items = None
        
    def fit(self, user_item_matrix):
        """Fit implicit ALS model"""
        # Convert to sparse matrix and transpose for implicit library
        import scipy.sparse as sp
        self.user_items = sp.csr_matrix(user_item_matrix)
        
        # Fit model
        self.model.fit(self.user_items)
        
        return self
    
    def recommend(self, user_id, n_recommendations=5):
        """Generate recommendations using implicit ALS"""
        recommendations = self.model.recommend(
            user_id, 
            self.user_items[user_id], 
            N=n_recommendations
        )
        
        return [item_id for item_id, score in recommendations]
```

### Real-time Serving

```python
import redis
import json
from datetime import datetime, timedelta

class RecommendationCache:
    def __init__(self, redis_host='localhost', redis_port=6379, ttl=3600):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.ttl = ttl  # Time to live in seconds
        
    def get_recommendations(self, user_id, n_recommendations=5):
        """Get cached recommendations for user"""
        cache_key = f"recommendations:{user_id}:{n_recommendations}"
        cached_recs = self.redis_client.get(cache_key)
        
        if cached_recs:
            return json.loads(cached_recs)
        return None
    
    def cache_recommendations(self, user_id, recommendations, n_recommendations=5):
        """Cache recommendations for user"""
        cache_key = f"recommendations:{user_id}:{n_recommendations}"
        cache_data = {
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'ttl': self.ttl
        }
        
        self.redis_client.setex(
            cache_key, 
            self.ttl, 
            json.dumps(cache_data)
        )
    
    def invalidate_user_cache(self, user_id):
        """Invalidate all cached recommendations for user"""
        pattern = f"recommendations:{user_id}:*"
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)

class RealTimeRecommendationService:
    def __init__(self, model, cache=None):
        self.model = model
        self.cache = cache
        
    def get_recommendations(self, user_id, n_recommendations=5, use_cache=True):
        """Get recommendations with caching support"""
        # Try cache first
        if use_cache and self.cache:
            cached_recs = self.cache.get_recommendations(user_id, n_recommendations)
            if cached_recs:
                return cached_recs['recommendations']
        
        # Generate fresh recommendations
        recommendations = self.model.recommend(user_id, n_recommendations)
        
        # Cache results
        if self.cache:
            self.cache.cache_recommendations(user_id, recommendations, n_recommendations)
        
        return recommendations
    
    def update_user_interaction(self, user_id, item_id, rating=None):
        """Update user interaction and invalidate cache"""
        # Update model (implementation depends on model type)
        # For online learning models, update incrementally
        
        # Invalidate cache for this user
        if self.cache:
            self.cache.invalidate_user_cache(user_id)
```

---

## 🎯 Advanced Techniques

### Deep Learning for Recommendations

```python
# Note: This requires TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class NeuralCollaborativeFiltering:
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_units=[128, 64]):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.model = self._build_model()
        
    def _build_model(self):
        """Build neural collaborative filtering model"""
        # User and item inputs
        user_input = layers.Input(shape=(), name='user_id')
        item_input = layers.Input(shape=(), name='item_id')
        
        # Embeddings
        user_embedding = layers.Embedding(self.n_users, self.embedding_dim)(user_input)
        item_embedding = layers.Embedding(self.n_items, self.embedding_dim)(item_input)
        
        # Flatten embeddings
        user_vec = layers.Flatten()(user_embedding)
        item_vec = layers.Flatten()(item_embedding)
        
        # Concatenate user and item vectors
        concat = layers.Concatenate()([user_vec, item_vec])
        
        # Hidden layers
        x = concat
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
        
        return model
    
    def fit(self, user_ids, item_ids, ratings, validation_split=0.2, epochs=50):
        """Train the neural collaborative filtering model"""
        # Normalize ratings to 0-1 range
        normalized_ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min())
        
        # Train model
        history = self.model.fit(
            [user_ids, item_ids], 
            normalized_ratings,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=256,
            verbose=1
        )
        
        return history
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        return self.model.predict([np.array([user_id]), np.array([item_id])])[0][0]
    
    def recommend(self, user_id, user_item_matrix, n_recommendations=5):
        """Generate recommendations for user"""
        user_ratings = user_item_matrix[user_id]
        unrated_items = np.where(user_ratings == 0)[0]
        
        # Predict ratings for unrated items
        user_ids = np.full(len(unrated_items), user_id)
        predictions = self.model.predict([user_ids, unrated_items])
        
        # Sort by predicted rating
        item_predictions = list(zip(unrated_items, predictions.flatten()))
        item_predictions.sort(key=lambda x: x[1], reverse=True)
        
        return [item_id for item_id, pred in item_predictions[:n_recommendations]]
```

---

## 🏗️ Infrastructure Setup

### Quick Start (5 minutes)

```bash
# 1. Navigate to day 29
cd days/day-29-recommendation-systems

# 2. Start the complete infrastructure
./setup.sh

# 3. Run interactive demo
python demo.py
```

### Infrastructure Components

**Recommendation Stack**:
- **PostgreSQL**: User profiles, item catalogs, and interaction storage
- **Elasticsearch**: Content-based search and item similarity
- **Redis**: Real-time recommendation caching and session storage
- **MLflow**: Model tracking and recommendation algorithm registry

**Production Services**:
- **FastAPI Server**: Production recommendation endpoints with multiple algorithms
- **Model Trainer**: Automated model training and updating service
- **Data Generator**: Realistic user-item interaction data with preferences
- **Jupyter Notebook**: Interactive analysis and experimentation environment

**Monitoring & Analytics**:
- **Prometheus**: Recommendation performance and system metrics
- **Grafana**: User engagement and recommendation quality dashboards
- **Health Checks**: Service availability and model performance monitoring

### Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Recommendation API | http://localhost:8000 | - |
| Grafana Dashboard | http://localhost:3000 | admin/recsys123 |
| Jupyter Notebook | http://localhost:8888 | token: recsys123 |
| Elasticsearch | http://localhost:9200 | - |
| MLflow Tracking | http://localhost:5000 | - |
| Prometheus | http://localhost:9090 | - |

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Get user recommendations
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "num_recommendations": 5, "algorithm": "hybrid"}'

# Get user profile
curl http://localhost:8000/users/1/profile

# Find similar items
curl http://localhost:8000/items/100/similar?num_similar=5
```

### Generated Data

The infrastructure provides realistic recommendation system data:

- **Users**: 10,000 users with demographics and preferences
- **Items**: 50,000 items with categories, descriptions, and ratings
- **Interactions**: 500,000 user-item interactions with realistic patterns
- **Categories**: Multiple content categories with user preference modeling

---

## 📚 Key Takeaways

- **Collaborative filtering** leverages user behavior patterns but suffers from sparsity and cold start problems
- **Content-based filtering** uses item features and works well for new items but may lack diversity
- **Matrix factorization** techniques like SVD can capture latent factors and handle sparse data effectively
- **Hybrid approaches** combine multiple methods to leverage their strengths and mitigate weaknesses
- **Cold start problems** require special handling through popularity-based recommendations and onboarding
- **Production systems** need caching, approximate algorithms, and real-time serving capabilities
- **Evaluation** should consider accuracy, diversity, coverage, and business metrics
- **Deep learning** approaches can capture complex non-linear patterns in user-item interactions
- **Real-time serving** requires efficient caching and approximate similarity search
- **Comprehensive infrastructure** supports production-grade recommendation systems

### Business Considerations

- **Data Quality**: Clean, consistent interaction data is crucial for good recommendations
- **Privacy**: Handle user data responsibly and comply with regulations
- **Bias**: Monitor for and mitigate algorithmic bias in recommendations
- **Explainability**: Provide reasons for recommendations to build user trust
- **A/B Testing**: Continuously test and optimize recommendation algorithms
- **Feedback Loops**: Implement mechanisms to learn from user feedback

---

## 🔄 What's Next?

Tomorrow, we'll explore **Ensemble Methods** where you'll learn advanced techniques for combining multiple machine learning models. We'll cover bagging, boosting, stacking, and how to build robust ensemble systems that outperform individual models.

The recommendation system techniques you've learned today will be valuable for understanding how ensemble methods can be applied to combine different recommendation approaches for better performance and robustness.