#!/usr/bin/env python3
"""
Recommendation Systems API
Production FastAPI server for recommendation systems with multiple algorithms
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
import json
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import psycopg2
import redis
from elasticsearch import Elasticsearch
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import joblib

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID for recommendations")
    num_recommendations: int = Field(10, description="Number of recommendations", ge=1, le=100)
    algorithm: str = Field("collaborative", description="Algorithm: collaborative, content, hybrid")
    exclude_seen: bool = Field(True, description="Exclude items user has already interacted with")

class ItemSimilarityRequest(BaseModel):
    item_id: int = Field(..., description="Item ID to find similar items")
    num_similar: int = Field(10, description="Number of similar items", ge=1, le=50)
    algorithm: str = Field("content", description="Similarity algorithm: content, collaborative")

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Union[int, float, str]]]
    algorithm_used: str
    generated_at: datetime
    metadata: Dict[str, Union[int, float]]

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    models_loaded: List[str]

# FastAPI app
app = FastAPI(
    title="Recommendation Systems API",
    description="Production API for recommendation systems with multiple algorithms",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationService:
    """Main recommendation service with multiple algorithms"""
    
    def __init__(self):
        self.setup_connections()
        self.models = {}
        self.load_data()
        self.train_models()
        
    def setup_connections(self):
        """Setup database connections"""
        # PostgreSQL
        self.pg_conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', 5432),
            database=os.getenv('POSTGRES_DB', 'recsys_db'),
            user=os.getenv('POSTGRES_USER', 'recsys_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'recsys_pass')
        )
        
        # Redis
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=True
        )
        
        # Elasticsearch
        self.es_client = Elasticsearch([{
            'host': os.getenv('ELASTICSEARCH_HOST', 'localhost'),
            'port': int(os.getenv('ELASTICSEARCH_PORT', 9200)),
            'scheme': 'http'
        }])
        
        # MLflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        mlflow.set_experiment("recommendation_systems")
    
    def load_data(self):
        """Load data from database"""
        try:
            # Load users
            self.users_df = pd.read_sql("SELECT * FROM users", self.pg_conn)
            
            # Load items
            self.items_df = pd.read_sql("SELECT * FROM items", self.pg_conn)
            
            # Load interactions
            self.interactions_df = pd.read_sql("SELECT * FROM interactions", self.pg_conn)
            
            logger.info(f"Loaded {len(self.users_df)} users, {len(self.items_df)} items, {len(self.interactions_df)} interactions")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Create empty dataframes as fallback
            self.users_df = pd.DataFrame()
            self.items_df = pd.DataFrame()
            self.interactions_df = pd.DataFrame()
    
    def train_models(self):
        """Train recommendation models"""
        if len(self.interactions_df) == 0:
            logger.warning("No interaction data available for training")
            return
        
        try:
            # Train collaborative filtering model (Matrix Factorization)
            self.train_collaborative_filtering()
            
            # Train content-based model
            self.train_content_based()
            
            logger.info("Models trained successfully")
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def train_collaborative_filtering(self):
        """Train collaborative filtering using Matrix Factorization"""
        try:
            # Create user-item matrix
            rating_interactions = self.interactions_df[
                self.interactions_df['rating'].notna()
            ].copy()
            
            if len(rating_interactions) == 0:
                logger.warning("No rating data for collaborative filtering")
                return
            
            # Create pivot table
            user_item_matrix = rating_interactions.pivot_table(
                index='user_id', 
                columns='item_id', 
                values='rating', 
                fill_value=0
            )
            
            # Train NMF model
            n_components = min(50, min(user_item_matrix.shape) - 1)
            nmf_model = NMF(n_components=n_components, random_state=42, max_iter=100)
            
            W = nmf_model.fit_transform(user_item_matrix)
            H = nmf_model.components_
            
            # Store model components
            self.models['collaborative'] = {
                'model': nmf_model,
                'user_factors': W,
                'item_factors': H,
                'user_index': user_item_matrix.index,
                'item_index': user_item_matrix.columns,
                'user_item_matrix': user_item_matrix
            }
            
            logger.info("Collaborative filtering model trained")
        except Exception as e:
            logger.error(f"Error training collaborative filtering: {e}")
    
    def train_content_based(self):
        """Train content-based filtering using item features"""
        try:
            if len(self.items_df) == 0:
                return
            
            # Create item feature matrix
            item_features = []
            for _, item in self.items_df.iterrows():
                features = f"{item['category']} {item['brand']} {item['description']}"
                item_features.append(features)
            
            # Train TF-IDF vectorizer
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            item_feature_matrix = tfidf.fit_transform(item_features)
            
            # Calculate item similarity matrix
            item_similarity = cosine_similarity(item_feature_matrix)
            
            self.models['content'] = {
                'tfidf': tfidf,
                'feature_matrix': item_feature_matrix,
                'similarity_matrix': item_similarity,
                'item_index': self.items_df['item_id'].values
            }
            
            logger.info("Content-based model trained")
        except Exception as e:
            logger.error(f"Error training content-based model: {e}")
    
    def get_collaborative_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[Dict]:
        """Get collaborative filtering recommendations"""
        if 'collaborative' not in self.models:
            return []
        
        try:
            model_data = self.models['collaborative']
            
            # Check if user exists in training data
            if user_id not in model_data['user_index']:
                # Cold start: return popular items
                return self.get_popular_recommendations(num_recommendations)
            
            # Get user index
            user_idx = list(model_data['user_index']).index(user_id)
            
            # Get user factors
            user_factors = model_data['user_factors'][user_idx]
            
            # Calculate scores for all items
            item_scores = np.dot(user_factors, model_data['item_factors'])
            
            # Get top recommendations
            item_indices = np.argsort(item_scores)[::-1]
            
            recommendations = []
            for idx in item_indices[:num_recommendations * 2]:  # Get more to filter
                item_id = model_data['item_index'][idx]
                score = item_scores[idx]
                
                # Get item details
                item_info = self.items_df[self.items_df['item_id'] == item_id]
                if len(item_info) > 0:
                    item = item_info.iloc[0]
                    recommendations.append({
                        'item_id': int(item_id),
                        'score': float(score),
                        'category': item['category'],
                        'brand': item['brand'],
                        'price': float(item['price']),
                        'avg_rating': float(item['avg_rating'])
                    })
                
                if len(recommendations) >= num_recommendations:
                    break
            
            return recommendations
        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return []
    
    def get_content_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[Dict]:
        """Get content-based recommendations"""
        if 'content' not in self.models:
            return []
        
        try:
            # Get user's interaction history
            user_interactions = self.interactions_df[
                self.interactions_df['user_id'] == user_id
            ]
            
            if len(user_interactions) == 0:
                return self.get_popular_recommendations(num_recommendations)
            
            # Get items user has interacted with
            user_items = user_interactions['item_id'].unique()
            
            # Calculate content-based scores
            model_data = self.models['content']
            similarity_matrix = model_data['similarity_matrix']
            item_index = model_data['item_index']
            
            # Aggregate similarity scores
            item_scores = {}
            for item_id in user_items:
                if item_id in item_index:
                    item_idx = list(item_index).index(item_id)
                    similarities = similarity_matrix[item_idx]
                    
                    for i, sim_score in enumerate(similarities):
                        similar_item_id = item_index[i]
                        if similar_item_id not in user_items:  # Don't recommend seen items
                            if similar_item_id not in item_scores:
                                item_scores[similar_item_id] = 0
                            item_scores[similar_item_id] += sim_score
            
            # Sort by score and get top recommendations
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for item_id, score in sorted_items[:num_recommendations]:
                item_info = self.items_df[self.items_df['item_id'] == item_id]
                if len(item_info) > 0:
                    item = item_info.iloc[0]
                    recommendations.append({
                        'item_id': int(item_id),
                        'score': float(score),
                        'category': item['category'],
                        'brand': item['brand'],
                        'price': float(item['price']),
                        'avg_rating': float(item['avg_rating'])
                    })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error in content recommendations: {e}")
            return []
    
    def get_popular_recommendations(self, num_recommendations: int = 10) -> List[Dict]:
        """Get popular items as fallback recommendations"""
        try:
            # Get popular items from cache
            popular_items_str = self.redis_client.get("recsys:popular_items")
            if popular_items_str:
                popular_items = json.loads(popular_items_str)
                
                recommendations = []
                for item_id, count in list(popular_items.items())[:num_recommendations]:
                    item_info = self.items_df[self.items_df['item_id'] == int(item_id)]
                    if len(item_info) > 0:
                        item = item_info.iloc[0]
                        recommendations.append({
                            'item_id': int(item_id),
                            'score': float(count),
                            'category': item['category'],
                            'brand': item['brand'],
                            'price': float(item['price']),
                            'avg_rating': float(item['avg_rating'])
                        })
                
                return recommendations
            
            # Fallback: highest rated items
            top_items = self.items_df.nlargest(num_recommendations, 'avg_rating')
            recommendations = []
            for _, item in top_items.iterrows():
                recommendations.append({
                    'item_id': int(item['item_id']),
                    'score': float(item['avg_rating']),
                    'category': item['category'],
                    'brand': item['brand'],
                    'price': float(item['price']),
                    'avg_rating': float(item['avg_rating'])
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting popular recommendations: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[Dict]:
        """Get hybrid recommendations combining collaborative and content-based"""
        try:
            # Get recommendations from both methods
            collab_recs = self.get_collaborative_recommendations(user_id, num_recommendations)
            content_recs = self.get_content_recommendations(user_id, num_recommendations)
            
            # Combine and weight the recommendations
            combined_scores = {}
            
            # Weight collaborative filtering recommendations
            for rec in collab_recs:
                item_id = rec['item_id']
                combined_scores[item_id] = combined_scores.get(item_id, 0) + 0.6 * rec['score']
            
            # Weight content-based recommendations
            for rec in content_recs:
                item_id = rec['item_id']
                combined_scores[item_id] = combined_scores.get(item_id, 0) + 0.4 * rec['score']
            
            # Sort by combined score
            sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Build final recommendations
            recommendations = []
            for item_id, score in sorted_items[:num_recommendations]:
                item_info = self.items_df[self.items_df['item_id'] == item_id]
                if len(item_info) > 0:
                    item = item_info.iloc[0]
                    recommendations.append({
                        'item_id': int(item_id),
                        'score': float(score),
                        'category': item['category'],
                        'brand': item['brand'],
                        'price': float(item['price']),
                        'avg_rating': float(item['avg_rating'])
                    })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return self.get_popular_recommendations(num_recommendations)

# Global service instance
recommendation_service = RecommendationService()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {}
    
    # Check PostgreSQL
    try:
        cursor = recommendation_service.pg_conn.cursor()
        cursor.execute("SELECT 1")
        services["postgresql"] = "healthy"
    except:
        services["postgresql"] = "unhealthy"
    
    # Check Redis
    try:
        recommendation_service.redis_client.ping()
        services["redis"] = "healthy"
    except:
        services["redis"] = "unhealthy"
    
    # Check Elasticsearch
    try:
        recommendation_service.es_client.ping()
        services["elasticsearch"] = "healthy"
    except:
        services["elasticsearch"] = "unhealthy"
    
    return HealthResponse(
        status="healthy" if all(s == "healthy" for s in services.values()) else "degraded",
        timestamp=datetime.now(),
        services=services,
        models_loaded=list(recommendation_service.models.keys())
    )

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations for a user"""
    try:
        if request.algorithm == "collaborative":
            recommendations = recommendation_service.get_collaborative_recommendations(
                request.user_id, request.num_recommendations
            )
        elif request.algorithm == "content":
            recommendations = recommendation_service.get_content_recommendations(
                request.user_id, request.num_recommendations
            )
        elif request.algorithm == "hybrid":
            recommendations = recommendation_service.get_hybrid_recommendations(
                request.user_id, request.num_recommendations
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid algorithm")
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            algorithm_used=request.algorithm,
            generated_at=datetime.now(),
            metadata={
                "total_recommendations": len(recommendations),
                "models_available": len(recommendation_service.models)
            }
        )
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: int):
    """Get user profile and interaction history"""
    try:
        # Get user info
        user_info = recommendation_service.users_df[
            recommendation_service.users_df['user_id'] == user_id
        ]
        
        if len(user_info) == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = user_info.iloc[0]
        
        # Get user interactions
        interactions = recommendation_service.interactions_df[
            recommendation_service.interactions_df['user_id'] == user_id
        ]
        
        return {
            "user_id": user_id,
            "profile": {
                "age": int(user['age']),
                "gender": user['gender'],
                "location": user['location'],
                "income_level": user['income_level'],
                "preferred_categories": user['preferred_categories'].split(',')
            },
            "interaction_summary": {
                "total_interactions": len(interactions),
                "interaction_types": interactions['interaction_type'].value_counts().to_dict(),
                "categories_interacted": interactions.merge(
                    recommendation_service.items_df, on='item_id'
                )['category'].value_counts().to_dict()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/items/{item_id}/similar")
async def get_similar_items(item_id: int, num_similar: int = 10):
    """Get similar items"""
    try:
        if 'content' not in recommendation_service.models:
            raise HTTPException(status_code=503, detail="Content model not available")
        
        model_data = recommendation_service.models['content']
        item_index = model_data['item_index']
        
        if item_id not in item_index:
            raise HTTPException(status_code=404, detail="Item not found")
        
        item_idx = list(item_index).index(item_id)
        similarities = model_data['similarity_matrix'][item_idx]
        
        # Get most similar items
        similar_indices = np.argsort(similarities)[::-1][1:num_similar+1]  # Exclude self
        
        similar_items = []
        for idx in similar_indices:
            similar_item_id = item_index[idx]
            similarity_score = similarities[idx]
            
            item_info = recommendation_service.items_df[
                recommendation_service.items_df['item_id'] == similar_item_id
            ]
            
            if len(item_info) > 0:
                item = item_info.iloc[0]
                similar_items.append({
                    'item_id': int(similar_item_id),
                    'similarity_score': float(similarity_score),
                    'category': item['category'],
                    'brand': item['brand'],
                    'price': float(item['price']),
                    'avg_rating': float(item['avg_rating'])
                })
        
        return {
            "item_id": item_id,
            "similar_items": similar_items
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Recommendation Systems API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/recommendations",
            "/users/{user_id}/profile",
            "/items/{item_id}/similar"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', 8000)),
        workers=int(os.getenv('API_WORKERS', 1)),
        reload=True
    )