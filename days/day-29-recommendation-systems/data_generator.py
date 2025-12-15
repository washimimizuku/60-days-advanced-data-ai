#!/usr/bin/env python3
"""
Recommendation Systems Data Generator
Generates realistic user-item interaction data for various recommendation scenarios
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
import redis
from elasticsearch import Elasticsearch
import json
import time
import logging
from typing import Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationDataGenerator:
    """Generate realistic recommendation system data"""
    
    def __init__(self):
        self.setup_connections()
        
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
    
    def generate_users(self, n_users: int = 10000) -> pd.DataFrame:
        """Generate user profiles with demographics"""
        
        np.random.seed(42)
        
        # User demographics
        ages = np.random.normal(35, 12, n_users).astype(int)
        ages = np.clip(ages, 18, 80)
        
        genders = np.random.choice(['M', 'F', 'O'], n_users, p=[0.48, 0.48, 0.04])
        
        locations = np.random.choice([
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
            'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'
        ], n_users, p=[0.15, 0.12, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05, 0.32])
        
        # Income levels (affects purchasing behavior)
        income_levels = np.random.choice(['low', 'medium', 'high'], n_users, p=[0.3, 0.5, 0.2])
        
        # User preferences (will influence recommendations)
        categories_preference = []
        for _ in range(n_users):
            # Each user has 1-3 preferred categories
            n_prefs = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            prefs = np.random.choice([
                'electronics', 'books', 'clothing', 'home', 'sports',
                'beauty', 'toys', 'automotive', 'music', 'movies'
            ], n_prefs, replace=False)
            categories_preference.append(','.join(prefs))
        
        users_df = pd.DataFrame({
            'user_id': range(1, n_users + 1),
            'age': ages,
            'gender': genders,
            'location': locations,
            'income_level': income_levels,
            'preferred_categories': categories_preference,
            'created_at': pd.date_range('2020-01-01', periods=n_users, freq='1H')
        })
        
        return users_df
    
    def generate_items(self, n_items: int = 50000) -> pd.DataFrame:
        """Generate item catalog with features"""
        
        np.random.seed(123)
        
        categories = ['electronics', 'books', 'clothing', 'home', 'sports',
                     'beauty', 'toys', 'automotive', 'music', 'movies']
        
        # Generate items
        item_categories = np.random.choice(categories, n_items)
        
        # Price distribution varies by category
        prices = []
        for category in item_categories:
            if category == 'electronics':
                price = np.random.lognormal(5, 1)  # Higher prices
            elif category == 'books':
                price = np.random.uniform(10, 50)  # Lower prices
            elif category == 'clothing':
                price = np.random.uniform(20, 200)
            else:
                price = np.random.lognormal(3, 1)
            prices.append(max(5, price))  # Minimum price $5
        
        # Brand popularity (affects recommendation likelihood)
        brands = []
        brand_names = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'Generic']
        brand_weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.15]  # Popular brands more likely
        
        for _ in range(n_items):
            brands.append(np.random.choice(brand_names, p=brand_weights))
        
        # Item ratings (average rating affects popularity)
        avg_ratings = np.random.beta(8, 2, n_items) * 4 + 1  # Skewed toward higher ratings
        num_ratings = np.random.exponential(50, n_items).astype(int)
        
        # Item descriptions for content-based filtering
        descriptions = []
        for i, category in enumerate(item_categories):
            if category == 'electronics':
                desc = f"High-quality {category} device with advanced features and modern design"
            elif category == 'books':
                desc = f"Engaging {category} with compelling storyline and excellent reviews"
            elif category == 'clothing':
                desc = f"Stylish {category} made from premium materials with comfortable fit"
            else:
                desc = f"Premium {category} product with excellent quality and durability"
            descriptions.append(desc)
        
        items_df = pd.DataFrame({
            'item_id': range(1, n_items + 1),
            'category': item_categories,
            'price': prices,
            'brand': brands,
            'avg_rating': avg_ratings,
            'num_ratings': num_ratings,
            'description': descriptions,
            'created_at': pd.date_range('2019-01-01', periods=n_items, freq='30min')
        })
        
        return items_df
    
    def generate_interactions(self, users_df: pd.DataFrame, items_df: pd.DataFrame, 
                            n_interactions: int = 500000) -> pd.DataFrame:
        """Generate user-item interactions with realistic patterns"""
        
        np.random.seed(456)
        
        interactions = []
        
        # Create interaction patterns based on user preferences
        for _, user in users_df.iterrows():
            user_id = user['user_id']
            preferred_cats = user['preferred_categories'].split(',')
            age = user['age']
            income = user['income_level']
            
            # Number of interactions per user (power law distribution)
            n_user_interactions = max(1, int(np.random.pareto(0.5) * 10))
            n_user_interactions = min(n_user_interactions, 100)  # Cap at 100
            
            for _ in range(n_user_interactions):
                # Choose item based on user preferences
                if np.random.random() < 0.7:  # 70% chance to interact with preferred category
                    preferred_cat = np.random.choice(preferred_cats)
                    candidate_items = items_df[items_df['category'] == preferred_cat]
                else:  # 30% chance for serendipitous discovery
                    candidate_items = items_df
                
                if len(candidate_items) == 0:
                    candidate_items = items_df
                
                # Price sensitivity based on income
                if income == 'low':
                    candidate_items = candidate_items[candidate_items['price'] < 100]
                elif income == 'medium':
                    candidate_items = candidate_items[candidate_items['price'] < 500]
                
                if len(candidate_items) == 0:
                    candidate_items = items_df
                
                # Select item (bias toward higher-rated items)
                weights = candidate_items['avg_rating'] ** 2  # Square to increase bias
                weights = weights / weights.sum()
                
                selected_item = candidate_items.sample(1, weights=weights).iloc[0]
                item_id = selected_item['item_id']
                
                # Generate interaction type and rating
                interaction_types = ['view', 'cart', 'purchase', 'rating']
                
                # Interaction probabilities (funnel effect)
                if np.random.random() < 0.8:  # 80% view
                    interaction_type = 'view'
                    rating = None
                elif np.random.random() < 0.3:  # 30% of remaining add to cart
                    interaction_type = 'cart'
                    rating = None
                elif np.random.random() < 0.5:  # 50% of remaining purchase
                    interaction_type = 'purchase'
                    # Rating given with purchase (biased toward item's avg rating)
                    rating = np.random.normal(selected_item['avg_rating'], 0.5)
                    rating = np.clip(rating, 1, 5)
                else:
                    interaction_type = 'rating'
                    rating = np.random.normal(selected_item['avg_rating'], 0.8)
                    rating = np.clip(rating, 1, 5)
                
                # Timestamp (recent interactions more likely)
                days_ago = int(np.random.exponential(30))  # Exponential decay
                days_ago = min(days_ago, 365)  # Cap at 1 year
                timestamp = datetime.now() - timedelta(days=days_ago)
                
                interactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'interaction_type': interaction_type,
                    'rating': rating,
                    'timestamp': timestamp
                })
        
        # Add some random interactions for diversity
        for _ in range(n_interactions // 10):  # 10% random interactions
            user_id = np.random.choice(users_df['user_id'])
            item_id = np.random.choice(items_df['item_id'])
            interaction_type = np.random.choice(['view', 'cart', 'purchase'])
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'interaction_type': interaction_type,
                'rating': None,
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
            })
        
        interactions_df = pd.DataFrame(interactions)
        
        # Remove duplicates and sort by timestamp
        interactions_df = interactions_df.drop_duplicates(['user_id', 'item_id', 'interaction_type'])
        interactions_df = interactions_df.sort_values('timestamp')
        
        return interactions_df
    
    def store_to_postgres(self, df: pd.DataFrame, table_name: str):
        """Store data to PostgreSQL"""
        try:
            df.to_sql(table_name, self.pg_conn, if_exists='replace', index=False, method='multi')
            logger.info(f"Stored {len(df)} records to PostgreSQL table: {table_name}")
        except Exception as e:
            logger.error(f"Error storing to PostgreSQL: {e}")
    
    def store_to_elasticsearch(self, items_df: pd.DataFrame):
        """Store item data to Elasticsearch for content-based search"""
        try:
            index_name = os.getenv('ELASTICSEARCH_INDEX', 'recommendations')
            
            # Create index if it doesn't exist
            if not self.es_client.indices.exists(index=index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "item_id": {"type": "integer"},
                            "category": {"type": "keyword"},
                            "brand": {"type": "keyword"},
                            "price": {"type": "float"},
                            "avg_rating": {"type": "float"},
                            "description": {"type": "text", "analyzer": "standard"}
                        }
                    }
                }
                self.es_client.indices.create(index=index_name, body=mapping)
            
            # Index items
            for _, item in items_df.iterrows():
                doc = {
                    'item_id': item['item_id'],
                    'category': item['category'],
                    'brand': item['brand'],
                    'price': item['price'],
                    'avg_rating': item['avg_rating'],
                    'description': item['description']
                }
                
                self.es_client.index(index=index_name, id=item['item_id'], body=doc)
            
            logger.info(f"Indexed {len(items_df)} items to Elasticsearch")
        except Exception as e:
            logger.error(f"Error storing to Elasticsearch: {e}")
    
    def cache_statistics(self, users_df: pd.DataFrame, items_df: pd.DataFrame, 
                        interactions_df: pd.DataFrame):
        """Cache dataset statistics to Redis"""
        try:
            stats = {
                'total_users': len(users_df),
                'total_items': len(items_df),
                'total_interactions': len(interactions_df),
                'avg_interactions_per_user': len(interactions_df) / len(users_df),
                'categories': list(items_df['category'].unique()),
                'interaction_types': list(interactions_df['interaction_type'].unique()),
                'last_updated': datetime.now().isoformat()
            }
            
            for key, value in stats.items():
                self.redis_client.set(f"recsys:stats:{key}", json.dumps(value))
            
            # Cache popular items
            popular_items = interactions_df['item_id'].value_counts().head(100).to_dict()
            self.redis_client.set("recsys:popular_items", json.dumps(popular_items))
            
            logger.info("Cached recommendation system statistics to Redis")
        except Exception as e:
            logger.error(f"Error caching statistics: {e}")
    
    def generate_all_data(self):
        """Generate complete recommendation system dataset"""
        logger.info("Starting recommendation system data generation...")
        
        # Generate data
        logger.info("Generating users...")
        users_df = self.generate_users(10000)
        
        logger.info("Generating items...")
        items_df = self.generate_items(50000)
        
        logger.info("Generating interactions...")
        interactions_df = self.generate_interactions(users_df, items_df, 500000)
        
        logger.info(f"Generated:")
        logger.info(f"  Users: {len(users_df)}")
        logger.info(f"  Items: {len(items_df)}")
        logger.info(f"  Interactions: {len(interactions_df)}")
        
        # Store data
        logger.info("Storing data to PostgreSQL...")
        self.store_to_postgres(users_df, 'users')
        self.store_to_postgres(items_df, 'items')
        self.store_to_postgres(interactions_df, 'interactions')
        
        logger.info("Storing items to Elasticsearch...")
        self.store_to_elasticsearch(items_df)
        
        logger.info("Caching statistics to Redis...")
        self.cache_statistics(users_df, items_df, interactions_df)
        
        # Save to CSV for backup
        users_df.to_csv('users.csv', index=False)
        items_df.to_csv('items.csv', index=False)
        interactions_df.to_csv('interactions.csv', index=False)
        
        logger.info("Data generation completed!")
    
    def close_connections(self):
        """Close all connections"""
        self.pg_conn.close()
        self.redis_client.close()

def main():
    """Main execution function"""
    generator = RecommendationDataGenerator()
    
    try:
        # Wait for services to be ready
        logger.info("Waiting for services to be ready...")
        time.sleep(30)
        
        # Generate data
        generator.generate_all_data()
        
    except Exception as e:
        logger.error(f"Error in data generation: {e}")
    finally:
        generator.close_connections()

if __name__ == "__main__":
    main()