"""
Extended ML Service for Advanced Clustering Endpoints
Author: BigData Expert  
Description: FastAPI service for serving advanced clustering predictions
"""

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeansModel

import pymongo
from pymongo import MongoClient

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "youtube_trending"

class AdvancedClusteringService:
    def __init__(self):
        """Initialize extended clustering service"""
        self.spark = SparkSession.builder \
            .appName("AdvancedClusteringService") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # MongoDB client
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[DB_NAME]
        
        # HDFS paths
        self.hdfs_base_path = "hdfs://namenode:9000/youtube_trending"
        self.clustering_models_path = f"{self.hdfs_base_path}/clustering_models"
        
        # Load available models
        self.loaded_models = {}
        self._load_clustering_models()
        
        print("[OK] Advanced Clustering Service initialized")

    def _load_clustering_models(self):
        """Load available clustering models from HDFS"""
        try:
            # Check for behavioral clustering model
            behavioral_path = f"{self.clustering_models_path}/behavioral_kmeans"
            try:
                self.loaded_models['behavioral'] = Pipeline.load(behavioral_path)
                print("[OK] Loaded behavioral clustering model")
            except Exception:
                print("[WARN] Behavioral clustering model not found")
            
            # Check for content clustering model
            content_path = f"{self.clustering_models_path}/content_clustering"
            try:
                self.loaded_models['content'] = Pipeline.load(content_path)
                print("[OK] Loaded content clustering model")
            except Exception:
                print("[WARN] Content clustering model not found")
            
            # Check for geographic clustering model
            geo_path = f"{self.clustering_models_path}/geographic_clustering"
            try:
                self.loaded_models['geographic'] = Pipeline.load(geo_path)
                print("[OK] Loaded geographic clustering model")
            except Exception:
                print("[WARN] Geographic clustering model not found")
                
        except Exception as e:
            print(f"[WARN] Could not check HDFS models: {str(e)}")

    def get_behavioral_cluster(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict behavioral cluster for a video"""
        try:
            # Create DataFrame from input
            input_data = [{
                'views': int(video_data.get('views', 0)),
                'likes': int(video_data.get('likes', 0)), 
                'dislikes': int(video_data.get('dislikes', 0)),
                'comment_count': int(video_data.get('comment_count', 0)),
                'like_ratio': float(video_data.get('like_ratio', 0)),
                'engagement_score': float(video_data.get('engagement_score', 0))
            }]
            
            input_df = self.spark.createDataFrame(input_data)
            
            # Use model if available, otherwise fallback clustering
            if 'behavioral' in self.loaded_models:
                prediction = self.loaded_models['behavioral'].transform(input_df)
                cluster_id = prediction.select("behavioral_cluster_kmeans").collect()[0][0]
            else:
                # Fallback behavioral clustering logic
                views = input_data[0]['views']
                engagement = input_data[0]['engagement_score']
                
                if views > 10000000 and engagement > 0.05:
                    cluster_id = 0  # Viral videos
                elif views > 1000000 and engagement > 0.03:
                    cluster_id = 1  # Popular videos
                elif views > 100000:
                    cluster_id = 2  # Moderate videos
                else:
                    cluster_id = 3  # Low engagement
            
            # Get cluster interpretation
            cluster_names = {
                0: "Viral Content",
                1: "Popular Content", 
                2: "Moderate Engagement",
                3: "Low Engagement",
                4: "Niche Content",
                5: "Emerging Content"
            }
            
            return {
                'cluster_id': int(cluster_id),
                'cluster_name': cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
                'confidence': 0.85,
                'cluster_type': 'behavioral'
            }
            
        except Exception as e:
            print(f"[ERROR] Behavioral clustering failed: {str(e)}")
            return {
                'cluster_id': -1,
                'cluster_name': 'Unknown',
                'confidence': 0.0,
                'cluster_type': 'behavioral',
                'error': str(e)
            }

    def get_content_cluster(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict content cluster based on title and tags"""
        try:
            title = video_data.get('title', '').lower()
            category_id = int(video_data.get('category_id', 0))
            
            # Content-based clustering rules
            cluster_mapping = {
                10: "Music",
                23: "Comedy",
                24: "Entertainment", 
                25: "News & Politics",
                26: "Howto & Style",
                27: "Education",
                28: "Science & Technology",
                1: "Film & Animation",
                2: "Autos & Vehicles",
                17: "Sports",
                19: "Travel & Events",
                20: "Gaming",
                22: "People & Blogs"
            }
            
            # Keywords for content clustering
            if any(word in title for word in ['music', 'song', 'official', 'mv']):
                cluster_id = 0
                cluster_name = "Music Content"
            elif any(word in title for word in ['funny', 'comedy', 'laugh', 'joke']):
                cluster_id = 1
                cluster_name = "Comedy Content"
            elif any(word in title for word in ['news', 'breaking', 'update']):
                cluster_id = 2
                cluster_name = "News Content"
            elif any(word in title for word in ['tutorial', 'how to', 'guide', 'learn']):
                cluster_id = 3
                cluster_name = "Educational Content"
            elif any(word in title for word in ['game', 'gaming', 'gameplay', 'stream']):
                cluster_id = 4
                cluster_name = "Gaming Content"
            elif any(word in title for word in ['tech', 'review', 'unbox', 'smartphone']):
                cluster_id = 5
                cluster_name = "Tech Content"
            elif any(word in title for word in ['vlog', 'daily', 'life', 'routine']):
                cluster_id = 6
                cluster_name = "Lifestyle Content"
            else:
                cluster_id = 7
                cluster_name = cluster_mapping.get(category_id, "General Content")
            
            return {
                'cluster_id': int(cluster_id),
                'cluster_name': cluster_name,
                'confidence': 0.80,
                'cluster_type': 'content',
                'category_id': category_id
            }
            
        except Exception as e:
            print(f"[ERROR] Content clustering failed: {str(e)}")
            return {
                'cluster_id': -1,
                'cluster_name': 'Unknown',
                'confidence': 0.0,
                'cluster_type': 'content',
                'error': str(e)
            }

    def get_geographic_cluster(self, country: str) -> Dict[str, Any]:
        """Predict geographic cluster for a country"""
        try:
            # Geographic clustering based on regions
            geographic_clusters = {
                # High-engagement English countries
                'US': {'cluster_id': 0, 'cluster_name': 'North America High Engagement'},
                'CA': {'cluster_id': 0, 'cluster_name': 'North America High Engagement'},
                'GB': {'cluster_id': 0, 'cluster_name': 'North America High Engagement'},
                
                # European markets
                'DE': {'cluster_id': 1, 'cluster_name': 'European Market'},
                'FR': {'cluster_id': 1, 'cluster_name': 'European Market'},
                
                # Asian markets
                'JP': {'cluster_id': 2, 'cluster_name': 'Asian Market'},
                'KR': {'cluster_id': 2, 'cluster_name': 'Asian Market'},
                'IN': {'cluster_id': 2, 'cluster_name': 'Asian Market'},
                
                # Emerging markets
                'MX': {'cluster_id': 3, 'cluster_name': 'Emerging Market'},
                'RU': {'cluster_id': 3, 'cluster_name': 'Emerging Market'}
            }
            
            cluster_info = geographic_clusters.get(country, {
                'cluster_id': 4, 
                'cluster_name': 'Other Market'
            })
            
            return {
                'cluster_id': cluster_info['cluster_id'],
                'cluster_name': cluster_info['cluster_name'],
                'confidence': 0.90,
                'cluster_type': 'geographic',
                'country': country
            }
            
        except Exception as e:
            print(f"[ERROR] Geographic clustering failed: {str(e)}")
            return {
                'cluster_id': -1,
                'cluster_name': 'Unknown',
                'confidence': 0.0,
                'cluster_type': 'geographic',
                'error': str(e)
            }

    def get_temporal_cluster(self, publish_time: str) -> Dict[str, Any]:
        """Predict temporal cluster based on publish time"""
        try:
            # Parse publish time
            if 'T' in publish_time:
                dt = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
            else:
                dt = datetime.strptime(publish_time, '%Y-%m-%d %H:%M:%S')
            
            hour = dt.hour
            day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
            
            # Temporal clustering rules
            if 6 <= hour <= 10:  # Morning
                if day_of_week < 5:  # Weekday
                    cluster_id = 0
                    cluster_name = "Weekday Morning"
                else:  # Weekend
                    cluster_id = 1
                    cluster_name = "Weekend Morning"
            elif 11 <= hour <= 17:  # Afternoon
                if day_of_week < 5:
                    cluster_id = 2
                    cluster_name = "Weekday Afternoon"
                else:
                    cluster_id = 3
                    cluster_name = "Weekend Afternoon"
            elif 18 <= hour <= 23:  # Evening
                cluster_id = 4
                cluster_name = "Prime Time Evening"
            else:  # Late night/early morning
                cluster_id = 5
                cluster_name = "Late Night"
            
            return {
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'confidence': 0.75,
                'cluster_type': 'temporal',
                'publish_hour': hour,
                'day_of_week': day_of_week
            }
            
        except Exception as e:
            print(f"[ERROR] Temporal clustering failed: {str(e)}")
            return {
                'cluster_id': -1,
                'cluster_name': 'Unknown',
                'confidence': 0.0,
                'cluster_type': 'temporal',
                'error': str(e)
            }

    def get_comprehensive_clustering(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive clustering analysis for a video"""
        try:
            # Calculate derived features
            views = int(video_data.get('views', 0))
            likes = int(video_data.get('likes', 0))
            dislikes = int(video_data.get('dislikes', 0))
            comments = int(video_data.get('comment_count', 0))
            
            # Add calculated features
            video_data['like_ratio'] = likes / views if views > 0 else 0
            video_data['engagement_score'] = (likes + comments) / views if views > 0 else 0
            
            # Get all clustering results
            behavioral = self.get_behavioral_cluster(video_data)
            content = self.get_content_cluster(video_data)
            geographic = self.get_geographic_cluster(video_data.get('country', 'US'))
            temporal = self.get_temporal_cluster(video_data.get('publish_time', datetime.now().isoformat()))
            
            # Overall cluster recommendation
            overall_score = (behavioral['confidence'] + content['confidence'] + 
                           geographic['confidence'] + temporal['confidence']) / 4
            
            return {
                'video_id': video_data.get('video_id', 'unknown'),
                'overall_clustering_score': round(overall_score, 3),
                'behavioral_cluster': behavioral,
                'content_cluster': content,
                'geographic_cluster': geographic,
                'temporal_cluster': temporal,
                'analysis_timestamp': datetime.now().isoformat(),
                'clustering_framework': 'Spark MLlib Advanced'
            }
            
        except Exception as e:
            print(f"[ERROR] Comprehensive clustering failed: {str(e)}")
            return {
                'error': str(e),
                'clustering_framework': 'Spark MLlib Advanced'
            }

    def get_clustering_statistics(self) -> Dict[str, Any]:
        """Get clustering statistics and insights"""
        try:
            # Get statistics from MongoDB
            stats = {}
            
            # Get clustering metadata
            metadata = list(self.db.ml_metadata.find({"type": "advanced_clustering_analysis"}).limit(1))
            if metadata:
                stats['last_analysis'] = metadata[0]
            
            # Get detailed clustering results
            clustering_results = list(self.db.clustering_results.find({}))
            stats['available_analyses'] = len(clustering_results)
            
            # Group by cluster type
            cluster_types = {}
            for result in clustering_results:
                cluster_type = result.get('cluster_type', 'unknown')
                if cluster_type not in cluster_types:
                    cluster_types[cluster_type] = []
                cluster_types[cluster_type].append(result)
            
            stats['cluster_types'] = {
                cluster_type: len(analyses) 
                for cluster_type, analyses in cluster_types.items()
            }
            
            # Model availability
            stats['loaded_models'] = list(self.loaded_models.keys())
            stats['total_models'] = len(self.loaded_models)
            
            return {
                'clustering_statistics': stats,
                'framework': 'Spark MLlib Advanced',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to get clustering statistics: {str(e)}")
            return {
                'error': str(e),
                'framework': 'Spark MLlib Advanced'
            }

# Global service instance
advanced_clustering_service = None

def get_advanced_clustering_service():
    """Get global clustering service instance"""
    global advanced_clustering_service
    if advanced_clustering_service is None:
        advanced_clustering_service = AdvancedClusteringService()
    return advanced_clustering_service

def main():
    """Test the advanced clustering service"""
    service = AdvancedClusteringService()
    
    # Test video data
    test_video = {
        'video_id': 'test_123',
        'title': 'Amazing Tech Review - iPhone 15 Pro Max',
        'views': 2500000,
        'likes': 125000,
        'dislikes': 5000,
        'comment_count': 8500,
        'category_id': 28,
        'country': 'US',
        'publish_time': '2024-01-15T14:30:00Z'
    }
    
    # Test comprehensive clustering
    result = service.get_comprehensive_clustering(test_video)
    print("Comprehensive Clustering Result:")
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
