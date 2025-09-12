"""
Spark MLlib Service for FastAPI Backend
Description: ML prediction service using Spark MLlib models from HDFS
"""

import math
from typing import Dict, Any

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

from pymongo import MongoClient

class SparkMLlibService:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "youtube_trending"):
        """Initialize Spark MLlib prediction service"""
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("SparkMLlibService") \
            .master("local[*]") \
            .config("spark.driver.memory", "3g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("ERROR")
        
        # Model components
        self.models = {}
        self.hdfs_model_path = "hdfs://localhost:9000/youtube_trending/models"
        self.is_loaded = False
        
        # Try to load models
        self.load_mllib_models()

    def load_mllib_models(self) -> bool:
        """Load trained Spark MLlib models from HDFS"""
        try:
            print("[ML] Loading MLlib models from HDFS...")
            
            model_types = ["trending_prediction", "regression", "clustering"]
            
            for model_type in model_types:
                model_path = f"{self.hdfs_model_path}/{model_type}"
                
                try:
                    # Load Spark MLlib pipeline model
                    model = PipelineModel.load(model_path)
                    self.models[model_type] = model
                    print(f"[OK] Loaded {model_type} model from HDFS")
                    
                except Exception as e:
                    print(f"[WARN] Could not load {model_type} model: {str(e)}")
                    continue
            
            self.is_loaded = len(self.models) > 0
            
            if self.is_loaded:
                print(f"[SUCCESS] Loaded {len(self.models)} MLlib models")
                return True
            else:
                print("[WARN] No models loaded - predictions will use fallback")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to load MLlib models: {str(e)}")
            self.is_loaded = False
            return False

    def predict_trending(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict if a video will be trending"""
        try:
            if "trending_prediction" not in self.models:
                return {"trending_probability": 0.5, "confidence": "low", "method": "fallback"}
            
            # Prepare features
            features = {
                "like_ratio": video_data.get("likes", 0) / max(video_data.get("views", 1), 1),
                "dislike_ratio": video_data.get("dislikes", 0) / max(video_data.get("views", 1), 1),
                "comment_ratio": video_data.get("comment_count", 0) / max(video_data.get("views", 1), 1),
                "engagement_score": (video_data.get("likes", 0) + video_data.get("comment_count", 0)) / max(video_data.get("views", 1), 1),
                "title_length": len(video_data.get("title", "")),
                "has_caps": 1 if any(c.isupper() for c in video_data.get("title", "")[:10]) else 0,
                "tag_count": len(video_data.get("tags", "").split("|")) if video_data.get("tags") else 0,
                "category_id": video_data.get("category_id", 0)
            }
            
            # Create Spark DataFrame
            df = self.spark.createDataFrame([features])
            
            # Make prediction
            model = self.models["trending_prediction"]
            predictions = model.transform(df)
            
            # Get probability
            result = predictions.select("prediction", "probability").collect()[0]
            probability = float(result.probability[1])  # Probability of class 1 (trending)
            
            return {
                "trending_probability": probability,
                "prediction": int(result.prediction),
                "confidence": "high" if probability > 0.7 or probability < 0.3 else "medium",
                "method": "spark_mllib"
            }
            
        except Exception as e:
            print(f"[ERROR] Trending prediction failed: {str(e)}")
            return {"trending_probability": 0.5, "confidence": "low", "method": "fallback"}

    def predict_views(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict view count for a video"""
        try:
            if "regression" not in self.models:
                # Fallback prediction based on engagement
                likes = video_data.get("likes", 0)
                comments = video_data.get("comment_count", 0)
                estimated_views = (likes * 50) + (comments * 100)
                return {"predicted_views": estimated_views, "confidence": "low", "method": "fallback"}
            
            # Prepare features
            features = {
                "likes": video_data.get("likes", 0),
                "dislikes": video_data.get("dislikes", 0),
                "comment_count": video_data.get("comment_count", 0),
                "like_ratio": video_data.get("likes", 0) / max(video_data.get("views", 1), 1),
                "engagement_score": (video_data.get("likes", 0) + video_data.get("comment_count", 0)) / max(video_data.get("views", 1), 1),
                "title_length": len(video_data.get("title", "")),
                "tag_count": len(video_data.get("tags", "").split("|")) if video_data.get("tags") else 0,
                "category_id": video_data.get("category_id", 0)
            }
            
            # Create Spark DataFrame
            df = self.spark.createDataFrame([features])
            
            # Make prediction
            model = self.models["regression"]
            predictions = model.transform(df)
            
            # Get predicted log_views and convert back
            log_views = predictions.select("prediction").collect()[0][0]
            predicted_views = int(math.exp(log_views) - 1)
            
            return {
                "predicted_views": max(0, predicted_views),
                "confidence": "high",
                "method": "spark_mllib"
            }
            
        except Exception as e:
            print(f"[ERROR] Views prediction failed: {str(e)}")
            likes = video_data.get("likes", 0)
            comments = video_data.get("comment_count", 0)
            estimated_views = (likes * 50) + (comments * 100)
            return {"predicted_views": estimated_views, "confidence": "low", "method": "fallback"}

    def predict_cluster(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict cluster for a video"""
        try:
            if "clustering" not in self.models:
                # Fallback clustering based on view ranges
                views = video_data.get("views", 0)
                if views < 10000:
                    cluster = 0  # Low engagement
                elif views < 100000:
                    cluster = 1  # Medium engagement
                elif views < 1000000:
                    cluster = 2  # High engagement
                elif views < 10000000:
                    cluster = 3  # Viral content
                else:
                    cluster = 4  # Mega viral
                return {"cluster": cluster, "confidence": "low", "method": "fallback"}
            
            # Prepare features
            features = {
                "views": video_data.get("views", 0),
                "likes": video_data.get("likes", 0),
                "comment_count": video_data.get("comment_count", 0),
                "like_ratio": video_data.get("likes", 0) / max(video_data.get("views", 1), 1),
                "engagement_score": (video_data.get("likes", 0) + video_data.get("comment_count", 0)) / max(video_data.get("views", 1), 1),
                "title_length": len(video_data.get("title", "")),
                "tag_count": len(video_data.get("tags", "").split("|")) if video_data.get("tags") else 0
            }
            
            # Add log features
            import math
            features["log_views"] = math.log(features["views"] + 1)
            features["log_likes"] = math.log(features["likes"] + 1)
            features["log_comments"] = math.log(features["comment_count"] + 1)
            
            # Create Spark DataFrame
            df = self.spark.createDataFrame([features])
            
            # Make prediction
            model = self.models["clustering"]
            predictions = model.transform(df)
            
            # Get cluster
            cluster = predictions.select("cluster").collect()[0][0]
            
            return {
                "cluster": int(cluster),
                "confidence": "high",
                "method": "spark_mllib"
            }
            
        except Exception as e:
            print(f"[ERROR] Clustering prediction failed: {str(e)}")
            # Fallback clustering
            views = video_data.get("views", 0)
            if views < 10000:
                cluster = 0
            elif views < 100000:
                cluster = 1
            elif views < 1000000:
                cluster = 2
            elif views < 10000000:
                cluster = 3
            else:
                cluster = 4
            return {"cluster": cluster, "confidence": "low", "method": "fallback"}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "loaded_models": list(self.models.keys()),
            "is_loaded": self.is_loaded,
            "model_path": self.hdfs_model_path,
            "spark_version": self.spark.version,
            "total_models": len(self.models)
        }

# Global MLlib service instance
_mllib_service = None

def get_mllib_service() -> SparkMLlibService:
    """Get or create MLlib service instance"""
    global _mllib_service
    if _mllib_service is None:
        _mllib_service = SparkMLlibService()
    return _mllib_service

def initialize_mllib_service():
    """Initialize MLlib service"""
    global _mllib_service
    _mllib_service = SparkMLlibService()
    return _mllib_service