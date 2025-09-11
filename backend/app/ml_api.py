"""
Spark MLlib Service for FastAPI Backend
Author: BigData Expert
Description: ML prediction service using Spark MLlib models
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler

import pymongo
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
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("ERROR")
        
        # Model components
        self.models = {}
        self.hdfs_model_path = "hdfs://namenode:9000/youtube_trending/models"
        self.is_loaded = False
        
        # Try to load models
        self.load_mllib_models()

    def load_mllib_models(self) -> bool:
        """Load trained Spark MLlib models from HDFS"""
        try:
            # Get model metadata from MongoDB
            metadata = self.db.ml_metadata.find_one({"type": "spark_mllib_models"})
            
            if not metadata:
                print("[WARN] No Spark MLlib models found. Train models first.")
                return False
            
            available_models = metadata.get("models", [])
            models_loaded = 0
            
            for model_name in available_models:
                model_path = f"{self.hdfs_model_path}/{model_name}"
                
                try:
                    # Load pipeline model from HDFS
                    pipeline_model = PipelineModel.load(model_path)
                    self.models[model_name] = pipeline_model
                    models_loaded += 1
                    print(f"[OK] Loaded {model_name} from HDFS: {model_path}")
                    
                except Exception as e:
                    print(f"[WARN] Could not load {model_name}: {str(e)}")
                    continue
            
            if models_loaded == 0:
                print("[WARN] No MLlib models could be loaded from HDFS")
                return False
            
            self.is_loaded = True
            print(f"[OK] Spark MLlib Service loaded {models_loaded} models")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load MLlib models: {str(e)}")
            return False

    def predict_trending(self, video_data: Dict, model_name: str = "logistic_regression") -> Dict:
        """Make trending prediction using Spark MLlib model"""
        try:
            if not self.is_loaded:
                return {"error": "MLlib models not loaded", "prediction": None}
            
            if model_name not in self.models:
                available_models = list(self.models.keys())
                return {
                    "error": f"Model {model_name} not found. Available: {available_models}",
                    "prediction": None
                }
            
            # Convert input to Spark DataFrame
            df = self.spark.createDataFrame([video_data])
            
            # Make prediction
            model = self.models[model_name]
            predictions = model.transform(df)
            
            # Extract prediction results
            result = predictions.select("probability", "prediction").collect()[0]
            
            # Get prediction probability
            probability = float(result.probability[1]) if result.probability else 0.0
            prediction = int(result.prediction)
            
            return {
                "prediction": prediction,
                "probability": probability,
                "model_used": model_name,
                "framework": "Spark MLlib",
                "is_trending": prediction == 1,
                "confidence": probability if prediction == 1 else (1 - probability),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "prediction": None
            }

    def get_clustering_analysis(self, video_data: List[Dict]) -> Dict:
        """Perform clustering analysis on multiple videos"""
        try:
            if not self.is_loaded:
                return {"error": "MLlib models not loaded"}
            
            # Check if clustering model exists
            clustering_models = [name for name in self.models.keys() if 'cluster' in name.lower()]
            
            if not clustering_models:
                return {"error": "No clustering models available"}
            
            # Convert to Spark DataFrame
            df = self.spark.createDataFrame(video_data)
            
            results = {}
            
            for cluster_model_name in clustering_models:
                try:
                    model = self.models[cluster_model_name]
                    clustered = model.transform(df)
                    
                    # Get cluster statistics
                    cluster_stats = clustered.groupBy("cluster_prediction").agg(
                        count("*").alias("count"),
                        avg("views").alias("avg_views"),
                        avg("like_ratio").alias("avg_like_ratio")
                    ).collect()
                    
                    results[cluster_model_name] = [row.asDict() for row in cluster_stats]
                    
                except Exception as e:
                    results[cluster_model_name] = {"error": str(e)}
            
            return {
                "clustering_results": results,
                "framework": "Spark MLlib",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Clustering analysis failed: {str(e)}"}

    def predict_views(self, video_data: Dict, model_name: str = "linear_regression") -> Dict:
        """Predict view count using regression model"""
        try:
            if not self.is_loaded:
                return {"error": "MLlib models not loaded", "predicted_views": None}
            
            # Look for regression models
            regression_models = [name for name in self.models.keys() if 'regression' in name.lower()]
            
            if not regression_models:
                return {"error": "No regression models available", "predicted_views": None}
            
            # Use first available regression model
            reg_model_name = regression_models[0]
            
            # Convert to Spark DataFrame
            df = self.spark.createDataFrame([video_data])
            
            # Make prediction
            model = self.models[reg_model_name]
            predictions = model.transform(df)
            
            # Extract prediction
            result = predictions.select("prediction").collect()[0]
            log_views_prediction = float(result.prediction)
            
            # Convert back from log scale
            predicted_views = int(exp(log_views_prediction) - 1)
            
            return {
                "predicted_views": predicted_views,
                "log_prediction": log_views_prediction,
                "model_used": reg_model_name,
                "framework": "Spark MLlib",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"View prediction failed: {str(e)}",
                "predicted_views": None
            }

    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            "loaded_models": list(self.models.keys()),
            "is_loaded": self.is_loaded,
            "framework": "Spark MLlib",
            "hdfs_path": self.hdfs_model_path,
            "spark_version": self.spark.version,
            "model_count": len(self.models)
        }

    def close(self):
        """Close Spark session and MongoDB connection"""
        if hasattr(self, 'spark'):
            self.spark.stop()
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()

# Global service instance
_mllib_service = None

def get_mllib_service() -> SparkMLlibService:
    """Get global MLlib service instance"""
    global _mllib_service
    if _mllib_service is None:
        _mllib_service = SparkMLlibService()
    return _mllib_service

def initialize_mllib_service(mongo_uri: str, db_name: str) -> SparkMLlibService:
    """Initialize MLlib service with specific configuration"""
    global _mllib_service
    _mllib_service = SparkMLlibService(mongo_uri, db_name)
    return _mllib_service
