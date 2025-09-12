"""
Machine Learning Service
Description: AI predictions using Spark MLlib models from HDFS
"""

from typing import Dict, Any
from fastapi import HTTPException
from pymongo import MongoClient
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
import os


class MLService:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "youtube_trending"):
        """Initialize ML service with Spark MLlib"""
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        
        # Spark session for ML
        self.spark = None
        self.models = {}
        self.is_trained = False
        
        # Initialize Spark and load models
        self._init_spark()
        self.load_models_from_hdfs()

    def _init_spark(self):
        """Initialize Spark session for ML operations"""
        try:
            self.spark = SparkSession.builder \
                .appName("YouTubeMLService") \
                .master("local[*]") \
                .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            
            # Set log level to reduce noise
            self.spark.sparkContext.setLogLevel("WARN")
            print("✅ Spark session initialized for ML service")
            
        except Exception as e:
            print(f"❌ Failed to initialize Spark: {str(e)}")
            raise HTTPException(status_code=503, detail="Spark initialization failed")

    def load_models_from_hdfs(self) -> bool:
        """Load trained Spark MLlib models from HDFS"""
        try:
            if not self.spark:
                raise HTTPException(status_code=503, detail="Spark session not initialized")
            
            hdfs_model_paths = {
                "trending_classifier": "hdfs://localhost:9000/youtube_trending/models/trending_prediction",
                "views_regressor": "hdfs://localhost:9000/youtube_trending/models/regression",
                "content_clusterer": "hdfs://localhost:9000/youtube_trending/models/clustering"
            }
            
            loaded_count = 0
            for model_name, hdfs_path in hdfs_model_paths.items():
                try:
                    model = PipelineModel.load(hdfs_path)
                    self.models[model_name] = model
                    loaded_count += 1
                    print(f"✅ Loaded {model_name} from HDFS ({len(model.stages)} stages)")
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {str(e)}")
            
            self.is_trained = loaded_count == 3
            
            if self.is_trained:
                print(f"🚀 All {loaded_count} Spark MLlib models loaded from HDFS")
                return True
            else:
                print(f"⚠️ Only {loaded_count}/3 models loaded. Train models first.")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load models from HDFS: {str(e)}")
            self.is_trained = False
            return False

    def predict_trending(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict if a video will be trending using Spark MLlib"""
        try:
            if not self.is_trained or "trending_classifier" not in self.models:
                raise HTTPException(status_code=503, detail="Trending classifier not available. Train models first.")
            
            # Create DataFrame from input data
            input_df = self._create_spark_dataframe(video_data)
            
            # Get model and predict
            model = self.models["trending_classifier"]
            predictions = model.transform(input_df)
            
            # Extract results
            result = predictions.select("prediction", "probability").collect()[0]
            prediction = int(result["prediction"])
            probability = float(result["probability"][1])  # Probability of class 1 (trending)
            
            return {
                "trending_probability": probability,
                "prediction": prediction,
                "confidence": "high" if probability > 0.7 or probability < 0.3 else "medium",
                "method": "spark_mllib"
            }
                
        except Exception as e:
            print(f"❌ Trending prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    def predict_views(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict view count using Spark MLlib regression"""
        try:
            if not self.is_trained or "views_regressor" not in self.models:
                raise HTTPException(status_code=503, detail="Views regressor not available. Train models first.")
            
            # Create DataFrame from input data
            input_df = self._create_regression_dataframe(video_data)
            
            # Get model and predict
            model = self.models["views_regressor"]
            predictions = model.transform(input_df)
            
            # Extract results
            result = predictions.select("prediction").collect()[0]
            predicted_log_views = float(result["prediction"])
            predicted_views = int(max(0, predicted_log_views))  # Convert from log scale if needed
            
            return {
                "predicted_views": predicted_views,
                "confidence": "high",
                "method": "spark_mllib"
            }
                
        except Exception as e:
            print(f"❌ Views prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    def predict_cluster(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict content cluster using Spark MLlib clustering"""
        try:
            if not self.is_trained or "content_clusterer" not in self.models:
                raise HTTPException(status_code=503, detail="Content clusterer not available. Train models first.")
            
            # Create DataFrame from input data
            input_df = self._create_clustering_dataframe(video_data)
            
            # Get model and predict
            model = self.models["content_clusterer"]
            predictions = model.transform(input_df)
            
            # Extract results
            result = predictions.select("cluster").collect()[0]
            cluster = int(result["cluster"])
            
            # Map cluster to content type
            cluster_types = {
                0: "Entertainment",
                1: "Educational", 
                2: "Music",
                3: "Gaming",
                4: "News",
                5: "Tech",
                6: "Sports",
                7: "Other"
            }
            
            cluster_type = cluster_types.get(cluster, "Other")
            
            return {
                "cluster": cluster,
                "cluster_type": cluster_type,
                "method": "spark_mllib"
            }
                
        except Exception as e:
            print(f"❌ Clustering failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    def _create_spark_dataframe(self, video_data: Dict[str, Any]):
        """Create Spark DataFrame for trending prediction"""
        return self._create_trending_dataframe(video_data)

    def _create_trending_dataframe(self, video_data: Dict[str, Any]):
        """Create DataFrame for trending classification model"""
        try:
            views = max(float(video_data.get("views", 1)), 1)
            likes = float(video_data.get("likes", 0))
            dislikes = float(video_data.get("dislikes", 0))
            comment_count = float(video_data.get("comment_count", 0))
            title = video_data.get("title", "")
            
            import re
            has_caps = 1.0 if re.search(r'[A-Z]{3,}', title) else 0.0
            
            features = {
                "like_ratio": likes / views,
                "dislike_ratio": dislikes / views,
                "comment_ratio": comment_count / views,
                "engagement_score": (likes + comment_count) / views,
                "title_length": float(len(title)),
                "has_caps": has_caps,
                "tag_count": float(len(video_data.get("tags", "").split("|")) if video_data.get("tags") else 0),
                "category_id": float(video_data.get("category_id", 0))
            }
            
            schema = StructType([
                StructField("like_ratio", DoubleType(), True),
                StructField("dislike_ratio", DoubleType(), True),
                StructField("comment_ratio", DoubleType(), True),
                StructField("engagement_score", DoubleType(), True),
                StructField("title_length", DoubleType(), True),
                StructField("has_caps", DoubleType(), True),
                StructField("tag_count", DoubleType(), True),
                StructField("category_id", DoubleType(), True)
            ])
            
            return self.spark.createDataFrame([tuple(features.values())], schema)
            
        except Exception as e:
            print(f"❌ Failed to create trending DataFrame: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid input data format")

    def _create_regression_dataframe(self, video_data: Dict[str, Any]):
        """Create DataFrame for views regression model"""
        try:
            views = max(float(video_data.get("views", 1)), 1)
            likes = float(video_data.get("likes", 0))
            dislikes = float(video_data.get("dislikes", 0))
            comment_count = float(video_data.get("comment_count", 0))
            title = video_data.get("title", "")
            
            features = {
                "likes": likes,
                "dislikes": dislikes,
                "comment_count": comment_count,
                "like_ratio": likes / views,
                "engagement_score": (likes + comment_count) / views,
                "title_length": float(len(title)),
                "tag_count": float(len(video_data.get("tags", "").split("|")) if video_data.get("tags") else 0),
                "category_id": float(video_data.get("category_id", 0))
            }
            
            schema = StructType([
                StructField("likes", DoubleType(), True),
                StructField("dislikes", DoubleType(), True),
                StructField("comment_count", DoubleType(), True),
                StructField("like_ratio", DoubleType(), True),
                StructField("engagement_score", DoubleType(), True),
                StructField("title_length", DoubleType(), True),
                StructField("tag_count", DoubleType(), True),
                StructField("category_id", DoubleType(), True)
            ])
            
            return self.spark.createDataFrame([tuple(features.values())], schema)
            
        except Exception as e:
            print(f"❌ Failed to create regression DataFrame: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid input data format")

    def _create_clustering_dataframe(self, video_data: Dict[str, Any]):
        """Create DataFrame for content clustering model"""
        try:
            views = max(float(video_data.get("views", 1)), 1)
            likes = float(video_data.get("likes", 0))
            comment_count = float(video_data.get("comment_count", 0))
            title = video_data.get("title", "")
            
            import math
            log_views = math.log1p(views)
            log_likes = math.log1p(likes)
            log_comments = math.log1p(comment_count)
            
            features = {
                "log_views": float(log_views),
                "log_likes": float(log_likes),
                "log_comments": float(log_comments),
                "like_ratio": likes / views,
                "engagement_score": (likes + comment_count) / views,
                "title_length": float(len(title)),
                "tag_count": float(len(video_data.get("tags", "").split("|")) if video_data.get("tags") else 0)
            }
            
            schema = StructType([
                StructField("log_views", DoubleType(), True),
                StructField("log_likes", DoubleType(), True),
                StructField("log_comments", DoubleType(), True),
                StructField("like_ratio", DoubleType(), True),
                StructField("engagement_score", DoubleType(), True),
                StructField("title_length", DoubleType(), True),
                StructField("tag_count", DoubleType(), True)
            ])
            
            return self.spark.createDataFrame([tuple(features.values())], schema)
            
        except Exception as e:
            print(f"❌ Failed to create clustering DataFrame: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid input data format")

    def train_models(self) -> bool:
        """Trigger model training using Spark job"""
        try:
            if not self.spark:
                raise HTTPException(status_code=503, detail="Spark session not available")
            
            # Check training data availability
            training_data_count = self.db.ml_features.count_documents({})
            if training_data_count < 1000:
                raise HTTPException(status_code=400, detail=f"Insufficient training data: {training_data_count} records")
            
            print(f"🤖 Triggering Spark MLlib model training with {training_data_count} records...")
            
            # Run Spark training job
            import subprocess
            import sys
            
            result = subprocess.run(
                [sys.executable, "spark/train_models.py"], 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                # Reload models after training
                success = self.load_models_from_hdfs()
                if success:
                    return {
                        "status": "success",
                        "message": "Spark MLlib models trained successfully",
                        "training_data_count": training_data_count,
                        "models": list(self.models.keys()),
                        "framework": "spark_mllib"
                    }
                else:
                    raise HTTPException(status_code=500, detail="Training completed but model loading failed")
            else:
                raise HTTPException(status_code=500, detail=f"Training failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded Spark MLlib models"""
        model_details = {}
        for name, model in self.models.items():
            model_details[name] = {
                "stages": len(model.stages) if hasattr(model, 'stages') else 0,
                "type": "PipelineModel"
            }
        
        return {
            "loaded_models": list(self.models.keys()),
            "is_trained": self.is_trained,
            "model_type": "spark_mllib",
            "framework": "Apache Spark MLlib",
            "storage": "HDFS",
            "model_details": model_details,
            "total_models": len(self.models),
            "spark_session": self.spark is not None
        }

    def __del__(self):
        """Cleanup Spark session"""
        if hasattr(self, 'spark') and self.spark:
            try:
                self.spark.stop()
            except:
                pass


# Global ML service instance
_ml_service = None

def get_ml_service() -> MLService:
    """Get or create ML service instance"""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service

def initialize_ml_service():
    """Initialize ML service"""
    global _ml_service
    _ml_service = MLService()
    return _ml_service