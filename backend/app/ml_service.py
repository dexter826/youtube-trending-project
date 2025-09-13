"""
Machine Learning Service
"""

from typing import Dict, Any
from fastapi import HTTPException
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, DoubleType
import math

from spark.core.spark_manager import get_spark_session, PRODUCTION_CONFIGS
from spark.core.database_manager import get_database_connection


class MLService:
    def __init__(self):
        """Initialize with production configs"""
        self.db = get_database_connection()
        self.spark = None
        self.models = {}
        self.is_trained = False
        self.metrics = {}
        self.cluster_names = {}  # Add cluster names storage

        self._init_spark()
        self.load_models_from_hdfs()
        self.load_metrics()
        self._load_cluster_names()  # Load dynamic cluster names

    def _init_spark(self):
        try:
            self.spark = get_spark_session("YouTubeMLService", PRODUCTION_CONFIGS["ml_inference"])
        except Exception as e:
            raise HTTPException(status_code=503, detail="Spark initialization failed")

    def load_models_from_hdfs(self) -> bool:
        """Load trained models from HDFS"""
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
                except Exception:
                    pass
            
            self.is_trained = loaded_count == len(hdfs_model_paths)
            return self.is_trained
                
        except Exception:
            self.is_trained = False
            return False

    def load_metrics(self):
        """Load model metrics from JSON file"""
        import json
        import os
        try:
            metrics_file = os.path.join(os.path.dirname(__file__), "../../spark/model_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                print(f"Loaded metrics from {metrics_file}")
            else:
                print("Metrics file not found, using default values")
                self.metrics = {}
        except Exception as e:
            print(f"Failed to load metrics: {e}")
            self.metrics = {}

    def predict_trending(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self.is_trained or "trending_classifier" not in self.models:
                raise HTTPException(status_code=503, detail="Trending classifier not available. Train models first.")
            
            input_df = self._create_spark_dataframe(video_data)
            
            model = self.models["trending_classifier"]
            predictions = model.transform(input_df)
            
            result = predictions.select("prediction", "probability").collect()[0]
            prediction = int(result["prediction"])
            probability = float(result["probability"][1])
            
            # Adjust threshold for imbalanced data
            if probability > 0.1:  # Lower threshold for trending prediction
                prediction = 1
            else:
                prediction = 0
            
            return {
                "trending_probability": probability,
                "prediction": prediction,
                "confidence": "high" if probability > 0.7 or probability < 0.3 else "medium",
                "method": "spark_mllib"
            }
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    def predict_views(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self.is_trained or "views_regressor" not in self.models:
                raise HTTPException(status_code=503, detail="Views regressor not available. Train models first.")
            
            input_df = self._create_regression_dataframe(video_data)
            
            model = self.models["views_regressor"]
            predictions = model.transform(input_df)
            
            result = predictions.select("prediction").collect()[0]
            predicted_log_views = float(result["prediction"])
            print(f"DEBUG: Raw prediction (log_views): {predicted_log_views}")
            predicted_views = int(max(0, math.exp(predicted_log_views) - 1))
            
            return {
                "predicted_views": predicted_views,
                "confidence": "high",
                "method": "spark_mllib"
            }
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    def predict_cluster(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self.is_trained or "content_clusterer" not in self.models:
                raise HTTPException(status_code=503, detail="Content clusterer not available. Train models first.")
            
            input_df = self._create_clustering_dataframe(video_data)
            
            model = self.models["content_clusterer"]
            predictions = model.transform(input_df)
            
            result = predictions.select("cluster").collect()[0]
            cluster = int(result["cluster"])
            
            # Use dynamic cluster names instead of hard-coded
            cluster_type = self.cluster_names.get(str(cluster), f"Cluster {cluster}")
            
            return {
                "cluster": cluster,
                "cluster_type": cluster_type,
                "method": "spark_mllib"
            }
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    def _create_spark_dataframe(self, video_data: Dict[str, Any]):
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
            
            import math
            log_views = math.log1p(views)
            
            # New time-based features
            publish_hour = float(video_data.get("publish_hour", 12))  # Default to noon
            video_age_proxy = float(video_data.get("video_age_proxy", 2))  # Default to category 2
            
            features = {
                "log_views": float(log_views),
                "like_ratio": likes / views,
                "dislike_ratio": dislikes / views,
                "comment_ratio": comment_count / views,
                "engagement_score": (likes + comment_count) / views,
                "title_length": float(len(title)),
                "has_caps": has_caps,
                "tag_count": float(len(video_data.get("tags", "").split("|")) if video_data.get("tags") else 0),
                "category_id": float(video_data.get("category_id", 0)),
                "publish_hour": publish_hour,
                "video_age_proxy": video_age_proxy
            }
            
            schema = StructType([
                StructField("log_views", DoubleType(), True),
                StructField("like_ratio", DoubleType(), True),
                StructField("dislike_ratio", DoubleType(), True),
                StructField("comment_ratio", DoubleType(), True),
                StructField("engagement_score", DoubleType(), True),
                StructField("title_length", DoubleType(), True),
                StructField("has_caps", DoubleType(), True),
                StructField("tag_count", DoubleType(), True),
                StructField("category_id", DoubleType(), True),
                StructField("publish_hour", DoubleType(), True),
                StructField("video_age_proxy", DoubleType(), True)
            ])
            
            return self.spark.createDataFrame([tuple(features.values())], schema)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid input data format")

    def _create_regression_dataframe(self, video_data: Dict[str, Any]):
        try:
            views = max(float(video_data.get("views", 1)), 1)
            likes = float(video_data.get("likes", 0))
            dislikes = float(video_data.get("dislikes", 0))
            comment_count = float(video_data.get("comment_count", 0))
            title = video_data.get("title", "")
            
            # New time-based features
            publish_hour = float(video_data.get("publish_hour", 12))
            video_age_proxy = float(video_data.get("video_age_proxy", 2))
            
            features = {
                "views": views,
                "likes": likes,
                "dislikes": dislikes,
                "comment_count": comment_count,
                "like_ratio": likes / views,
                "engagement_score": (likes + comment_count) / views,
                "title_length": float(len(title)),
                "tag_count": float(len(video_data.get("tags", "").split("|")) if video_data.get("tags") else 0),
                "category_id": float(video_data.get("category_id", 0)),
                "publish_hour": publish_hour,
                "video_age_proxy": video_age_proxy
            }
            
            schema = StructType([
                StructField("views", DoubleType(), True),
                StructField("likes", DoubleType(), True),
                StructField("dislikes", DoubleType(), True),
                StructField("comment_count", DoubleType(), True),
                StructField("like_ratio", DoubleType(), True),
                StructField("engagement_score", DoubleType(), True),
                StructField("title_length", DoubleType(), True),
                StructField("tag_count", DoubleType(), True),
                StructField("category_id", DoubleType(), True),
                StructField("publish_hour", DoubleType(), True),
                StructField("video_age_proxy", DoubleType(), True)
            ])
            
            return self.spark.createDataFrame([tuple(features.values())], schema)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid input data format")

    def _create_clustering_dataframe(self, video_data: Dict[str, Any]):
        try:
            views = max(float(video_data.get("views", 1)), 1)
            likes = float(video_data.get("likes", 0))
            dislikes = float(video_data.get("dislikes", 0))
            comment_count = float(video_data.get("comment_count", 0))
            title = video_data.get("title", "")
            
            # New time-based features
            publish_hour = float(video_data.get("publish_hour", 12))
            video_age_proxy = float(video_data.get("video_age_proxy", 2))
            
            import math
            log_views = math.log1p(views)
            log_likes = math.log1p(likes)
            log_dislikes = math.log1p(dislikes)
            log_comments = math.log1p(comment_count)
            
            features = {
                "log_views": float(log_views),
                "log_likes": float(log_likes),
                "log_dislikes": float(log_dislikes),
                "log_comments": float(log_comments),
                "like_ratio": likes / views,
                "engagement_score": (likes + comment_count) / views,
                "title_length": float(len(title)),
                "tag_count": float(len(video_data.get("tags", "").split("|")) if video_data.get("tags") else 0),
                "category_id": float(video_data.get("category_id", 0)),
                "publish_hour": publish_hour,
                "video_age_proxy": video_age_proxy
            }
            
            schema = StructType([
                StructField("log_views", DoubleType(), True),
                StructField("log_likes", DoubleType(), True),
                StructField("log_dislikes", DoubleType(), True),
                StructField("log_comments", DoubleType(), True),
                StructField("like_ratio", DoubleType(), True),
                StructField("engagement_score", DoubleType(), True),
                StructField("title_length", DoubleType(), True),
                StructField("tag_count", DoubleType(), True),
                StructField("category_id", DoubleType(), True),
                StructField("publish_hour", DoubleType(), True),
                StructField("video_age_proxy", DoubleType(), True)
            ])
            
            df = self.spark.createDataFrame([tuple(features.values())], schema)
            return df
            
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid input data format")

    def train_models(self) -> bool:
        """Trigger model training using Spark job"""
        try:
            import subprocess
            import os
            
            if not self.spark:
                raise HTTPException(status_code=503, detail="Spark session not available")
            
            # Check training data availability
            training_data_count = self.db.ml_features.count_documents({})
            if training_data_count < 1000:
                raise HTTPException(status_code=400, detail=f"Insufficient training data: {training_data_count} records")
            
            # Path to training script - use absolute path from project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            script_path = os.path.join(project_root, 'spark', 'train_models.py')
            
            # Check if script exists
            if not os.path.exists(script_path):
                raise HTTPException(status_code=500, detail=f"Training script not found at: {script_path}")
            
            # Check if spark-submit is available
            import shutil
            spark_submit_path = shutil.which('spark-submit')
            if not spark_submit_path:
                raise HTTPException(status_code=500, detail="spark-submit not found in PATH. Please ensure Apache Spark is installed and in PATH.")
            
            # Run training with spark-submit
            cmd = f"spark-submit {script_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                # Reload models after training
                self.load_models_from_hdfs()
                return True
            else:
                raise HTTPException(status_code=500, detail=f"Training failed: {result.stderr}")
            
        except Exception as e:
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
            "spark_session": self.spark is not None,
            "metrics": self.metrics
        }

    def _load_cluster_names(self):
        """Load dynamic cluster names from JSON file"""
        import json
        import os
        try:
            # Fix path to point to project root
            current_dir = os.path.dirname(__file__)
            cluster_file = os.path.join(current_dir, "../../cluster_names.json")
            print(f"Current dir: {current_dir}")
            print(f"Looking for cluster file at: {cluster_file}")
            print(f"Absolute path: {os.path.abspath(cluster_file)}")
            if os.path.exists(cluster_file):
                with open(cluster_file, 'r', encoding='utf-8') as f:
                    self.cluster_names = json.load(f)
                print(f"Loaded dynamic cluster names from {cluster_file}")
                print(f"Cluster names: {self.cluster_names}")
            else:
                print("Cluster names file not found, using fallback names")
                self.cluster_names = self._get_fallback_cluster_names()
        except Exception as e:
            print(f"Failed to load cluster names: {e}, using fallback")
            self.cluster_names = self._get_fallback_cluster_names()

    def _get_fallback_cluster_names(self):
        """Fallback cluster names if dynamic loading fails"""
        return {
            "0": "Nội dung Tác động Cao",
            "1": "Nội dung Đại chúng", 
            "2": "Nội dung Tiềm năng",
            "3": "Nội dung Ổn định"
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