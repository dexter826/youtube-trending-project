"""
Model Loading Service
"""

from typing import Dict, Any
from fastapi import HTTPException
from pyspark.ml import PipelineModel
import json
import os
import sys

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


sys.path.insert(0, project_root)

from spark.core.spark_manager import get_spark_session, PRODUCTION_CONFIGS


class ModelLoader:
    def __init__(self, spark_session=None):
        self.spark = spark_session
        self.models = {}
        self.is_trained = False
        self.metrics = {}
        self.cluster_names = {}

        if self.spark:
            self.load_models_from_hdfs()
            self.load_metrics()
            self._load_cluster_names()

    def load_models_from_hdfs(self) -> bool:
        """Load trained models from HDFS"""
        try:
            if not self.spark:
                raise HTTPException(status_code=503, detail="Spark session not initialized")

            hdfs_model_paths = {
                "content_clusterer": "hdfs://localhost:9000/youtube_trending/models/clustering",
                # days regressor for predicting days in trending
                "days_regressor": "hdfs://localhost:9000/youtube_trending/models/days_regression",
            }

            loaded_count = 0
            for model_name, hdfs_path in hdfs_model_paths.items():
                try:
                    model = PipelineModel.load(hdfs_path)
                    self.models[model_name] = model
                    loaded_count += 1
                except Exception as e:
                    pass

            # Consider trained if at least clustering is available; other models are optional
            self.is_trained = loaded_count >= 1
            return self.is_trained

        except Exception as e:
            self.is_trained = False
            return False

    def load_metrics(self):
        """Load model metrics from JSON file"""
        try:
            # Import path_config for proper path management
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            sys.path.insert(0, str(project_root))
            from config.paths import path_config
            
            metrics_file = path_config.SPARK_METRICS_DIR / "model_metrics.json"
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.metrics = json.load(f)
            else:
                self.metrics = {}
        except Exception as e:
            self.metrics = {}

    def _load_cluster_names(self):
        """Load dynamic cluster names from JSON file"""
        try:
            # Import path_config for proper path management
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            sys.path.insert(0, str(project_root))
            from config.paths import path_config
            
            cluster_file = path_config.SPARK_ANALYSIS_DIR / "cluster_names.json"
            if os.path.exists(cluster_file):
                with open(cluster_file, 'r', encoding='utf-8') as f:
                    self.cluster_names = json.load(f)
            else:
                self.cluster_names = self._get_fallback_cluster_names()
        except Exception as e:
            self.cluster_names = self._get_fallback_cluster_names()

    def _get_fallback_cluster_names(self):
        """Fallback cluster names if dynamic loading fails"""
        return {
            "0": "Nội dung Tác động Cao",
            "1": "Nội dung Đại chúng",
            "2": "Nội dung Tiềm năng",
            "3": "Nội dung Ổn định"
        }

    def get_model(self, model_name: str):
        """Get a specific model"""
        return self.models.get(model_name)

    def get_cluster_name(self, cluster_id: int) -> str:
        """Get cluster name by ID"""
        return self.cluster_names.get(str(cluster_id), f"Cluster {cluster_id}")