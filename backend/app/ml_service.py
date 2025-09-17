"""
Machine Learning Service - Refactored to use modular services
"""

from typing import Dict, Any
from fastapi import HTTPException
import os
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from spark.core.spark_manager import get_spark_session, PRODUCTION_CONFIGS
from spark.core.database_manager import get_database_connection

from backend.app.services.model_loader import ModelLoader
from backend.app.services.feature_processor import FeatureProcessor
from backend.app.services.predictor import Predictor
from backend.app.services.model_evaluator import ModelEvaluator


class MLService:
    def __init__(self):
        """Initialize with production configs"""
        self.db = get_database_connection()
        self.spark = None

        self._init_spark()

        # Initialize services
        self.model_loader = ModelLoader(self.spark)
        self.feature_processor = FeatureProcessor(self.spark)
        self.predictor = Predictor(self.model_loader, self.feature_processor)
        self.evaluator = ModelEvaluator(self.model_loader, self.db)
        
        # Set log level to reduce noise
        self.spark.sparkContext.setLogLevel("ERROR")

    def _init_spark(self):
        try:
            self.spark = get_spark_session("YouTubeMLService", PRODUCTION_CONFIGS["ml_inference"])
        except Exception as e:
            raise HTTPException(status_code=503, detail="Spark initialization failed")

    # Delegate methods to services
    def predict_trending(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.predictor.predict_trending(video_data)

    def predict_views(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.predictor.predict_views(video_data)

    def predict_cluster(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.predictor.predict_cluster(video_data)

    def train_models(self) -> bool:
        return self.evaluator.train_models()

    def get_model_info(self) -> Dict[str, Any]:
        return self.evaluator.get_model_info()

    # Legacy compatibility methods
    @property
    def is_trained(self):
        return self.model_loader.is_trained

    @property
    def models(self):
        return self.model_loader.models

    @property
    def metrics(self):
        return self.model_loader.metrics

    @property
    def cluster_names(self):
        return self.model_loader.cluster_names

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

# Backward compatibility alias
initialize_ml_service = get_ml_service