"""
Machine Learning Routes for YouTube Analytics API
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.models.request_models import VideoMLInput

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

db = None
get_ml_service = None

class UrlInput(BaseModel):
    url: str

def _get_ml_service():
    """Helper to get and validate ML service"""
    if router.get_ml_service is None:
        raise HTTPException(status_code=500, detail="ML service not initialized")
    ml_service = router.get_ml_service()
    if not ml_service:
        raise HTTPException(status_code=500, detail="ML service not available")
    return ml_service

def _format_prediction_response(result: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to format prediction response"""
    return {
        "prediction": result,
        "input_data": input_data,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/health")
async def health_check():
    """Health check endpoint for ML service"""
    try:
        if router.db is None:
            raise HTTPException(status_code=500, detail="Database connection not initialized")
        if router.get_ml_service is None:
            raise HTTPException(status_code=500, detail="ML service not initialized")

        router.db.client.admin.command('ping')

        ml_service = _get_ml_service()

        # Get comprehensive model info from evaluator
        return ml_service.get_model_info()

    except Exception as e:
        # Return error response but still include basic structure
        return {
            "loaded_models": [],
            "is_trained": False,
            "model_type": "spark_mllib",
            "framework": "Apache Spark MLlib",
            "storage": "HDFS",
            "model_details": {},
            "total_models": 0,
            "spark_session": False,
            "metrics": {},
            "error": str(e)
        }

@router.post("/train")
async def train_ml_models():
    """Train ML models"""
    try:
        ml_service = _get_ml_service()
        return await ml_service.train_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train ML models: {str(e)}")

@router.post("/predict/days")
async def predict_days(video_data: VideoMLInput):
    """Predict number of days a video may stay on trending"""
    try:
        ml_service = _get_ml_service()
        video_dict = video_data.dict()
        result = ml_service.predict_days(video_dict)
        return _format_prediction_response(result, video_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Days prediction failed: {str(e)}")

@router.post("/predict/cluster")
async def predict_cluster(video_data: VideoMLInput):
    """Predict content cluster for a video"""
    try:
        ml_service = _get_ml_service()
        video_dict = video_data.dict()
        result = ml_service.predict_cluster(video_dict)
        return _format_prediction_response(result, video_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

@router.post("/predict/url")
async def predict_from_url(payload: UrlInput):
    """Predict using a YouTube URL."""
    try:
        ml_service = _get_ml_service()
        result = ml_service.predict_from_url(payload.url)
        return {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction from URL failed: {str(e)}")