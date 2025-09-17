"""
Machine Learning Routes for YouTube Analytics API
"""

from fastapi import HTTPException, APIRouter
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

db = None
get_ml_service = None

class VideoMLInput(BaseModel):
    title: str
    views: int = 0
    likes: int = 0
    dislikes: int = 0
    comment_count: int = 0
    category_id: int = 0
    tags: str = ""
    publish_hour: int = 12
    video_age_proxy: int = 2

@router.get("/health")
async def health_check():
    """Health check endpoint for ML service"""
    try:
        if router.db is None:
            raise HTTPException(status_code=500, detail="Database connection not initialized")
        if router.get_ml_service is None:
            raise HTTPException(status_code=500, detail="ML service not initialized")
        
        router.db.client.admin.command('ping')
        
        ml_service = router.get_ml_service()
        if ml_service is None:
            raise HTTPException(status_code=500, detail="ML service not available")
        
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
        if router.get_ml_service is None:
            raise HTTPException(status_code=500, detail="ML service not initialized")
            
        ml_service = router.get_ml_service()
        if ml_service is None:
            raise HTTPException(status_code=500, detail="ML service not available")
        
        return await ml_service.train_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to train ML models: {str(e)}")

@router.post("/predict/trending")
async def predict_trending(video_data: VideoMLInput):
    """Predict if a video will be trending"""
    try:
        ml_service = router.get_ml_service()
        if not ml_service:
            raise HTTPException(status_code=500, detail="ML service not initialized")

        if not ml_service.is_trained:
            raise HTTPException(
                status_code=503,
                detail="Models not trained. Please train models first."
            )

        video_dict = video_data.dict()
        result = ml_service.predict_trending(video_dict)

        return {
            "prediction": result,
            "input_data": video_dict,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/views")
async def predict_views(video_data: VideoMLInput):
    """Predict view count for a video"""
    try:
        ml_service = router.get_ml_service()
        if not ml_service:
            raise HTTPException(status_code=500, detail="ML service not initialized")

        video_dict = video_data.dict()
        result = ml_service.predict_views(video_dict)

        return {
            "prediction": result,
            "input_data": video_dict,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Views prediction failed: {str(e)}")

@router.post("/predict/cluster")
async def predict_cluster(video_data: VideoMLInput):
    """Predict content cluster for a video"""
    try:
        ml_service = router.get_ml_service()
        if not ml_service:
            raise HTTPException(status_code=500, detail="ML service not initialized")

        video_dict = video_data.dict()
        result = ml_service.predict_cluster(video_dict)

        return {
            "prediction": result,
            "input_data": video_dict,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")