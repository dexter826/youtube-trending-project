"""
Machine Learning Routes for YouTube Analytics API
"""

from fastapi import HTTPException
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel

# Global variables (will be imported from main)
db = None
get_ml_service = None

def set_database(database):
    """Set database connection"""
    global db
    db = database

def set_ml_service_getter(getter_func):
    """Set ML service getter function"""
    global get_ml_service
    get_ml_service = getter_func

class VideoMLInput(BaseModel):
    title: str
    views: int = 0
    likes: int = 0
    dislikes: int = 0
    comment_count: int = 0
    category_id: int = 0
    tags: str = ""

async def ml_health():
    """ML service health check"""
    try:
        ml_service = get_ml_service()
        return ml_service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML health check failed: {str(e)}")

async def train_ml_models():
    """Train ML models using database data"""
    try:
        ml_service = get_ml_service()
        training_data_count = db.ml_features.count_documents({})

        if training_data_count < 1000:
            raise HTTPException(status_code=400, detail=f"Insufficient training data: {training_data_count} records")

        success = ml_service.train_models()

        if success:
            return {
                "status": "success",
                "message": "ML models trained successfully",
                "training_data_count": training_data_count,
                "models": list(ml_service.models.keys())
            }
        else:
            raise HTTPException(status_code=500, detail="Training failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

async def predict_trending(video_data: VideoMLInput):
    """Predict if a video will be trending"""
    try:
        ml_service = get_ml_service()
        video_dict = video_data.dict()
        result = ml_service.predict_trending(video_dict)

        return {
            "prediction": result,
            "input_data": video_dict,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

async def predict_views(video_data: VideoMLInput):
    """Predict view count for a video"""
    try:
        ml_service = get_ml_service()
        video_dict = video_data.dict()
        result = ml_service.predict_views(video_dict)

        return {
            "prediction": result,
            "input_data": video_dict,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Views prediction failed: {str(e)}")

async def predict_cluster(video_data: VideoMLInput):
    """Predict content cluster for a video"""
    try:
        ml_service = get_ml_service()
        video_dict = video_data.dict()
        result = ml_service.predict_cluster(video_dict)

        return {
            "prediction": result,
            "input_data": video_dict,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")