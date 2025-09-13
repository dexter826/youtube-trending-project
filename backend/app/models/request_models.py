"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class VideoMLInput(BaseModel):
    title: str
    views: int = 0
    likes: int = 0
    dislikes: int = 0
    comment_count: int = 0
    category_id: int = 0
    tags: str = ""

class HealthResponse(BaseModel):
    status: str
    database: str
    ml_service: Optional[str] = None
    timestamp: str

class PredictionResponse(BaseModel):
    prediction: Any
    input_data: Dict[str, Any]
    timestamp: str

class TrainingResponse(BaseModel):
    status: str
    message: str
    training_data_count: int
    models: list

class DatabaseStatsResponse(BaseModel):
    collections: Dict[str, int]