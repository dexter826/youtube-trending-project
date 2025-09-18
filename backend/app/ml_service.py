"""
Machine Learning Service
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import requests
from fastapi import HTTPException

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.services.feature_processor import FeatureProcessor
from backend.app.services.model_evaluator import ModelEvaluator
from backend.app.services.model_loader import ModelLoader
from backend.app.services.predictor import Predictor
from spark.core.database_manager import get_database_connection
from spark.core.spark_manager import PRODUCTION_CONFIGS, get_spark_session


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
    def predict_cluster(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.predictor.predict_cluster(video_data)

    def predict_days(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.predictor.predict_days(video_data)

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

    # Predict from YouTube URL via YouTube Data API v3
    def predict_from_url(self, url: str) -> Dict[str, Any]:
        """Fetch video metadata from YouTube and run predictions."""
        try:
            # Strictly use API key from environment
            api_key = os.getenv("YOUTUBE_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY is not configured on the server")

            video_id = self._extract_video_id(url)
            if not video_id:
                raise HTTPException(status_code=400, detail="Invalid YouTube URL")

            # Fetch video details
            api_url = (
                "https://www.googleapis.com/youtube/v3/videos"
                f"?part=snippet,statistics,contentDetails&id={video_id}&key={api_key}"
            )
            resp = requests.get(api_url, timeout=15)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail="YouTube API request failed")
            data = resp.json()
            items = data.get("items", [])
            if not items:
                raise HTTPException(status_code=404, detail="Video not found or not accessible")

            item = items[0]
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})

            title = snippet.get("title", "")
            tags_list = snippet.get("tags", []) or []
            tags = "|".join(tags_list)
            category_id = int(snippet.get("categoryId", 0) or 0)
            published_at = snippet.get("publishedAt")
            description = snippet.get("description", "")
            channel_title = snippet.get("channelTitle", "")
            content_details = item.get("contentDetails", {})
            duration = content_details.get("duration", "")

            # Stats
            views = int(statistics.get("viewCount", 0)) if statistics.get("viewCount") else 0
            likes = int(statistics.get("likeCount", 0)) if statistics.get("likeCount") else 0  # May be hidden; default 0
            comment_count = int(statistics.get("commentCount", 0)) if statistics.get("commentCount") else 0

            # Time features
            publish_hour = 12
            video_age_proxy = 2
            if published_at:
                try:
                    dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    publish_hour = dt.hour
                    days_old = (datetime.now(timezone.utc) - dt).days
                    if days_old <= 1:
                        video_age_proxy = 1
                    elif days_old <= 7:
                        video_age_proxy = 2
                    elif days_old <= 30:
                        video_age_proxy = 3
                    else:
                        video_age_proxy = 4
                except Exception:
                    pass

            # Build unified feature dict expected by FeatureProcessor
            video_data = {
                "title": title,
                "views": views,
                "likes": likes,
                "comment_count": comment_count,
                "category_id": category_id,
                "tags": tags,
                "publish_hour": publish_hour,
                "video_age_proxy": video_age_proxy,
                "description": description,
                "channel_title": channel_title,
                "duration": duration,
                "description_length": len(description),
                "has_tags": 1 if tags_list else 0,
            }

            days_pred = self.predictor.predict_days(video_data)
            cluster_pred = self.predictor.predict_cluster(video_data)

            return {
                "input_video": {
                    "id": video_id,
                    "title": title,
                    "category_id": category_id,
                    "publish_hour": publish_hour,
                    "video_age_proxy": video_age_proxy,
                    "views": views,
                    "likes": likes,
                    "comment_count": comment_count,
                    "description": description,
                    "channel_title": channel_title,
                    "duration": duration,
                    "description_length": len(description),
                    "has_tags": bool(tags_list),
                },
                "prediction": {
                    "days": days_pred,
                    "cluster": cluster_pred,
                },
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction from URL failed: {str(e)}")

    def _extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from various URL formats."""
        try:
            parsed = urlparse(url)
            if parsed.netloc in ("youtu.be",):
                # Short URL: youtu.be/<id>
                vid = parsed.path.lstrip("/")
                return vid
            if parsed.netloc.endswith("youtube.com"):
                qs = parse_qs(parsed.query)
                vid = qs.get("v", [None])[0]
                if not vid and parsed.path.startswith("/embed/"):
                    vid = parsed.path.split("/embed/")[-1]
                return vid
            return None
        except Exception:
            return None


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