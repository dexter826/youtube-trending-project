"""
Feature Processing Service
"""

from typing import Dict, Any
from fastapi import HTTPException
from pyspark.sql.types import StructType, StructField, DoubleType
import math
import re


class FeatureProcessor:
    def __init__(self, spark_session):
        self.spark = spark_session

    def create_trending_dataframe(self, video_data: Dict[str, Any]):
        """Create DataFrame for trending classification model"""
        try:
            views = max(float(video_data.get("views", 1)), 1)
            likes = float(video_data.get("likes", 0))
            dislikes = float(video_data.get("dislikes", 0))
            comment_count = float(video_data.get("comment_count", 0))
            title = video_data.get("title", "")

            has_caps = 1.0 if re.search(r'[A-Z]{3,}', title) else 0.0
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

    def create_regression_dataframe(self, video_data: Dict[str, Any]):
        """Create DataFrame for views regression model"""
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

    def create_clustering_dataframe(self, video_data: Dict[str, Any]):
        """Create DataFrame for content clustering model"""
        try:
            views = max(float(video_data.get("views", 1)), 1)
            likes = float(video_data.get("likes", 0))
            dislikes = float(video_data.get("dislikes", 0))
            comment_count = float(video_data.get("comment_count", 0))
            title = video_data.get("title", "")

            # New time-based features
            publish_hour = float(video_data.get("publish_hour", 12))
            video_age_proxy = float(video_data.get("video_age_proxy", 2))

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

            return self.spark.createDataFrame([tuple(features.values())], schema)

        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid input data format")