"""
Feature Processing Service
"""

from typing import Dict, Any
from fastapi import HTTPException
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
import math


class FeatureProcessor:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.nlp_pipeline = None
        self._init_nlp_pipeline()

    def _init_nlp_pipeline(self):
        """Initialize NLP pipeline for text processing"""
        try:
            title_tokenizer = Tokenizer(inputCol="title", outputCol="title_words")
            title_remover = StopWordsRemover(inputCol="title_words", outputCol="title_filtered")
            title_hashingTF = HashingTF(inputCol="title_filtered", outputCol="title_raw_features", numFeatures=1000)
            title_idf = IDF(inputCol="title_raw_features", outputCol="title_tfidf")

            desc_tokenizer = Tokenizer(inputCol="description", outputCol="desc_words")
            desc_remover = StopWordsRemover(inputCol="desc_words", outputCol="desc_filtered")
            desc_hashingTF = HashingTF(inputCol="desc_filtered", outputCol="desc_raw_features", numFeatures=2000)
            desc_idf = IDF(inputCol="desc_raw_features", outputCol="desc_tfidf")

            self.nlp_pipeline = Pipeline(stages=[
                title_tokenizer, title_remover, title_hashingTF, title_idf,
                desc_tokenizer, desc_remover, desc_hashingTF, desc_idf
            ])
        except Exception:
            # If NLP fails, set to None
            self.nlp_pipeline = None

    def _add_nlp_features(self, df):
        """Add NLP features if pipeline is available - disabled for now"""
        # Temporarily disable NLP to match trained model
        df = df.withColumn("title_tfidf", [0.0] * df.count()) \
               .withColumn("desc_tfidf", [0.0] * df.count())
        return df

    def create_days_regression_dataframe(self, video_data: Dict[str, Any]):
        """Create DataFrame for days-in-trending regression model"""
        try:
            views = max(float(video_data.get("views", 1)), 1)
            likes = float(video_data.get("likes", 0))
            comment_count = float(video_data.get("comment_count", 0))
            title = video_data.get("title", "")
            description = video_data.get("description", "")

            # New time-based features
            publish_hour = float(video_data.get("publish_hour", 12))
            video_age_proxy = float(video_data.get("video_age_proxy", 2))

            features = {
                "views": views,
                "likes": likes,
                "comment_count": comment_count,
                "like_ratio": likes / views,
                "engagement_score": (likes + comment_count) / views,
                "title_length": float(len(title)),
                "tag_count": float(len(video_data.get("tags", "").split("|")) if video_data.get("tags") else 0),
                "category_id": float(video_data.get("category_id", 0)),
                "publish_hour": publish_hour,
                "video_age_proxy": video_age_proxy,
                "title": title,
                "description": description
            }

            schema = StructType([
                StructField("views", DoubleType(), True),
                StructField("likes", DoubleType(), True),
                StructField("comment_count", DoubleType(), True),
                StructField("like_ratio", DoubleType(), True),
                StructField("engagement_score", DoubleType(), True),
                StructField("title_length", DoubleType(), True),
                StructField("tag_count", DoubleType(), True),
                StructField("category_id", DoubleType(), True),
                StructField("publish_hour", DoubleType(), True),
                StructField("video_age_proxy", DoubleType(), True),
                StructField("title", StringType(), True),
                StructField("description", StringType(), True)
            ])

            df = self.spark.createDataFrame([tuple(features.values())], schema)
            # df = self._add_nlp_features(df)  # Disabled to match trained model

            return df

        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid input data format")

    def create_clustering_dataframe(self, video_data: Dict[str, Any]):
        """Create DataFrame for content clustering model"""
        try:
            views = max(float(video_data.get("views", 1)), 1)
            likes = float(video_data.get("likes", 0))
            comment_count = float(video_data.get("comment_count", 0))
            title = video_data.get("title", "")
            description = video_data.get("description", "")

            # New time-based features
            publish_hour = float(video_data.get("publish_hour", 12))
            video_age_proxy = float(video_data.get("video_age_proxy", 2))

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
                "tag_count": float(len(video_data.get("tags", "").split("|")) if video_data.get("tags") else 0),
                "category_id": float(video_data.get("category_id", 0)),
                "publish_hour": publish_hour,
                "video_age_proxy": video_age_proxy,
                "title": title,
                "description": description
            }

            schema = StructType([
                StructField("log_views", DoubleType(), True),
                StructField("log_likes", DoubleType(), True),
                StructField("log_comments", DoubleType(), True),
                StructField("like_ratio", DoubleType(), True),
                StructField("engagement_score", DoubleType(), True),
                StructField("title_length", DoubleType(), True),
                StructField("tag_count", DoubleType(), True),
                StructField("category_id", DoubleType(), True),
                StructField("publish_hour", DoubleType(), True),
                StructField("video_age_proxy", DoubleType(), True),
                StructField("title", StringType(), True),
                StructField("description", StringType(), True)
            ])

            df = self.spark.createDataFrame([tuple(features.values())], schema)
            # df = self._add_nlp_features(df)  # Disabled to match trained model

            return df

        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid input data format")