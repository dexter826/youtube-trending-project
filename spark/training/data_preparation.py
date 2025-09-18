"""
Data Preparation Module for ML Training (Clustering + Days Regression)
"""

import pandas as pd
from pyspark.sql.functions import col, lit, when, log, count as F_count, max as F_max


class DataPreparation:
    def __init__(self, db_connection):
        self.db = db_connection

    def load_training_data_from_mongodb(self):
        """Load training data from MongoDB ml_features collection"""
        raw_data = list(self.db.ml_features.find({}))

        if not raw_data:
            return None

        cleaned_data = []
        for record in raw_data:
            if '_id' in record:
                del record['_id']

            cleaned_record = {}
            for key, value in record.items():
                if value is None:
                    if key in ['views', 'likes', 'dislikes', 'comment_count', 'category_id']:
                        cleaned_record[key] = 0
                    else:
                        cleaned_record[key] = ""
                else:
                    cleaned_record[key] = value

            cleaned_data.append(cleaned_record)

        pandas_df = pd.DataFrame(cleaned_data)
        return pandas_df

    # Removed legacy trending classification preparation

    def prepare_features_for_clustering(self, df):
        """Prepare features for clustering model"""
        # Ensure required columns exist
        if "publish_hour" not in df.columns:
            df = df.withColumn("publish_hour", lit(12))
        if "video_age_proxy" not in df.columns:
            df = df.withColumn("video_age_proxy",
                              when(col("engagement_score") > 0.1, 1).otherwise(2))

        # Add log transformations
        df = df.withColumn("log_views", log(col("views") + 1)) \
               .withColumn("log_likes", log(col("likes") + 1)) \
               .withColumn("log_dislikes", log(col("dislikes") + 1)) \
               .withColumn("log_comments", log(col("comment_count") + 1))

        cluster_features = [
            "log_views", "log_likes", "log_dislikes", "log_comments",
            "like_ratio", "engagement_score", "title_length", "tag_count", "category_id",
            "publish_hour", "video_age_proxy"
        ]

        data = df.select(cluster_features).na.fill(0)
        return data, cluster_features

    # Removed legacy views regression preparation

    def prepare_features_for_days_regression(self, df):
        """Prepare features for days-in-trending regression.
        Aggregates per video_id and computes label days_in_trending.
        """
        # Ensure required columns exist
        if "publish_hour" not in df.columns:
            df = df.withColumn("publish_hour", lit(12))
        if "video_age_proxy" not in df.columns:
            df = df.withColumn("video_age_proxy",
                              when(col("engagement_score") > 0.1, 1).otherwise(2))

        # Aggregate per video to get label and representative features
        # Note: 'title' and 'tags' are not present in ml_features; aggregate
        # existing numeric proxies instead: title_length, tag_count
        agg = df.groupBy("video_id").agg(
            F_max("category_id").alias("category_id"),
            F_max("views").alias("views"),
            F_max("likes").alias("likes"),
            F_max("dislikes").alias("dislikes"),
            F_max("comment_count").alias("comment_count"),
            F_max("publish_hour").alias("publish_hour"),
            F_max("video_age_proxy").alias("video_age_proxy"),
            F_max("title_length").alias("title_length"),
            F_max("tag_count").alias("tag_count"),
            F_count(lit(1)).alias("days_in_trending"),
        )

        # Derived features consistent with inference
        agg = agg.withColumn("like_ratio", (col("likes") / (col("views") + lit(1e-9)))) \
                 .withColumn("engagement_score", (col("likes") + col("comment_count")) / (col("views") + lit(1e-9)))

        feature_cols = [
            "views", "likes", "dislikes", "comment_count", "like_ratio",
            "engagement_score", "title_length", "tag_count", "category_id",
            "publish_hour", "video_age_proxy"
        ]

        data = agg.select(feature_cols + ["days_in_trending"]).na.fill(0)
        return data, feature_cols