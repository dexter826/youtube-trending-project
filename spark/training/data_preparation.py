"""
Data Preparation Module for ML Training (Clustering + Days Regression)
"""

import pandas as pd
from pyspark.sql.functions import col, lit, when, log, count as F_count, max as F_max
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml import Pipeline


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

    def prepare_nlp_features(self, df):
        """Prepare NLP features using Spark MLlib for title and description"""
        # Ensure text columns exist, default to empty string if missing
        if "title" not in df.columns:
            df = df.withColumn("title", lit(""))
        if "description" not in df.columns:
            df = df.withColumn("description", lit(""))

        # Check if text columns have data
        title_count = df.filter(col("title") != "").count()
        desc_count = df.filter(col("description") != "").count()

        if title_count == 0 and desc_count == 0:
            # No text data, skip NLP
            df = df.withColumn("title_tfidf", lit(0.0)) \
                   .withColumn("desc_tfidf", lit(0.0))
            return df

        # NLP Pipeline for title (only if has data)
        if title_count > 0:
            title_tokenizer = Tokenizer(inputCol="title", outputCol="title_words")
            title_remover = StopWordsRemover(inputCol="title_words", outputCol="title_filtered")
            title_hashingTF = HashingTF(inputCol="title_filtered", outputCol="title_raw_features", numFeatures=1000)
            title_idf = IDF(inputCol="title_raw_features", outputCol="title_tfidf")

            title_pipeline = Pipeline(stages=[title_tokenizer, title_remover, title_hashingTF, title_idf])
            title_model = title_pipeline.fit(df)
            df = title_model.transform(df)
        else:
            df = df.withColumn("title_tfidf", lit(0.0))

        # NLP Pipeline for description (only if has data)
        if desc_count > 0:
            desc_tokenizer = Tokenizer(inputCol="description", outputCol="desc_words")
            desc_remover = StopWordsRemover(inputCol="desc_words", outputCol="desc_filtered")
            desc_hashingTF = HashingTF(inputCol="desc_filtered", outputCol="desc_raw_features", numFeatures=2000)
            desc_idf = IDF(inputCol="desc_raw_features", outputCol="desc_tfidf")

            desc_pipeline = Pipeline(stages=[desc_tokenizer, desc_remover, desc_hashingTF, desc_idf])
            desc_model = desc_pipeline.fit(df)
            df = desc_model.transform(df)
        else:
            df = df.withColumn("desc_tfidf", lit(0.0))

        return df

    # Removed legacy trending classification preparation

    def prepare_features_for_clustering(self, df):
        """Prepare features for clustering model"""
        # Add NLP features first
        df = self.prepare_nlp_features(df)

        # Ensure required columns exist
        if "publish_hour" not in df.columns:
            df = df.withColumn("publish_hour", lit(12))
        if "video_age_proxy" not in df.columns:
            df = df.withColumn("video_age_proxy",
                              when(col("engagement_score") > 0.1, 1).otherwise(2))

        # Add log transformations
        df = df.withColumn("log_views", log(col("views") + 1)) \
               .withColumn("log_likes", log(col("likes") + 1)) \
               .withColumn("log_comments", log(col("comment_count") + 1))

        # Base features
        cluster_features = [
            "log_views", "log_likes", "log_comments",
            "like_ratio", "engagement_score", "title_length", "tag_count", "category_id",
            "publish_hour", "video_age_proxy"
        ]

        # Add NLP features only if they exist and have data
        if "title_tfidf" in df.columns and df.filter(col("title_tfidf").isNotNull()).count() > 0:
            cluster_features.append("title_tfidf")
        if "desc_tfidf" in df.columns and df.filter(col("desc_tfidf").isNotNull()).count() > 0:
            cluster_features.append("desc_tfidf")

        # Return df with features, let model_training handle assembling
        data = df.select(cluster_features).na.fill(0)
        return data, cluster_features

    # Removed legacy views regression preparation

    def prepare_features_for_days_regression(self, df):
        """Prepare features for days-in-trending regression.
        Aggregates per video_id and computes label days_in_trending.
        """
        # Add NLP features first
        df = self.prepare_nlp_features(df)

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
            F_max("comment_count").alias("comment_count"),
            F_max("publish_hour").alias("publish_hour"),
            F_max("video_age_proxy").alias("video_age_proxy"),
            F_max("title_length").alias("title_length"),
            F_max("tag_count").alias("tag_count"),
            F_max("title_tfidf").alias("title_tfidf"),
            F_max("desc_tfidf").alias("desc_tfidf"),
            F_count(lit(1)).alias("days_in_trending"),
        )

        # Derived features consistent with inference
        agg = agg.withColumn("like_ratio", (col("likes") / (col("views") + lit(1e-9)))) \
                 .withColumn("engagement_score", (col("likes") + col("comment_count")) / (col("views") + lit(1e-9)))

        # Base features
        feature_cols = [
            "views", "likes", "comment_count", "like_ratio",
            "engagement_score", "title_length", "tag_count", "category_id",
            "publish_hour", "video_age_proxy"
        ]

        # Add NLP features only if they exist and have data
        if "title_tfidf" in agg.columns and agg.filter(col("title_tfidf").isNotNull()).count() > 0:
            feature_cols.append("title_tfidf")
        if "desc_tfidf" in agg.columns and agg.filter(col("desc_tfidf").isNotNull()).count() > 0:
            feature_cols.append("desc_tfidf")

        # Return df with features and label, let model_training handle assembling
        data = agg.select(feature_cols + ["days_in_trending"]).na.fill(0)
        return data, feature_cols