"""
YouTube Trending ML Training
"""

import os
import sys
from datetime import datetime
import pandas as pd

from pyspark.sql.functions import *
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator, ClusteringEvaluator

from core.spark_manager import get_spark_session, PRODUCTION_CONFIGS
from core.database_manager import get_database_connection


class YouTubeMLTrainer:
    def __init__(self):
        """Initialize ML trainer"""
        self.spark = get_spark_session("YouTubeMLTrainer", PRODUCTION_CONFIGS["ml_training"])
        self.db = get_database_connection()
        # Get the MongoDB client from the manager for proper cleanup
        from core.database_manager import _manager
        self.mongo_client = _manager._client
        self.hdfs_base_path = "hdfs://localhost:9000/youtube_trending"
        self.models_path = f"{self.hdfs_base_path}/models"

    def load_training_data(self):
        """Load training data from MongoDB"""
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
        df = self.spark.createDataFrame(pandas_df)

        return df

    def train_trending_prediction_model(self, df):
        """Train trending prediction model"""
        feature_cols = [
            "like_ratio", "dislike_ratio", "comment_ratio", "engagement_score",
            "title_length", "has_caps", "tag_count", "category_id"
        ]

        data = df.select(feature_cols + ["is_trending"]).na.fill(0)

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
        rf_classifier = RandomForestClassifier(
            featuresCol="scaledFeatures",
            labelCol="is_trending",
            predictionCol="prediction",
            numTrees=50,
            maxDepth=10,
            seed=42
        )

        pipeline = Pipeline(stages=[assembler, scaler, rf_classifier])

        train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
        model = pipeline.fit(train_data)

        predictions = model.transform(test_data)
        evaluator = BinaryClassificationEvaluator(labelCol="is_trending")
        auc = evaluator.evaluate(predictions)

        model_path = f"{self.models_path}/trending_prediction"
        model.write().overwrite().save(model_path)

        return model

    def train_clustering_model(self, df):
        """Train clustering model"""
        category_counts = df.groupBy("category_id").count().orderBy("count", ascending=False)
        category_counts.show(10)
        
        feature_cols = [
            "views", "likes", "dislikes", "comment_count",
            "like_ratio", "engagement_score", "title_length", "tag_count", "category_id"
        ]

        data = df.select(feature_cols).na.fill(0)

        data = data.withColumn("log_views", log(col("views") + 1)) \
                  .withColumn("log_likes", log(col("likes") + 1)) \
                  .withColumn("log_dislikes", log(col("dislikes") + 1)) \
                  .withColumn("log_comments", log(col("comment_count") + 1))

        cluster_features = [
            "log_views", "log_likes", "log_dislikes", "log_comments",
            "like_ratio", "engagement_score", "title_length", "tag_count", "category_id"
        ]

        assembler = VectorAssembler(inputCols=cluster_features, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
        kmeans = KMeans(featuresCol="scaledFeatures", predictionCol="cluster", k=3, seed=42, maxIter=200)

        pipeline = Pipeline(stages=[assembler, scaler, kmeans])
        model = pipeline.fit(data)

        # Evaluate clustering
        predictions = model.transform(data)
        evaluator = ClusteringEvaluator(predictionCol="cluster", featuresCol="scaledFeatures")
        silhouette = evaluator.evaluate(predictions)
        print(f"Silhouette score: {silhouette}")

        model_path = f"{self.models_path}/clustering"
        model.write().overwrite().save(model_path)

        return model

    def train_regression_model(self, df):
        """Train regression model"""
        feature_cols = [
            "views", "likes", "dislikes", "comment_count", "like_ratio",
            "engagement_score", "title_length", "tag_count", "category_id"
        ]

        data = df.select(feature_cols + ["log_views"]).na.fill(0)

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
        rf_regressor = RandomForestRegressor(
            featuresCol="scaledFeatures",
            labelCol="log_views",
            predictionCol="prediction",
            numTrees=50,
            maxDepth=10,
            seed=42
        )

        pipeline = Pipeline(stages=[assembler, scaler, rf_regressor])

        train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
        model = pipeline.fit(train_data)

        model_path = f"{self.models_path}/regression"
        model.write().overwrite().save(model_path)

        return model

    def run_training_pipeline(self):
        """Run complete model training pipeline"""
        try:
            df = self.load_training_data()
            if df is None:
                return False

            df.cache()

            trending_model = self.train_trending_prediction_model(df)
            clustering_model = self.train_clustering_model(df)
            regression_model = self.train_regression_model(df)

            return all([trending_model, clustering_model, regression_model])

        except Exception as e:
            return False
        finally:
            self.spark.stop()
            self.mongo_client.close()

def main():
    """Main execution function"""
    trainer = YouTubeMLTrainer()
    success = trainer.run_training_pipeline()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()