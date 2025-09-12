"""
YouTube Trending ML Model Training Pipeline
Author: BigData Expert  
Description: Train and save all ML models to HDFS
"""

import os
import sys
from datetime import datetime
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# ML imports
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator, RegressionEvaluator

import pymongo
from pymongo import MongoClient

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "youtube_trending"

class YouTubeMLTrainer:
    def __init__(self):
        """Initialize Spark session for ML training"""
        self.spark = SparkSession.builder \
            .appName("YouTubeMLTrainer") \
            .master("local[*]") \
            .config("spark.driver.memory", "6g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # MongoDB client
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[DB_NAME]
        
        # HDFS paths
        self.hdfs_base_path = "hdfs://localhost:9000/youtube_trending"
        self.models_path = f"{self.hdfs_base_path}/models"
        
        print("[OK] ML Trainer initialized with HDFS support")

    def load_training_data(self):
        """Load and prepare training data from MongoDB"""
        print("📁 Loading training data from MongoDB...")
        
        try:
            # Get data from MongoDB
            raw_data = list(self.db.raw_videos.find({}))
            
            if not raw_data:
                print("[ERROR] No training data found!")
                return None
            
            # Clean data
            cleaned_data = []
            for record in raw_data:
                if '_id' in record:
                    del record['_id']
                
                # Handle null values
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
            
            # Convert to Spark DataFrame
            pandas_df = pd.DataFrame(cleaned_data)
            df = self.spark.createDataFrame(pandas_df)
            
            print(f"[OK] Loaded {df.count()} training records")
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to load training data: {str(e)}")
            return None

    def engineer_features(self, df):
        """Engineer features for ML models"""
        print("🔧 Engineering features...")
        
        # 1. Engagement Features
        df = df.withColumn("like_ratio", 
                          when(col("views") > 0, col("likes") / col("views")).otherwise(0))
        
        df = df.withColumn("dislike_ratio", 
                          when(col("views") > 0, col("dislikes") / col("views")).otherwise(0))
        
        df = df.withColumn("comment_ratio", 
                          when(col("views") > 0, col("comment_count") / col("views")).otherwise(0))
        
        df = df.withColumn("engagement_score", 
                          when(col("views") > 0, 
                               (col("likes") + col("comment_count")) / col("views")).otherwise(0))
        
        # 2. Content Features
        df = df.withColumn("title_length", length(col("title")))
        
        df = df.withColumn("has_caps", 
                          when(col("title").rlike("[A-Z]{3,}"), 1).otherwise(0))
        
        df = df.withColumn("tag_count", 
                          when(col("tags").isNotNull(), 
                               size(split(col("tags"), "\\|"))).otherwise(0))
        
        # 3. Trending Target Variable
        window_spec = Window.partitionBy("country", "trending_date_parsed").orderBy(col("views").desc())
        df = df.withColumn("view_rank", row_number().over(window_spec))
        
        total_videos = df.groupBy("country", "trending_date_parsed").agg(
            count("*").alias("total_videos_per_day")
        )
        
        df = df.join(total_videos, ["country", "trending_date_parsed"], "left")
        
        df = df.withColumn("trending_threshold", 
                          greatest(lit(1), least(lit(20), (col("total_videos_per_day") * 0.2).cast("int"))))
        
        df = df.withColumn("is_trending", 
                          when(col("view_rank") <= col("trending_threshold"), 1).otherwise(0))
        
        # 4. Views prediction target (log scale)
        df = df.withColumn("log_views", log(col("views") + 1))
        
        print("[OK] Features engineered")
        return df

    def train_trending_prediction_model(self, df):
        """Train trending prediction classification model"""
        print("🎯 Training Trending Prediction Model...")
        
        try:
            # Select features for trending prediction
            feature_cols = [
                "like_ratio", "dislike_ratio", "comment_ratio", "engagement_score",
                "title_length", "has_caps", "tag_count", "category_id"
            ]
            
            # Prepare data
            data = df.select(feature_cols + ["is_trending"]).na.fill(0)
            
            # Assemble features
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            
            # Scale features
            scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
            
            # Classification model
            rf_classifier = RandomForestClassifier(
                featuresCol="scaledFeatures",
                labelCol="is_trending",
                predictionCol="prediction",
                numTrees=50,
                maxDepth=10,
                seed=42
            )
            
            # Create pipeline
            pipeline = Pipeline(stages=[assembler, scaler, rf_classifier])
            
            # Split data
            train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
            
            # Train model
            model = pipeline.fit(train_data)
            
            # Evaluate
            predictions = model.transform(test_data)
            evaluator = BinaryClassificationEvaluator(labelCol="is_trending")
            auc = evaluator.evaluate(predictions)
            
            print(f"[TRENDING MODEL] AUC: {auc:.4f}")
            
            # Save model to HDFS
            model_path = f"{self.models_path}/trending_prediction"
            model.write().overwrite().save(model_path)
            
            print(f"[OK] Trending prediction model saved to HDFS: {model_path}")
            return model
            
        except Exception as e:
            print(f"[ERROR] Failed to train trending prediction model: {str(e)}")
            return None

    def train_clustering_model(self, df):
        """Train clustering model for video categorization"""
        print("🔗 Training Clustering Model...")
        
        try:
            # Select features for clustering
            feature_cols = [
                "views", "likes", "dislikes", "comment_count",
                "like_ratio", "engagement_score", "title_length", "tag_count"
            ]
            
            # Prepare data
            data = df.select(feature_cols).na.fill(0)
            
            # Log transform for views, likes, etc.
            data = data.withColumn("log_views", log(col("views") + 1)) \
                      .withColumn("log_likes", log(col("likes") + 1)) \
                      .withColumn("log_comments", log(col("comment_count") + 1))
            
            cluster_features = [
                "log_views", "log_likes", "log_comments",
                "like_ratio", "engagement_score", "title_length", "tag_count"
            ]
            
            # Assemble features
            assembler = VectorAssembler(inputCols=cluster_features, outputCol="features")
            
            # Scale features
            scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
            
            # K-means clustering
            kmeans = KMeans(
                featuresCol="scaledFeatures",
                predictionCol="cluster",
                k=5,
                seed=42,
                maxIter=100
            )
            
            # Create pipeline
            pipeline = Pipeline(stages=[assembler, scaler, kmeans])
            
            # Train model
            model = pipeline.fit(data)
            
            # Evaluate
            predictions = model.transform(data)
            evaluator = ClusteringEvaluator(featuresCol="scaledFeatures", predictionCol="cluster")
            silhouette = evaluator.evaluate(predictions)
            
            print(f"[CLUSTERING MODEL] Silhouette Score: {silhouette:.4f}")
            
            # Save model to HDFS
            model_path = f"{self.models_path}/clustering"
            model.write().overwrite().save(model_path)
            
            print(f"[OK] Clustering model saved to HDFS: {model_path}")
            return model
            
        except Exception as e:
            print(f"[ERROR] Failed to train clustering model: {str(e)}")
            return None

    def train_regression_model(self, df):
        """Train regression model for views prediction"""
        print("📈 Training Regression Model...")
        
        try:
            # Select features for regression
            feature_cols = [
                "likes", "dislikes", "comment_count", "like_ratio", 
                "engagement_score", "title_length", "tag_count", "category_id"
            ]
            
            # Prepare data (predict log_views to handle large numbers)
            data = df.select(feature_cols + ["log_views"]).na.fill(0)
            
            # Assemble features
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            
            # Scale features
            scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
            
            # Random Forest Regressor
            rf_regressor = RandomForestRegressor(
                featuresCol="scaledFeatures",
                labelCol="log_views",
                predictionCol="prediction",
                numTrees=50,
                maxDepth=10,
                seed=42
            )
            
            # Create pipeline
            pipeline = Pipeline(stages=[assembler, scaler, rf_regressor])
            
            # Split data
            train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
            
            # Train model
            model = pipeline.fit(train_data)
            
            # Evaluate
            predictions = model.transform(test_data)
            evaluator = RegressionEvaluator(labelCol="log_views", predictionCol="prediction", metricName="rmse")
            rmse = evaluator.evaluate(predictions)
            
            print(f"[REGRESSION MODEL] RMSE: {rmse:.4f}")
            
            # Save model to HDFS
            model_path = f"{self.models_path}/regression"
            model.write().overwrite().save(model_path)
            
            print(f"[OK] Regression model saved to HDFS: {model_path}")
            return model
            
        except Exception as e:
            print(f"[ERROR] Failed to train regression model: {str(e)}")
            return None

    def run_training_pipeline(self):
        """Run complete model training pipeline"""
        try:
            print("🚀 Starting ML Model Training Pipeline")
            print("=" * 60)
            
            # Step 1: Load training data
            df = self.load_training_data()
            if df is None:
                return False
            
            # Step 2: Engineer features
            df = self.engineer_features(df)
            
            # Cache for multiple model training
            df.cache()
            
            # Step 3: Train models
            trending_model = self.train_trending_prediction_model(df)
            clustering_model = self.train_clustering_model(df)
            regression_model = self.train_regression_model(df)
            
            print("=" * 60)
            print("[SUCCESS] All ML models trained and saved to HDFS!")
            print(f"[MODELS PATH] {self.models_path}")
            
            return all([trending_model, clustering_model, regression_model])
            
        except Exception as e:
            print(f"[ERROR] Training pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
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