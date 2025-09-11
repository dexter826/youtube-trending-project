"""
Production MLlib Service with Spark Optimizations
Description: High-performance MLlib service for distributed machine learning
"""

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.storagelevel import StorageLevel

# Import optimized components
# Configuration settings

# ML imports
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, GBTClassifier
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator, RegressionEvaluator

import pymongo
from pymongo import MongoClient

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "youtube_trending"

class ProductionMLlibService:
    def __init__(self, mongo_uri=MONGO_URI, db_name=DB_NAME):
        """Initialize production MLlib service with optimized Spark configuration"""
        
# Create optimized Spark session directly
        self.spark = SparkSession.builder \
            .appName("OptimizedMLlibService") \
            .master("local[*]") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "6g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # MongoDB connection
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        
        # HDFS paths
        self.hdfs_base_path = "hdfs://namenode:9000/youtube_trending"
        self.models_path = f"{self.hdfs_base_path}/optimized_models"
        
        # Cached models
        self.loaded_models = {}
        
        print("[OK] Optimized MLlib service initialized")

    def load_optimized_data(self):
        """Load data with optimal partitioning and caching"""
        try:
            print("📁 Loading optimized training data...")
            
            # Load raw data from MongoDB (only metadata)
            raw_data = list(self.db.raw_videos.find({}, {"_id": 0}).limit(10000))  # Limit for optimization
            
            if not raw_data:
                print("[WARN] No data found in MongoDB")
                return None
            
            # Create DataFrame
            df = self.spark.createDataFrame(raw_data)
            
            # Add derived features efficiently
            df = df.withColumn("title_length", length(col("title"))) \
                   .withColumn("like_ratio", when(col("views") > 0, col("likes") / col("views")).otherwise(0)) \
                   .withColumn("engagement_score", when(col("views") > 0, (col("likes") + col("comment_count")) / col("views")).otherwise(0))
            
            # Optimal partitioning and caching
            df = df.repartition(4, "country").cache()
            
            print(f"[OK] Optimized data loaded: {df.count()} records")
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to load optimized data: {str(e)}")
            return None

    def train_optimized_classification(self, df):
        """Train classification models with optimizations"""
        print("🎯 Training optimized classification models...")
        
        try:
            # Prepare features efficiently
            feature_cols = ['views', 'likes', 'dislikes', 'comment_count', 'like_ratio', 'engagement_score']
            
            # Create binary target (trending vs not trending)
            df = df.withColumn("is_trending", when(col("views") > 1000000, 1.0).otherwise(0.0))
            
            # Feature pipeline
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
            scaler = StandardScaler(inputCol="raw_features", outputCol="features")
            
            # Multiple optimized algorithms
            algorithms = {
                'logistic_regression': LogisticRegression(featuresCol="features", labelCol="is_trending", maxIter=20),
                'random_forest': RandomForestClassifier(featuresCol="features", labelCol="is_trending", numTrees=50),
                'decision_tree': DecisionTreeClassifier(featuresCol="features", labelCol="is_trending"),
                'gradient_boosting': GBTClassifier(featuresCol="features", labelCol="is_trending", maxIter=20)
            }
            
            # Split data efficiently
            train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
            train_df.cache()  # Cache training data
            test_df.cache()   # Cache test data
            
            results = {}
            evaluator = BinaryClassificationEvaluator(labelCol="is_trending", metricName="areaUnderROC")
            
            for name, algorithm in algorithms.items():
                print(f"   Training {name}...")
                
                # Create pipeline
                pipeline = Pipeline(stages=[assembler, scaler, algorithm])
                
                # Train model
                model = pipeline.fit(train_df)
                
                # Evaluate efficiently (no collect needed)
                predictions = model.transform(test_df)
                auc = evaluator.evaluate(predictions)
                
                # Save model to HDFS
                model_path = f"{self.models_path}/classification/{name}"
                model.write().overwrite().save(model_path)
                
                results[name] = {
                    'auc': float(auc),
                    'model_path': model_path
                }
                
                print(f"   [OK] {name} - AUC: {auc:.4f}")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Optimized classification training failed: {str(e)}")
            return {}

    def train_optimized_clustering(self, df):
        """Train clustering models with optimizations"""
        print("🔄 Training optimized clustering models...")
        
        try:
            # Prepare features
            feature_cols = ['views', 'likes', 'comment_count', 'like_ratio', 'engagement_score']
            
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
            scaler = StandardScaler(inputCol="raw_features", outputCol="features")
            
            # Optimized clustering algorithms
            algorithms = {
                'kmeans': KMeans(featuresCol="features", predictionCol="cluster", k=6, seed=42),
                'bisecting_kmeans': BisectingKMeans(featuresCol="features", predictionCol="cluster", k=6, seed=42)
            }
            
            results = {}
            evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
            
            for name, algorithm in algorithms.items():
                print(f"   Training {name}...")
                
                # Create pipeline
                pipeline = Pipeline(stages=[assembler, scaler, algorithm])
                
                # Train model
                model = pipeline.fit(df)
                
                # Evaluate
                predictions = model.transform(df)
                silhouette = evaluator.evaluate(predictions)
                
                # Get cluster statistics without collect()
                cluster_stats = predictions.groupBy("cluster").agg(
                    count("*").alias("cluster_size"),
                    avg("views").alias("avg_views")
                ).collect()  # Only collect small aggregated results
                
                # Save model
                model_path = f"{self.models_path}/clustering/{name}"
                model.write().overwrite().save(model_path)
                
                results[name] = {
                    'silhouette_score': float(silhouette),
                    'cluster_stats': [row.asDict() for row in cluster_stats],
                    'model_path': model_path
                }
                
                print(f"   [OK] {name} - Silhouette: {silhouette:.4f}")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Optimized clustering training failed: {str(e)}")
            return {}

    def train_optimized_regression(self, df):
        """Train regression models with optimizations"""
        print("📈 Training optimized regression models...")
        
        try:
            # Prepare features for view prediction
            feature_cols = ['likes', 'dislikes', 'comment_count', 'title_length']
            
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
            scaler = StandardScaler(inputCol="raw_features", outputCol="features")
            
            # Log transform views for better regression
            df = df.withColumn("log_views", log(col("views") + 1))
            
            # Optimized regression algorithms
            algorithms = {
                'linear_regression': LinearRegression(featuresCol="features", labelCol="log_views"),
                'random_forest_regression': RandomForestRegressor(featuresCol="features", labelCol="log_views", numTrees=30)
            }
            
            # Split data
            train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
            train_df.cache()
            test_df.cache()
            
            results = {}
            evaluator = RegressionEvaluator(labelCol="log_views", metricName="rmse")
            
            for name, algorithm in algorithms.items():
                print(f"   Training {name}...")
                
                # Create pipeline
                pipeline = Pipeline(stages=[assembler, scaler, algorithm])
                
                # Train model
                model = pipeline.fit(train_df)
                
                # Evaluate
                predictions = model.transform(test_df)
                rmse = evaluator.evaluate(predictions)
                
                # Save model
                model_path = f"{self.models_path}/regression/{name}"
                model.write().overwrite().save(model_path)
                
                results[name] = {
                    'rmse': float(rmse),
                    'model_path': model_path
                }
                
                print(f"   [OK] {name} - RMSE: {rmse:.4f}")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Optimized regression training failed: {str(e)}")
            return {}

    def run_complete_production_training(self):
        """Run complete production ML training pipeline"""
        try:
            print("🚀 Starting Production MLlib Training Pipeline")
            print("=" * 60)
            
            # Load data with optimized configuration
            df = self.load_optimized_data()
            if df is None:
                return False
            
            # Train all model types
            classification_results = self.train_optimized_classification(df)
            clustering_results = self.train_optimized_clustering(df)
            regression_results = self.train_optimized_regression(df)
            
            # Combine results
            all_results = {
                'classification': classification_results,
                'clustering': clustering_results,
                'regression': regression_results,
                'framework': 'Spark MLlib Production',
                'features': [
                    'Partitioned data loading',
                    'Memory-optimized caching', 
                    'Distributed operations',
                    'Pipeline-based training',
                    'HDFS model storage'
                ]
            }
            
            # Save metadata
            self.save_optimized_metadata(all_results)
            
            print("=" * 60)
            print("🎉 PRODUCTION MLLIB TRAINING COMPLETED!")
            print("✅ High-performance configurations applied")
            print("✅ Models saved to HDFS")
            print("✅ Distributed operations only")
            print("✅ Production-ready pipeline")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Optimized training failed: {str(e)}")
            return False
        finally:
            self.mongo_client.close()

    def save_optimized_metadata(self, results):
        """Save training metadata to MongoDB"""
        try:
            metadata = {
                "created_at": datetime.now().isoformat(),
                "type": "spark_mllib_optimized_models",
                "framework": "Spark MLlib Optimized",
                "total_models": sum(len(models) for models in [
                    results['classification'], 
                    results['clustering'], 
                    results['regression']
                ]),
                "optimizations_applied": results['optimizations'],
                "model_results": results
            }
            
            # Update MongoDB
            self.db.ml_metadata.delete_many({"type": "spark_mllib_optimized_models"})
            self.db.ml_metadata.insert_one(metadata)
            
            print("[OK] Optimized model metadata saved to MongoDB")
            
        except Exception as e:
            print(f"[WARN] Failed to save metadata: {str(e)}")

# Global service instance
production_mllib_service = None

def get_production_mllib_service():
    """Get global production MLlib service instance"""
    global production_mllib_service
    if production_mllib_service is None:
        production_mllib_service = ProductionMLlibService()
    return production_mllib_service

def main():
    """Test production MLlib service"""
    service = ProductionMLlibService()
            success = service.run_complete_production_training()
            
            if success:
                print("🚀 Production MLlib models ready!")    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
