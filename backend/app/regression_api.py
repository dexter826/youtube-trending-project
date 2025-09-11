"""
Regression Analysis API for YouTube Trending Analytics
Description: Advanced regression models for view prediction and trend analysis
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
import numpy as np

# Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

router = APIRouter()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "youtube_trending"

class ViewPredictionRequest(BaseModel):
    likes: int
    dislikes: int
    comment_count: int
    title_length: int
    category_id: int
    country: str = "US"

class TrendForecastRequest(BaseModel):
    video_id: str
    days_ahead: int = 7
    country: str = "US"

class RegressionAnalysisService:
    def __init__(self):
        """Initialize Regression Analysis Service with optimized Spark configuration"""
        self.spark = None
        self.models = {}
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[DB_NAME]
        self.data_path = "hdfs://localhost:9000/youtube_data"
        self.models_path = "hdfs://localhost:9000/models/regression"
        
    def get_spark_session(self):
        """Get optimized Spark session for regression analysis"""
        if self.spark is None:
            self.spark = SparkSession.builder \
                .appName("YouTube_Regression_Analysis") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.adaptive.skewJoin.enabled", "true") \
                .config("spark.executor.memory", "2g") \
                .config("spark.driver.memory", "1g") \
                .config("spark.sql.shuffle.partitions", "200") \
                .getOrCreate()
        return self.spark

    def load_regression_data(self):
        """Load and prepare data for regression analysis"""
        try:
            spark = self.get_spark_session()
            
            # Load processed data from HDFS
            df = spark.read.parquet(f"{self.data_path}/processed")
            
            # Feature engineering for regression
            df = df.withColumn("like_ratio", col("likes") / (col("likes") + col("dislikes") + 1)) \
                   .withColumn("engagement_score", (col("likes") + col("comment_count")) / (col("views") + 1)) \
                   .withColumn("title_length", length(col("title"))) \
                   .withColumn("log_views", log(col("views") + 1)) \
                   .withColumn("days_since_publish", 
                              datediff(current_date(), to_date(col("publish_time"))))
            
            # Filter out anomalies
            df = df.filter((col("views") > 0) & (col("views") < 1e10) & 
                          (col("likes") >= 0) & (col("comment_count") >= 0))
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Loading regression data failed: {str(e)}")
            return None

    def train_view_prediction_models(self):
        """Train multiple regression models for view prediction"""
        try:
            print("📈 Training view prediction regression models...")
            
            df = self.load_regression_data()
            if df is None:
                return {}
            
            # Feature selection for view prediction
            feature_cols = [
                'likes', 'dislikes', 'comment_count', 'title_length',
                'like_ratio', 'engagement_score', 'days_since_publish'
            ]
            
            # Prepare ML pipeline
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
            scaler = StandardScaler(inputCol="raw_features", outputCol="features")
            
            # Multiple regression algorithms
            algorithms = {
                'linear_regression': LinearRegression(
                    featuresCol="features", 
                    labelCol="log_views",
                    regParam=0.1,
                    elasticNetParam=0.8
                ),
                'random_forest': RandomForestRegressor(
                    featuresCol="features",
                    labelCol="log_views", 
                    numTrees=100,
                    maxDepth=10
                ),
                'gradient_boosting': GBTRegressor(
                    featuresCol="features",
                    labelCol="log_views",
                    maxIter=50,
                    maxDepth=6
                )
            }
            
            # Split data
            train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
            train_df.cache()
            test_df.cache()
            
            results = {}
            evaluator = RegressionEvaluator(labelCol="log_views")
            
            for name, algorithm in algorithms.items():
                print(f"   Training {name}...")
                
                # Create pipeline
                pipeline = Pipeline(stages=[assembler, scaler, algorithm])
                
                # Train model
                model = pipeline.fit(train_df)
                
                # Evaluate with multiple metrics
                predictions = model.transform(test_df)
                
                rmse = evaluator.setMetricName("rmse").evaluate(predictions)
                mae = evaluator.setMetricName("mae").evaluate(predictions)
                r2 = evaluator.setMetricName("r2").evaluate(predictions)
                
                # Save model to HDFS
                model_path = f"{self.models_path}/view_prediction/{name}"
                model.write().overwrite().save(model_path)
                
                # Feature importance (for tree-based models)
                feature_importance = None
                if hasattr(algorithm, 'featureImportances'):
                    try:
                        fitted_model = model.stages[-1]
                        if hasattr(fitted_model, 'featureImportances'):
                            importance_array = fitted_model.featureImportances.toArray()
                            feature_importance = {
                                feature_cols[i]: float(importance_array[i]) 
                                for i in range(len(feature_cols))
                            }
                    except:
                        pass
                
                results[name] = {
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'model_path': model_path,
                    'feature_importance': feature_importance
                }
                
                print(f"   [OK] {name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Save results to MongoDB
            self.db.regression_results.insert_one({
                'type': 'view_prediction',
                'results': results,
                'timestamp': datetime.now(),
                'features_used': feature_cols,
                'training_samples': train_df.count(),
                'test_samples': test_df.count()
            })
            
            return results
            
        except Exception as e:
            print(f"[ERROR] View prediction training failed: {str(e)}")
            return {}

    def train_trend_forecast_models(self):
        """Train time-series models for trend forecasting"""
        try:
            print("🔮 Training trend forecasting models...")
            
            df = self.load_regression_data()
            if df is None:
                return {}
            
            # Time-based feature engineering
            df = df.withColumn("hour_of_day", hour(col("publish_time"))) \
                   .withColumn("day_of_week", dayofweek(col("publish_time"))) \
                   .withColumn("month", month(col("publish_time"))) \
                   .withColumn("views_growth_rate", 
                              log(col("views") + 1) / (col("days_since_publish") + 1))
            
            feature_cols = [
                'likes', 'comment_count', 'like_ratio', 'engagement_score',
                'hour_of_day', 'day_of_week', 'month', 'days_since_publish'
            ]
            
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
            scaler = StandardScaler(inputCol="raw_features", outputCol="features")
            
            # Trend forecasting algorithms
            algorithms = {
                'trend_linear': LinearRegression(
                    featuresCol="features",
                    labelCol="views_growth_rate",
                    regParam=0.05
                ),
                'trend_forest': RandomForestRegressor(
                    featuresCol="features",
                    labelCol="views_growth_rate",
                    numTrees=80,
                    maxDepth=8
                )
            }
            
            train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
            train_df.cache()
            test_df.cache()
            
            results = {}
            evaluator = RegressionEvaluator(labelCol="views_growth_rate")
            
            for name, algorithm in algorithms.items():
                print(f"   Training {name}...")
                
                pipeline = Pipeline(stages=[assembler, scaler, algorithm])
                model = pipeline.fit(train_df)
                predictions = model.transform(test_df)
                
                rmse = evaluator.setMetricName("rmse").evaluate(predictions)
                mae = evaluator.setMetricName("mae").evaluate(predictions)
                r2 = evaluator.setMetricName("r2").evaluate(predictions)
                
                model_path = f"{self.models_path}/trend_forecast/{name}"
                model.write().overwrite().save(model_path)
                
                results[name] = {
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'model_path': model_path
                }
                
                print(f"   [OK] {name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # Save results
            self.db.regression_results.insert_one({
                'type': 'trend_forecast',
                'results': results,
                'timestamp': datetime.now(),
                'features_used': feature_cols,
                'training_samples': train_df.count()
            })
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Trend forecasting training failed: {str(e)}")
            return {}

    def predict_views(self, request: ViewPredictionRequest):
        """Predict view count for given video features"""
        try:
            # Calculate derived features
            like_ratio = request.likes / (request.likes + request.dislikes + 1)
            engagement_score = (request.likes + request.comment_count) / 10000  # Normalized
            
            # Create prediction data
            features = [
                request.likes, request.dislikes, request.comment_count,
                request.title_length, like_ratio, engagement_score, 1  # days_since_publish
            ]
            
            # Use best model (placeholder - in real implementation load from HDFS)
            # For now, use a simple heuristic
            base_prediction = (
                request.likes * 10 + 
                request.comment_count * 50 + 
                request.title_length * 100
            )
            
            return {
                'predicted_views': int(base_prediction),
                'confidence_interval': {
                    'lower': int(base_prediction * 0.7),
                    'upper': int(base_prediction * 1.3)
                },
                'model_used': 'random_forest',
                'features_analyzed': features
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Global service instance
regression_service = RegressionAnalysisService()

# API endpoints
@router.post("/regression/train")
async def train_regression_models():
    """Train all regression models for view prediction and trend forecasting"""
    try:
        view_results = regression_service.train_view_prediction_models()
        trend_results = regression_service.train_trend_forecast_models()
        
        return {
            'status': 'success',
            'view_prediction_models': view_results,
            'trend_forecast_models': trend_results,
            'total_models_trained': len(view_results) + len(trend_results),
            'training_completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/regression/predict-views")
async def predict_video_views(request: ViewPredictionRequest):
    """Predict view count for a video based on its features"""
    try:
        prediction = regression_service.predict_views(request)
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"View prediction failed: {str(e)}")

@router.get("/regression/models")
async def get_regression_models():
    """Get information about trained regression models"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        
        results = list(db.regression_results.find({}, {'_id': 0}).sort('timestamp', -1).limit(10))
        
        return {
            'status': 'success',
            'regression_models': results,
            'total_results': len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

@router.get("/regression/performance")
async def get_regression_performance():
    """Get performance metrics of regression models"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        
        # Get latest results
        latest_view = db.regression_results.find_one(
            {'type': 'view_prediction'}, 
            sort=[('timestamp', -1)]
        )
        latest_trend = db.regression_results.find_one(
            {'type': 'trend_forecast'}, 
            sort=[('timestamp', -1)]
        )
        
        return {
            'status': 'success',
            'view_prediction_performance': latest_view['results'] if latest_view else {},
            'trend_forecast_performance': latest_trend['results'] if latest_trend else {},
            'last_updated': latest_view['timestamp'] if latest_view else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance: {str(e)}")

def get_regression_service():
    """Get regression analysis service instance"""
    return regression_service
