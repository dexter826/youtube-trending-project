"""
FastAPI Backend for YouTube Trending Analytics
Description: REST API serving processed YouTube data using distributed Spark MLlib
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
from bson import ObjectId
import json

# Import ML services
from .ml_api import get_mllib_service, initialize_mllib_service
from .clustering_api import get_advanced_clustering_service
from .regression_api import get_regression_service, ViewPredictionRequest

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "youtube_trending"

app = FastAPI(
    title="YouTube Trending Analytics API - Spark MLlib",
    description="Big Data API for YouTube trending videos analysis using Spark MLlib",
    version="2.0.0"
)

# CORS middleware for React frontend
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
default_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()] if allowed_origins_env else default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB ObjectId and datetime"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)

@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    try:
        # Test MongoDB connection
        client.admin.command({'ping': 1})
        print("✅ MongoDB connection successful")
        
        # Initialize MLlib services
        mllib_service = initialize_mllib_service()
        advanced_clustering_service = get_advanced_clustering_service()
        regression_service = get_regression_service()
        print("🤖 Spark MLlib service initialized")
        print("🧠 Advanced clustering service initialized")
        print("📈 Regression analysis service initialized")
        
        # Check if data exists
        raw_count = db.raw_videos.count_documents({})
        trending_count = db.trending_results.count_documents({})
        wordcloud_count = db.wordcloud_data.count_documents({})
        ml_features_count = db.ml_features.count_documents({})
        
        print(f"📊 Database status:")
        print(f"   - Raw videos: {raw_count}")
        print(f"   - Trending results: {trending_count}")
        print(f"   - Wordcloud data: {wordcloud_count}")
        print(f"   - ML features: {ml_features_count}")
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    client.close()
    print("👋 MongoDB connection closed")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "YouTube Trending Analytics API - Spark MLlib",
        "framework": "Spark MLlib",
        "big_data_compliant": True,
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check MongoDB connection
        client.admin.command({'ping': 1})
        
        # Check data availability
        countries = list(db.trending_results.distinct("country"))
        dates = list(db.trending_results.distinct("date"))
        
        return {
            "status": "healthy",
            "mongodb": "connected",
            "framework": "Spark MLlib",
            "big_data_compliant": True,
            "data": {
                "countries_available": countries,
                "dates_available": sorted(dates) if dates else [],
                "total_countries": len(countries),
                "total_dates": len(dates)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/countries")
async def get_available_countries():
    """Get list of available countries"""
    try:
        countries = list(db.trending_results.distinct("country"))
        return {
            "countries": sorted(countries),
            "count": len(countries)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch countries: {str(e)}")

@app.get("/dates")
async def get_available_dates(country: Optional[str] = None):
    """Get list of available dates, optionally filtered by country"""
    try:
        filter_query = {}
        if country:
            filter_query["country"] = country
            
        dates = list(db.trending_results.distinct("date", filter_query))
        return {
            "dates": sorted(dates) if dates else [],
            "count": len(dates),
            "country": country
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dates: {str(e)}")

@app.get("/trending/{country}")
async def get_trending_by_country(
    country: str,
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    limit: int = Query(10, ge=1, le=50, description="Number of results to return")
):
    """Get trending videos for a specific country"""
    try:
        filter_query = {"country": country.upper()}
        
        if date:
            filter_query["date"] = date
        
        # Get trending results
        results = list(db.trending_results.find(
            filter_query,
            {"_id": 0}
        ).limit(limit))
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No trending data found for country: {country}" + (f" on date: {date}" if date else "")
            )
        
        return {
            "country": country.upper(),
            "date": date,
            "results": results,
            "count": len(results)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending data: {str(e)}")

@app.get("/statistics/{country}")
async def get_statistics_by_country(
    country: str,
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")
):
    """Get statistics for a specific country"""
    try:
        filter_query = {"country": country.upper()}
        
        if date:
            filter_query["date"] = date
        
        # Get statistics from trending results
        result = db.trending_results.find_one(filter_query, {"_id": 0, "statistics": 1})
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No statistics found for country: {country}" + (f" on date: {date}" if date else "")
            )
        
        return {
            "country": country.upper(),
            "date": date,
            "statistics": result.get("statistics", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch statistics: {str(e)}")

@app.get("/wordcloud/{country}")
async def get_wordcloud_data(
    country: str,
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")
):
    """Get wordcloud data for a specific country"""
    try:
        filter_query = {"country": country.upper()}
        
        if date:
            filter_query["date"] = date
        
        # Get wordcloud data
        result = db.wordcloud_data.find_one(filter_query, {"_id": 0})
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No wordcloud data found for country: {country}" + (f" on date: {date}" if date else "")
            )
        
        return {
            "country": country.upper(),
            "date": date,
            "wordcloud": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch wordcloud data: {str(e)}")

@app.get("/videos")
async def get_videos(
    country: Optional[str] = Query(None, description="Filter by country"),
    category_id: Optional[int] = Query(None, description="Filter by category ID"),
    limit: int = Query(100, ge=1, le=1000, description="Number of videos to return"),
    skip: int = Query(0, ge=0, description="Number of videos to skip")
):
    """Get raw video data with optional filtering"""
    try:
        filter_query = {}
        
        if country:
            filter_query["country"] = country.upper()
        
        if category_id is not None:
            filter_query["category_id"] = category_id
        
        # Get videos from raw collection
        videos = list(db.raw_videos.find(
            filter_query,
            {"_id": 0}
        ).skip(skip).limit(limit))
        
        # Get total count for pagination
        total_count = db.raw_videos.count_documents(filter_query)
        
        return {
            "videos": videos,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "skip": skip,
                "current_page": (skip // limit) + 1,
                "total_pages": (total_count + limit - 1) // limit
            },
            "filters": {
                "country": country,
                "category_id": category_id
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch videos: {str(e)}")

# ===============================
# SPARK MLLIB ENDPOINTS ONLY
# ===============================

# MLlib Input Models
class VideoMLlibInput(BaseModel):
    title: str = "Sample Video"
    views: int = 1000
    likes: int = 50
    dislikes: int = 5
    comment_count: int = 20
    category_id: int = 1
    publish_hour: int = 12
    tags: str = "sample,video"
    comments_disabled: bool = False
    ratings_disabled: bool = False
    channel_title: str = "Sample Channel"

@app.get("/mllib/health")
async def mllib_health_check():
    """Check Spark MLlib service health"""
    try:
        mllib_service = get_mllib_service()
        model_info = mllib_service.get_model_info()
        
        return {
            "status": "ready" if mllib_service.is_loaded else "not_ready",
            "framework": "Spark MLlib",
            "loaded_models": model_info["loaded_models"],
            "model_count": model_info["model_count"],
            "spark_version": model_info["spark_version"],
            "hdfs_path": model_info["hdfs_path"],
            "big_data_compliant": True,
            "distributed_ml": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Spark MLlib service error: {str(e)}",
            "framework": "Spark MLlib",
            "big_data_compliant": False
        }

@app.post("/mllib/predict")
async def mllib_predict_trending(video_data: VideoMLlibInput):
    """Predict trending using Spark MLlib models"""
    try:
        mllib_service = get_mllib_service()
        
        # Use best performing model (random_forest preferred)
        model_preference = ["random_forest", "gradient_boosting", "logistic_regression", "decision_tree"]
        
        available_models = list(mllib_service.models.keys())
        selected_model = None
        
        for model in model_preference:
            if model in available_models:
                selected_model = model
                break
        
        if not selected_model:
            selected_model = available_models[0] if available_models else "logistic_regression"
        
        # Convert to dict and add derived features
        video_dict = video_data.dict()
        video_dict['title_length'] = len(video_dict['title'])
        video_dict['tag_count'] = len(video_dict['tags'].split(',')) if video_dict['tags'] else 0
        video_dict['like_ratio'] = video_dict['likes'] / max(video_dict['views'], 1)
        video_dict['comment_ratio'] = video_dict['comment_count'] / max(video_dict['views'], 1)
        video_dict['engagement_score'] = (video_dict['likes'] + video_dict['comment_count']) / max(video_dict['views'], 1)
        
        result = mllib_service.predict_trending(video_dict)
        
        return {
            "prediction": result,
            "input_data": video_dict,
            "framework": "Spark MLlib",
            "distributed": True,
            "big_data_technology": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLlib prediction failed: {str(e)}")

@app.post("/mllib/clustering")
async def mllib_clustering_analysis(videos_data: List[VideoMLlibInput]):
    """Perform clustering analysis using Spark MLlib"""
    try:
        mllib_service = get_mllib_service()
        
        # Convert input data
        videos_dict = []
        for video in videos_data:
            video_dict = video.dict()
            video_dict['title_length'] = len(video_dict['title'])
            video_dict['tag_count'] = len(video_dict['tags'].split(',')) if video_dict['tags'] else 0
            video_dict['like_ratio'] = video_dict['likes'] / max(video_dict['views'], 1)
            video_dict['comment_ratio'] = video_dict['comment_count'] / max(video_dict['views'], 1)
            video_dict['engagement_score'] = (video_dict['likes'] + video_dict['comment_count']) / max(video_dict['views'], 1)
            videos_dict.append(video_dict)
        
        result = mllib_service.get_clustering_analysis(videos_dict)
        
        return {
            "clustering_analysis": result,
            "framework": "Spark MLlib",
            "distributed": True,
            "video_count": len(videos_data),
            "algorithms": ["K-Means", "Bisecting K-Means"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLlib clustering failed: {str(e)}")

@app.post("/mllib/predict-views")
async def mllib_predict_views(video_data: VideoMLlibInput):
    """Predict view count using Spark MLlib regression"""
    try:
        mllib_service = get_mllib_service()
        
        # Convert to dict and add derived features
        video_dict = video_data.dict()
        video_dict['title_length'] = len(video_dict['title'])
        video_dict['tag_count'] = len(video_dict['tags'].split(',')) if video_dict['tags'] else 0
        video_dict['like_ratio'] = video_dict['likes'] / max(video_dict['views'], 1)
        video_dict['comment_ratio'] = video_dict['comment_count'] / max(video_dict['views'], 1)
        video_dict['engagement_score'] = (video_dict['likes'] + video_dict['comment_count']) / max(video_dict['views'], 1)
        
        result = mllib_service.predict_views(video_dict)
        
        return {
            "view_prediction": result,
            "input_data": video_dict,
            "framework": "Spark MLlib", 
            "distributed": True,
            "algorithm": "regression"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLlib view prediction failed: {str(e)}")

@app.get("/mllib/models")
async def get_mllib_models():
    """Get information about available Spark MLlib models"""
    try:
        mllib_service = get_mllib_service()
        model_info = mllib_service.get_model_info()
        
        # Get metadata from MongoDB
        metadata = db.ml_metadata.find_one({"type": "spark_mllib_models"})
        
        return {
            "model_info": model_info,
            "metadata": metadata,
            "framework": "Spark MLlib",
            "big_data_compliant": True,
            "hdfs_storage": True,
            "distributed_ml": True,
            "algorithms": {
                "classification": ["Logistic Regression", "Random Forest", "Decision Tree", "Gradient Boosting"],
                "clustering": ["K-Means", "Bisecting K-Means"],
                "regression": ["Linear Regression", "Random Forest Regression"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get MLlib models info: {str(e)}")

# ============================================================================
# CLUSTERING ENDPOINTS 
# ============================================================================

@app.post("/clustering/behavioral")
async def behavioral_clustering(video_data: dict):
    """Get behavioral cluster prediction for a video"""
    try:
        clustering_service = get_advanced_clustering_service()
        result = clustering_service.get_behavioral_cluster(video_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Behavioral clustering failed: {str(e)}")

@app.post("/clustering/content")
async def content_clustering(video_data: dict):
    """Get content cluster prediction for a video"""
    try:
        clustering_service = get_advanced_clustering_service()
        result = clustering_service.get_content_cluster(video_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content clustering failed: {str(e)}")

@app.get("/clustering/geographic/{country}")
async def geographic_clustering(country: str):
    """Get geographic cluster for a country"""
    try:
        clustering_service = get_advanced_clustering_service()
        result = clustering_service.get_geographic_cluster(country)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geographic clustering failed: {str(e)}")

@app.post("/clustering/temporal")
async def temporal_clustering(temporal_data: dict):
    """Get temporal cluster prediction"""
    try:
        clustering_service = get_advanced_clustering_service()
        publish_time = temporal_data.get('publish_time', datetime.now().isoformat())
        result = clustering_service.get_temporal_cluster(publish_time)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Temporal clustering failed: {str(e)}")

@app.post("/clustering/comprehensive")
async def comprehensive_clustering(video_data: dict):
    """Get comprehensive clustering analysis for a video"""
    try:
        clustering_service = get_advanced_clustering_service()
        result = clustering_service.get_comprehensive_clustering(video_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive clustering failed: {str(e)}")

@app.get("/clustering/statistics")
async def clustering_statistics():
    """Get clustering statistics and insights"""
    try:
        clustering_service = get_advanced_clustering_service()
        result = clustering_service.get_clustering_statistics()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get clustering statistics: {str(e)}")

@app.post("/clustering/train")
async def train_clustering():
    """Train clustering models"""
    try:
        # Check training data
        training_data_count = db.raw_videos.count_documents({})
        if training_data_count == 0:
            raise HTTPException(status_code=400, detail="No training data available")
        
        # Run clustering analysis
        import subprocess
        import sys
        
        script_path = "c:/BigData/youtube-trending-project/spark/services/clustering_service.py"
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd="c:/BigData/youtube-trending-project")
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Clustering models trained successfully",
                "training_data_count": training_data_count,
                "framework": "Spark MLlib",
                "algorithms": [
                    "Behavioral K-Means",
                    "Behavioral Bisecting K-Means", 
                    "Behavioral Gaussian Mixture",
                    "Content Word2Vec + K-Means",
                    "Geographic K-Means",
                    "Temporal K-Means"
                ],
                "output": result.stdout
            }
        else:
            return {
                "status": "error",
                "message": "Clustering training failed",
                "error": result.stderr,
                "output": result.stdout
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering training failed: {str(e)}")

@app.get("/clustering/models")
async def get_clustering_models():
    """Get information about available clustering models"""
    try:
        clustering_service = get_advanced_clustering_service()
        
        # Get clustering metadata
        metadata = db.ml_metadata.find_one({"type": "advanced_clustering_analysis"})
        
        # Get clustering results summary
        clustering_results = list(db.clustering_results.find({}))
        
        model_info = {
            "available_models": list(clustering_service.loaded_models.keys()),
            "total_models": len(clustering_service.loaded_models),
            "last_training": metadata.get("created_at") if metadata else None,
            "algorithms_used": metadata.get("algorithms_used") if metadata else {},
            "clustering_types": ["behavioral", "content", "geographic", "temporal"],
            "total_clustering_results": len(clustering_results),
            "framework": "Spark MLlib"
        }
        
        return {
            "clustering_info": model_info,
            "metadata": metadata,
            "big_data_compliant": True,
            "distributed_clustering": True,
            "hdfs_storage": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get advanced clustering models info: {str(e)}")

# ============================================================================
# SPARK OPTIMIZATION ENDPOINTS (STEP 4)
# ============================================================================

@app.post("/optimized/train-all")
async def train_optimized_models():
    """Train all models with Spark optimizations"""
    try:
        import subprocess
        import sys
        
        # Check training data
        training_data_count = db.raw_videos.count_documents({})
        if training_data_count == 0:
            raise HTTPException(status_code=400, detail="No training data available")
        
        # Run optimized training
        script_path = "c:/BigData/youtube-trending-project/spark/ml_models/optimized_mllib_service.py"
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, 
                              cwd="c:/BigData/youtube-trending-project")
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Optimized models trained successfully",
                "training_data_count": training_data_count,
                "framework": "Spark MLlib Optimized",
                "optimizations": [
                    "Partitioned data loading",
                    "Optimized caching strategy",
                    "No collect() operations",
                    "Pipeline-based training",
                    "HDFS model storage"
                ],
                "output": result.stdout
            }
        else:
            return {
                "status": "error", 
                "message": "Optimized training failed",
                "error": result.stderr,
                "output": result.stdout
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimized training failed: {str(e)}")

@app.get("/optimized/performance-metrics")
async def get_performance_metrics():
    """Get Spark performance metrics and optimizations"""
    try:
        # Get optimized model metadata
        metadata = db.ml_metadata.find_one({"type": "spark_mllib_optimized_models"})
        
        # Performance metrics
        performance_info = {
            "optimizations_applied": [
                "Adaptive Query Execution (AQE)",
                "Dynamic partitioning", 
                "Broadcast joins",
                "Column pruning",
                "Predicate pushdown",
                "Kryo serialization",
                "Memory-optimized storage"
            ],
            "data_optimizations": [
                "Parquet file format",
                "Snappy compression",
                "Partitioned storage",
                "Cached DataFrames",
                "Vectorized operations"
            ],
            "ml_optimizations": [
                "Pipeline-based training",
                "No collect() operations",
                "Distributed model training",
                "HDFS model storage",
                "Feature caching"
            ],
            "cluster_ready": True,
            "hdfs_integrated": True,
            "big_data_compliant": True
        }
        
        return {
            "performance_metrics": performance_info,
            "last_optimization": metadata.get("created_at") if metadata else None,
            "total_optimized_models": metadata.get("total_models") if metadata else 0,
            "framework": "Spark MLlib Optimized"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@app.get("/optimized/cluster-config")
async def get_cluster_configuration():
    """Get Spark cluster configuration for production deployment"""
    try:
        cluster_config = {
            "spark_master": "spark://spark-master:7077",
            "deployment_mode": "cluster",
            "resource_allocation": {
                "driver_memory": "8g",
                "executor_memory": "6g", 
                "executor_cores": "4",
                "executor_instances": "3"
            },
            "optimization_settings": {
                "adaptive_query_execution": True,
                "dynamic_allocation": True,
                "broadcast_threshold": "50MB",
                "shuffle_partitions": "200"
            },
            "storage_configuration": {
                "hdfs_replication": 2,
                "compression": "snappy",
                "file_format": "parquet"
            },
            "production_ready": True
        }
        
        return {
            "cluster_configuration": cluster_config,
            "framework": "Spark MLlib Optimized",
            "big_data_platform": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cluster config: {str(e)}")

# ================================
# REGRESSION ANALYSIS ENDPOINTS 
# ================================

@app.post("/regression/train")
async def train_regression_models():
    """Train all regression models for view prediction and trend forecasting"""
    try:
        regression_service = get_regression_service()
        view_results = regression_service.train_view_prediction_models()
        trend_results = regression_service.train_trend_forecast_models()
        
        return {
            'status': 'success',
            'view_prediction_models': view_results,
            'trend_forecast_models': trend_results,
            'total_models_trained': len(view_results) + len(trend_results),
            'training_completed_at': datetime.now().isoformat(),
            'framework': 'Spark MLlib Regression',
            'features': [
                'Advanced view prediction',
                'Time-series trend forecasting', 
                'Multiple regression algorithms',
                'HDFS model storage',
                'Feature importance analysis'
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regression training failed: {str(e)}")

@app.post("/regression/predict-views")
async def predict_video_views(request: ViewPredictionRequest):
    """Predict view count for a video based on its features"""
    try:
        regression_service = get_regression_service()
        prediction = regression_service.predict_views(request)
        
        return {
            'status': 'success',
            'prediction': prediction,
            'input_features': request.dict(),
            'model_type': 'regression_analysis',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"View prediction failed: {str(e)}")

@app.get("/regression/models")
async def get_regression_models():
    """Get information about trained regression models"""
    try:
        results = list(db.regression_results.find({}, {'_id': 0}).sort('timestamp', -1).limit(10))
        
        return {
            'status': 'success',
            'regression_models': results,
            'total_results': len(results),
            'model_types': ['view_prediction', 'trend_forecast'],
            'algorithms': ['linear_regression', 'random_forest', 'gradient_boosting'],
            'framework': 'Spark MLlib Regression'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get regression models: {str(e)}")

@app.get("/regression/performance")
async def get_regression_performance():
    """Get performance metrics of regression models"""
    try:
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
            'metrics_available': ['RMSE', 'MAE', 'R²', 'Feature Importance'],
            'last_updated': latest_view['timestamp'] if latest_view else None,
            'framework': 'Spark MLlib Regression Analysis'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get regression performance: {str(e)}")

@app.get("/regression/analysis")
async def get_regression_analysis():
    """Get comprehensive regression analysis results"""
    try:
        # Get all regression results
        view_models = list(db.regression_results.find({'type': 'view_prediction'}).sort('timestamp', -1).limit(5))
        trend_models = list(db.regression_results.find({'type': 'trend_forecast'}).sort('timestamp', -1).limit(5))
        
        # Calculate summary statistics
        total_models = len(view_models) + len(trend_models)
        best_view_model = None
        best_trend_model = None
        
        if view_models:
            best_view_model = max(view_models[0]['results'].items(), key=lambda x: x[1].get('r2', 0))
        
        if trend_models:
            best_trend_model = max(trend_models[0]['results'].items(), key=lambda x: x[1].get('r2', 0))
        
        return {
            'status': 'success',
            'analysis_summary': {
                'total_regression_models': total_models,
                'view_prediction_models': len(view_models),
                'trend_forecast_models': len(trend_models),
                'best_view_model': best_view_model[0] if best_view_model else None,
                'best_trend_model': best_trend_model[0] if best_trend_model else None
            },
            'recent_trainings': {
                'view_prediction': view_models,
                'trend_forecast': trend_models
            },
            'capabilities': [
                'Video view count prediction',
                'Trending forecast analysis',
                'Feature importance ranking',
                'Multi-algorithm comparison',
                'Time-series modeling'
            ],
            'framework': 'Advanced Spark MLlib Regression',
            'big_data_compliance': True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get regression analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
