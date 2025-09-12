"""
FastAPI Backend for YouTube Trending Analytics
Description: Clean REST API serving processed YouTube data with simple ML
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
from .ml_service import get_ml_service, initialize_ml_service

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "youtube_trending"

app = FastAPI(
    title="YouTube Trending Analytics API",
    description="Big Data API for YouTube trending videos analysis with ML predictions",
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
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

# Pydantic models for ML predictions
class VideoMLInput(BaseModel):
    title: str
    views: int = 0
    likes: int = 0
    dislikes: int = 0
    comment_count: int = 0
    category_id: int = 0
    tags: str = ""

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Test MongoDB connection
        client.admin.command({'ping': 1})
        print("✅ MongoDB connection successful")
        
        # Initialize ML services
        ml_service = initialize_ml_service()
        print("🤖 ML service initialized")
        
        # Database statistics
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
        print(f"❌ Startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    client.close()
    print("👋 MongoDB connection closed")

# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "YouTube Trending Analytics API",
        "framework": "Machine Learning with scikit-learn",
        "big_data_compliant": True,
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test database connection
        db_status = "healthy"
        try:
            client.admin.command({'ping': 1})
        except:
            db_status = "unhealthy"
        
        # Test ML service
        ml_service = get_ml_service()
        ml_status = "trained" if ml_service.is_trained else "untrained"
        
        return {
            "status": "healthy",
            "database": db_status,
            "ml_service": ml_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# ============================================================================
# DATA ENDPOINTS
# ============================================================================

@app.get("/countries")
async def get_countries():
    """Get list of available countries"""
    try:
        countries = list(db.trending_results.distinct("country"))
        return {"countries": sorted(countries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch countries: {str(e)}")

@app.get("/categories")
async def get_categories():
    """Get list of video categories"""
    try:
        # YouTube category mapping
        category_mapping = {
            1: "Film & Animation",
            2: "Autos & Vehicles", 
            10: "Music",
            15: "Pets & Animals",
            17: "Sports",
            19: "Travel & Events",
            20: "Gaming",
            22: "People & Blogs",
            23: "Comedy",
            24: "Entertainment",
            25: "News & Politics",
            26: "Howto & Style",
            27: "Education",
            28: "Science & Technology",
            29: "Nonprofits & Activism"
        }
        
        # Get unique category IDs from database
        category_ids = list(db.trending_results.aggregate([
            {"$unwind": "$top_videos"},
            {"$group": {"_id": "$top_videos.category_id"}},
            {"$sort": {"_id": 1}}
        ]))
        
        # Map IDs to names
        categories = []
        for cat_doc in category_ids:
            cat_id = cat_doc["_id"]
            if cat_id in category_mapping:
                categories.append({
                    "id": cat_id,
                    "name": category_mapping[cat_id]
                })
        
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")

@app.get("/dates")
async def get_dates(country: Optional[str] = None):
    """Get available dates for analysis"""
    try:
        filter_query = {}
        if country:
            filter_query["country"] = country
            
        # Get distinct dates for the country
        dates = list(db.trending_results.distinct("date", filter_query))
        
        # Sort dates in descending order (newest first)
        sorted_dates = sorted(dates, reverse=True)
        
        return {"dates": sorted_dates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dates: {str(e)}")

@app.get("/trending")
async def get_trending_videos(
    country: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = Query(100, le=1000)
):
    """Get trending videos with filters"""
    try:
        filter_query = {}
        
        if country:
            filter_query["country"] = country
        if category:
            filter_query["category_title"] = category
        
        results = list(db.trending_results.find(
            filter_query,
            {"_id": 0}
        ).sort("views", -1).limit(limit))
        
        return {
            "videos": results,
            "count": len(results),
            "filters": {"country": country, "category": category}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending videos: {str(e)}")

@app.get("/statistics")
async def get_statistics(country: Optional[str] = None):
    """Get trending statistics"""
    try:
        filter_query = {"country": country} if country else {}
        
        pipeline = [
            {"$match": filter_query},
            {"$group": {
                "_id": None,
                "total_videos": {"$sum": 1},
                "avg_views": {"$avg": "$views"},
                "avg_likes": {"$avg": "$likes"},
                "avg_comments": {"$avg": "$comment_count"},
                "max_views": {"$max": "$views"}
            }}
        ]
        
        result = list(db.trending_results.aggregate(pipeline))
        if result:
            stats = result[0]
            del stats["_id"]
            return {"statistics": stats, "country": country}
        else:
            return {"statistics": {}, "country": country}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.get("/wordcloud")
async def get_wordcloud_data(country: Optional[str] = None):
    """Get word cloud data"""
    try:
        filter_query = {"country": country} if country else {}
        
        result = db.wordcloud_data.find_one(filter_query, {"_id": 0})
        if result:
            return result
        else:
            return {"wordcloud_data": []}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get wordcloud data: {str(e)}")

# ============================================================================
# MACHINE LEARNING ENDPOINTS
# ============================================================================

@app.get("/ml/health")
async def ml_health():
    """ML service health check"""
    try:
        ml_service = get_ml_service()
        return ml_service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML health check failed: {str(e)}")

@app.post("/ml/train")
async def train_ml_models():
    """Train ML models using database data"""
    try:
        ml_service = get_ml_service()
        
        # Check training data availability
        training_data_count = db.ml_features.count_documents({})
        if training_data_count < 1000:
            raise HTTPException(status_code=400, detail=f"Insufficient training data: {training_data_count} records")
        
        # Train models
        success = ml_service.train_models()
        
        if success:
            return {
                "status": "success",
                "message": "ML models trained successfully",
                "training_data_count": training_data_count,
                "models": list(ml_service.models.keys())
            }
        else:
            raise HTTPException(status_code=500, detail="Training failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/ml/predict")
async def predict_trending(video_data: VideoMLInput):
    """Predict if a video will be trending"""
    try:
        ml_service = get_ml_service()
        
        # Convert to dict and add derived features
        video_dict = video_data.dict()
        
        result = ml_service.predict_trending(video_dict)
        
        return {
            "prediction": result,
            "input_data": video_dict,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/ml/predict-views")
async def predict_views(video_data: VideoMLInput):
    """Predict view count for a video"""
    try:
        ml_service = get_ml_service()
        
        video_dict = video_data.dict()
        result = ml_service.predict_views(video_dict)
        
        return {
            "prediction": result,
            "input_data": video_dict,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Views prediction failed: {str(e)}")

@app.post("/ml/clustering")
async def predict_cluster(video_data: VideoMLInput):
    """Predict content cluster for a video"""
    try:
        ml_service = get_ml_service()
        
        video_dict = video_data.dict()
        result = ml_service.predict_cluster(video_dict)
        
        return {
            "prediction": result,
            "input_data": video_dict,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

# ============================================================================
# DATA PROCESSING ENDPOINTS (Spark)
# ============================================================================

@app.post("/data/process")
async def process_data():
    """Process raw data using Spark (data processing only)"""
    try:
        import subprocess
        import sys
        
        # Check raw data availability
        raw_data_count = db.raw_videos.count_documents({})
        if raw_data_count == 0:
            raise HTTPException(status_code=400, detail="No raw data available for processing")
        
        # Run Spark data processing job
        script_path = "spark/jobs/process_trending.py"
        result = subprocess.run(
            [sys.executable, script_path], 
            capture_output=True, 
            text=True, 
            cwd="c:/BigData/youtube-trending-project"
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Data processing completed successfully",
                "framework": "Apache Spark",
                "input_records": raw_data_count,
                "output": result.stdout[-500:] if result.stdout else ""  # Last 500 chars
            }
        else:
            return {
                "status": "error",
                "message": "Data processing failed",
                "error": result.stderr,
                "output": result.stdout
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data processing failed: {str(e)}")

# ============================================================================
# ADMINISTRATION ENDPOINTS
# ============================================================================

@app.get("/admin/database-stats")
async def get_database_stats():
    """Get detailed database statistics"""
    try:
        collections = db.list_collection_names()
        stats = {}
        
        for collection in collections:
            count = db[collection].count_documents({})
            stats[collection] = count
        
        return {
            "collections": stats,
            "total_collections": len(collections),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)