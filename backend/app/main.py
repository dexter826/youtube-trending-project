"""
FastAPI Backend for YouTube Trending Analytics
Description: REST API to serve processed YouTube trending data
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

# Import ML service
from .ml_service import get_ml_service, initialize_ml_service

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "youtube_trending"

app = FastAPI(
    title="YouTube Trending Analytics API",
    description="Big Data API for YouTube trending videos analysis",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
        client.admin.command('ismaster')
        print("✅ MongoDB connection successful")
        
        # Initialize ML service
        ml_service = initialize_ml_service(MONGO_URI, DB_NAME)
        print("🤖 ML service initialized")
        
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
        "message": "YouTube Trending Analytics API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check MongoDB connection
        client.admin.command('ismaster')
        
        # Check data availability
        countries = list(db.trending_results.distinct("country"))
        dates = list(db.trending_results.distinct("date"))
        
        return {
            "status": "healthy",
            "mongodb": "connected",
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
            "country_filter": country
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dates: {str(e)}")

@app.get("/categories")
async def get_youtube_categories():
    """Get YouTube video categories from category mapping file"""
    try:
        import json
        import os
        
        # Load categories from US category file (most complete)
        category_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                   "data", "US_category_id.json")
        
        if not os.path.exists(category_file):
            # Fallback categories if file not found
            return {
                "categories": {
                    "1": "Film & Animation",
                    "2": "Autos & Vehicles", 
                    "10": "Music",
                    "15": "Pets & Animals",
                    "17": "Sports",
                    "19": "Travel & Events",
                    "20": "Gaming",
                    "22": "People & Blogs",
                    "23": "Comedy",
                    "24": "Entertainment",
                    "25": "News & Politics",
                    "26": "Howto & Style",
                    "27": "Education",
                    "28": "Science & Technology"
                },
                "source": "fallback"
            }
        
        with open(category_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract categories from YouTube API format
        categories = {}
        for item in data.get("items", []):
            category_id = item.get("id")
            title = item.get("snippet", {}).get("title", "Unknown")
            if category_id and title:
                categories[category_id] = title
        
        return {
            "categories": categories,
            "count": len(categories),
            "source": "US_category_id.json"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")

@app.get("/trending")
async def get_trending_videos(
    country: str = Query(..., description="Country code (e.g., US, GB, CA)"),
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    limit: int = Query(10, ge=1, le=50, description="Number of top videos to return")
):
    """Get top trending videos for a specific country and date"""
    try:
        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Query trending results
        result = db.trending_results.find_one({
            "country": country.upper(),
            "date": date
        })
        
        if not result:
            raise HTTPException(
                status_code=404, 
                detail=f"No trending data found for {country} on {date}"
            )
        
        # Limit top videos
        top_videos = result.get("top_videos", [])[:limit]
        
        response_data = {
            "country": result["country"],
            "date": result["date"],
            "processed_at": result.get("processed_at"),
            "statistics": result.get("statistics", {}),
            "top_videos": top_videos,
            "total_videos_returned": len(top_videos)
        }
        
        return JSONResponse(
            content=json.loads(json.dumps(response_data, cls=JSONEncoder))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending data: {str(e)}")

@app.get("/wordcloud")
async def get_wordcloud_data(
    country: str = Query(..., description="Country code (e.g., US, GB, CA)"),
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    limit: int = Query(50, ge=10, le=100, description="Number of words to return")
):
    """Get wordcloud data for a specific country and date"""
    try:
        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Query wordcloud data
        result = db.wordcloud_data.find_one({
            "country": country.upper(),
            "date": date
        })
        
        if not result:
            raise HTTPException(
                status_code=404, 
                detail=f"No wordcloud data found for {country} on {date}"
            )
        
        # Limit words
        words = result.get("words", [])[:limit]
        
        response_data = {
            "country": result["country"],
            "date": result["date"],
            "processed_at": result.get("processed_at"),
            "words": words,
            "total_words_returned": len(words)
        }
        
        return JSONResponse(
            content=json.loads(json.dumps(response_data, cls=JSONEncoder))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch wordcloud data: {str(e)}")

@app.get("/analytics")
async def get_analytics_overview(
    country: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get analytics overview with optional filters"""
    try:
        # Build query filter
        filter_query = {}
        if country:
            filter_query["country"] = country.upper()
        
        if start_date or end_date:
            date_filter = {}
            if start_date:
                try:
                    datetime.strptime(start_date, "%Y-%m-%d")
                    date_filter["$gte"] = start_date
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
            
            if end_date:
                try:
                    datetime.strptime(end_date, "%Y-%m-%d")
                    date_filter["$lte"] = end_date
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
            
            filter_query["date"] = date_filter
        
        # Aggregate statistics
        pipeline = [
            {"$match": filter_query},
            {
                "$group": {
                    "_id": None,
                    "total_datasets": {"$sum": 1},
                    "total_videos": {"$sum": "$statistics.total_videos"},
                    "total_views": {"$sum": "$statistics.total_views"},
                    "avg_views": {"$avg": "$statistics.average_views"},
                    "countries": {"$addToSet": "$country"},
                    "date_range": {
                        "$push": "$date"
                    }
                }
            }
        ]
        
        result = list(db.trending_results.aggregate(pipeline))
        
        if not result:
            return {
                "message": "No data found for the specified filters",
                "filters": {
                    "country": country,
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
        
        stats = result[0]
        dates = stats.get("date_range", [])
        
        response_data = {
            "overview": {
                "total_datasets": stats.get("total_datasets", 0),
                "total_videos_analyzed": stats.get("total_videos", 0),
                "total_views": int(stats.get("total_views", 0)),
                "average_views": round(stats.get("avg_views", 0), 2),
                "countries_covered": sorted(stats.get("countries", [])),
                "date_range": {
                    "earliest": min(dates) if dates else None,
                    "latest": max(dates) if dates else None,
                    "total_dates": len(set(dates))
                }
            },
            "filters_applied": {
                "country": country,
                "start_date": start_date,
                "end_date": end_date
            }
        }
        
        return JSONResponse(
            content=json.loads(json.dumps(response_data, cls=JSONEncoder))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")

@app.get("/video/{video_id}")
async def get_video_details(video_id: str):
    """Get detailed information about a specific video"""
    try:
        # Search in raw videos collection
        video = db.raw_videos.find_one({"video_id": video_id})
        
        if not video:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        # Remove MongoDB ObjectId
        video.pop("_id", None)
        
        return JSONResponse(
            content=json.loads(json.dumps(video, cls=JSONEncoder))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch video details: {str(e)}")

@app.get("/categories")
async def get_categories_stats(country: Optional[str] = None):
    """Get video categories statistics"""
    try:
        match_filter = {}
        if country:
            match_filter["country"] = country.upper()
        
        pipeline = [
            {"$match": match_filter},
            {
                "$group": {
                    "_id": "$category_id",
                    "count": {"$sum": 1},
                    "total_views": {"$sum": "$views"},
                    "avg_views": {"$avg": "$views"},
                    "total_likes": {"$sum": "$likes"}
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": 20}
        ]
        
        results = list(db.raw_videos.aggregate(pipeline))
        
        # Map category IDs to names (simplified mapping)
        category_names = {
            1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music",
            15: "Pets & Animals", 17: "Sports", 19: "Travel & Events",
            20: "Gaming", 22: "People & Blogs", 23: "Comedy",
            24: "Entertainment", 25: "News & Politics", 26: "Howto & Style",
            27: "Education", 28: "Science & Technology"
        }
        
        categories_stats = [
            {
                "category_id": result["_id"],
                "category_name": category_names.get(result["_id"], f"Category {result['_id']}"),
                "video_count": result["count"],
                "total_views": int(result["total_views"]) if result["total_views"] else 0,
                "average_views": round(result["avg_views"], 2) if result["avg_views"] else 0,
                "total_likes": int(result["total_likes"]) if result["total_likes"] else 0
            }
            for result in results
        ]
        
        return {
            "categories": categories_stats,
            "country_filter": country,
            "total_categories": len(categories_stats)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories stats: {str(e)}")

# ============================================================================
# MACHINE LEARNING ENDPOINTS
# ============================================================================

# Pydantic models for ML requests
class VideoInput(BaseModel):
    title: str
    views: int = 0
    likes: int = 0
    dislikes: int = 0
    comment_count: int = 0
    category_id: int = 1
    publish_time: Optional[str] = None
    tags: str = ""
    comments_disabled: bool = False
    ratings_disabled: bool = False
    channel_title: Optional[str] = None
    video_id: Optional[str] = None

class BatchVideoInput(BaseModel):
    videos: List[VideoInput]
    model_name: str = "random_forest"

@app.post("/ml/predict-trending")
async def predict_video_trending(video: VideoInput, model_name: str = "random_forest"):
    """Predict if a single video will be trending"""
    try:
        ml_service = get_ml_service()
        
        # Convert Pydantic model to dict
        video_data = video.dict()
        
        # Make prediction
        result = ml_service.predict_trending(video_data, model_name)
        
        if not result or not result.get('success', False):
            raise HTTPException(
                status_code=500, 
                detail=result.get('error', 'Prediction failed') if result else 'ML service error'
            )
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")

@app.post("/ml/predict-batch")
async def predict_videos_trending(batch_input: BatchVideoInput):
    """Predict trending for multiple videos"""
    try:
        ml_service = get_ml_service()
        
        # Convert Pydantic models to dicts
        videos_data = [video.dict() for video in batch_input.videos]
        
        # Make batch prediction
        results = ml_service.predict_batch(videos_data, batch_input.model_name)
        
        return {
            "success": True,
            "total_predictions": len(results),
            "model_used": batch_input.model_name,
            "predictions": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/ml/model-info")
async def get_ml_model_info():
    """Get information about loaded ML models"""
    try:
        ml_service = get_ml_service()
        model_info = ml_service.get_model_info()
        
        if not model_info.get('success', False):
            raise HTTPException(
                status_code=404, 
                detail=model_info.get('error', 'Model info not available')
            )
        
        return JSONResponse(content=model_info)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/ml/health")
async def ml_health_check():
    """Check ML service health"""
    try:
        ml_service = get_ml_service()
        
        if not ml_service.is_loaded:
            return {
                "status": "not_ready",
                "message": "ML models not loaded. Please train models first.",
                "models_available": 0
            }
        
        return {
            "status": "ready",
            "message": "ML service is ready",
            "models_available": len(ml_service.models),
            "available_models": list(ml_service.models.keys())
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"ML service error: {str(e)}",
            "models_available": 0
        }

@app.post("/ml/quick-predict")
async def quick_predict_trending(
    title: str,
    views: int = 1000,
    likes: int = 100,
    comments: int = 10,
    category_id: int = 28
):
    """Quick prediction endpoint with minimal required fields"""
    try:
        ml_service = get_ml_service()
        
        # Create simple video data
        video_data = {
            'title': title,
            'views': views,
            'likes': likes,
            'dislikes': max(1, views // 100),  # Estimate dislikes
            'comment_count': comments,
            'category_id': category_id,
            'publish_time': datetime.now().isoformat(),
            'tags': '',
            'comments_disabled': False,
            'ratings_disabled': False
        }
        
        result = ml_service.predict_trending(video_data, 'random_forest')
        
        if not result or not result.get('success', False):
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Simplified response
        prediction = result['prediction']
        return {
            "title": title,
            "is_trending": prediction['is_trending'],
            "trending_probability": f"{prediction['trending_probability']:.1%}",
            "confidence": prediction['confidence_level'],
            "recommendation": result['recommendation']['trending_probability_assessment'],
            "advice": result['recommendation']['recommendations'][:2]  # Top 2 recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
