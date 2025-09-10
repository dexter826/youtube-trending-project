"""
FastAPI Backend for YouTube Trending Analytics
Author: BigData Expert
Description: REST API to serve processed YouTube trending data
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import os
from bson import ObjectId
import json

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
        
        # Check if data exists
        raw_count = db.raw_videos.count_documents({})
        trending_count = db.trending_results.count_documents({})
        wordcloud_count = db.wordcloud_data.count_documents({})
        
        print(f"📊 Database status:")
        print(f"   - Raw videos: {raw_count}")
        print(f"   - Trending results: {trending_count}")
        print(f"   - Wordcloud data: {wordcloud_count}")
        
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

@app.get("/ml-analysis")
async def get_ml_analysis():
    """Get Machine Learning analysis results"""
    try:
        # Get the latest ML analysis results
        ml_result = db.ml_analysis.find_one(
            {"analysis_type": "comprehensive_ml_analysis"},
            sort=[("analysis_timestamp", -1)]
        )
        
        if not ml_result:
            raise HTTPException(status_code=404, detail="No ML analysis results found. Please run Spark job first.")
        
        # Clean the result for JSON serialization
        clean_result = json.loads(JSONEncoder().encode(ml_result))
        
        return {
            "ml_analysis": clean_result,
            "status": "success",
            "last_updated": clean_result.get("analysis_timestamp")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch ML analysis: {str(e)}")

@app.get("/ml-analysis/clustering")
async def get_clustering_results():
    """Get video clustering results"""
    try:
        ml_result = db.ml_analysis.find_one(
            {"analysis_type": "comprehensive_ml_analysis"},
            sort=[("analysis_timestamp", -1)]
        )
        
        if not ml_result or "clustering" not in ml_result:
            raise HTTPException(status_code=404, detail="No clustering results found")
        
        return {
            "clustering_results": ml_result["clustering"],
            "description": "Video clusters based on engagement metrics (views, likes, comments)",
            "cluster_interpretation": {
                "0": "Low Engagement Videos",
                "1": "Medium Engagement Videos", 
                "2": "High Engagement Videos",
                "3": "Viral Videos",
                "4": "Niche High-Engagement Videos"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch clustering results: {str(e)}")

@app.get("/ml-analysis/predictions")
async def get_prediction_results():
    """Get popularity prediction model results"""
    try:
        ml_result = db.ml_analysis.find_one(
            {"analysis_type": "comprehensive_ml_analysis"},
            sort=[("analysis_timestamp", -1)]
        )
        
        if not ml_result or "predictions" not in ml_result:
            raise HTTPException(status_code=404, detail="No prediction results found")
        
        return {
            "prediction_model": ml_result["predictions"],
            "description": "Machine learning model to predict video popularity based on metadata",
            "popularity_levels": {
                "0": "Low Popularity",
                "1": "Medium Popularity", 
                "2": "High Popularity"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch prediction results: {str(e)}")

@app.get("/ml-analysis/sentiment")
async def get_sentiment_analysis():
    """Get title sentiment analysis results"""
    try:
        ml_result = db.ml_analysis.find_one(
            {"analysis_type": "comprehensive_ml_analysis"},
            sort=[("analysis_timestamp", -1)]
        )
        
        if not ml_result or "sentiment" not in ml_result:
            raise HTTPException(status_code=404, detail="No sentiment analysis results found")
        
        return {
            "sentiment_analysis": ml_result["sentiment"],
            "description": "Sentiment analysis of video titles using keyword-based approach",
            "interpretation": {
                "positive_score": "Average positive words per title",
                "negative_score": "Average negative words per title",
                "sentiment_score": "Overall sentiment (positive - negative)",
                "clickbait_score": "Average clickbait indicators per title"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch sentiment analysis: {str(e)}")

@app.get("/ml-analysis/anomalies")
async def get_anomaly_detection():
    """Get anomaly detection results"""
    try:
        ml_result = db.ml_analysis.find_one(
            {"analysis_type": "comprehensive_ml_analysis"},
            sort=[("analysis_timestamp", -1)]
        )
        
        if not ml_result or "anomalies" not in ml_result:
            raise HTTPException(status_code=404, detail="No anomaly detection results found")
        
        return {
            "anomaly_detection": ml_result["anomalies"],
            "description": "Detection of videos with unusual engagement patterns",
            "criteria": [
                "Views exceeding 3 standard deviations from mean",
                "Likes exceeding 3 standard deviations from mean", 
                "High dislike ratio (>50%)"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch anomaly detection: {str(e)}")

@app.get("/ml-analysis/categories")
async def get_category_analysis():
    """Get category performance analysis"""
    try:
        ml_result = db.ml_analysis.find_one(
            {"analysis_type": "comprehensive_ml_analysis"},
            sort=[("analysis_timestamp", -1)]
        )
        
        if not ml_result or "category_analysis" not in ml_result:
            raise HTTPException(status_code=404, detail="No category analysis results found")
        
        return {
            "category_analysis": ml_result["category_analysis"],
            "description": "Performance patterns and statistics by video category",
            "metrics_explained": {
                "avg_views": "Average views per video in category",
                "avg_like_rate": "Average like rate (likes/views)",
                "view_variance": "Standard deviation of views (consistency indicator)"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch category analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
