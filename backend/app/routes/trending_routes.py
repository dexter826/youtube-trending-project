"""
Trending Data Routes for YouTube Analytics API
"""

from fastapi import HTTPException, Query, APIRouter
from datetime import datetime
from typing import Optional
import os
import json
import logging

logger = logging.getLogger(__name__)

# Global variables
CATEGORY_MAPPINGS = {}

# Create router
router = APIRouter()

db = None

def load_category_mappings():
    """Load category mappings from JSON files"""
    global CATEGORY_MAPPINGS
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')

    for filename in os.listdir(data_dir):
        if filename.endswith('_category_id.json'):
            country = filename.split('_')[0].upper()
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    mapping = {}
                    for item in data.get('items', []):
                        cat_id = int(item['id'])
                        title = item['snippet']['title']
                        mapping[cat_id] = title
                    CATEGORY_MAPPINGS[country] = mapping
            except Exception as e:
                pass  # Skip invalid files

# Load mappings on import
load_category_mappings()

@router.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "YouTube Trending Analytics API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        if router.db is None:
            raise HTTPException(status_code=500, detail="Database connection not initialized")
        
        router.db.client.admin.command('ping')
        return {
            "status": "healthy",
            "database": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/countries")
async def get_countries():
    """Get list of available countries"""
    try:
        if router.db is None:
            raise HTTPException(status_code=500, detail="Database connection not initialized")
        
        countries = list(router.db.trending_results.distinct("country"))
        return {"countries": countries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get countries: {str(e)}")

@router.get("/categories")
async def get_categories(country: Optional[str] = None):
    """Get available categories"""
    try:
        if router.db is None:
            raise HTTPException(status_code=500, detail="Database connection not initialized")

        filter_query = {"country": country} if country else {}
        categories = list(router.db.trending_results.distinct("top_videos.category_id", filter_query))
        category_names = {str(cat_id): CATEGORY_MAPPINGS.get(str(cat_id), f"Category {cat_id}") 
                         for cat_id in sorted(categories)}
        return {"categories": category_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")

@router.get("/dates")
async def get_dates():
    """Get list of available dates"""
    try:
        if router.db is None:
            raise HTTPException(status_code=500, detail="Database connection not initialized")

        dates = list(router.db.trending_results.distinct("trending_date"))
        return {"dates": sorted(dates, reverse=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dates: {str(e)}")

@router.get("/trending")
async def get_trending_videos(
    country: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = Query(100, le=1000)
):
    """Get trending videos with filters"""
    try:
        if router.db is None:
            logger.error("Database connection is None")
            raise HTTPException(status_code=500, detail="Database connection not initialized")

        pipeline = []

        match_stage = {}
        if country:
            match_stage["country"] = country
        if match_stage:
            pipeline.append({"$match": match_stage})

        pipeline.extend([
            {"$unwind": "$top_videos"},
            {
                "$replaceRoot": {
                    "newRoot": {
                        "$mergeObjects": [
                            "$top_videos",
                            {"country": "$country", "date": "$date", "processed_at": "$processed_at"}
                        ]
                    }
                }
            }
        ])

        if category:
            pipeline.append({"$match": {"category_id": int(category)}})

        pipeline.extend([
            {"$sort": {"views": -1}},
            {"$limit": limit}
        ])

        results = list(router.db.trending_results.aggregate(pipeline))
        return {"videos": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Failed to fetch trending videos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending videos: {str(e)}")

@router.get("/statistics")
async def get_statistics(country: Optional[str] = None):
    """Get trending statistics"""
    try:
        if router.db is None:
            raise HTTPException(status_code=500, detail="Database connection not initialized")

        filter_query = {"country": country} if country else {}
        total_videos = router.db.raw_videos.count_documents(filter_query)

        if total_videos == 0:
            return {"statistics": {}, "country": country}

        videos = list(router.db.raw_videos.find(filter_query, {"views": 1, "likes": 1, "comment_count": 1}))

        total_views = sum(video.get("views", 0) for video in videos)
        total_likes = sum(video.get("likes", 0) for video in videos)
        total_comments = sum(video.get("comment_count", 0) for video in videos)
        max_views = max((video.get("views", 0) for video in videos), default=0)

        stats = {
            "total_videos": total_videos,
            "avg_views": total_views / total_videos if total_videos > 0 else 0,
            "avg_likes": total_likes / total_videos if total_videos > 0 else 0,
            "avg_comments": total_comments / total_videos if total_videos > 0 else 0,
            "max_views": max_views,
            "total_views": total_views,
            "total_likes": total_likes,
            "total_comments": total_comments
        }

        return {"statistics": stats, "country": country}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.get("/wordcloud")
async def get_wordcloud_data(country: Optional[str] = None):
    """Get word cloud data"""
    try:
        if router.db is None:
            logger.error("Database connection is None")
            raise HTTPException(status_code=500, detail="Database connection not initialized")

        filter_query = {"country": country} if country else {}
        result = router.db.wordcloud_data.find_one(filter_query, {"_id": 0})
        return result or {"wordcloud_data": []}
    except Exception as e:
        logger.error(f"Failed to get wordcloud data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get wordcloud data: {str(e)}")

@router.get("/admin/database-stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        if router.db is None:
            raise HTTPException(status_code=500, detail="Database connection not initialized")

        stats = {
            "collections": router.db.list_collection_names(),
            "raw_videos_count": router.db.raw_videos.count_documents({}),
            "trending_results_count": router.db.trending_results.count_documents({}),
            "wordcloud_data_count": router.db.wordcloud_data.count_documents({})
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")