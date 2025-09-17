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

# Create router
router = APIRouter()

db = None

class CategoryService:
    """Service for managing category mappings with caching"""
    _mappings = {}
    _loaded = False
    
    @classmethod
    def load_category_mappings(cls):
        """Load category mappings from JSON files with caching"""
        if cls._loaded:
            return cls._mappings
            
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
                            cat_id = str(item['id'])
                            title = item['snippet']['title']
                            mapping[cat_id] = title
                        cls._mappings[country] = mapping
                except Exception as e:
                    logger.warning(f"Failed to load category mapping for {country}: {e}")
                    continue
        
        cls._loaded = True
        return cls._mappings
    
    @classmethod
    def get_category_mapping(cls, country='US'):
        """Get category mapping for specific country"""
        if not cls._loaded:
            cls.load_category_mappings()
        return cls._mappings.get(country, {})

# Initialize category service
category_service = CategoryService()

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
        
        # Use cached category mapping
        mapping = category_service.get_category_mapping('US')
        
        category_names = {str(cat_id): mapping.get(str(cat_id), f"Category {cat_id}") 
                         for cat_id in sorted(categories)}
        return {"categories": category_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")

@router.get("/dates")
async def get_dates(country: Optional[str] = None):
    """Get list of available dates, optionally filtered by country.
    Supports legacy field names by checking both 'date' and 'trending_date'.
    """
    try:
        if router.db is None:
            raise HTTPException(status_code=500, detail="Database connection not initialized")

        filter_query = {"country": country} if country else {}
        dates_primary = list(router.db.trending_results.distinct("date", filter_query))
        dates_legacy = list(router.db.trending_results.distinct("trending_date", filter_query))
        # Merge and deduplicate
        all_dates = sorted(list({*(d for d in dates_primary if d), *(d for d in dates_legacy if d)}), reverse=True)
        return {"dates": all_dates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dates: {str(e)}")

@router.get("/trending")
async def get_trending_videos(
    country: Optional[str] = None,
    category: Optional[str] = None,
    date: Optional[str] = None,
    sort_by: Optional[str] = Query("views", description="views|likes|comments|engagement"),
    order: Optional[str] = Query("desc", description="asc|desc"),
    limit: int = Query(200, le=1000)
):
    """Get trending videos with filters, date selection and sorting.
    - date: filter by trending_date (string as stored in DB)
    - sort_by: one of views, likes, comments (comment_count), engagement
    - order: asc or desc
    - limit: internal cap, hidden from UI
    """
    try:
        if router.db is None:
            logger.error("Database connection is None")
            raise HTTPException(status_code=500, detail="Database connection not initialized")

        pipeline = []

        # Initial match on document-level fields
        match_stage = {}
        if country:
            match_stage["country"] = country
        if date:
            # Support both 'date' and legacy 'trending_date' fields
            pipeline.append({
                "$match": {
                    "$or": [
                        {"date": date},
                        {"trending_date": date}
                    ],
                    **({"country": country} if country else {})
                }
            })
        else:
            if match_stage:
                pipeline.append({"$match": match_stage})

        # Flatten top_videos and surface fields
        pipeline.extend([
            {"$unwind": "$top_videos"},
            {
                "$replaceRoot": {
                    "newRoot": {
                        "$mergeObjects": [
                            "$top_videos",
                            {"video_id": "$top_videos.video_id"},
                            {"country": "$country", "date": {"$ifNull": ["$date", "$trending_date"]}, "processed_at": "$processed_at"}
                        ]
                    }
                }
            }
        ])

        # Optional category match (now on flattened docs)
        if category:
            pipeline.append({"$match": {"category_id": int(category)}})

        # Sorting logic
        sort_field_map = {
            "views": "views",
            "likes": "likes",
            "comments": "comment_count",
            "engagement": "engagement"
        }
        sort_dir = 1 if (order or "desc").lower() == "asc" else -1

        if (sort_by or "views").lower() == "engagement":
            # Compute engagement safely: (likes + comments) / views
            pipeline.append({
                "$addFields": {
                    "engagement": {
                        "$cond": [
                            {"$gt": ["$views", 0]},
                            {"$divide": [{"$add": [{"$ifNull": ["$likes", 0]}, {"$ifNull": ["$comment_count", 0]}]}, "$views"]},
                            0
                        ]
                    }
                }
            })
        
        sort_field = sort_field_map.get((sort_by or "views").lower(), "views")
        pipeline.extend([
            {"$sort": {sort_field: sort_dir}},
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