"""
Trending Data Routes for YouTube Analytics API
"""

from fastapi import HTTPException, Query
from datetime import datetime
from typing import Optional
import os
import json

# Global variables (will be imported from main)
CATEGORY_MAPPINGS = {}
db = None

def set_database(database):
    """Set database connection"""
    global db
    db = database

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

async def root():
    """Health check endpoint"""
    return {
        "message": "YouTube Trending Analytics API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

async def health_check():
    """Detailed health check"""
    try:
        db.admin.command({'ping': 1})
        return {
            "status": "healthy",
            "database": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

async def get_countries():
    """Get list of available countries"""
    try:
        countries = list(db.trending_results.distinct("country"))
        return {"countries": sorted(countries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch countries: {str(e)}")

async def get_categories():
    """Get list of video categories"""
    try:
        category_mapping = {
            1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music",
            15: "Pets & Animals", 17: "Sports", 19: "Travel & Events",
            20: "Gaming", 22: "People & Blogs", 23: "Comedy",
            24: "Entertainment", 25: "News & Politics", 26: "Howto & Style",
            27: "Education", 28: "Science & Technology", 29: "Nonprofits & Activism"
        }

        category_ids = list(db.trending_results.aggregate([
            {"$unwind": "$top_videos"},
            {"$group": {"_id": "$top_videos.category_id"}},
            {"$sort": {"_id": 1}}
        ]))

        categories = []
        for cat_doc in category_ids:
            cat_id = cat_doc["_id"]
            if cat_id in category_mapping:
                categories.append({"id": cat_id, "name": category_mapping[cat_id]})

        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")

async def get_dates(country: Optional[str] = None):
    """Get available dates for analysis"""
    try:
        filter_query = {"country": country} if country else {}
        dates = list(db.trending_results.distinct("date", filter_query))
        sorted_dates = sorted(dates, reverse=True)
        return {"dates": sorted_dates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dates: {str(e)}")

async def get_trending_videos(
    country: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = Query(100, le=1000)
):
    """Get trending videos with filters"""
    try:
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

        results = list(db.trending_results.aggregate(pipeline))
        return {"videos": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending videos: {str(e)}")

async def get_statistics(country: Optional[str] = None):
    """Get trending statistics"""
    try:
        filter_query = {"country": country} if country else {}
        total_videos = db.raw_videos.count_documents(filter_query)

        if total_videos == 0:
            return {"statistics": {}, "country": country}

        videos = list(db.raw_videos.find(filter_query, {"views": 1, "likes": 1, "comment_count": 1}))

        total_views = sum(video.get("views", 0) for video in videos)
        total_likes = sum(video.get("likes", 0) for video in videos)
        total_comments = sum(video.get("comment_count", 0) for video in videos)
        max_views = max((video.get("views", 0) for video in videos), default=0)

        stats = {
            "total_videos": total_videos,
            "avg_views": total_views / total_videos if total_videos > 0 else 0,
            "avg_likes": total_likes / total_videos if total_videos > 0 else 0,
            "avg_comments": total_comments / total_videos if total_videos > 0 else 0,
            "max_views": max_views
        }

        return {"statistics": stats, "country": country}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

async def get_wordcloud_data(country: Optional[str] = None):
    """Get word cloud data"""
    try:
        filter_query = {"country": country} if country else {}
        result = db.wordcloud_data.find_one(filter_query, {"_id": 0})
        return result or {"wordcloud_data": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get wordcloud data: {str(e)}")

async def get_database_stats():
    """Get database statistics"""
    try:
        stats = {
            "raw_videos": db.raw_videos.count_documents({}),
            "trending_results": db.trending_results.count_documents({}),
            "wordcloud_data": db.wordcloud_data.count_documents({}),
            "ml_features": db.ml_features.count_documents({})
        }
        return {"collections": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")