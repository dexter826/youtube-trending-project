"""
FastAPI Backend for YouTube Trending Analytics
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from datetime import datetime
import os
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.ml_service import get_ml_service, initialize_ml_service
from backend.app.routes.trending_routes import *
from backend.app.routes.ml_routes import *
from backend.app.utils.response_utils import JSONEncoder

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

# Set database connections for routes
set_database(db)
set_ml_service_getter(get_ml_service)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        client.admin.command({'ping': 1})
        initialize_ml_service()
    except Exception as e:
        pass

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    client.close()

# Include routes
app.get("/")(root)
app.get("/health")(health_check)
app.get("/countries")(get_countries)
app.get("/categories")(get_categories)
app.get("/dates")(get_dates)
app.get("/trending")(get_trending_videos)
app.get("/statistics")(get_statistics)
app.get("/wordcloud")(get_wordcloud_data)
app.get("/admin/database-stats")(get_database_stats)

# ML routes
app.get("/ml/health")(ml_health)
app.post("/ml/train")(train_ml_models)
app.post("/ml/predict")(predict_trending)
app.post("/ml/predict-views")(predict_views)
app.post("/ml/clustering")(predict_cluster)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)