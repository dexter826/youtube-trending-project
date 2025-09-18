"""
FastAPI Backend for YouTube Trending Analytics
"""

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from datetime import datetime
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
load_dotenv(dotenv_path=project_root / ".env")

from backend.app.ml_service import get_ml_service, initialize_ml_service
from backend.app.routes.trending_routes import router as trending_router
from backend.app.routes.ml_routes import router as ml_router
from backend.app.utils.response_utils import JSONEncoder
from backend.app.services.health_service import health_service

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
trending_router.db = db
ml_router.db = db
ml_router.get_ml_service = get_ml_service

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        client.admin.command({'ping': 1})
        initialize_ml_service()
        
        # Register database health check
        async def check_db_health():
            try:
                client.admin.command('ping')
                return {"healthy": True, "message": "Database connected"}
            except Exception as e:
                return {"healthy": False, "error": str(e)}
        
        health_service.register_health_check("database", check_db_health)
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    client.close()

@app.get("/health")
async def health_check():
    """Health check endpoint using centralized service"""
    return await health_service.check_health()

# Include routes
app.include_router(trending_router, prefix="", tags=["trending"])
app.include_router(ml_router, prefix="/ml", tags=["ml"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)