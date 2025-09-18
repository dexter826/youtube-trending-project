# YouTube Trending Analytics

## Overview

YouTube trending video analysis using Apache Spark, HDFS, and Machine Learning with React frontend.

## Architecture

```
CSV Data → HDFS → Spark Processing → MongoDB → FastAPI → React Frontend
                      ↓
              Spark MLlib Training → HDFS Model Storage
```

## Tech Stack

- **Big Data**: Apache Spark, HDFS, MongoDB
- **ML**: Spark MLlib (RandomForest for days regression, KMeans for clustering)
- **Backend**: FastAPI, Python
- **Frontend**: React, TailwindCSS

## Project Structure

```
youtube-trending-project/
├── data/                    # Raw CSV data (10 countries)
├── spark/                   # Spark jobs and ML models
├── backend/                 # FastAPI application
├── frontend/                # React dashboard
├── run_pipeline.py          # Main pipeline runner
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.8+, Node.js 16+, Java 8/11
- MongoDB, Apache Spark 3.x, Hadoop/HDFS
- Windows/Linux/Mac

### Installation & Setup

```bash
# Clone repository
git clone https://github.com/dexter826/youtube-trending-project.git
cd youtube-trending-project

# One-click setup (installs all dependencies)
python setup.py
```

### Running the Project

```bash
# Start backend + frontend only
python run.py app

# Start full stack (infrastructure + pipeline + app)
python run.py all

# Start infrastructure only (HDFS + MongoDB)
python run.py infrastructure

# Run data pipeline only
python run.py pipeline

# Check service status
python run.py status
```

### Access Points

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **HDFS**: http://localhost:9870

## Features

- Multi-country trending analysis (10 countries)
- ML predictions: clustering and days-in-trending only
- Predict by YouTube URL using YouTube Data API v3 (provide your API key at runtime)
- Interactive dashboard with charts
- Real-time data processing with Spark

## ML Models

- **Days-in-Trending Regressor**: RandomForest regression (label: days_in_trending)
- **Content Clusterer**: KMeans clustering

## API Endpoints

### Data
- `GET /countries` - Available countries
- `GET /trending` - Trending videos
- `GET /statistics` - Analytics data

### ML
- `POST /ml/train` - Train models (clustering + days-in-trending)
- `POST /ml/predict/days` - Predict number of days a video may stay on trending (structured input)
- `POST /ml/predict/cluster` - Predict content cluster (structured input)
- `POST /ml/predict/url` - Predict by YouTube URL (requires body: { url, api_key })

## Configuration

### Spark
```python
spark_config = {
    "spark.master": "local[*]",
    "spark.hadoop.fs.defaultFS": "hdfs://localhost:9000"
}
```

### MongoDB
```python
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "youtube_trending"
```

## Performance

- **Data Volume**: 375K+ records
- **Processing**: ~10K records/second
- **API Response**: < 200ms average
- **Concurrent Users**: 100+ supported

## Developer

Trần Công Minh - MSSV: 2001222641

## Migration Guide

### From Old Scripts to New Simplified Setup

**Old way (deprecated - removed):**
```bash
python run_pipeline.py     # ❌ Old pipeline script
start-full.bat            # ❌ Old batch script  
start-app.bat             # ❌ Old batch script
```

**New simplified way:**
```bash
# One-time setup
python setup.py           # ✅ Install everything

# Run as needed
python run.py app         # ✅ Start backend + frontend
python run.py all         # ✅ Start full stack
```

### File Structure Changes
- ✅ **setup.py**: One-click environment setup
- ✅ **run.py**: Flexible runner for different modes
- ✅ **Removed**: 3 confusing files → 2 clear files

### Benefits of New Approach
- ✅ **Simpler**: Just 2 files instead of 3 confusing ones
- ✅ **Clear separation**: Setup vs Run
- ✅ **Environment-aware**: Uses .env file for configuration
- ✅ **Better error handling**: Prerequisites checking, helpful messages
- ✅ **Flexible**: Run only what you need
