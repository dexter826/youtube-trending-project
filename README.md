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
- **ML**: Spark MLlib (RandomForest, KMeans)
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

### Installation

```bash
# Clone repository
git clone https://github.com/dexter826/youtube-trending-project.git
cd youtube-trending-project

# Install dependencies
pip install -r backend/requirements.txt
pip install -r spark/requirements.txt
cd frontend && npm install

# Start services
python run_pipeline.py
cd backend && python -m app.main
cd frontend && npm start
```

## Access Points

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **HDFS**: http://localhost:9870

## Features

- Multi-country trending analysis (10 countries)
- ML predictions (trending, views, clustering)
- Interactive dashboard with charts
- Real-time data processing with Spark

## ML Models

- **Trending Classifier**: RandomForest binary classification
- **Views Regressor**: RandomForest regression
- **Content Clusterer**: KMeans clustering

## API Endpoints

### Data
- `GET /countries` - Available countries
- `GET /trending` - Trending videos
- `GET /statistics` - Analytics data

### ML
- `POST /ml/train` - Train models
- `POST /ml/predict` - Trending prediction
- `POST /ml/predict-views` - View prediction
- `POST /ml/clustering` - Content clustering

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
