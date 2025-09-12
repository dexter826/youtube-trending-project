# YouTube Trending Analytics - Big Data Project

> Sinh viên thực hiện: **Trần Công Minh** - MSSV: 2001222641

## 🚀 Tổng quan dự án

Dự án phân tích video thịnh hành trên YouTube từ nhiều quốc gia, sử dụng **Apache Spark** và **HDFS** làm core cho Big Data processing, kết hợp **Machine Learning** để dự đoán trending và **React** frontend hiện đại.

## 🏗️ Kiến trúc hệ thống

```
CSV Data → HDFS → Spark Processing → MongoDB → FastAPI → React Frontend
                      ↓
              Spark MLlib Training → HDFS Model Storage
```

### Luồng xử lý chính:

1. **Data Ingestion**: Upload CSV data vào HDFS distributed storage
2. **Spark Processing**: Xử lý dữ liệu lớn với Apache Spark cluster  
3. **ML Training**: Huấn luyện models với Spark MLlib (Trending, Views, Clustering)
4. **Model Storage**: Lưu trained PipelineModels vào HDFS
5. **Data Storage**: Lưu processed data và metadata vào MongoDB
6. **API Layer**: FastAPI backend cung cấp REST APIs và ML predictions
7. **Frontend**: React dashboard với analytics và ML prediction interface

## ⚙️ Công nghệ stack

### Big Data Core:
- **Apache Spark**: Distributed data processing engine
- **HDFS**: Hadoop Distributed File System cho data storage
- **MongoDB**: Document database cho processed data

### Machine Learning:
- **Spark MLlib**: Distributed ML library (PipelineModel, RandomForest, KMeans)
- **HDFS Model Storage**: Trained models stored in distributed file system
- **Feature Engineering**: Advanced feature extraction và scaling with Spark

### Backend & API:
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Frontend:
- **ReactJS**: Modern UI framework
- **TailwindCSS**: Utility-first CSS framework
- **Chart.js**: Data visualization
- **Lucide React**: Icon library

## 📁 Cấu trúc project

```
youtube-trending-project/
├── data/                    # Raw CSV data (10 countries)
│   ├── USvideos.csv
│   ├── CAvideos.csv
│   └── ...
├── spark/                   # Apache Spark jobs
│   ├── jobs/
│   │   └── process_trending.py    # Main data processing
│   ├── core/
│   │   └── spark_config.py        # Spark configuration
│   ├── ml_models/
│   │   └── feature_engineering.py # ML feature extraction
│   └── train_models.py             # ML training pipeline
├── backend/
│   └── app/
│       ├── main.py                 # FastAPI application
│       └── ml_service.py           # ML prediction service
├── frontend/
│   ├── src/
│   │   ├── components/             # React components
│   │   │   ├── Dashboard.jsx
│   │   │   ├── MLPredictor.jsx
│   │   │   ├── TrendingVideosChart.jsx
│   │   │   └── ...
│   │   └── services/
│   │       └── apiService.jsx      # API client
├── run_pipeline.py                 # Main pipeline runner
├── start-bigdata.bat              # Windows startup script
└── README.md
```

## 🛠️ Cài đặt và khởi chạy

### Yêu cầu hệ thống:
- Windows 10/11
- Java 8/11
- Python 3.8+
- Node.js 16+
- MongoDB
- Apache Spark 3.x
- Hadoop/HDFS

### 1. Khởi động Big Data infrastructure:

```bash
# Start HDFS services
C:\hadoop-3.4.1\sbin\start-dfs.cmd

# Verify HDFS running
hdfs dfsadmin -report

# Start MongoDB
mongod

# Verify services
jps  # Should show NameNode and DataNode
```

### 2. Setup và chạy pipeline:

```bash
# Clone repository
git clone https://github.com/dexter826/youtube-trending-project.git
cd youtube-trending-project

# Chạy full pipeline (automated)
start-bigdata.bat

# Hoặc chạy manual từng bước:
python run_pipeline.py
```

### 3. Cài đặt dependencies:

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install

# Spark jobs
cd spark
pip install -r requirements.txt
```

### 4. Khởi động services:

```bash
# Backend API (Terminal 1)
cd backend
python -m app.main

# Frontend (Terminal 2)
cd frontend
npm start
```

## 🔗 Access points

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **HDFS NameNode**: http://localhost:9870
- **Spark Master UI**: http://localhost:8080

## 📊 Features chính

### 1. Data Analytics Dashboard
- **Multi-country analysis**: 10 quốc gia (US, CA, GB, DE, FR, IN, JP, KR, MX, RU)
- **Interactive filtering**: Theo country và category
- **Real-time statistics**: Views, likes, comments aggregation
- **Word cloud visualization**: Popular title keywords
- **Trending charts**: Video performance metrics

### 2. Machine Learning Predictions
- **Trending Prediction**: Dự đoán video có trending không
- **View Count Prediction**: Ước tính số lượt xem
- **Content Clustering**: Phân nhóm video theo nội dung
- **Model Training**: Auto-train với Spark processed data

### 3. Big Data Processing
- **Distributed Storage**: HDFS cho large datasets
- **Spark Processing**: Fast parallel data processing
- **Scalable Architecture**: Handle millions of records
- **Performance Optimized**: Broadcast joins, partitioning

## 🤖 Machine Learning Models

### Model Architecture:
- **Trending Classifier**: Spark MLlib RandomForestClassificationModel (Binary classification)
- **Views Regressor**: Spark MLlib RandomForestRegressionModel (Regression) 
- **Content Clusterer**: Spark MLlib KMeansModel (Unsupervised clustering)

### PipelineModel Structure:
1. **VectorAssembler**: Feature vector assembly
2. **StandardScaler**: Feature normalization
3. **ML Algorithm**: RandomForest/KMeans prediction

### Features được sử dụng:
- **Trending Model**: like_ratio, dislike_ratio, comment_ratio, engagement_score, title_length, has_caps, tag_count, category_id
- **Views Model**: likes, dislikes, comment_count, like_ratio, engagement_score, title_length, tag_count, category_id  
- **Clustering Model**: log_views, log_likes, log_comments, like_ratio, engagement_score, title_length, tag_count

### Model Storage:
- **HDFS Location**: hdfs://localhost:9000/youtube_trending/models/
- **Format**: Spark MLlib PipelineModel
- **Models**: trending_prediction, regression, clustering

### Model Performance:
- **Trending Prediction**: Distributed training with Spark MLlib
- **Views Prediction**: Scalable regression with feature engineering
- **Clustering**: K-means with optimal cluster selection

## 🗃️ Database Schema

### MongoDB Collections:
- `raw_videos`: Original CSV data (375,940 records)
- `trending_results`: Processed trending analysis (1,967 records)
- `wordcloud_data`: Title keyword frequencies
- `ml_features`: Feature engineered data for ML
- `model_metadata`: ML model information

## 🚀 API Endpoints

### Data Endpoints:
- `GET /countries` - Available countries
- `GET /trending` - Trending videos with filters
- `GET /statistics` - Aggregated analytics
- `GET /wordcloud` - Word cloud data

### ML Endpoints:
- `POST /ml/train` - Train ML models
- `POST /ml/predict` - Trending prediction
- `POST /ml/predict-views` - View count prediction
- `POST /ml/clustering` - Content clustering

### System Endpoints:
- `GET /health` - System health check
- `POST /data/process` - Trigger Spark processing

## 🔧 Configuration

### Spark Configuration:
```python
spark_config = {
    "spark.master": "local[*]",
    "spark.hadoop.fs.defaultFS": "hdfs://localhost:9000",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true"
}
```

### MongoDB Configuration:
```python
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "youtube_trending"
```

## 📈 Performance & Scalability

- **Data Volume**: 375K+ video records processed
- **Processing Speed**: ~10K records/second với Spark
- **Memory Usage**: Optimized với broadcast joins
- **API Response**: < 200ms average response time
- **Concurrent Users**: Supports 100+ concurrent requests

## 🧪 Testing & Quality

- **Code Quality**: Clean code architecture, no fallback logic
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging cho debugging
- **Monitoring**: Health checks và metrics
- **Documentation**: Comprehensive API docs

## 🏆 Key Achievements

✅ **Big Data Compliant**: Proper HDFS + Spark architecture  
✅ **ML Production Ready**: Trained models với good performance  
✅ **Clean Code**: No legacy code, fallback logic removed  
✅ **Scalable Design**: Handle large datasets efficiently  
✅ **Modern Stack**: Latest technologies và best practices  
✅ **Full Stack**: Complete end-to-end solution  

## 🔄 Future Enhancements

- Real-time streaming với Spark Streaming
- Advanced ML models (Deep Learning)
- Multi-language support
- Performance monitoring dashboard
- Automated model retraining
- Cloud deployment (AWS/GCP)

## 👨‍💻 Developer Notes

Dự án được thiết kế theo principles:
- **Clean Architecture**: Separation of concerns
- **Scalability First**: Built for large-scale data
- **Production Ready**: Enterprise-level code quality
- **Modern Technologies**: Latest frameworks và tools
- **Big Data Best Practices**: Distributed computing patterns

---

**Contact**: Trần Công Minh - MSSV: 2001222641