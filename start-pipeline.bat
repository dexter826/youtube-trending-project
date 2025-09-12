@echo off
echo ========================================
echo YouTube Trending Big Data Pipeline
echo Author: BigData Expert
echo ========================================

echo.
echo [STEP 1] Starting Hadoop Services...
echo ----------------------------------------
start hadoop namenode -format -force
timeout /t 3
start hadoop namenode
timeout /t 3  
start hadoop datanode
timeout /t 3
start hadoop resourcemanager
timeout /t 3
start hadoop nodemanager
timeout /t 5

echo.
echo [STEP 2] Starting MongoDB...
echo ----------------------------------------
start mongod --dbpath "C:\data\db"
timeout /t 5

echo.
echo [STEP 3] Setting up HDFS directories...
echo ----------------------------------------
hdfs dfs -mkdir -p /youtube_trending
hdfs dfs -mkdir -p /youtube_trending/raw_data
hdfs dfs -mkdir -p /youtube_trending/processed  
hdfs dfs -mkdir -p /youtube_trending/models

echo.
echo [STEP 4] Copying data to HDFS...
echo ----------------------------------------
for %%c in (US CA GB DE FR IN JP KR MX RU) do (
    echo Copying %%c data...
    hdfs dfs -mkdir -p /youtube_trending/raw_data/%%c
    hdfs dfs -put -f "data\%%c*.csv" /youtube_trending/raw_data/%%c/
    hdfs dfs -put -f "data\%%c*.json" /youtube_trending/raw_data/%%c/
)

echo.
echo [STEP 5] Running Spark Data Processing...
echo ----------------------------------------
spark-submit spark\jobs\process_trending.py data

echo.
echo [STEP 6] Training ML Models...
echo ----------------------------------------
spark-submit spark\train_models.py

echo.
echo [STEP 7] Starting Backend API...
echo ----------------------------------------
cd backend
start python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
cd ..
timeout /t 5

echo.
echo [STEP 8] Starting Frontend...
echo ----------------------------------------
cd frontend
start npm start
cd ..

echo.
echo ========================================
echo PIPELINE COMPLETED!
echo ========================================
echo Frontend: http://localhost:3000
echo Backend API: http://localhost:8000  
echo API Docs: http://localhost:8000/docs
echo ========================================

pause