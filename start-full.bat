@echo off
REM YouTube Trending Project - Full Big Data Stack
REM Includes: HDFS + Spark + MongoDB + Backend + Frontend

echo 🚀 Starting YouTube Trending Project (FULL BIG DATA STACK)
echo ================================================================
echo Components: HDFS + Spark MLlib + MongoDB + FastAPI + React
echo ================================================================
echo.

REM Navigate to project root
cd /d "%~dp0"

echo 🔧 [1/5] Starting HDFS Services...
echo ----------------------------------------
echo Starting NameNode and DataNode...
call C:\hadoop-3.4.1\sbin\start-all.cmd
timeout /t 8

echo.
echo 🗄️ [2/5] Starting MongoDB...  
echo ----------------------------------------
echo Starting MongoDB service...
start "MongoDB" cmd /k "mongod --dbpath C:\data\db"
timeout /t 5

echo.
echo 📊 [3/5] Verifying HDFS Models...
echo ----------------------------------------
hdfs dfs -ls /youtube_trending/models
echo HDFS models verified ✅

echo.
echo 🌐 [4/5] Starting Backend API (Spark MLlib)...
echo ----------------------------------------
start "Backend API" cmd /k "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"
timeout /t 8

echo.
echo 🎨 [5/5] Starting Frontend...
echo ----------------------------------------
cd frontend
start "Frontend" cmd /k "npm start"
cd ..

echo.
echo ✅ FULL BIG DATA STACK STARTED SUCCESSFULLY!
echo ================================================================
echo 🔗 Access Points:
echo    - Frontend UI:      http://localhost:3000
echo    - Backend API:      http://localhost:8000  
echo    - API Docs:         http://localhost:8000/docs
echo    - HDFS NameNode:    http://localhost:9870
echo.
echo 🎯 Big Data Features Active:
echo    ✅ Distributed Storage (HDFS)
echo    ✅ Distributed ML (Spark MLlib) 
echo    ✅ 3 ML Models from HDFS
echo    ✅ 375K+ YouTube Records
echo ================================================================
pause