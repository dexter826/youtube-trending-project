@echo off
REM YouTube Trending Project - Start All Services with HDFS + Spark MLlib
REM Author: BigData Expert

echo 🚀 Starting YouTube Trending Project (Big Data Edition)...
echo Framework: Spark MLlib + HDFS + MongoDB
echo.

REM Navigate to project root
cd /d "%~dp0"

echo 📋 Project Components:
echo    ✅ HDFS - Distributed File System
echo    ✅ Spark MLlib - Distributed Machine Learning  
echo    ✅ MongoDB - NoSQL Database
echo    ✅ FastAPI - REST API Backend
echo    ✅ React - Frontend UI
echo.

echo 🔧 Step 1: Starting HDFS + Big Data Infrastructure...
cd infra
call start-hdfs-cluster.bat

echo.
echo 🤖 Step 2: Training Spark MLlib Models...
cd ..\spark\ml_models
python trending_predictor.py

echo.
echo 🌐 Step 3: Starting Backend API...
cd ..\..\backend
start "Backend API" cmd /k "uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo.
echo 🎨 Step 4: Starting Frontend...
cd ..\frontend  
start "Frontend" cmd /k "npm start"

echo.
echo ✅ YouTube Trending Project Started Successfully!
echo.
echo 🔗 Access Points:
echo    - Frontend UI:      http://localhost:3000
echo    - Backend API:      http://localhost:8000
echo    - API Docs:         http://localhost:8000/docs
echo    - HDFS NameNode:    http://localhost:9870
echo    - Spark Master:     http://localhost:8080
echo    - MongoDB Express:  http://localhost:8081
echo.
echo 🎯 Big Data Features:
echo    - Distributed storage with HDFS
echo    - Distributed ML with Spark MLlib
echo    - Multiple algorithms: Classification, Clustering, Regression
echo    - Models stored on HDFS
echo.
pause
