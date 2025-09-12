@echo off
REM YouTube Trending Project - Backend + Frontend Only
REM Requires: HDFS and MongoDB already running

echo 🚀 Starting YouTube Trending Application
echo ================================================
echo Components: Backend API + Frontend UI
echo ================================================
echo.

REM Navigate to project root
cd /d "%~dp0"

echo 🔍 Checking Prerequisites...
echo ----------------------------------------
jps | findstr "NameNode" >nul
if %errorlevel% neq 0 (
    echo ❌ HDFS not running! Please start HDFS first.
    echo Run: start-full.bat for complete setup
    pause
    exit /b 1
)
echo ✅ HDFS is running

echo.
echo 🌐 [1/2] Starting Backend API...
echo ----------------------------------------
echo Starting FastAPI with Spark MLlib integration...
start "Backend API" cmd /k "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"
timeout /t 8

echo.
echo 🎨 [2/2] Starting Frontend...
echo ----------------------------------------
echo Starting React development server...
cd frontend
start "Frontend" cmd /k "npm start"
cd ..

echo.
echo ✅ APPLICATION STARTED SUCCESSFULLY!
echo ================================================
echo 🔗 Access Points:
echo    - Frontend UI:      http://localhost:3000
echo    - Backend API:      http://localhost:8000
echo    - API Docs:         http://localhost:8000/docs
echo.
echo 💡 Features Available:
echo    ✅ YouTube Analytics Dashboard
echo    ✅ ML Predictions (Spark MLlib)
echo    ✅ Real-time Data Visualization
echo ================================================
pause