@echo off
echo 🚀 Starting YouTube Trending Analytics Project
echo.

echo Choose an option:
echo 1. Quick Start (just run services - use existing data)
echo 2. Full Rebuild (recreate all data from scratch)
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" goto quick_start
if "%choice%"=="2" goto full_rebuild
echo Invalid choice. Defaulting to Quick Start...
goto quick_start

:quick_start
echo.
echo 📦 Starting Infrastructure (MongoDB + Spark)...
cd infra
docker compose up -d
if %errorlevel% neq 0 (
    echo ❌ Failed to start infrastructure
    pause
    exit /b 1
)
echo ✅ Infrastructure started successfully!
goto start_services

:full_rebuild
echo.
echo 📦 Step 1: Starting Infrastructure (MongoDB + Spark)...
cd infra
docker compose up -d
if %errorlevel% neq 0 (
    echo ❌ Failed to start infrastructure
    pause
    exit /b 1
)

echo.
echo ✅ Infrastructure started successfully!
echo.
echo 📊 Step 2: Running Spark ETL Pipeline (This will recreate all data)...
cd ..\spark
python run_spark_job.py ..\data
if %errorlevel% neq 0 (
    echo ❌ Spark job failed
    pause
    exit /b 1
)
echo ✅ Data processing completed!

:start_services
echo.
echo 🚀 Step 3: Starting Backend API...
cd ..\backend
start "Backend API" cmd /k "python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo.
echo ⚛️ Step 4: Starting Frontend Dashboard...
cd ..\frontend
start "Frontend Dashboard" cmd /k "npm start"

echo.
echo 🎉 All services are starting...
echo.
echo 📍 URLs:
echo   - Dashboard: http://localhost:3000
echo   - API Docs:  http://localhost:8000/docs
echo   - MongoDB:   http://localhost:8081
echo.
echo Press any key to exit...
pause > nul
