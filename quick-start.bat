@echo off
echo 🚀 Quick Start - YouTube Trending Analytics
echo.
echo 📦 Starting Infrastructure...
cd infra
docker compose up -d
if %errorlevel% neq 0 (
    echo ❌ Failed to start infrastructure
    pause
    exit /b 1
)

echo.
echo ✅ Infrastructure started!
echo.
echo 🚀 Starting Backend API...
cd ..\backend
start "Backend API" cmd /k "python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo.
echo ⚛️ Starting Frontend Dashboard...
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
echo ℹ️ Note: This uses existing data in MongoDB
echo    To recreate data, use start.bat option 2
echo.
echo Press any key to exit...
pause > nul
