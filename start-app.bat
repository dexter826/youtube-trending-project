@echo off
REM YouTube Trending Project - Backend + Frontend Only

cd /d "%~dp0"

jps | findstr "NameNode" >nul
if %errorlevel% neq 0 (
    echo HDFS not running! Please start HDFS first.
    pause
    exit /b 1
)

echo Starting Backend API...
start "Backend API" cmd /k "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"
timeout /t 8

echo Starting Frontend...
cd frontend
start "Frontend" cmd /k "npm start"
cd ..

echo Application started successfully!
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8000
pause