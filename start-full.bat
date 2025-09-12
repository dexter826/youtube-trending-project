@echo off
REM YouTube Trending Project - Full Big Data Stack

cd /d "%~dp0"

echo Starting HDFS Services...
call C:\hadoop-3.4.1\sbin\start-all.cmd
timeout /t 8

echo Starting MongoDB...
start "MongoDB" cmd /k "mongod --dbpath C:\data\db"
timeout /t 5

echo Verifying HDFS Models...
hdfs dfs -ls /youtube_trending/models

echo Starting Backend API...
start "Backend API" cmd /k "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"
timeout /t 8

echo Starting Frontend...
cd frontend
start "Frontend" cmd /k "npm start"
cd ..

echo Full Big Data Stack started successfully!
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8000
echo HDFS: http://localhost:9870
pause