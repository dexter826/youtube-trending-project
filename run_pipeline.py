"""
Complete YouTube Trending Big Data Pipeline
Author: BigData Expert
Description: Run complete pipeline from data processing to model training
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(cmd, description):
    """Run command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout:
                print("Output:", result.stdout)
        else:
            print(f"❌ {description} - FAILED")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ {description} - EXCEPTION: {str(e)}")
        return False
    
    return True

def check_hdfs_directories():
    """Create necessary HDFS directories"""
    print("\n📁 Setting up HDFS directories...")
    
    directories = [
        "/youtube_trending",
        "/youtube_trending/raw_data",
        "/youtube_trending/processed",
        "/youtube_trending/models"
    ]
    
    for directory in directories:
        cmd = f"hdfs dfs -mkdir -p {directory}"
        if not run_command(cmd, f"Creating HDFS directory: {directory}"):
            print(f"⚠️  Warning: Could not create {directory} (may already exist)")

def copy_data_to_hdfs():
    """Copy CSV data to HDFS"""
    print("\n📤 Copying data to HDFS...")
    
    data_dir = "C:\\BigData\\youtube-trending-project\\data"
    
    # Countries to copy
    countries = ['US', 'CA', 'GB', 'DE', 'FR', 'IN', 'JP', 'KR', 'MX', 'RU']
    
    for country in countries:
        csv_file = f"{data_dir}\\{country}videos.csv"
        category_file = f"{data_dir}\\{country}_category_id.json"
        
        if os.path.exists(csv_file):
            # Create country directory in HDFS
            hdfs_country_dir = f"/youtube_trending/raw_data/{country}"
            run_command(f"hdfs dfs -mkdir -p {hdfs_country_dir}", f"Creating {country} directory")
            
            # Copy CSV file
            hdfs_csv_path = f"{hdfs_country_dir}/{country}videos.csv"
            run_command(f"hdfs dfs -put -f {csv_file} {hdfs_csv_path}", f"Copying {country} videos CSV")
            
            # Copy category file
            if os.path.exists(category_file):
                hdfs_cat_path = f"{hdfs_country_dir}/{country}_category_id.json"
                run_command(f"hdfs dfs -put -f {category_file} {hdfs_cat_path}", f"Copying {country} categories JSON")
        else:
            print(f"⚠️  Warning: {csv_file} not found")

def run_spark_data_processing():
    """Run Spark data processing job"""
    print("\n🔥 Running Spark Data Processing...")
    
    spark_script = "C:\\BigData\\youtube-trending-project\\spark\\jobs\\process_trending.py"
    fallback_data = "C:\\BigData\\youtube-trending-project\\data"
    
    cmd = f"spark-submit {spark_script} {fallback_data}"
    
    return run_command(cmd, "Spark Data Processing Job")

def run_model_training():
    """Run ML model training"""
    print("\n🤖 Running ML Model Training...")
    
    training_script = "C:\\BigData\\youtube-trending-project\\spark\\train_models.py"
    
    cmd = f"spark-submit {training_script}"
    
    return run_command(cmd, "ML Model Training Job")

def start_backend_api():
    """Start FastAPI backend"""
    print("\n🚀 Starting Backend API...")
    
    backend_dir = "C:\\BigData\\youtube-trending-project\\backend"
    
    # Change to backend directory and start
    cmd = f"cd {backend_dir} && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    
    print(f"Command: {cmd}")
    print("Backend will start in background...")
    
    try:
        # Start backend in background
        subprocess.Popen(cmd, shell=True, cwd=backend_dir)
        time.sleep(3)  # Give it time to start
        print("✅ Backend API started at http://localhost:8000")
        return True
    except Exception as e:
        print(f"❌ Failed to start backend: {str(e)}")
        return False

def start_frontend():
    """Start React frontend"""
    print("\n🌐 Starting Frontend...")
    
    frontend_dir = "C:\\BigData\\youtube-trending-project\\frontend"
    
    # Install dependencies and start
    cmd = f"cd {frontend_dir} && npm install && npm start"
    
    print(f"Command: {cmd}")
    print("Frontend will start in background...")
    
    try:
        # Start frontend in background
        subprocess.Popen(cmd, shell=True, cwd=frontend_dir)
        time.sleep(5)  # Give it time to start
        print("✅ Frontend started at http://localhost:3000")
        return True
    except Exception as e:
        print(f"❌ Failed to start frontend: {str(e)}")
        return False

def main():
    """Main pipeline execution"""
    print("🌟 YouTube Trending Big Data Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print("=" * 60)
    
    # Step 1: Setup HDFS
    check_hdfs_directories()
    
    # Step 2: Copy data to HDFS
    copy_data_to_hdfs()
    
    # Step 3: Run Spark data processing
    if not run_spark_data_processing():
        print("❌ Data processing failed. Stopping pipeline.")
        return False
    
    # Step 4: Train ML models
    if not run_model_training():
        print("❌ Model training failed. Stopping pipeline.")
        return False
    
    # Step 5: Start backend API
    if not start_backend_api():
        print("❌ Backend startup failed. Stopping pipeline.")
        return False
    
    # Step 6: Start frontend
    if not start_frontend():
        print("❌ Frontend startup failed.")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("📊 Frontend: http://localhost:3000")
    print("🔗 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)