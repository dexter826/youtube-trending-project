"""
YouTube Trending Big Data Pipeline
"""

import os
import sys
import subprocess
import time
from config.paths import path_config

def run_command(cmd, description):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def check_hdfs_directories():
    directories = [
        "/youtube_trending",
        "/youtube_trending/raw_data",
        "/youtube_trending/models"
    ]
    
    for directory in directories:
        cmd = f"hdfs dfs -mkdir -p {directory}"
        run_command(cmd, f"Creating HDFS directory: {directory}")

def copy_data_to_hdfs():
    data_dir = str(path_config.DATA_DIR)
    countries = ['US', 'CA', 'GB', 'DE', 'FR', 'IN', 'JP', 'KR', 'MX', 'RU']
    
    for country in countries:
        csv_file = f"{data_dir}\\{country}videos.csv"
        category_file = f"{data_dir}\\{country}_category_id.json"
        
        if os.path.exists(csv_file):
            hdfs_country_dir = f"/youtube_trending/raw_data/{country}"
            run_command(f"hdfs dfs -mkdir -p {hdfs_country_dir}", f"Creating {country} directory")
            
            hdfs_csv_path = f"{hdfs_country_dir}/{country}videos.csv"
            run_command(f"hdfs dfs -put -f {csv_file} {hdfs_csv_path}", f"Copying {country} videos CSV")
            
            if os.path.exists(category_file):
                hdfs_cat_path = f"{hdfs_country_dir}/{country}_category_id.json"
                run_command(f"hdfs dfs -put -f {category_file} {hdfs_cat_path}", f"Copying {country} categories JSON")

def run_spark_data_processing():
    spark_script = str(path_config.SPARK_PROCESS_SCRIPT)
    fallback_data = str(path_config.DATA_DIR)
    
    cmd = f"spark-submit {spark_script} {fallback_data}"
    return run_command(cmd, "Spark Data Processing Job")

def run_model_training():
    training_script = str(path_config.SPARK_TRAIN_SCRIPT)
    cmd = f"spark-submit {training_script}"
    return run_command(cmd, "ML Model Training Job")

def start_backend_api():
    backend_dir = str(path_config.BACKEND_DIR)
    cmd = f"cd {backend_dir} && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    
    try:
        subprocess.Popen(cmd, shell=True, cwd=backend_dir)
        time.sleep(3)
        return True
    except Exception:
        return False

def start_frontend():
    frontend_dir = str(path_config.FRONTEND_DIR)
    cmd = f"cd {frontend_dir} && npm install && npm start"
    
    try:
        subprocess.Popen(cmd, shell=True, cwd=frontend_dir)
        time.sleep(5)
        return True
    except Exception:
        return False

def main():
    # Setup HDFS
    check_hdfs_directories()
    
    # Copy data to HDFS
    copy_data_to_hdfs()
    
    # Run Spark data processing
    if not run_spark_data_processing():
        return False
    
    # Train ML models
    if not run_model_training():
        return False
    
    # Start backend API
    if not start_backend_api():
        return False
    
    # Start frontend
    if not start_frontend():
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)