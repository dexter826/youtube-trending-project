#!/usr/bin/env python3
"""
YouTube Trending Project Runner
Start backend, frontend, infrastructure, and data pipeline
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.paths import path_config


class ProjectRunner:
    """Project runner for different modes"""

    def __init__(self):
        self.project_root = path_config.PROJECT_ROOT

    def run_command(self, cmd, description="", cwd=None, check=True):
        """Run shell command with proper error handling"""
        try:
            print(f"🔄 {description}")
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd or self.project_root
            )

            if check and result.returncode != 0:
                print(f"❌ Failed: {description}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False

            if result.stdout and not result.stdout.startswith("WARNING"):
                print(f"✅ {description}")
            return True

        except Exception as e:
            print(f"❌ Error running command: {e}")
            return False

    def start_infrastructure(self):
        """Start infrastructure services (HDFS, MongoDB)"""
        print("🏗️  Starting infrastructure services...")

        # Start HDFS
        hadoop_home = os.getenv("HADOOP_HOME", "C:\\hadoop-3.4.1")
        hdfs_cmd = f"{hadoop_home}\\sbin\\start-all.cmd"

        if not self.run_command(hdfs_cmd, "Starting HDFS services"):
            print("⚠️  HDFS start failed - check HADOOP_HOME environment variable")
            return False

        # Start MongoDB
        mongo_data_path = os.getenv("MONGO_DATA_PATH", "C:\\data\\db")
        mongo_cmd = f'start "MongoDB" cmd /k "mongod --dbpath {mongo_data_path}"'

        if not self.run_command(mongo_cmd, "Starting MongoDB"):
            print("⚠️  MongoDB start failed - check MONGO_DATA_PATH environment variable")
            return False

        # Wait for services to start
        print("⏳ Waiting for services to initialize...")
        time.sleep(10)

        return True

    def run_pipeline(self):
        """Run the complete data processing pipeline"""
        print("🔄 Running data processing pipeline...")

        # Create HDFS directories
        directories = [
            "/youtube_trending",
            "/youtube_trending/raw_data",
            "/youtube_trending/models"
        ]

        for directory in directories:
            self.run_command(
                f"hdfs dfs -mkdir -p {directory}",
                f"Creating HDFS directory: {directory}"
            )

        # Copy data to HDFS
        countries = ['US', 'CA', 'GB', 'DE', 'FR', 'IN', 'JP', 'KR', 'MX', 'RU']
        data_dir = str(path_config.DATA_DIR)

        for country in countries:
            csv_file = f"{data_dir}\\{country}videos.csv"
            category_file = f"{data_dir}\\{country}_category_id.json"

            if os.path.exists(csv_file):
                hdfs_country_dir = f"/youtube_trending/raw_data/{country}"
                self.run_command(
                    f"hdfs dfs -mkdir -p {hdfs_country_dir}",
                    f"Creating {country} directory"
                )

                hdfs_csv_path = f"{hdfs_country_dir}/{country}videos.csv"
                self.run_command(
                    f"hdfs dfs -put -f {csv_file} {hdfs_csv_path}",
                    f"Copying {country} videos CSV"
                )

                if os.path.exists(category_file):
                    hdfs_cat_path = f"{hdfs_country_dir}/{country}_category_id.json"
                    self.run_command(
                        f"hdfs dfs -put -f {category_file} {hdfs_cat_path}",
                        f"Copying {country} categories JSON"
                    )

        # Run Spark data processing
        spark_script = str(path_config.SPARK_PROCESS_SCRIPT)
        fallback_data = str(path_config.DATA_DIR)
        cmd = f"spark-submit {spark_script} {fallback_data}"

        if not self.run_command(cmd, "Running Spark data processing"):
            return False

        # Train ML models
        training_script = str(path_config.SPARK_TRAIN_SCRIPT)
        if not self.run_command(
            f"spark-submit {training_script}",
            "Training ML models"
        ):
            return False

        print("✅ Pipeline completed successfully!")
        return True

    def start_app(self):
        """Start backend and frontend services"""
        print("🌐 Starting application services...")

        # Start backend API in background
        backend_dir = str(path_config.BACKEND_DIR)
        api_host = os.getenv("API_HOST", "0.0.0.0")
        api_port = os.getenv("API_PORT", "8000")
        backend_cmd = f"cd {backend_dir} && python -m uvicorn app.main:app --reload --host {api_host} --port {api_port}"

        try:
            print("🔄 Starting backend API...")
            backend_process = subprocess.Popen(
                backend_cmd, 
                shell=True, 
                cwd=backend_dir,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            # Wait for backend to be ready
            print("⏳ Waiting for backend to be ready...")
            if not self._wait_for_backend_ready(api_port, timeout=30):
                print("❌ Backend failed to become ready within timeout")
                return False
                
            print(f"✅ Backend API ready on http://{api_host}:{api_port}")
                
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False

        # Start frontend only after backend is confirmed ready
        frontend_dir = str(path_config.FRONTEND_DIR)
        frontend_port = os.getenv("FRONTEND_PORT", "3000")
        frontend_cmd = f"cd {frontend_dir} && npm start"

        try:
            print("🔄 Starting frontend...")
            frontend_process = subprocess.Popen(
                frontend_cmd, 
                shell=True, 
                cwd=frontend_dir,
                env=dict(os.environ, PORT=frontend_port),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            # Give frontend time to start
            time.sleep(5)
            print(f"✅ Frontend started on http://localhost:{frontend_port}")
                
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False

        print("\n🎉 Services started successfully!")
        print("💡 Services are running in background. Use 'python run.py status' to check status.")
        print("💡 Press Ctrl+C in their respective terminals to stop services.")
        
        return True

    def _wait_for_backend_ready(self, port, timeout=30):
        """Wait for backend to be ready by checking health endpoint"""
        import requests
        import time
        
        start_time = time.time()
        health_url = f"http://localhost:{port}/docs"  # Use /docs as health check
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print(".", end="", flush=True)
            time.sleep(1)
        
        print()  # New line after dots
        return False

    def show_status(self):
        """Show current status of services"""
        print("📊 Service Status:")

        # Check HDFS
        if self.run_command("hdfs dfs -ls /", "Checking HDFS", check=False):
            print("✅ HDFS: Running")
        else:
            print("❌ HDFS: Not running")

        # Check MongoDB
        try:
            import pymongo
            mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
            client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=1000)
            client.admin.command('ping')
            print("✅ MongoDB: Running")
        except:
            print("❌ MongoDB: Not running")

        # Check backend
        api_port = os.getenv("API_PORT", "8000")
        if self.run_command(f"curl -s http://localhost:{api_port}/docs >nul 2>&1", "Checking backend API", check=False):
            print(f"✅ Backend API: Running (http://localhost:{api_port})")
        else:
            print(f"❌ Backend API: Not running")

        # Check frontend
        frontend_port = os.getenv("FRONTEND_PORT", "3000")
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', int(frontend_port)))
        sock.close()
        if result == 0:
            print(f"✅ Frontend: Running (http://localhost:{frontend_port})")
        else:
            print(f"❌ Frontend: Not running")


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Trending Project Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py app              # Start backend + frontend
  python run.py infrastructure   # Start HDFS + MongoDB
  python run.py pipeline         # Run data processing pipeline
  python run.py all              # Start everything (infrastructure + pipeline + app)
  python run.py status           # Show service status
        """
    )

    parser.add_argument(
        'mode',
        choices=['app', 'infrastructure', 'pipeline', 'all', 'status'],
        help='Mode to run'
    )

    args = parser.parse_args()

    runner = ProjectRunner()

    if args.mode == 'infrastructure':
        success = runner.start_infrastructure()
    elif args.mode == 'pipeline':
        success = runner.run_pipeline()
    elif args.mode == 'app':
        success = runner.start_app()
    elif args.mode == 'all':
        # Full stack: infrastructure -> pipeline -> app
        if runner.start_infrastructure():
            if runner.run_pipeline():
                success = runner.start_app()
            else:
                success = False
        else:
            success = False
    elif args.mode == 'status':
        runner.show_status()
        success = True

    if args.mode != 'status':
        if success:
            print("\n🎉 Success!")
            if args.mode in ['app', 'all']:
                api_port = os.getenv("API_PORT", "8000")
                frontend_port = os.getenv("FRONTEND_PORT", "3000")
                print(f"🌐 Frontend: http://localhost:{frontend_port}")
                print(f"🔌 Backend API: http://localhost:{api_port}")
                print(f"📊 API Docs: http://localhost:{api_port}/docs")
                print("🗂️  HDFS: http://localhost:9870")
        else:
            print("\n❌ Command failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()