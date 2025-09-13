#!/usr/bin/env python3
"""
YouTube Trending Project Setup Script
One-click setup for the entire development environment
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.paths import path_config


class ProjectSetup:
    """Complete project setup automation"""

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

    def check_prerequisites(self):
        """Check if required tools are installed"""
        print("🔍 Checking prerequisites...")

        checks = [
            ("python --version", "Python 3.8+"),
            ("java -version", "Java 8/11"),
            ("node --version", "Node.js 16+"),
            ("npm --version", "npm"),
            ("hadoop version", "Hadoop/HDFS"),
            ("spark-submit --version", "Apache Spark"),
        ]

        missing = []
        for cmd, name in checks:
            if not self.run_command(cmd, f"Checking {name}", check=False):
                missing.append(name)

        if missing:
            print(f"⚠️  Missing prerequisites: {', '.join(missing)}")
            print("\n📋 Please install missing tools:")
            print("- Python: https://python.org")
            print("- Java: https://adoptium.net")
            print("- Node.js: https://nodejs.org")
            print("- Hadoop: https://hadoop.apache.org")
            print("- Spark: https://spark.apache.org")
            return False

        print("✅ All prerequisites found!")
        return True

    def setup_python_environment(self):
        """Setup Python dependencies"""
        print("🐍 Setting up Python environment...")

        # Backend dependencies
        backend_req = path_config.BACKEND_DIR / "requirements.txt"
        if backend_req.exists():
            if not self.run_command(
                f"pip install -r {backend_req}",
                "Installing backend dependencies"
            ):
                return False

        # Spark dependencies
        spark_req = path_config.SPARK_DIR / "requirements.txt"
        if spark_req.exists():
            if not self.run_command(
                f"pip install -r {spark_req}",
                "Installing Spark dependencies"
            ):
                return False

        return True

    def setup_frontend_environment(self):
        """Setup frontend dependencies"""
        print("⚛️  Setting up frontend environment...")

        frontend_dir = path_config.FRONTEND_DIR
        if not frontend_dir.exists():
            print(f"❌ Frontend directory not found: {frontend_dir}")
            return False

        # Install npm dependencies
        if not self.run_command(
            "npm install",
            "Installing frontend dependencies",
            cwd=frontend_dir
        ):
            return False

        return True

    def setup_infrastructure(self):
        """Setup infrastructure configuration"""
        print("🏗️  Setting up infrastructure...")

        # Ensure all directories exist
        path_config.ensure_paths_exist()

        # Validate project structure
        try:
            path_config.validate_project_structure()
            print("✅ Project structure validated")
        except FileNotFoundError as e:
            print(f"⚠️  Missing files: {e}")
            print("Some features may not work until files are created.")

        return True

    def create_environment_file(self):
        """Create .env file with default configuration"""
        print("📝 Creating environment configuration...")

        env_content = """# YouTube Trending Project Environment Configuration

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/
DB_NAME=youtube_trending

# HDFS Configuration
HDFS_BASE_PATH=hdfs://localhost:9000/youtube_trending

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Frontend Configuration
FRONTEND_PORT=3000

# Hadoop Home (Windows)
HADOOP_HOME=C:\\hadoop-3.4.1

# MongoDB Data Path (Windows)
MONGO_DATA_PATH=C:\\data\\db
"""

        env_file = self.project_root / ".env"
        if not env_file.exists():
            with open(env_file, 'w') as f:
                f.write(env_content)
            print("✅ Created .env file with default configuration")
        else:
            print("ℹ️  .env file already exists")

        return True

    def run_initial_pipeline(self):
        """Run initial data pipeline setup"""
        print("🔄 Running initial pipeline setup...")

        # This is optional - user can run this separately with run.py
        print("ℹ️  Skipping initial pipeline. Use 'python run.py pipeline' to run it later.")
        return True

    def show_completion_message(self):
        """Show setup completion message"""
        print("\n🎉 Setup Complete!")
        print("\n📋 Next Steps:")
        print("1. Start infrastructure: python run.py infrastructure")
        print("2. Run data pipeline: python run.py pipeline")
        print("3. Start application: python run.py app")
        print("\n🌐 Access URLs:")
        print("- Frontend: http://localhost:3000")
        print("- Backend API: http://localhost:8000")
        print("- API Docs: http://localhost:8000/docs")


def main():
    print("🚀 YouTube Trending Project Setup")
    print("=" * 40)

    setup = ProjectSetup()

    # Check prerequisites
    if not setup.check_prerequisites():
        print("\n❌ Setup failed - missing prerequisites")
        sys.exit(1)

    # Setup Python environment
    if not setup.setup_python_environment():
        print("\n❌ Setup failed - Python environment setup failed")
        sys.exit(1)

    # Setup frontend environment
    if not setup.setup_frontend_environment():
        print("\n❌ Setup failed - Frontend environment setup failed")
        sys.exit(1)

    # Setup infrastructure
    if not setup.setup_infrastructure():
        print("\n❌ Setup failed - Infrastructure setup failed")
        sys.exit(1)

    # Create environment file
    setup.create_environment_file()

    # Show completion message
    setup.show_completion_message()


if __name__ == "__main__":
    main()