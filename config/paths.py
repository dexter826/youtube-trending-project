"""
Configuration management for YouTube Trending Project
"""

import os
from pathlib import Path


class PathConfig:
    """Centralized path configuration"""

    # Project root - dynamically determined
    PROJECT_ROOT = Path(__file__).parent.parent

    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    SPARK_DIR = PROJECT_ROOT / "spark"
    BACKEND_DIR = PROJECT_ROOT / "backend"
    FRONTEND_DIR = PROJECT_ROOT / "frontend"

    # Spark subdirectories
    SPARK_CORE_DIR = SPARK_DIR / "core"
    SPARK_JOBS_DIR = SPARK_DIR / "jobs"
    SPARK_ANALYSIS_DIR = SPARK_DIR / "analysis"
    SPARK_MODELS_DIR = SPARK_DIR / "models"
    SPARK_METRICS_DIR = SPARK_MODELS_DIR / "metrics"

    # Backend subdirectories
    BACKEND_APP_DIR = BACKEND_DIR / "app"

    # Script files
    SPARK_PROCESS_SCRIPT = SPARK_JOBS_DIR / "process_trending.py"
    SPARK_TRAIN_SCRIPT = SPARK_DIR / "train_models.py"
    SPARK_ANALYZE_CLUSTERS_SCRIPT = SPARK_ANALYSIS_DIR / "analyze_clusters.py"

    # Config files
    SPARK_MANAGER = SPARK_CORE_DIR / "spark_manager.py"
    DATABASE_MANAGER = SPARK_CORE_DIR / "database_manager.py"

    @classmethod
    def ensure_paths_exist(cls):
        """Ensure all required directories exist"""
        dirs_to_check = [
            cls.DATA_DIR,
            cls.SPARK_DIR,
            cls.BACKEND_DIR,
            cls.FRONTEND_DIR,
            cls.SPARK_CORE_DIR,
            cls.SPARK_JOBS_DIR,
            cls.BACKEND_APP_DIR
        ]

        for dir_path in dirs_to_check:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_data_files(cls, pattern="*.csv"):
        """Get all data files matching pattern"""
        if cls.DATA_DIR.exists():
            return list(cls.DATA_DIR.glob(pattern))
        return []

    @classmethod
    def validate_project_structure(cls):
        """Validate that project structure is correct"""
        required_files = [
            cls.SPARK_PROCESS_SCRIPT,
            cls.SPARK_TRAIN_SCRIPT,
            cls.SPARK_ANALYZE_CLUSTERS_SCRIPT,
            cls.SPARK_MANAGER,
            cls.DATABASE_MANAGER
        ]

        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")

        return True


# Global instance
path_config = PathConfig()