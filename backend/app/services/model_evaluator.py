"""
Model Evaluation and Training Service
"""

from typing import Dict, Any
from fastapi import HTTPException
import subprocess
import os
import shutil
import sys

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class ModelEvaluator:
    def __init__(self, model_loader, db_connection):
        self.model_loader = model_loader
        self.db = db_connection

    def train_models(self) -> bool:
        """Trigger model training using Spark job"""
        try:
            # Check training data availability
            training_data_count = self.db.ml_features.count_documents({})
            if training_data_count < 1000:
                raise HTTPException(status_code=400, detail=f"Insufficient training data: {training_data_count} records")

            # Path to training script - use absolute path from project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            script_path = os.path.join(project_root, 'spark', 'train_models.py')

            # Check if script exists
            if not os.path.exists(script_path):
                raise HTTPException(status_code=500, detail=f"Training script not found at: {script_path}")

            # Check if spark-submit is available
            spark_submit_path = shutil.which('spark-submit')
            if not spark_submit_path:
                raise HTTPException(status_code=500, detail="spark-submit not found in PATH. Please ensure Apache Spark is installed and in PATH.")

            # Run training with spark-submit
            cmd = f"spark-submit {script_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=project_root)

            if result.returncode == 0:
                # Reload models after training
                self.model_loader.load_models_from_hdfs()
                return True
            else:
                raise HTTPException(status_code=500, detail=f"Training failed: {result.stderr}")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded Spark MLlib models"""
        model_details = {}
        for name, model in self.model_loader.models.items():
            model_details[name] = {
                "stages": len(model.stages) if hasattr(model, 'stages') else 0,
                "type": "PipelineModel"
            }

        return {
            "loaded_models": list(self.model_loader.models.keys()),
            "is_trained": self.model_loader.is_trained,
            "model_type": "spark_mllib",
            "framework": "Apache Spark MLlib",
            "storage": "HDFS",
            "model_details": model_details,
            "total_models": len(self.model_loader.models),
            "spark_session": True,  # Assuming spark is available
            "metrics": self.model_loader.metrics
        }