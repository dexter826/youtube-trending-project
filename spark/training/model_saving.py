"""
Model Saving Module
"""

import json
from datetime import datetime
import os


class ModelSaving:
    def __init__(self, hdfs_base_path):
        self.hdfs_base_path = hdfs_base_path
        self.models_path = f"{self.hdfs_base_path}/models"
        
        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from config.paths import path_config
        
        self.local_metrics_dir = path_config.SPARK_METRICS_DIR
        os.makedirs(self.local_metrics_dir, exist_ok=True)

    def save_model(self, model, model_type):
        """Save trained model to HDFS"""
        model_path = f"{self.models_path}/{model_type}"
        model.write().overwrite().save(model_path)
        return model_path

    def save_metrics_to_json(self, trending_metrics, clustering_metrics, regression_metrics, dataset_size):
        """Save all model metrics to JSON file"""
        all_metrics = {
            "trending": trending_metrics,
            "clustering": clustering_metrics,
            "regression": regression_metrics,
            "training_timestamp": datetime.now().isoformat(),
            "dataset_size": dataset_size
        }

        # Save to local file in organized directory
        local_metrics_path = self.local_metrics_dir / "model_metrics.json"
        with open(local_metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        return str(local_metrics_path)

    def save_metrics_to_mongodb(self, db, trending_metrics, clustering_metrics, regression_metrics, dataset_size):
        """Save metrics to MongoDB"""
        metrics_doc = {
            "trending": trending_metrics,
            "clustering": clustering_metrics,
            "regression": regression_metrics,
            "training_timestamp": datetime.now().isoformat(),
            "dataset_size": dataset_size
        }

        # Save to model_metrics collection
        db.model_metrics.insert_one(metrics_doc)
        return True