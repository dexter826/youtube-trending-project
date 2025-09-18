"""
Prediction Service
"""

from typing import Dict, Any
from fastapi import HTTPException
import math


class Predictor:
    def __init__(self, model_loader, feature_processor):
        self.model_loader = model_loader
        self.feature_processor = feature_processor

    def predict_days(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict number of days a video may stay on trending."""
        try:
            model = self.model_loader.models.get("days_regressor")
            if model is not None:
                input_df = self.feature_processor.create_days_regression_dataframe(video_data)
                predictions = model.transform(input_df)
                result = predictions.select("prediction").collect()[0]
                predicted_days = max(0.0, float(result["prediction"]))
            else:
                # Heuristic fallback using engagement and scale
                views = max(float(video_data.get("views", 0)), 1.0)
                likes = float(video_data.get("likes", 0))
                comments = float(video_data.get("comment_count", 0))
                engagement = (likes + comments) / views
                # Simple scoring: base on log scale and engagement
                score = math.log1p(views) * (0.5 + engagement)
                predicted_days = min(10.0, max(0.0, score))

            return {
                "predicted_days": float(round(predicted_days, 2)),
                "confidence": "medium" if model is None else "high",
                "method": "spark_mllib" if model is not None else "heuristic"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Days prediction failed: {str(e)}")

    def predict_cluster(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict content cluster for a video"""
        try:
            if "content_clusterer" not in self.model_loader.models:
                raise HTTPException(status_code=503, detail="Content clusterer not available.")

            input_df = self.feature_processor.create_clustering_dataframe(video_data)

            model = self.model_loader.models["content_clusterer"]
            predictions = model.transform(input_df)

            result = predictions.select("cluster").collect()[0]
            cluster = int(result["cluster"])

            # Use dynamic cluster names
            cluster_type = self.model_loader.get_cluster_name(cluster)

            return {
                "cluster": cluster,
                "cluster_type": cluster_type,
                "method": "spark_mllib"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")