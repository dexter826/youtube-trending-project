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

    def predict_trending(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict if a video will be trending"""
        try:
            print(f"DEBUG: Starting predict_trending with video_data: {video_data}")
            
            if not self.model_loader.is_trained or "trending_classifier" not in self.model_loader.models:
                print("DEBUG: Trending classifier not available. Train models first.")
                raise HTTPException(status_code=503, detail="Trending classifier not available. Train models first.")

            input_df = self.feature_processor.create_trending_dataframe(video_data)
            print(f"DEBUG: Created dataframe: {input_df.count()} rows")

            model = self.model_loader.models["trending_classifier"]
            print(f"DEBUG: Model type: {type(model)}")
            print(f"DEBUG: Model is None: {model is None}")
            
            predictions = model.transform(input_df)
            print("DEBUG: Model transform completed")

            result = predictions.select("prediction", "probability").collect()[0]
            print(f"DEBUG: Result: {result}")
            
            prediction = int(result["prediction"])
            probability = float(result["probability"][1])

            # Adjust threshold for imbalanced data
            if probability > 0.1:  # Lower threshold for trending prediction
                prediction = 1
            else:
                prediction = 0

            return {
                "trending_probability": probability,
                "prediction": prediction,
                "confidence": "high" if probability > 0.7 or probability < 0.3 else "medium",
                "method": "spark_mllib"
            }

        except Exception as e:
            print(f"DEBUG: Error in predict_trending: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    def predict_views(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict view count for a video"""
        try:
            if not self.model_loader.is_trained or "views_regressor" not in self.model_loader.models:
                raise HTTPException(status_code=503, detail="Views regressor not available. Train models first.")

            input_df = self.feature_processor.create_regression_dataframe(video_data)

            model = self.model_loader.models["views_regressor"]
            predictions = model.transform(input_df)

            result = predictions.select("prediction").collect()[0]
            predicted_log_views = float(result["prediction"])
            predicted_views = int(max(0, math.exp(predicted_log_views) - 1))

            return {
                "predicted_views": predicted_views,
                "confidence": "high",
                "method": "spark_mllib"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    def predict_cluster(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict content cluster for a video"""
        try:
            if not self.model_loader.is_trained or "content_clusterer" not in self.model_loader.models:
                raise HTTPException(status_code=503, detail="Content clusterer not available. Train models first.")

            input_df = self.feature_processor.create_clustering_dataframe(video_data)

            model = self.model_loader.models["content_clusterer"]
            predictions = model.transform(input_df)

            result = predictions.select("cluster").collect()[0]
            cluster = int(result["cluster"])

            # Use dynamic cluster names instead of hard-coded
            cluster_type = self.model_loader.get_cluster_name(cluster)

            return {
                "cluster": cluster,
                "cluster_type": cluster_type,
                "method": "spark_mllib"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")