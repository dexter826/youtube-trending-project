"""
Machine Learning Service
Description: AI predictions for YouTube trending analytics using scikit-learn
"""

import math
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from pymongo import MongoClient


class MLService:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "youtube_trending"):
        """Initialize ML service"""
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Try to load pre-trained models
        self.load_models()

    def load_models(self) -> bool:
        """Load pre-trained models from disk"""
        try:
            model_dir = "models/"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                return False
            
            model_files = {
                "trending_classifier": f"{model_dir}trending_classifier.joblib",
                "views_regressor": f"{model_dir}views_regressor.joblib", 
                "content_clusterer": f"{model_dir}content_clusterer.joblib",
                "scaler": f"{model_dir}feature_scaler.joblib"
            }
            
            loaded_count = 0
            for model_name, file_path in model_files.items():
                if os.path.exists(file_path):
                    try:
                        if model_name == "scaler":
                            self.scalers["main"] = joblib.load(file_path)
                        else:
                            self.models[model_name] = joblib.load(file_path)
                        loaded_count += 1
                        print(f"[OK] Loaded {model_name}")
                    except Exception as e:
                        print(f"[WARN] Could not load {model_name}: {str(e)}")
            
            self.is_trained = loaded_count >= 3  # At least 3 models
            
            if self.is_trained:
                print(f"[SUCCESS] Loaded {loaded_count} ML models")
                return True
            else:
                print("[INFO] No pre-trained models found, will use rule-based predictions")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to load models: {str(e)}")
            self.is_trained = False
            return False

    def train_models(self) -> bool:
        """Train models using MongoDB data"""
        try:
            print("🤖 Training ML models...")
            
            # Load training data from MongoDB
            data = list(self.db.ml_features.find({}, {"_id": 0}))
            if len(data) < 1000:
                print("[WARN] Insufficient training data")
                return False
            
            print(f"[OK] Loaded {len(data)} training samples")
            
            # Prepare features
            features = []
            trending_labels = []
            view_targets = []
            
            for record in data:
                # Extract features
                feature_row = [
                    record.get("likes", 0),
                    record.get("dislikes", 0), 
                    record.get("comment_count", 0),
                    record.get("like_ratio", 0),
                    record.get("engagement_score", 0),
                    record.get("title_length", 0),
                    record.get("tag_count", 0),
                    record.get("category_id", 0)
                ]
                
                features.append(feature_row)
                
                # Labels for classification (trending prediction)
                is_trending = 1 if record.get("views", 0) > 100000 else 0
                trending_labels.append(is_trending)
                
                # Targets for regression (view prediction)
                views = max(1, record.get("views", 1))
                log_views = math.log(views)
                view_targets.append(log_views)
            
            # Convert to numpy arrays
            X = np.array(features)
            y_trending = np.array(trending_labels)
            y_views = np.array(view_targets)
            
            # Feature scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers["main"] = scaler
            
            # Split data
            X_train, X_test, y_trend_train, y_trend_test = train_test_split(
                X_scaled, y_trending, test_size=0.2, random_state=42
            )
            _, _, y_views_train, y_views_test = train_test_split(
                X_scaled, y_views, test_size=0.2, random_state=42
            )
            
            # Train trending classifier
            trending_model = RandomForestClassifier(n_estimators=100, random_state=42)
            trending_model.fit(X_train, y_trend_train)
            trend_score = trending_model.score(X_test, y_trend_test)
            self.models["trending_classifier"] = trending_model
            print(f"[OK] Trending classifier accuracy: {trend_score:.4f}")
            
            # Train views regressor
            views_model = RandomForestRegressor(n_estimators=100, random_state=42)
            views_model.fit(X_train, y_views_train)
            views_score = views_model.score(X_test, y_views_test)
            self.models["views_regressor"] = views_model
            print(f"[OK] Views regressor R²: {views_score:.4f}")
            
            # Train content clusterer
            clusterer = KMeans(n_clusters=5, random_state=42)
            clusterer.fit(X_scaled)
            self.models["content_clusterer"] = clusterer
            print(f"[OK] Content clusterer trained with 5 clusters")
            
            # Save models
            self.save_models()
            self.is_trained = True
            
            print("✅ ML models trained successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Training failed: {str(e)}")
            return False

    def save_models(self):
        """Save trained models to disk"""
        try:
            model_dir = "models/"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save models
            for model_name, model in self.models.items():
                file_path = f"{model_dir}{model_name}.joblib"
                joblib.dump(model, file_path)
                print(f"[OK] Saved {model_name}")
            
            # Save scaler
            if "main" in self.scalers:
                joblib.dump(self.scalers["main"], f"{model_dir}feature_scaler.joblib")
                print(f"[OK] Saved feature scaler")
                
        except Exception as e:
            print(f"[ERROR] Failed to save models: {str(e)}")

    def predict_trending(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict if a video will be trending"""
        try:
            # Extract features
            features = self._extract_features(video_data)
            
            if self.is_trained and "trending_classifier" in self.models:
                # Use trained model
                X = np.array([features])
                if "main" in self.scalers:
                    X = self.scalers["main"].transform(X)
                
                model = self.models["trending_classifier"]
                prediction = model.predict(X)[0]
                probability = model.predict_proba(X)[0][1]  # Probability of trending
                
                return {
                    "trending_probability": float(probability),
                    "prediction": int(prediction),
                    "confidence": "high" if probability > 0.7 or probability < 0.3 else "medium",
                    "method": "scikit_learn"
                }
            else:
                # Fallback rule-based prediction
                views = video_data.get("views", 0)
                likes = video_data.get("likes", 0)
                engagement = likes / max(views, 1)
                
                trending_prob = min(0.9, engagement * 10 + 0.1)
                
                return {
                    "trending_probability": trending_prob,
                    "prediction": 1 if trending_prob > 0.5 else 0,
                    "confidence": "low",
                    "method": "rule_based"
                }
                
        except Exception as e:
            print(f"[ERROR] Trending prediction failed: {str(e)}")
            return {"trending_probability": 0.5, "confidence": "low", "method": "fallback"}

    def predict_views(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict view count for a video"""
        try:
            features = self._extract_features(video_data)
            
            if self.is_trained and "views_regressor" in self.models:
                # Use trained model
                X = np.array([features])
                if "main" in self.scalers:
                    X = self.scalers["main"].transform(X)
                
                model = self.models["views_regressor"]
                log_views = model.predict(X)[0]
                predicted_views = int(math.exp(log_views))
                
                return {
                    "predicted_views": max(0, predicted_views),
                    "confidence": "high",
                    "method": "scikit_learn"
                }
            else:
                # Fallback prediction
                likes = video_data.get("likes", 0)
                comments = video_data.get("comment_count", 0)
                estimated_views = (likes * 50) + (comments * 100)
                
                return {
                    "predicted_views": estimated_views,
                    "confidence": "low", 
                    "method": "rule_based"
                }
                
        except Exception as e:
            print(f"[ERROR] Views prediction failed: {str(e)}")
            return {"predicted_views": 1000, "confidence": "low", "method": "fallback"}

    def predict_cluster(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict content cluster for a video"""
        try:
            features = self._extract_features(video_data)
            
            if self.is_trained and "content_clusterer" in self.models:
                # Use trained model
                X = np.array([features])
                if "main" in self.scalers:
                    X = self.scalers["main"].transform(X)
                
                model = self.models["content_clusterer"]
                cluster = model.predict(X)[0]
                
                return {
                    "cluster": int(cluster),
                    "confidence": "high",
                    "method": "scikit_learn"
                }
            else:
                # Fallback clustering based on view ranges
                views = video_data.get("views", 0)
                if views < 10000:
                    cluster = 0  # Low engagement
                elif views < 100000:
                    cluster = 1  # Medium engagement
                elif views < 1000000:
                    cluster = 2  # High engagement
                elif views < 10000000:
                    cluster = 3  # Viral content
                else:
                    cluster = 4  # Mega viral
                    
                return {
                    "cluster": cluster,
                    "confidence": "medium",
                    "method": "rule_based"
                }
                
        except Exception as e:
            print(f"[ERROR] Clustering failed: {str(e)}")
            return {"cluster": 0, "confidence": "low", "method": "fallback"}

    def _extract_features(self, video_data: Dict[str, Any]) -> list:
        """Extract numerical features from video data"""
        return [
            video_data.get("likes", 0),
            video_data.get("dislikes", 0),
            video_data.get("comment_count", 0),
            video_data.get("likes", 0) / max(video_data.get("views", 1), 1),  # like_ratio
            (video_data.get("likes", 0) + video_data.get("comment_count", 0)) / max(video_data.get("views", 1), 1),  # engagement_score
            len(video_data.get("title", "")),  # title_length
            len(video_data.get("tags", "").split("|")) if video_data.get("tags") else 0,  # tag_count
            video_data.get("category_id", 0)
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "loaded_models": list(self.models.keys()),
            "is_trained": self.is_trained,
            "model_type": "scikit_learn",
            "scalers": list(self.scalers.keys()),
            "total_models": len(self.models)
        }


# Global ML service instance
_ml_service = None

def get_ml_service() -> MLService:
    """Get or create ML service instance"""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service

def initialize_ml_service():
    """Initialize ML service"""
    global _ml_service
    _ml_service = MLService()
    return _ml_service