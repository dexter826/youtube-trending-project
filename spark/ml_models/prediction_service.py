"""
ML Model Inference for YouTube Trending Prediction
Author: BigData Expert
Description: Load trained models and make predictions
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

import pymongo
from pymongo import MongoClient

# MongoDB connection settings
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "youtube_trending"

class TrendingPredictionService:
    def __init__(self):
        """Initialize prediction service"""
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[DB_NAME]
        
        # Model components
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.label_encoders = {}
        self.feature_columns = []
        
        self.load_models()
        print("[OK] Prediction service initialized")

    def load_models(self):
        """Load trained models and preprocessors"""
        try:
            # Get model metadata from MongoDB
            metadata = self.db.ml_metadata.find_one({"type": "trending_prediction"})
            
            if not metadata:
                print("[ERROR] No trained models found! Train models first.")
                return False
            
            # Load models
            for model_name, model_info in metadata["models"].items():
                model_path = model_info["model_path"]
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    print(f"[OK] Loaded {model_name} model")
            
            # Load preprocessors
            preprocessors = metadata["preprocessors"]
            
            if os.path.exists(preprocessors["scaler_path"]):
                self.scaler = joblib.load(preprocessors["scaler_path"])
            
            if os.path.exists(preprocessors["imputer_path"]):
                self.imputer = joblib.load(preprocessors["imputer_path"])
            
            if os.path.exists(preprocessors["encoders_path"]):
                self.label_encoders = joblib.load(preprocessors["encoders_path"])
            
            self.feature_columns = metadata.get("feature_columns", [])
            
            print(f"[OK] Loaded {len(self.models)} models and preprocessors")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {str(e)}")
            return False

    def prepare_prediction_features(self, video_data):
        """Prepare features for a single video prediction"""
        try:
            # Convert to DataFrame if it's a dict
            if isinstance(video_data, dict):
                df = pd.DataFrame([video_data])
            else:
                df = video_data.copy()
            
            # Calculate engagement features
            df['like_ratio'] = np.where(df['views'] > 0, df['likes'] / df['views'], 0)
            df['dislike_ratio'] = np.where(df['views'] > 0, df['dislikes'] / df['views'], 0)
            df['comment_ratio'] = np.where(df['views'] > 0, df['comment_count'] / df['views'], 0)
            df['engagement_score'] = np.where(df['views'] > 0, 
                                            (df['likes'] + df['comment_count']) / df['views'], 0)
            
            # Content features
            df['title_length'] = df['title'].str.len()
            df['has_caps'] = df['title'].str.contains(r'[A-Z]{3,}', regex=True).astype(int)
            df['tag_count'] = df['tags'].str.count(r'\|') + 1
            df['tag_count'] = df['tag_count'].fillna(0)
            
            # Time features
            df['publish_hour'] = pd.to_datetime(df['publish_time']).dt.hour
            
            # Log features
            df['log_views'] = np.log1p(df['views'])
            df['log_likes'] = np.log1p(df['likes'])
            df['log_comments'] = np.log1p(df['comment_count'])
            
            # Default channel features (would be populated from historical data)
            df['channel_avg_views'] = df['views']  # Simplified
            df['channel_video_count'] = 1  # Simplified
            
            # Define feature order
            numerical_features = [
                'views', 'likes', 'dislikes', 'comment_count',
                'like_ratio', 'dislike_ratio', 'comment_ratio', 'engagement_score',
                'title_length', 'tag_count', 'publish_hour',
                'channel_avg_views', 'channel_video_count',
                'log_views', 'log_likes', 'log_comments'
            ]
            
            categorical_features = [
                'category_id', 'has_caps', 'comments_disabled', 'ratings_disabled'
            ]
            
            # Select and order features
            X_num = df[numerical_features].copy()
            X_cat = df[categorical_features].copy()
            
            # Handle missing values
            X_num = pd.DataFrame(
                self.imputer.transform(X_num),
                columns=numerical_features
            )
            
            # Encode categorical features
            for col in categorical_features:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    X_cat[col] = X_cat[col].astype(str)
                    mask = X_cat[col].isin(le.classes_)
                    X_cat.loc[~mask, col] = le.classes_[0]  # Use first class for unseen values
                    X_cat[col] = le.transform(X_cat[col])
                else:
                    X_cat[col] = 0  # Default value
            
            # Combine features
            X = pd.concat([X_num, X_cat], axis=1)
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
            
            return X
            
        except Exception as e:
            print(f"[ERROR] Feature preparation failed: {str(e)}")
            return None

    def predict_trending(self, video_data, model_name='random_forest'):
        """Predict if a video will be trending"""
        try:
            # Prepare features
            X = self.prepare_prediction_features(video_data)
            if X is None:
                return None
            
            # Check if model exists
            if model_name not in self.models:
                available_models = list(self.models.keys())
                if available_models:
                    model_name = available_models[0]
                    print(f"[WARN] Using {model_name} model instead")
                else:
                    print("[ERROR] No models available")
                    return None
            
            model = self.models[model_name]
            
            # Scale features if using logistic regression
            if model_name == 'logistic' and self.scaler:
                X = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=X.columns
                )
            
            # Make prediction
            prediction = model.predict(X)[0]
            prediction_proba = model.predict_proba(X)[0]
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                feature_importance = importance_df.head(5).to_dict('records')
            
            result = {
                'prediction': int(prediction),
                'trending_probability': float(prediction_proba[1]),
                'non_trending_probability': float(prediction_proba[0]),
                'confidence': float(max(prediction_proba)),
                'model_used': model_name,
                'prediction_time': datetime.now().isoformat(),
                'feature_importance': feature_importance
            }
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {str(e)}")
            return None

    def predict_batch(self, videos_data, model_name='random_forest'):
        """Predict trending for multiple videos"""
        try:
            results = []
            
            for i, video_data in enumerate(videos_data):
                result = self.predict_trending(video_data, model_name)
                if result:
                    result['video_index'] = i
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Batch prediction failed: {str(e)}")
            return []

    def get_model_info(self):
        """Get information about loaded models"""
        try:
            metadata = self.db.ml_metadata.find_one({"type": "trending_prediction"})
            
            if not metadata:
                return None
            
            model_info = {
                'available_models': list(self.models.keys()),
                'trained_at': metadata.get('created_at'),
                'models_performance': {}
            }
            
            for model_name, model_data in metadata.get('models', {}).items():
                if 'metrics' in model_data:
                    model_info['models_performance'][model_name] = {
                        'accuracy': model_data['metrics'].get('accuracy'),
                        'f1_score': model_data['metrics'].get('f1_score'),
                        'precision': model_data['metrics'].get('precision'),
                        'recall': model_data['metrics'].get('recall'),
                        'roc_auc': model_data['metrics'].get('roc_auc')
                    }
            
            return model_info
            
        except Exception as e:
            print(f"[ERROR] Failed to get model info: {str(e)}")
            return None

def main():
    """Test the prediction service"""
    service = TrendingPredictionService()
    
    # Test prediction with sample data
    sample_video = {
        'title': 'Amazing Video Tutorial - How to Code!',
        'views': 1000,
        'likes': 150,
        'dislikes': 5,
        'comment_count': 25,
        'category_id': 28,
        'publish_time': '2023-01-01T12:00:00Z',
        'tags': 'tutorial|coding|programming',
        'comments_disabled': False,
        'ratings_disabled': False
    }
    
    result = service.predict_trending(sample_video)
    
    if result:
        print("\n=== PREDICTION RESULT ===")
        print(f"Trending Prediction: {'YES' if result['prediction'] == 1 else 'NO'}")
        print(f"Trending Probability: {result['trending_probability']:.2%}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Model Used: {result['model_used']}")
    
    # Get model info
    info = service.get_model_info()
    if info:
        print(f"\n=== MODEL INFO ===")
        print(f"Available Models: {info['available_models']}")
        print(f"Trained At: {info['trained_at']}")

if __name__ == "__main__":
    main()
