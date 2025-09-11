"""
ML Service for FastAPI Backend
Author: BigData Expert
Description: ML prediction service integrated with FastAPI
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

import pymongo
from pymongo import MongoClient

class MLPredictionService:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "youtube_trending"):
        """Initialize ML prediction service for FastAPI"""
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        
        # Model components
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.label_encoders = {}
        self.feature_columns = []
        self.is_loaded = False
        
        # Try to load models
        self.load_models()

    def load_models(self) -> bool:
        """Load trained models and preprocessors"""
        try:
            # Get model metadata from MongoDB
            metadata = self.db.ml_metadata.find_one({"type": "trending_prediction"})
            
            if not metadata:
                print("[WARN] No trained ML models found. Train models first.")
                return False
            
            # Load single model (updated structure)
            # Resolve models directory: ENV MODELS_DIR or default to <repo_root>/spark/saved_models
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            models_dir = os.getenv("MODELS_DIR", os.path.join(repo_root, "spark", "saved_models"))
            models_loaded = 0
            model_path = os.path.join(models_dir, "trending_predictor.pkl")
            
            if os.path.exists(model_path):
                self.models["logistic"] = joblib.load(model_path)
                models_loaded = 1
                print(f"[OK] Loaded model: {model_path}")
            else:
                print(f"[ERROR] Model file not found: {model_path}")
                return False
            
            # Load preprocessors with correct paths
            scaler_path = os.path.join(models_dir, "scaler.pkl")
            imputer_path = os.path.join(models_dir, "imputer.pkl")
            encoders_path = os.path.join(models_dir, "label_encoders.pkl")
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"[OK] Loaded scaler: {scaler_path}")
            
            if os.path.exists(imputer_path):
                self.imputer = joblib.load(imputer_path)
                print(f"[OK] Loaded imputer: {imputer_path}")
            
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
                print(f"[OK] Loaded encoders: {encoders_path}")
            
            if models_loaded == 0:
                print("[WARN] No model files found on disk")
                return False
            
            self.feature_columns = metadata.get("feature_columns", [])
            self.is_loaded = True
            
            print(f"[OK] ML Service loaded {models_loaded} models")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load ML models: {str(e)}")
            return False

    def prepare_features(self, video_data: Dict) -> Optional[pd.DataFrame]:
        """Prepare features for prediction"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([video_data])
            
            # Ensure required fields exist with defaults
            required_fields = {
                'views': 0,
                'likes': 0,
                'dislikes': 0,
                'comment_count': 0,
                'title': '',
                'tags': '',
                'category_id': 1,
                'publish_time': datetime.now().isoformat(),
                'comments_disabled': False,
                'ratings_disabled': False
            }
            
            for field, default_value in required_fields.items():
                if field not in df.columns or df[field].isna().any():
                    df[field] = default_value
            
            # Calculate engagement features
            df['like_ratio'] = np.where(df['views'] > 0, df['likes'] / df['views'], 0)
            df['dislike_ratio'] = np.where(df['views'] > 0, df['dislikes'] / df['views'], 0)
            df['comment_ratio'] = np.where(df['views'] > 0, df['comment_count'] / df['views'], 0)
            df['engagement_score'] = np.where(df['views'] > 0, 
                                            (df['likes'] + df['dislikes'] + df['comment_count']) / df['views'], 0)
            
            # Content features
            df['title_length'] = df['title'].astype(str).str.len()
            df['has_caps'] = df['title'].astype(str).str.contains(r'[A-Z]{3,}', regex=True, na=False).astype(int)
            
            # Tag features
            df['tag_count'] = df['tags'].astype(str).str.count(r'\|') + 1
            df['tag_count'] = np.where(df['tags'].astype(str) == '', 0, df['tag_count'])
            
            # Time features
            try:
                df['publish_hour'] = pd.to_datetime(df['publish_time']).dt.hour
            except:
                df['publish_hour'] = 12  # Default to noon
            
            # Log features
            df['log_views'] = np.log1p(df['views'])
            df['log_likes'] = np.log1p(df['likes'])
            df['log_comments'] = np.log1p(df['comment_count'])
            
            # Channel features (simplified for demo)
            df['channel_avg_views'] = df['views']
            df['channel_video_count'] = 1
            
            # Define feature order (must match training)
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
            
            # Select features
            X_num = df[numerical_features].copy()
            X_cat = df[categorical_features].copy()
            
            # Handle missing values in numerical features
            if self.imputer:
                X_num = pd.DataFrame(
                    self.imputer.transform(X_num),
                    columns=numerical_features
                )
            else:
                X_num = X_num.fillna(0)
            
            # Encode categorical features
            for col in categorical_features:
                if col in self.label_encoders and self.label_encoders[col]:
                    le = self.label_encoders[col]
                    X_cat[col] = X_cat[col].astype(str)
                    
                    # Handle unseen categories
                    mask = X_cat[col].isin(le.classes_)
                    if not mask.all():
                        X_cat.loc[~mask, col] = le.classes_[0]
                    
                    X_cat[col] = le.transform(X_cat[col])
                else:
                    # Convert boolean to int, category_id to int
                    if col in ['comments_disabled', 'ratings_disabled']:
                        X_cat[col] = X_cat[col].astype(bool).astype(int)
                    else:
                        X_cat[col] = pd.to_numeric(X_cat[col], errors='coerce').fillna(0).astype(int)
            
            # Combine features
            X = pd.concat([X_num, X_cat], axis=1)
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
            
            return X
            
        except Exception as e:
            print(f"[ERROR] Feature preparation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def predict_trending(self, video_data: Dict, model_name: str = 'logistic') -> Optional[Dict]:
        """Predict if a video will be trending"""
        if not self.is_loaded:
            return {
                'error': 'ML models not loaded. Please train models first.',
                'success': False
            }
        
        try:
            # Prepare features
            X = self.prepare_features(video_data)
            if X is None:
                return {
                    'error': 'Failed to prepare features',
                    'success': False
                }
            
            # Use available model if specified model doesn't exist
            if model_name not in self.models:
                available_models = list(self.models.keys())
                if available_models:
                    model_name = available_models[0]
                else:
                    return {
                        'error': 'No models available',
                        'success': False
                    }
            
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
            
            # Create interpretation
            trending_probability = float(prediction_proba[1])
            confidence_level = "High" if max(prediction_proba) > 0.8 else "Medium" if max(prediction_proba) > 0.6 else "Low"
            
            # Generate recommendation
            recommendation = self._generate_recommendation(video_data, trending_probability)
            
            result = {
                'success': True,
                'prediction': {
                    'is_trending': bool(prediction),
                    'trending_probability': trending_probability,
                    'non_trending_probability': float(prediction_proba[0]),
                    'confidence': float(max(prediction_proba)),
                    'confidence_level': confidence_level
                },
                'model_info': {
                    'model_used': model_name,
                    'prediction_time': datetime.now().isoformat()
                },
                'feature_importance': feature_importance,
                'recommendation': recommendation,
                'input_summary': {
                    'views': video_data.get('views', 0),
                    'likes': video_data.get('likes', 0),
                    'dislikes': video_data.get('dislikes', 0),
                    'comments': video_data.get('comment_count', 0),
                    'total_engagement': video_data.get('likes', 0) + video_data.get('dislikes', 0) + video_data.get('comment_count', 0),
                    'engagement_rate': round((video_data.get('likes', 0) + video_data.get('dislikes', 0) + video_data.get('comment_count', 0)) / max(video_data.get('views', 1), 1) * 100, 2)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'error': f'Prediction failed: {str(e)}',
                'success': False
            }

    def _generate_recommendation(self, video_data: Dict, trending_probability: float) -> Dict:
        """Generate actionable recommendations based on prediction"""
        recommendations = []
        
        # Analyze current metrics
        views = video_data.get('views', 0)
        likes = video_data.get('likes', 0)
        comments = video_data.get('comment_count', 0)
        
        engagement_rate = likes / max(views, 1)
        comment_rate = comments / max(views, 1)
        
        if trending_probability < 0.3:
            recommendations.append("❌ Khả năng trending thấp. Cần cải thiện content quality.")
            
            if engagement_rate < 0.02:
                recommendations.append("📈 Tăng engagement: Tạo content hấp dẫn hơn để có nhiều likes")
            
            if comment_rate < 0.01:
                recommendations.append("💬 Khuyến khích tương tác: Đặt câu hỏi hoặc call-to-action")
            
            if len(video_data.get('title', '')) < 50:
                recommendations.append("📝 Tối ưu title: Làm title dài và mô tả hơn")
                
        elif trending_probability < 0.7:
            recommendations.append("⚡ Khả năng trending trung bình. Một vài cải thiện có thể giúp!")
            
            if engagement_rate < 0.05:
                recommendations.append("👍 Tăng likes: Nhắc nhở viewers like video")
            
            recommendations.append("⏰ Timing: Thử post vào giờ peak (19-21h)")
            
        else:
            recommendations.append("🚀 Khả năng trending cao! Content có tiềm năng viral.")
            recommendations.append("📊 Theo dõi metrics: Monitor performance sau publish")
            recommendations.append("🔥 Promote: Share trên social media để boost ban đầu")
        
        return {
            'trending_probability_assessment': 'High' if trending_probability > 0.7 else 'Medium' if trending_probability > 0.3 else 'Low',
            'recommendations': recommendations,
            'current_metrics': {
                'engagement_rate': f"{engagement_rate:.2%}",
                'comment_rate': f"{comment_rate:.2%}",
                'title_length': len(video_data.get('title', ''))
            }
        }

    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        if not self.is_loaded:
            return {
                'success': False,
                'error': 'Models not loaded'
            }
        
        try:
            metadata = self.db.ml_metadata.find_one({"type": "trending_prediction"})
            
            if not metadata:
                return {
                    'success': False,
                    'error': 'Model metadata not found'
                }
            
            model_info = {
                'success': True,
                'available_models': list(self.models.keys()),
                'trained_at': metadata.get('created_at'),
                'models_performance': {},
                'total_models': len(self.models)
            }
            
            # Prefer flat 'metrics' (as saved by current training pipeline)
            flat_metrics = metadata.get('metrics')
            if flat_metrics:
                model_info['models_performance']['logistic'] = {
                    'accuracy': round(flat_metrics.get('accuracy', 0), 4),
                    'f1_score': round(flat_metrics.get('f1_score', 0), 4),
                    'precision': round(flat_metrics.get('precision', 0), 4),
                    'recall': round(flat_metrics.get('recall', 0), 4),
                    'roc_auc': round(flat_metrics.get('roc_auc', 0), 4)
                }
            else:
                # Fallback to legacy structure with 'models'
                for model_name, model_data in metadata.get('models', {}).items():
                    if 'metrics' in model_data:
                        metrics = model_data['metrics']
                        model_info['models_performance'][model_name] = {
                            'accuracy': round(metrics.get('accuracy', 0), 4),
                            'f1_score': round(metrics.get('f1_score', 0), 4),
                            'precision': round(metrics.get('precision', 0), 4),
                            'recall': round(metrics.get('recall', 0), 4),
                            'roc_auc': round(metrics.get('roc_auc', 0), 4)
                        }
            
            return model_info
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to get model info: {str(e)}'
            }

    def predict_batch(self, videos_data: List[Dict], model_name: str = 'logistic') -> List[Dict]:
        """Predict trending for multiple videos"""
        if not self.is_loaded:
            return [{'error': 'ML models not loaded', 'success': False}]
        
        results = []
        
        for i, video_data in enumerate(videos_data):
            result = self.predict_trending(video_data, model_name)
            if result:
                result['video_index'] = i
                result['video_id'] = video_data.get('video_id', f'video_{i}')
                results.append(result)
        
        return results

# Global ML service instance
ml_service = None

def get_ml_service() -> MLPredictionService:
    """Get global ML service instance"""
    global ml_service
    if ml_service is None:
        ml_service = MLPredictionService()
    return ml_service

def initialize_ml_service(mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "youtube_trending"):
    """Initialize ML service with custom settings"""
    global ml_service
    ml_service = MLPredictionService(mongo_uri, db_name)
    return ml_service
