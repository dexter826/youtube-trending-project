"""
ML Model Training for YouTube Trending Prediction
Author: BigData Expert
Description: Train and evaluate trending prediction models
"""

import os
import sys
import pickle
import joblib
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.impute import SimpleImputer

import pymongo
from pymongo import MongoClient

# MongoDB connection settings
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "youtube_trending"

class TrendingPredictor:
    def __init__(self):
        """Initialize ML models and MongoDB connection"""
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[DB_NAME]
        
        # Initialize single model - Logistic Regression only
        self.model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            solver='liblinear',  # Faster solver
            n_jobs=-1  # Use all CPU cores
        )
        
        # Initialize preprocessors
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        
        # Model storage
        self.trained_model = None
        self.model_metrics = {}
        
        print("[OK] ML Predictor initialized")

    def load_ml_features(self):
        """Load engineered features from MongoDB"""
        print("📁 Loading ML features from MongoDB...")
        
        try:
            # Get features from MongoDB
            features_data = list(self.db.ml_features.find({}))
            
            if not features_data:
                print("[ERROR] No ML features found! Run feature engineering first.")
                return None
            
            df = pd.DataFrame(features_data)
            
            # Drop MongoDB _id field
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            print(f"[OK] Loaded {len(df)} feature records")
            print(f"[INFO] Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to load ML features: {str(e)}")
            return None

    def prepare_training_data(self, df):
        """Prepare data for ML training"""
        print("🔧 Preparing training data...")
        
        # Define feature columns
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
        
        target_column = 'is_trending'
        
        # Check if target exists
        if target_column not in df.columns:
            print(f"[ERROR] Target column '{target_column}' not found!")
            return None, None, None, None
        
        # Filter available columns
        available_num_features = [col for col in numerical_features if col in df.columns]
        available_cat_features = [col for col in categorical_features if col in df.columns]
        
        print(f"[INFO] Using {len(available_num_features)} numerical features")
        print(f"[INFO] Using {len(available_cat_features)} categorical features")
        
        # Prepare features
        X_num = df[available_num_features].copy()
        X_cat = df[available_cat_features].copy()
        y = df[target_column].copy()
        
        # Handle missing values in numerical features
        X_num = pd.DataFrame(
            self.imputer.fit_transform(X_num),
            columns=available_num_features
        )
        
        # Encode categorical features
        for col in available_cat_features:
            le = LabelEncoder()
            X_cat[col] = le.fit_transform(X_cat[col].astype(str))
            self.label_encoders[col] = le
        
        # Combine features
        X = pd.concat([X_num, X_cat], axis=1)
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"[OK] Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"[INFO] Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, available_num_features, available_cat_features

    def train_model(self, X, y):
        """Train single Logistic Regression model"""
        print("🚀 Training Logistic Regression model...")
        
        # Split data
        print("   - Splitting data (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        print("   - Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("   - Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        print("   - Making predictions...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        print("📊 Calculating performance metrics...")
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Cross-validation
        print("   - Cross-validation...")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=3, scoring='f1')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        # Feature importance (using coefficients)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': abs(self.model.coef_[0])
        }).sort_values('importance', ascending=False)
        metrics['feature_importance'] = feature_importance.head(10).to_dict('records')
        
        # Store results
        self.trained_model = self.model
        self.model_metrics = metrics
        
        # Print results
        print("\n" + "="*50)
        print("🎯 MODEL PERFORMANCE RESULTS")
        print("="*50)
        print(f"✅ Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"🎯 Precision:   {metrics['precision']:.4f}")
        print(f"📈 Recall:      {metrics['recall']:.4f}")
        print(f"⚖️  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"🔮 ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"🔄 CV F1:       {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
        
        print(f"\n📊 Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"   True Neg: {cm[0][0]:,}  |  False Pos: {cm[0][1]:,}")
        print(f"   False Neg: {cm[1][0]:,} |  True Pos: {cm[1][1]:,}")
        
        print(f"\n🔝 Top 5 Important Features:")
        for i, feat in enumerate(feature_importance.head(5).itertuples(), 1):
            print(f"   {i}. {feat.feature}: {feat.importance:.4f}")
        
        return metrics

    def save_model(self):
        """Save trained model and metadata"""
        print("💾 Saving model to disk...")
        
        # Create models directory
        models_dir = "saved_models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model files
        model_path = os.path.join(models_dir, "trending_predictor.pkl")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        imputer_path = os.path.join(models_dir, "imputer.pkl")
        encoders_path = os.path.join(models_dir, "label_encoders.pkl")
        
        joblib.dump(self.trained_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.imputer, imputer_path)
        joblib.dump(self.label_encoders, encoders_path)
        
        # Save metadata to MongoDB
        model_metadata = {
            "created_at": datetime.now().isoformat(),
            "type": "trending_prediction",
            "model_name": "logistic_regression",
            "model_path": os.path.abspath(model_path),
            "scaler_path": os.path.abspath(scaler_path),
            "imputer_path": os.path.abspath(imputer_path),
            "encoders_path": os.path.abspath(encoders_path),
            "metrics": self.model_metrics,
            "feature_count": len(self.model_metrics.get('feature_importance', [])),
            "training_completed": True
        }
        
        # Update MongoDB
        self.db.ml_metadata.delete_many({"type": "trending_prediction"})
        self.db.ml_metadata.insert_one(model_metadata)
        
        print(f"[OK] Model saved to: {model_path}")
        print(f"[OK] Scaler saved to: {scaler_path}")
        print(f"[OK] Metadata saved to MongoDB")
        
        return model_metadata

    def run_training_pipeline(self):
        """Run complete simplified training pipeline"""
        try:
            print("🚀 Starting SIMPLIFIED ML Training Pipeline")
            print("="*60)
            
            # Step 1: Load features
            df = self.load_ml_features()
            if df is None:
                return False
            
            # Step 2: Prepare training data
            X, y, num_features, cat_features = self.prepare_training_data(df)
            if X is None:
                return False
            
            # Step 3: Train single model
            metrics = self.train_model(X, y)
            
            # Step 4: Save model
            metadata = self.save_model()
            
            print("="*60)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print(f"🎯 Final F1-Score: {metrics['f1_score']:.4f}")
            print(f"🎯 Final Accuracy: {metrics['accuracy']:.4f}")
            print("✅ Model ready for predictions!")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Training pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.mongo_client.close()

def main():
    """Main execution function"""
    predictor = TrendingPredictor()
    success = predictor.run_training_pipeline()
    
    if success:
        print("\n🚀 Ready to start backend and test predictions!")
        print("💡 Next steps:")
        print("   1. cd ../backend")
        print("   2. uvicorn app.main:app --reload")
        print("   3. cd ../frontend && npm start")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
