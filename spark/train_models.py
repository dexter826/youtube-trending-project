"""
YouTube Trending ML Training - Refactored
"""

import os
import sys
from datetime import datetime
import time

from training.data_preparation import DataPreparation
from training.model_training import ModelTraining
from training.model_evaluation import ModelEvaluation
from training.model_saving import ModelSaving


class YouTubeMLTrainer:
    def __init__(self):
        """Initialize ML trainer with modular components"""
        # Import here to avoid circular imports
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from core.spark_manager import get_spark_session, PRODUCTION_CONFIGS
        from core.database_manager import get_database_connection

        self.spark = get_spark_session("YouTubeMLTrainer", PRODUCTION_CONFIGS["ml_training"])
        self.db = get_database_connection()

        # Initialize modular components
        self.data_prep = DataPreparation(self.db)
        self.model_training = ModelTraining(self.spark)
        self.model_evaluation = ModelEvaluation(self.spark)
        self.model_saving = ModelSaving("hdfs://localhost:9000/youtube_trending")

        # Keep references for backward compatibility
        from core.database_manager import _manager
        self.mongo_client = _manager._client

    def load_training_data(self):
        """Load training data using DataPreparation module"""
        pandas_df = self.data_prep.load_training_data_from_mongodb()
        if pandas_df is None:
            return None

        df = self.spark.createDataFrame(pandas_df)
        return df

    def train_trending_prediction_model(self, df):
        return None, {}

    def train_clustering_model(self, df):
        """Train clustering model using modular components"""
        # Prepare data
        data, feature_cols = self.data_prep.prepare_features_for_clustering(df)

        # Show category distribution for analysis
        category_dist = self.model_evaluation.get_category_distribution(df)
        category_dist.show(10)

        # Train model
        model, predictions = self.model_training.train_clustering_model(data, feature_cols)

        # Evaluate model
        metrics = self.model_evaluation.evaluate_clustering_model(predictions, feature_cols)

        # Save model
        model_path = self.model_saving.save_model(model, "clustering")

        return model, metrics

    def train_days_regression_model(self, df):
        """Train days-in-trending regression model"""
        # Prepare data
        data, feature_cols = self.data_prep.prepare_features_for_days_regression(df)

        # Train model
        model, train_data, test_data, predictions = self.model_training.train_days_regression_model(data, feature_cols)

        # Evaluate model
        metrics = self.model_evaluation.evaluate_days_regression_model(predictions, feature_cols)

        # Save model
        model_path = self.model_saving.save_model(model, "days_regression")

        return model, metrics

    def run_training_pipeline(self):
        """Run complete model training pipeline using modular components"""
        try:
            print("🤖 Starting ML Training Pipeline...")
            start_time = time.time()

            # Step 1: Load training data
            print("\n📥 Step 1/3: Loading training data...")
            step_start = time.time()
            df = self.load_training_data()
            if df is None:
                return False

            df.cache()
            record_count = df.count()
            print(f"   [INFO] Loaded {record_count} training records")

            # Step 2: Train models
            print("\n🎯 Step 2/3: Training ML models (Clustering k=3, Days Regression)...")
            step_start = time.time()

            # Train clustering model
            print("   [TRAINING] Training clustering model...")
            clustering_model, clustering_metrics = self.train_clustering_model(df)
            print(f"      [METRICS] Silhouette Score: {clustering_metrics['silhouette_score']:.3f}")

            # Train days-in-trending regression model
            print("   [TRAINING] Training days-in-trending regression model...")
            days_model, days_metrics = self.train_days_regression_model(df)
            print(f"      [METRICS] R² Score: {days_metrics['r2_score']:.3f}, RMSE: {days_metrics['rmse']:.4f}")

            # Step 3: Save results
            print("\n💾 Step 3/3: Saving models and metrics...")
            step_start = time.time()
            dataset_size = df.count()
            # Compose metrics for backward compatibility keys
            trending_metrics = {}
            regression_metrics = days_metrics  # repurpose 'regression' key as days metrics
            metrics_path = self.model_saving.save_metrics_to_json(
                trending_metrics, clustering_metrics, regression_metrics, dataset_size
            )

            # Also save to MongoDB
            self.model_saving.save_metrics_to_mongodb(
                self.db, trending_metrics, clustering_metrics, regression_metrics, dataset_size
            )
            print(f"   [SAVE] Metrics saved to: {metrics_path}")

            # Summary
            total_time = time.time() - start_time
            print(
                "\n[SUCCESS] ML Training Pipeline completed successfully!"
            )
            print(f"   [INFO] Dataset size: {dataset_size} records")
            print("   [INFO] Models trained: 2 (Clustering k=3, Days Regression)")
            return all([clustering_model, days_model])

        except Exception as e:
            print(f"\n[ERROR] Training pipeline failed with error: {e}")
            return False
        finally:
            print("\n[CLEANUP] Cleaning up resources...")
            self.spark.stop()
            self.mongo_client.close()
            print("[SUCCESS] Resources cleaned up")

def main():
    """Main execution function"""
    trainer = YouTubeMLTrainer()
    success = trainer.run_training_pipeline()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()