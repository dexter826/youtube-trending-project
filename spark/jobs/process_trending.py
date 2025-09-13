"""
YouTube Trending Data Processing - Refactored
"""

import os
import sys
from datetime import datetime
import time

# Add processors directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'processors'))

from processors.data_loader import DataLoader
from processors.data_validator import DataValidator
from processors.data_processor import DataProcessor
from processors.data_exporter import DataExporter


class YouTubeTrendingProcessor:
    def __init__(self):
        """Initialize processor with modular components"""
        self.data_loader = DataLoader()
        self.data_validator = DataValidator(self.data_loader.spark)
        self.data_processor = DataProcessor(self.data_loader.spark)
        self.data_exporter = DataExporter(self.data_loader.db)

        # Keep references for backward compatibility
        self.spark = self.data_loader.spark
        self.db = self.data_loader.db

    def load_csv_data_from_hdfs(self):
        """Load CSV files from HDFS using DataLoader"""
        return self.data_loader.load_csv_data_from_hdfs()

    def load_csv_data_from_local(self, data_path):
        """Load CSV files from local directory using DataLoader"""
        return self.data_loader.load_csv_data_from_local(data_path)

    def save_raw_data_to_mongodb(self, df):
        """Save raw data to MongoDB using DataExporter"""
        self.data_exporter.save_raw_data_to_mongodb(df)

    def process_trending_analysis(self, df):
        """Process trending analysis using DataProcessor"""
        results = self.data_processor.process_trending_analysis(df)
        self.data_exporter.save_trending_results(results)
        return results

    def generate_wordcloud_data(self, df):
        """Generate wordcloud data using DataProcessor"""
        results = self.data_processor.generate_wordcloud_data(df)
        self.data_exporter.save_wordcloud_data(results)
        return results

    def create_ml_features(self, df):
        """Create ML features using DataProcessor and save using DataExporter"""
        ml_features = self.data_processor.create_ml_features(df)
        self.data_exporter.save_ml_features(ml_features)
        return len(ml_features)

    def validate_and_clean_data(self, df):
        """Validate and clean data using DataValidator"""
        # Validate schema
        self.data_validator.validate_schema(df)

        # Check data quality
        quality_report = self.data_validator.check_data_quality(df)
        print(f"Data quality report: {quality_report}")

        # Clean data
        df_clean = self.data_validator.clean_data(df)
        return df_clean

    def run_full_pipeline(self, data_path=None):
        """Run the complete data processing pipeline"""
        try:
            print("[START] Starting YouTube Trending Data Processing Pipeline...")
            start_time = time.time()

            # Step 1: Load data from HDFS only
            print("\n[STEP 1/4] Loading data from HDFS...")
            step_start = time.time()
            df = self.load_csv_data_from_hdfs()

            if df is None:
                print("[ERROR] Failed to load data from HDFS")
                return False

            print(f"   [INFO] Loaded {df.count()} records from {len(self.data_loader.countries)} countries")

            # Step 2: Validate and clean data
            print("\n[STEP 2/4] Validating and cleaning data...")
            step_start = time.time()
            df = self.validate_and_clean_data(df)
            print("   [SUCCESS] Data validation and cleaning completed")

            # Step 3: Process and save data
            print("\n[STEP 3/4] Processing trending analysis...")
            step_start = time.time()
            self.save_raw_data_to_mongodb(df)
            ml_features_count = self.create_ml_features(df)
            trending_results = self.process_trending_analysis(df)
            wordcloud_results = self.generate_wordcloud_data(df)
            print(f"   [INFO] Created {ml_features_count} ML features")
            print(f"   [INFO] Generated {len(trending_results)} trending reports")
            print(f"   [INFO] Generated {len(wordcloud_results)} wordcloud datasets")

            # Step 4: Summary
            total_time = time.time() - start_time
            print("\n[SUCCESS] Step 4/4: Pipeline completed successfully!")
            print(f"   [INFO] Total records processed: {df.count()}")
            print(f"   [INFO] Countries processed: {len(self.data_loader.countries)}")
            print(f"   [INFO] ML features created: {ml_features_count}")

            return True

        except Exception as e:
            print(f"\n[ERROR] Pipeline failed with error: {e}")
            return False
        finally:
            print("\n[CLEANUP] Cleaning up resources...")
            self.spark.stop()
            self.data_loader.mongo_client.close()
            print("[SUCCESS] Resources cleaned up")

def main():
    """Main execution function"""
    if len(sys.argv) > 1:
        print("Usage: python process_trending.py")
        print("Note: Data must be loaded to HDFS first using run.py pipeline")
        sys.exit(1)

    processor = YouTubeTrendingProcessor()
    success = processor.run_full_pipeline()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
