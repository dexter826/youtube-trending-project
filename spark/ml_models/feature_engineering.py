"""
ML Feature Engineering for YouTube Trending Prediction
Author: BigData Expert
Description: Extract and engineer features for trending prediction model
"""

import os
import sys
from datetime import datetime
import re
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pymongo
from pymongo import MongoClient

# MongoDB connection settings
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "youtube_trending"

class TrendingFeatureEngine:
    def __init__(self):
        """Initialize Spark session and MongoDB connection"""
        self.spark = SparkSession.builder \
            .appName("TrendingFeatureEngine") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # MongoDB client
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[DB_NAME]
        
        print("[OK] Feature Engineering - Spark session initialized")

    def load_raw_data(self):
        """Load raw video data from MongoDB"""
        print("📁 Loading raw data from MongoDB...")
        
        try:
            # Get data from MongoDB
            raw_data = list(self.db.raw_videos.find({}))
            
            if not raw_data:
                print("[ERROR] No raw data found in MongoDB!")
                return None
            
            # Remove MongoDB ObjectId fields and convert to proper types
            cleaned_data = []
            for record in raw_data:
                # Remove _id field
                if '_id' in record:
                    del record['_id']
                
                # Handle null values and convert types
                cleaned_record = {}
                for key, value in record.items():
                    if value is None:
                        if key in ['views', 'likes', 'dislikes', 'comment_count', 'category_id']:
                            cleaned_record[key] = 0
                        else:
                            cleaned_record[key] = ""
                    else:
                        cleaned_record[key] = value
                
                cleaned_data.append(cleaned_record)
            
            # Convert to Pandas first, then to Spark DataFrame
            pandas_df = pd.DataFrame(cleaned_data)
            
            # Convert to Spark DataFrame
            df = self.spark.createDataFrame(pandas_df)
            
            print(f"[OK] Loaded {df.count()} raw records")
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to load raw data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def engineer_features(self, df):
        """Engineer ML features from raw data"""
        print("🔧 Engineering features...")
        
        # 1. Engagement Metrics
        df = df.withColumn("like_ratio", 
                          when(col("views") > 0, col("likes") / col("views")).otherwise(0))
        
        df = df.withColumn("dislike_ratio", 
                          when(col("views") > 0, col("dislikes") / col("views")).otherwise(0))
        
        df = df.withColumn("comment_ratio", 
                          when(col("views") > 0, col("comment_count") / col("views")).otherwise(0))
        
        df = df.withColumn("engagement_score", 
                          when(col("views") > 0, 
                               (col("likes") + col("comment_count")) / col("views")).otherwise(0))
        
        # 2. Content Features
        df = df.withColumn("title_length", length(col("title")))
        
        df = df.withColumn("has_caps", 
                          when(col("title").rlike("[A-Z]{3,}"), 1).otherwise(0))
        
        # Count tags (split by |)
        df = df.withColumn("tag_count", 
                          when(col("tags").isNotNull(), 
                               size(split(col("tags"), "\\|"))).otherwise(0))
        
        # Extract publish hour
        df = df.withColumn("publish_hour", 
                          hour(to_timestamp(col("publish_time"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")))
        
        # 3. Channel popularity (average views per channel)
        channel_stats = df.groupBy("channel_title").agg(
            avg("views").alias("channel_avg_views"),
            count("*").alias("channel_video_count")
        )
        
        df = df.join(channel_stats, "channel_title", "left")
        
        # 4. Create target variable (is_trending)
        # We'll define trending as top 20% by views for each country-date combination
        window_spec = Window.partitionBy("country", "trending_date_parsed").orderBy(col("views").desc())
        
        df = df.withColumn("view_rank", row_number().over(window_spec))
        
        # Calculate total videos per country-date
        total_videos = df.groupBy("country", "trending_date_parsed").agg(
            count("*").alias("total_videos_per_day")
        )
        
        df = df.join(total_videos, ["country", "trending_date_parsed"], "left")
        
        # Mark top 20% as trending (minimum 1, maximum 50)
        df = df.withColumn("trending_threshold", 
                          greatest(lit(1), least(lit(50), (col("total_videos_per_day") * 0.2).cast("int"))))
        
        df = df.withColumn("is_trending", 
                          when(col("view_rank") <= col("trending_threshold"), 1).otherwise(0))
        
        # 5. Log transform high-skew numerical features
        df = df.withColumn("log_views", log1p(col("views")))
        df = df.withColumn("log_likes", log1p(col("likes")))
        df = df.withColumn("log_comments", log1p(col("comment_count")))
        
        print("[OK] Feature engineering completed")
        return df

    def select_ml_features(self, df):
        """Select final features for ML model"""
        feature_columns = [
            # Identifiers
            "video_id", "country", "trending_date_parsed", "title",
            
            # Target variable
            "is_trending",
            
            # Numerical features
            "views", "likes", "dislikes", "comment_count",
            "like_ratio", "dislike_ratio", "comment_ratio", "engagement_score",
            "title_length", "tag_count", "publish_hour",
            "channel_avg_views", "channel_video_count",
            "log_views", "log_likes", "log_comments",
            
            # Categorical features
            "category_id", "channel_title", "has_caps",
            
            # Additional context
            "comments_disabled", "ratings_disabled"
        ]
        
        return df.select(*feature_columns)

    def save_ml_features(self, df):
        """Save engineered features to MongoDB"""
        print("💾 Saving ML features to MongoDB...")
        
        # Clear existing ML features
        self.db.ml_features.delete_many({})
        
        # Convert to Pandas and save
        pandas_df = df.toPandas()
        
        # Handle NaN values and convert datetime
        records = pandas_df.to_dict('records')
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif key == 'trending_date_parsed' and value is not None:
                    record[key] = value.strftime('%Y-%m-%d') if hasattr(value, 'strftime') else str(value)
        
        # Add metadata
        feature_metadata = {
            "created_at": datetime.now().isoformat(),
            "total_records": len(records),
            "trending_count": len([r for r in records if r.get('is_trending') == 1]),
            "non_trending_count": len([r for r in records if r.get('is_trending') == 0]),
            "features": [
                "like_ratio", "dislike_ratio", "comment_ratio", "engagement_score",
                "title_length", "tag_count", "publish_hour", "channel_avg_views",
                "log_views", "log_likes", "log_comments", "category_id", "has_caps"
            ]
        }
        
        if records:
            # Save features
            self.db.ml_features.insert_many(records)
            
            # Save metadata
            self.db.ml_metadata.delete_many({"type": "features"})
            self.db.ml_metadata.insert_one({**feature_metadata, "type": "features"})
            
            print(f"[OK] Saved {len(records)} ML feature records")
            print(f"[STATS] Trending: {feature_metadata['trending_count']}, Non-trending: {feature_metadata['non_trending_count']}")
        
        return feature_metadata

    def generate_feature_summary(self, df):
        """Generate feature summary statistics"""
        print("📊 Generating feature summary...")
        
        # Basic statistics
        summary_stats = df.describe([
            "views", "likes", "like_ratio", "engagement_score", 
            "title_length", "tag_count", "channel_avg_views"
        ]).toPandas()
        
        # Trending vs non-trending comparison
        trending_stats = df.groupBy("is_trending").agg(
            count("*").alias("count"),
            avg("views").alias("avg_views"),
            avg("like_ratio").alias("avg_like_ratio"),
            avg("engagement_score").alias("avg_engagement"),
            avg("title_length").alias("avg_title_length")
        ).toPandas()
        
        # Category distribution
        category_dist = df.groupBy("category_id", "is_trending").count().toPandas()
        
        print("\n=== FEATURE SUMMARY ===")
        print("\n1. Basic Statistics:")
        print(summary_stats.to_string())
        
        print("\n2. Trending vs Non-Trending:")
        print(trending_stats.to_string())
        
        print("\n3. Category Distribution (first 10):")
        print(category_dist.head(10).to_string())
        
        return {
            "basic_stats": summary_stats.to_dict(),
            "trending_comparison": trending_stats.to_dict(),
            "category_distribution": category_dist.to_dict()
        }

    def run_feature_pipeline(self):
        """Run complete feature engineering pipeline"""
        try:
            print("🚀 Starting Feature Engineering Pipeline")
            print("=" * 50)
            
            # Step 1: Load raw data
            df = self.load_raw_data()
            if df is None:
                return False
            
            # Step 2: Engineer features
            df_features = self.engineer_features(df)
            
            # Step 3: Select ML features
            df_ml = self.select_ml_features(df_features)
            
            # Step 4: Generate summary
            summary = self.generate_feature_summary(df_ml)
            
            # Step 5: Save to MongoDB
            metadata = self.save_ml_features(df_ml)
            
            print("=" * 50)
            print("[SUCCESS] Feature engineering completed!")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Feature pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.spark.stop()
            self.mongo_client.close()

def main():
    """Main execution function"""
    feature_engine = TrendingFeatureEngine()
    success = feature_engine.run_feature_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
