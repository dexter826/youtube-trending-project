"""
YouTube Trending Data Processing with PySpark
Author: BigData Expert
Description: Process YouTube trending videos CSV data and store results in MongoDB
"""

import os
import sys
from datetime import datetime, timedelta
import re
from collections import Counter
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pymongo
from pymongo import MongoClient

# MongoDB connection settings
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "youtube_trending"

class YouTubeTrendingProcessor:
    def __init__(self):
        """Initialize Spark session and MongoDB connection"""
        self.spark = SparkSession.builder \
            .appName("YouTubeTrendingProcessor") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.mongodb.input.uri", f"{MONGO_URI}{DB_NAME}.raw_videos") \
            .config("spark.mongodb.output.uri", f"{MONGO_URI}{DB_NAME}.trending_results") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # MongoDB client for direct operations
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[DB_NAME]
        
        print("[OK] Spark session and MongoDB connection initialized")

    def extract_country_from_filename(self, filename):
        """Extract country code from CSV filename (e.g., USvideos.csv -> US)"""
        match = re.match(r'([A-Z]{2})videos\.csv', os.path.basename(filename))
        return match.group(1) if match else 'UNKNOWN'

    def load_csv_data(self, data_path):
        """Load and process CSV files from data directory"""
        print(f"📁 Loading CSV files from: {data_path}")
        
        # Define schema for better performance
        schema = StructType([
            StructField("video_id", StringType(), True),
            StructField("trending_date", StringType(), True),
            StructField("title", StringType(), True),
            StructField("channel_title", StringType(), True),
            StructField("category_id", IntegerType(), True),
            StructField("publish_time", StringType(), True),
            StructField("tags", StringType(), True),
            StructField("views", LongType(), True),
            StructField("likes", LongType(), True),
            StructField("dislikes", LongType(), True),
            StructField("comment_count", LongType(), True),
            StructField("thumbnail_link", StringType(), True),
            StructField("comments_disabled", BooleanType(), True),
            StructField("ratings_disabled", BooleanType(), True),
            StructField("video_error_or_removed", BooleanType(), True),
            StructField("description", StringType(), True)
        ])
        
        all_data = []
        csv_files = [f for f in os.listdir(data_path) if f.endswith('videos.csv')]
        
        if not csv_files:
            print("[ERROR] No CSV files found in data directory!")
            return None
            
        for csv_file in csv_files:
            country = self.extract_country_from_filename(csv_file)
            file_path = os.path.join(data_path, csv_file)
            
            print(f"[PROCESSING] {csv_file} (Country: {country})")
            
            try:
                df = self.spark.read.csv(
                    file_path,
                    header=True,
                    schema=schema,
                    timestampFormat="yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
                )
                
                # Add country column
                df = df.withColumn("country", lit(country))
                
                # Convert trending_date to proper date format
                df = df.withColumn(
                    "trending_date_parsed",
                    to_date(col("trending_date"), "yy.dd.MM")
                )
                
                # Clean and validate data
                df = df.filter(
                    col("video_id").isNotNull() &
                    col("title").isNotNull() &
                    col("views").isNotNull() &
                    (col("views") >= 0)
                )
                
                all_data.append(df)
                print(f"[OK] Loaded {df.count()} records from {csv_file}")
                
            except Exception as e:
                print(f"[ERROR] Error processing {csv_file}: {str(e)}")
                continue
        
        if all_data:
            # Union all dataframes
            combined_df = all_data[0]
            for df in all_data[1:]:
                combined_df = combined_df.union(df)
            
            print(f"[SUCCESS] Total records loaded: {combined_df.count()}")
            return combined_df
        
        return None

    def save_raw_data_to_mongodb(self, df):
        """Save raw data to MongoDB for API access"""
        print("💾 Saving raw data to MongoDB...")
        
        # Clear existing raw data
        self.db.raw_videos.delete_many({})
        
        # Convert to Pandas and save to MongoDB
        pandas_df = df.toPandas()
        
        # Convert to dict and insert
        records = pandas_df.to_dict('records')
        
        # Handle NaN values and datetime conversion
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif key == 'trending_date_parsed' and value is not None:
                    # Convert datetime.date to string for MongoDB
                    record[key] = value.strftime('%Y-%m-%d') if hasattr(value, 'strftime') else str(value)
        
        if records:
            self.db.raw_videos.insert_many(records)
            print(f"[OK] Saved {len(records)} raw records to MongoDB")

    def process_trending_analysis(self, df):
        """Process trending analysis by country and date"""
        print("📈 Processing trending analysis...")
        
        results = []
        
        # Get unique countries and dates
        countries_dates = df.select("country", "trending_date_parsed") \
                           .distinct() \
                           .collect()
        
        for row in countries_dates:
            country = row.country
            date = row.trending_date_parsed
            
            if date is None:
                continue
                
            print(f"🔍 Analyzing {country} for {date}")
            
            # Filter data for specific country and date
            filtered_df = df.filter(
                (col("country") == country) & 
                (col("trending_date_parsed") == date)
            )
            
            # Get top 10 trending videos by views
            top_videos = filtered_df.orderBy(col("views").desc()) \
                                   .limit(10) \
                                   .select(
                                       "video_id", "title", "channel_title",
                                       "views", "likes", "dislikes", "comment_count",
                                       "category_id", "tags"
                                   ) \
                                   .collect()
            
            # Calculate statistics
            stats = filtered_df.agg(
                count("*").alias("total_videos"),
                sum("views").alias("total_views"),
                avg("views").alias("avg_views"),
                max("views").alias("max_views"),
                sum("likes").alias("total_likes"),
                sum("comment_count").alias("total_comments")
            ).collect()[0]
            
            # Prepare result document
            result_doc = {
                "country": country,
                "date": date.strftime("%Y-%m-%d") if date else None,
                "processed_at": datetime.now().isoformat(),
                "statistics": {
                    "total_videos": stats.total_videos,
                    "total_views": int(stats.total_views) if stats.total_views else 0,
                    "average_views": float(stats.avg_views) if stats.avg_views else 0,
                    "max_views": int(stats.max_views) if stats.max_views else 0,
                    "total_likes": int(stats.total_likes) if stats.total_likes else 0,
                    "total_comments": int(stats.total_comments) if stats.total_comments else 0
                },
                "top_videos": [
                    {
                        "video_id": video.video_id,
                        "title": video.title,
                        "channel_title": video.channel_title,
                        "views": int(video.views) if video.views else 0,
                        "likes": int(video.likes) if video.likes else 0,
                        "dislikes": int(video.dislikes) if video.dislikes else 0,
                        "comment_count": int(video.comment_count) if video.comment_count else 0,
                        "category_id": video.category_id,
                        "tags": video.tags
                    } for video in top_videos
                ]
            }
            
            results.append(result_doc)
        
        # Save to MongoDB
        if results:
            self.db.trending_results.delete_many({})  # Clear existing
            self.db.trending_results.insert_many(results)
            print(f"[OK] Saved {len(results)} trending analysis results")
        
        return results

    def generate_wordcloud_data(self, df):
        """Generate wordcloud data from video titles"""
        print("[WORDCLOUD] Generating wordcloud data...")
        
        import pandas as pd
        
        results = []
        
        # Get unique countries and dates
        countries_dates = df.select("country", "trending_date_parsed") \
                           .distinct() \
                           .collect()
        
        # Common stop words to filter out
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        for row in countries_dates:
            country = row.country
            date = row.trending_date_parsed
            
            if date is None:
                continue
                
            print(f"[WORDCLOUD] Generating wordcloud for {country} - {date}")
            
            # Filter data for specific country and date
            filtered_df = df.filter(
                (col("country") == country) & 
                (col("trending_date_parsed") == date)
            )
            
            # Get all titles
            titles = filtered_df.select("title").collect()
            
            # Process titles to extract words
            all_words = []
            for title_row in titles:
                if title_row.title:
                    # Clean title: remove special characters, convert to lowercase
                    clean_title = re.sub(r'[^\w\s]', ' ', title_row.title.lower())
                    words = clean_title.split()
                    
                    # Filter out stop words and short words
                    filtered_words = [
                        word for word in words 
                        if len(word) > 2 and word not in stop_words
                    ]
                    all_words.extend(filtered_words)
            
            # Count word frequencies
            word_counts = Counter(all_words)
            
            # Get top 50 words
            top_words = word_counts.most_common(50)
            
            # Prepare wordcloud data
            wordcloud_data = {
                "country": country,
                "date": date.strftime("%Y-%m-%d") if date else None,
                "processed_at": datetime.now().isoformat(),
                "words": [
                    {"text": word, "value": count}
                    for word, count in top_words
                ]
            }
            
            results.append(wordcloud_data)
        
        # Save to MongoDB
        if results:
            self.db.wordcloud_data.delete_many({})  # Clear existing
            self.db.wordcloud_data.insert_many(results)
            print(f"[OK] Saved {len(results)} wordcloud datasets")
        
        return results

    def run_full_pipeline(self, data_path):
        """Run the complete data processing pipeline"""
        try:
            print("🚀 Starting YouTube Trending Data Processing Pipeline")
            print("=" * 60)
            
            # Step 1: Load CSV data
            df = self.load_csv_data(data_path)
            if df is None:
                print("[ERROR] Failed to load data. Exiting.")
                return False
            
            # Step 2: Save raw data to MongoDB
            self.save_raw_data_to_mongodb(df)
            
            # Step 3: Process trending analysis
            trending_results = self.process_trending_analysis(df)
            
            # Step 4: Generate wordcloud data
            wordcloud_results = self.generate_wordcloud_data(df)
            
            print("=" * 60)
            print("[SUCCESS] Pipeline completed successfully!")
            print(f"[RESULTS] Processed {len(trending_results)} trending analysis results")
            print(f"[WORDCLOUD] Generated {len(wordcloud_results)} wordcloud datasets")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.spark.stop()
            self.mongo_client.close()

def main():
    """Main execution function"""
    if len(sys.argv) != 2:
        print("Usage: spark-submit process_trending.py <data_directory>")
        print("Example: spark-submit process_trending.py /opt/bitnami/spark/data")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Data directory not found: {data_path}")
        sys.exit(1)
    
    processor = YouTubeTrendingProcessor()
    success = processor.run_full_pipeline(data_path)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
