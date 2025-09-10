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
from pyspark.ml.feature import CountVectorizer, IDF, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
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
            
            # Step 5: Perform Machine Learning Analysis
            ml_success = self.perform_ml_analysis(df)
            
            print("=" * 60)
            print("[SUCCESS] Pipeline completed successfully!")
            print(f"[RESULTS] Processed {len(trending_results)} trending analysis results")
            print(f"[WORDCLOUD] Generated {len(wordcloud_results)} wordcloud datasets")
            print(f"[ML] Machine Learning analysis: {'✅ Completed' if ml_success else '❌ Failed'}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.spark.stop()
            self.mongo_client.close()

    def perform_ml_analysis(self, df):
        """Perform Machine Learning analysis on the data"""
        print("🤖 Starting Machine Learning Analysis...")
        
        try:
            # 1. VIDEO CLUSTERING ANALYSIS
            print("📊 Performing video clustering based on engagement metrics...")
            clustering_results = self.perform_video_clustering(df)
            
            # 2. POPULARITY PREDICTION
            print("🔮 Building popularity prediction model...")
            prediction_results = self.build_popularity_prediction_model(df)
            
            # 3. SENTIMENT ANALYSIS (Basic)
            print("💭 Performing sentiment analysis on video titles...")
            sentiment_results = self.perform_title_sentiment_analysis(df)
            
            # 4. ANOMALY DETECTION
            print("🕵️ Detecting anomalous videos...")
            anomaly_results = self.detect_anomalous_videos(df)
            
            # 5. CATEGORY PERFORMANCE ANALYSIS
            print("📈 Analyzing category performance patterns...")
            category_analysis = self.analyze_category_performance(df)
            
            # Save ML results to MongoDB
            self.save_ml_results_to_mongodb({
                'clustering': clustering_results,
                'predictions': prediction_results,
                'sentiment': sentiment_results,
                'anomalies': anomaly_results,
                'category_analysis': category_analysis
            })
            
            print("✅ Machine Learning analysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] ML Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def perform_video_clustering(self, df):
        """Cluster videos based on engagement metrics"""
        try:
            # Prepare features for clustering
            feature_cols = ['views', 'likes', 'dislikes', 'comment_count']
            
            # Remove null values and create engagement rate
            ml_df = df.filter(col("views") > 0).withColumn(
                "like_rate", col("likes") / col("views")
            ).withColumn(
                "comment_rate", col("comment_count") / col("views")
            ).withColumn(
                "engagement_score", 
                (col("likes") + col("comment_count")) / col("views")
            )
            
            # Assemble features
            assembler = VectorAssembler(
                inputCols=['views', 'likes', 'dislikes', 'comment_count', 'like_rate', 'comment_rate'],
                outputCol='raw_features'
            )
            
            # Scale features
            scaler = StandardScaler(inputCol='raw_features', outputCol='features')
            
            # K-means clustering
            kmeans = KMeans(k=5, featuresCol='features', predictionCol='cluster')
            
            # Create pipeline
            pipeline = Pipeline(stages=[assembler, scaler, kmeans])
            
            # Fit and transform
            model = pipeline.fit(ml_df)
            clustered_df = model.transform(ml_df)
            
            # Analyze clusters
            cluster_summary = clustered_df.groupBy("cluster").agg(
                avg("views").alias("avg_views"),
                avg("likes").alias("avg_likes"),
                avg("like_rate").alias("avg_like_rate"),
                avg("engagement_score").alias("avg_engagement"),
                count("*").alias("video_count")
            ).collect()
            
            return [row.asDict() for row in cluster_summary]
            
        except Exception as e:
            print(f"[ERROR] Clustering failed: {str(e)}")
            return []

    def build_popularity_prediction_model(self, df):
        """Build a model to predict video popularity"""
        try:
            # Define popularity categories based on views
            ml_df = df.filter(col("views").isNotNull() & (col("views") > 0))
            
            # Calculate view percentiles for popularity classification
            quantiles = ml_df.approxQuantile("views", [0.33, 0.66], 0.05)
            low_threshold, high_threshold = quantiles
            
            # Create popularity labels
            ml_df = ml_df.withColumn(
                "popularity_label",
                when(col("views") <= low_threshold, 0)  # Low popularity
                .when(col("views") <= high_threshold, 1)  # Medium popularity
                .otherwise(2)  # High popularity
            )
            
            # Prepare features
            string_indexer = StringIndexer(inputCol="category_id", outputCol="category_index")
            
            # Create time-based features
            ml_df = ml_df.withColumn("title_length", length(col("title")))
            ml_df = ml_df.withColumn("has_caps", when(col("title").rlike("[A-Z]{2,}"), 1).otherwise(0))
            
            # Assemble features
            assembler = VectorAssembler(
                inputCols=['category_index', 'title_length', 'has_caps'],
                outputCol='features'
            )
            
            # Random Forest Classifier
            rf = RandomForestClassifier(
                featuresCol='features',
                labelCol='popularity_label',
                predictionCol='prediction',
                numTrees=50
            )
            
            # Create pipeline
            pipeline = Pipeline(stages=[string_indexer, assembler, rf])
            
            # Split data
            train_data, test_data = ml_df.randomSplit([0.8, 0.2], seed=42)
            
            # Train model
            model = pipeline.fit(train_data)
            predictions = model.transform(test_data)
            
            # Evaluate model
            evaluator = MulticlassClassificationEvaluator(
                labelCol='popularity_label',
                predictionCol='prediction',
                metricName='accuracy'
            )
            
            accuracy = evaluator.evaluate(predictions)
            
            # Feature importance (simplified)
            feature_importance = model.stages[-1].featureImportances.toArray().tolist()
            
            return {
                'model_accuracy': float(accuracy),
                'feature_importance': feature_importance,
                'low_threshold': float(low_threshold),
                'high_threshold': float(high_threshold),
                'total_predictions': predictions.count()
            }
            
        except Exception as e:
            print(f"[ERROR] Prediction model failed: {str(e)}")
            return {}

    def perform_title_sentiment_analysis(self, df):
        """Basic sentiment analysis of video titles"""
        try:
            # Simple sentiment keywords (can be expanded)
            positive_keywords = ['amazing', 'best', 'awesome', 'great', 'fantastic', 'love', 'wonderful', 'incredible']
            negative_keywords = ['worst', 'hate', 'terrible', 'awful', 'bad', 'horrible', 'disgusting']
            clickbait_keywords = ['shocking', 'unbelievable', 'you won\'t believe', 'amazing', 'incredible']
            
            # Create sentiment scores
            sentiment_df = df.withColumn("title_lower", lower(col("title")))
            
            # Count positive words
            positive_pattern = "|".join(positive_keywords)
            sentiment_df = sentiment_df.withColumn(
                "positive_score",
                size(split(col("title_lower"), positive_pattern)) - 1
            )
            
            # Count negative words  
            negative_pattern = "|".join(negative_keywords)
            sentiment_df = sentiment_df.withColumn(
                "negative_score",
                size(split(col("title_lower"), negative_pattern)) - 1
            )
            
            # Count clickbait words
            clickbait_pattern = "|".join(clickbait_keywords)
            sentiment_df = sentiment_df.withColumn(
                "clickbait_score",
                size(split(col("title_lower"), clickbait_pattern)) - 1
            )
            
            # Overall sentiment
            sentiment_df = sentiment_df.withColumn(
                "sentiment_score",
                col("positive_score") - col("negative_score")
            )
            
            # Aggregate results
            sentiment_summary = sentiment_df.agg(
                avg("sentiment_score").alias("avg_sentiment"),
                avg("positive_score").alias("avg_positive"),
                avg("negative_score").alias("avg_negative"),
                avg("clickbait_score").alias("avg_clickbait"),
                count("*").alias("total_videos")
            ).collect()[0]
            
            return sentiment_summary.asDict()
            
        except Exception as e:
            print(f"[ERROR] Sentiment analysis failed: {str(e)}")
            return {}

    def detect_anomalous_videos(self, df):
        """Detect videos with unusual patterns"""
        try:
            # Calculate statistical measures
            stats = df.select(
                mean("views").alias("mean_views"),
                stddev("views").alias("std_views"),
                mean("likes").alias("mean_likes"),
                stddev("likes").alias("std_likes")
            ).collect()[0]
            
            # Define anomaly thresholds (3 standard deviations)
            view_threshold = stats["mean_views"] + 3 * stats["std_views"]
            like_threshold = stats["mean_likes"] + 3 * stats["std_likes"]
            
            # Find anomalous videos
            anomalous_df = df.filter(
                (col("views") > view_threshold) | 
                (col("likes") > like_threshold) |
                (col("dislikes") / (col("likes") + 1) > 0.5)  # High dislike ratio
            )
            
            anomaly_summary = anomalous_df.agg(
                count("*").alias("anomalous_count"),
                max("views").alias("max_views"),
                max("likes").alias("max_likes"),
                avg("dislikes").alias("avg_dislikes")
            ).collect()[0]
            
            # Get top anomalous videos
            top_anomalies = anomalous_df.select(
                "video_id", "title", "views", "likes", "dislikes", "country"
            ).orderBy(desc("views")).limit(10).collect()
            
            return {
                'summary': anomaly_summary.asDict(),
                'top_anomalies': [row.asDict() for row in top_anomalies],
                'thresholds': {
                    'view_threshold': float(view_threshold),
                    'like_threshold': float(like_threshold)
                }
            }
            
        except Exception as e:
            print(f"[ERROR] Anomaly detection failed: {str(e)}")
            return {}

    def analyze_category_performance(self, df):
        """Analyze performance patterns by category"""
        try:
            # Performance metrics by category
            category_stats = df.groupBy("category_id").agg(
                count("*").alias("video_count"),
                avg("views").alias("avg_views"),
                avg("likes").alias("avg_likes"),
                avg("comment_count").alias("avg_comments"),
                (sum("likes") / sum("views")).alias("avg_like_rate"),
                stddev("views").alias("view_variance")
            ).orderBy(desc("avg_views"))
            
            results = category_stats.collect()
            
            # Find best and worst performing categories
            best_category = results[0] if results else None
            worst_category = results[-1] if results else None
            
            return {
                'category_performance': [row.asDict() for row in results],
                'best_category': best_category.asDict() if best_category else None,
                'worst_category': worst_category.asDict() if worst_category else None,
                'total_categories': len(results)
            }
            
        except Exception as e:
            print(f"[ERROR] Category analysis failed: {str(e)}")
            return {}

    def save_ml_results_to_mongodb(self, ml_results):
        """Save ML analysis results to MongoDB"""
        try:
            # Clear existing ML results
            self.db.ml_analysis.delete_many({})
            
            # Add timestamp
            ml_results['analysis_timestamp'] = datetime.now()
            ml_results['analysis_type'] = 'comprehensive_ml_analysis'
            
            # Insert results
            self.db.ml_analysis.insert_one(ml_results)
            
            print("💾 ML analysis results saved to MongoDB")
            
        except Exception as e:
            print(f"[ERROR] Failed to save ML results: {str(e)}")

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
