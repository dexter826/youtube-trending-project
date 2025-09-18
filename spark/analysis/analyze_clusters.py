"""
Cluster Analysis and Dynamic Naming
Tech Lead Implementation: Pure PySpark approach for big data consistency
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

from pyspark.sql.functions import col, count, mean, isnan, log
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

# Import existing managers for consistency
from ..core.spark_manager import get_spark_session, PRODUCTION_CONFIGS
from ..core.database_manager import get_database_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cluster_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    """
    Advanced cluster analysis with dynamic naming based on feature centroids.
    Tech Lead Decision: Separate concerns - analysis, naming, persistence.
    """

    def __init__(self):
        """Initialize with production configs"""
        self.spark = get_spark_session("ClusterAnalyzer", PRODUCTION_CONFIGS["ml_inference"])
        self.db = get_database_connection()
        self.mongo_client = self.db.client  # For proper cleanup
        self.hdfs_model_path = "hdfs://localhost:9000/youtube_trending/models/clustering"
        
        # Import path_config for proper path management
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        from config.paths import path_config
        
        self.output_path = path_config.SPARK_ANALYSIS_DIR / "cluster_analysis_results.json"
        self.cluster_names_path = path_config.SPARK_ANALYSIS_DIR / "cluster_names.json"

        # Feature columns used in clustering (from train_models.py)
        self.cluster_features = [
            "log_views", "log_likes", "log_dislikes", "log_comments",
            "like_ratio", "engagement_score", "title_length", "tag_count",
            "category_id", "publish_hour", "video_age_proxy"
        ]

        # Category mapping for context (from YouTube API)
        self.category_mapping = {
            1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music",
            15: "Pets & Animals", 17: "Sports", 18: "Short Movies",
            19: "Travel & Events", 20: "Gaming", 21: "Videoblogging",
            22: "People & Blogs", 23: "Comedy", 24: "Entertainment",
            25: "News & Politics", 26: "Howto & Style", 27: "Education",
            28: "Science & Technology", 29: "Nonprofits & Activism",
            30: "Movies", 31: "Anime/Animation", 32: "Action/Adventure",
            33: "Classics", 34: "Comedy", 35: "Documentary",
            36: "Drama", 37: "Family", 38: "Foreign", 39: "Horror",
            40: "Sci-Fi/Fantasy", 41: "Thriller", 42: "Shorts",
            43: "Shows", 44: "Trailers"
        }

    def load_model(self) -> PipelineModel:
        """Load trained clustering model from HDFS"""
        try:
            logger.info(f"Loading clustering model from {self.hdfs_model_path}")
            model = PipelineModel.load(self.hdfs_model_path)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_training_data(self) -> DataFrame:
        """Load training data from MongoDB with pure PySpark approach"""
        try:
            logger.info("Loading training data from MongoDB")
            raw_data = list(self.db.ml_features.find({}))

            if not raw_data:
                raise ValueError("No training data found in MongoDB")

            # Define explicit schema for consistency
            schema = StructType([
                StructField("views", DoubleType(), True),
                StructField("likes", DoubleType(), True),
                StructField("dislikes", DoubleType(), True),
                StructField("comment_count", DoubleType(), True),
                StructField("like_ratio", DoubleType(), True),
                StructField("engagement_score", DoubleType(), True),
                StructField("title_length", DoubleType(), True),
                StructField("tag_count", DoubleType(), True),
                StructField("category_id", DoubleType(), True),
                StructField("publish_hour", DoubleType(), True),
                StructField("video_age_proxy", DoubleType(), True),
                StructField("title", StringType(), True),
                StructField("tags", StringType(), True)
            ])

            # Clean and prepare data
            cleaned_data = []
            for record in raw_data:
                if '_id' in record:
                    del record['_id']
                
                cleaned_record = {}
                for key, value in record.items():
                    if value is None:
                        if key in ['views', 'likes', 'dislikes', 'comment_count', 'category_id', 'publish_hour', 'video_age_proxy', 'like_ratio', 'engagement_score', 'title_length', 'tag_count']:
                            cleaned_record[key] = 0.0
                        else:
                            cleaned_record[key] = ""
                    else:
                        # Ensure numeric types
                        if key in ['views', 'likes', 'dislikes', 'comment_count', 'category_id', 'publish_hour', 'video_age_proxy', 'like_ratio', 'engagement_score', 'title_length', 'tag_count']:
                            try:
                                cleaned_record[key] = float(value)
                            except (ValueError, TypeError):
                                cleaned_record[key] = 0.0
                        else:
                            cleaned_record[key] = str(value)
                
                cleaned_data.append(cleaned_record)

            # Create Spark DataFrame directly (no pandas)
            df = self.spark.createDataFrame(cleaned_data, schema)
            
            # Apply preprocessing (same as training)
            df = df.withColumn("log_views", log(col("views") + 1)) \
                   .withColumn("log_likes", log(col("likes") + 1)) \
                   .withColumn("log_dislikes", log(col("dislikes") + 1)) \
                   .withColumn("log_comments", log(col("comment_count") + 1))

            # Fill any remaining nulls
            df = df.na.fill(0.0)

            logger.info(f"Loaded {df.count()} training records")
            return df

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise

    def analyze_clusters(self, model: PipelineModel, df: DataFrame) -> Dict[str, Any]:
        """Analyze cluster centroids and assign dynamic names"""
        try:
            logger.info("Starting cluster analysis")

            # Transform and cache predictions
            predictions = model.transform(df).cache()
            total_records = predictions.count()
            logger.info(f"Total records for analysis: {total_records}")

            # Calculate cluster statistics
            cluster_stats = predictions.groupBy("cluster").agg(
                count("*").alias("count"),
                mean("log_views").alias("avg_log_views"),
                mean("log_likes").alias("avg_log_likes"),
                mean("log_dislikes").alias("avg_log_dislikes"),
                mean("log_comments").alias("avg_log_comments"),
                mean("like_ratio").alias("avg_like_ratio"),
                mean("engagement_score").alias("avg_engagement_score"),
                mean("title_length").alias("avg_title_length"),
                mean("tag_count").alias("avg_tag_count"),
                mean("category_id").alias("avg_category_id"),
                mean("publish_hour").alias("avg_publish_hour"),
                mean("video_age_proxy").alias("avg_video_age_proxy")
            ).orderBy("cluster")

            # Collect stats for analysis
            stats_list = cluster_stats.collect()
            cluster_analysis = {}

            for row in stats_list:
                cluster_id = row["cluster"]
                # Use asDict() for safe access to all fields
                stats = row.asDict()
                del stats["cluster"]  # Remove cluster key from stats

                # Determine dominant category
                avg_cat = stats["avg_category_id"]
                dominant_category = self.category_mapping.get(round(avg_cat), "Unknown")

                # Assign dynamic name based on feature analysis
                cluster_name = self._assign_cluster_name(stats, dominant_category)

                cluster_analysis[str(cluster_id)] = {
                    "name": cluster_name,
                    "stats": stats,
                    "dominant_category": dominant_category,
                    "sample_size": stats["count"]
                }

            logger.info("Cluster analysis completed")
            return cluster_analysis

        except Exception as e:
            logger.error(f"Cluster analysis failed: {e}")
            raise
        finally:
            try:
                predictions.unpersist()
            except Exception:
                pass

    def _assign_cluster_name(self, stats: Dict[str, float], dominant_category: str) -> str:
        """
        Dynamic naming based on feature percentiles and patterns with k=4
        Tech Lead Decision: Simplified naming for better user experience
        """
        name_parts = []

        # Analyze view engagement
        if stats["avg_log_views"] > 12:  # High views (log scale)
            name_parts.append("Popular")
        elif stats["avg_log_views"] > 10:
            name_parts.append("Trending")
        else:
            name_parts.append("Niche")

        # Analyze engagement
        if stats["avg_engagement_score"] > 0.05:
            name_parts.append("High-Engagement")
        elif stats["avg_engagement_score"] > 0.02:
            name_parts.append("Medium-Engagement")
        else:
            name_parts.append("Low-Engagement")

        # Add category context for specificity
        if dominant_category != "Unknown":
            if dominant_category == "Entertainment":
                name_parts.append("Entertainment")
            elif dominant_category == "Gaming":
                name_parts.append("Gaming")
            elif dominant_category == "Music":
                name_parts.append("Music")
            elif dominant_category == "News & Politics":
                name_parts.append("News")
            elif dominant_category == "Science & Technology":
                name_parts.append("Tech")
            elif dominant_category == "Education":
                name_parts.append("Educational")
            else:
                name_parts.append("Content")

        return " ".join(name_parts)

    def save_results(self, analysis: Dict[str, Any]):
        """Save analysis results to JSON file"""
        try:
            result = {
                "analysis_timestamp": datetime.now().isoformat(),
                "clusters": analysis,
                "metadata": {
                    "features_analyzed": self.cluster_features,
                    "total_clusters": len(analysis),
                    "method": "centroid-based dynamic naming"
                }
            }

            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to {self.output_path}")

            # Also save simplified mapping for backend use
            mapping = {cid: data["name"] for cid, data in analysis.items()}
            with open(self.cluster_names_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)

            logger.info(f"Cluster name mapping saved to {self.cluster_names_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

    def run_analysis(self):
        """Main execution method with proper error handling"""
        try:
            logger.info("Starting cluster analysis pipeline")

            # Load components
            model = self.load_model()
            df = self.load_training_data()

            # Perform analysis
            analysis = self.analyze_clusters(model, df)

            # Save results
            self.save_results(analysis)

            # Print summary
            print("\n=== CLUSTER ANALYSIS SUMMARY ===")
            for cluster_id, data in analysis.items():
                print(f"Cluster {cluster_id}: {data['name']}")
                print(f"  - Sample size: {data['stats']['count']}")
                print(f"  - Avg views (log): {data['stats']['avg_log_views']:.2f}")
                print(f"  - Avg engagement: {data['stats']['avg_engagement_score']:.4f}")
                print(f"  - Dominant category: {data['dominant_category']}")
                print()

            logger.info("Analysis completed successfully")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            sys.exit(1)
        finally:
            self.spark.stop()
            self.mongo_client.close()

def main():
    """Entry point"""
    analyzer = ClusterAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
