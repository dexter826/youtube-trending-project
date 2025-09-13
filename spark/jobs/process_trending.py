"""
YouTube Trending Data Processing
"""

import os
import sys
from datetime import datetime

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pandas as pd

from core.spark_manager import get_spark_session, PRODUCTION_CONFIGS
from core.database_manager import get_database_connection


class YouTubeTrendingProcessor:
    def __init__(self):
        """Initialize processor with Spark and database connections"""
        self.spark = get_spark_session("YouTubeTrendingProcessor", PRODUCTION_CONFIGS["data_processing"])
        self.db = get_database_connection()
        # Get the MongoDB client from the manager for proper cleanup
        from core.database_manager import _manager
        self.mongo_client = _manager._client
        self.hdfs_base_path = "hdfs://localhost:9000/youtube_trending"

    def load_csv_data_from_hdfs(self):
        """Load CSV files from HDFS"""
        countries = ['US', 'CA', 'GB', 'DE', 'FR', 'IN', 'JP', 'KR', 'MX', 'RU']
        all_data = []

        for country in countries:
            hdfs_path = f"{self.hdfs_base_path}/raw_data/{country}/{country}videos.csv"

            try:
                df = self.spark.read.csv(
                    hdfs_path,
                    header=True,
                    schema=self._get_schema(),
                    timestampFormat="yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
                )

                df = self._process_dataframe(df, country)
                all_data.append(df)

            except Exception as e:
                raise RuntimeError(f"HDFS loading failed for {country}")

        return self._combine_dataframes(all_data)

    def load_csv_data_from_local(self, data_path):
        """Load CSV files from local directory"""
        countries = ['US', 'CA', 'GB', 'DE', 'FR', 'IN', 'JP', 'KR', 'MX', 'RU']
        all_data = []

        for country in countries:
            csv_path = os.path.join(data_path, f"{country}videos.csv")

            if os.path.exists(csv_path):
                try:
                    df = self.spark.read.csv(
                        csv_path,
                        header=True,
                        schema=self._get_schema(),
                        timestampFormat="yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
                    )

                    df = self._process_dataframe(df, country)
                    all_data.append(df)

                except Exception as e:
                    continue

        return self._combine_dataframes(all_data)

    def _get_schema(self):
        """Get CSV schema definition"""
        return StructType([
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

    def _process_dataframe(self, df, country):
        """Process dataframe with common transformations"""
        return df.withColumn("country", lit(country)) \
                 .withColumn("trending_date_parsed", to_date(col("trending_date"), "yy.dd.MM")) \
                 .filter(col("video_id").isNotNull() & col("title").isNotNull() &
                        col("views").isNotNull() & (col("views") >= 0))

    def _combine_dataframes(self, dataframes):
        """Combine multiple dataframes"""
        if not dataframes:
            return None

        combined_df = dataframes[0]
        for df in dataframes[1:]:
            combined_df = combined_df.unionByName(df)

        return combined_df

    def save_raw_data_to_mongodb(self, df):
        """Save raw data to MongoDB"""
        self.db.raw_videos.delete_many({})

        batch = []
        batch_size = 5000

        def normalize_record(row_dict):
            for k, v in list(row_dict.items()):
                if isinstance(v, float) and v != v:
                    row_dict[k] = None
                if k == 'trending_date_parsed' and v is not None:
                    row_dict[k] = v.strftime('%Y-%m-%d') if hasattr(v, 'strftime') else str(v)
            return row_dict

        for row in df.toLocalIterator():
            rec = normalize_record(row.asDict(recursive=True))
            batch.append(rec)
            if len(batch) >= batch_size:
                self.db.raw_videos.insert_many(batch)
                batch = []

        if batch:
            self.db.raw_videos.insert_many(batch)

    def process_trending_analysis(self, df):
        """Process trending analysis by country and date"""
        results = []

        countries_dates = df.select("country", "trending_date_parsed").distinct().collect()

        for row in countries_dates:
            country = row.country
            date = row.trending_date_parsed

            if date is None:
                continue

            filtered_df = df.filter((col("country") == country) & (col("trending_date_parsed") == date))

            top_videos = filtered_df.orderBy(col("views").desc()) \
                                   .limit(10) \
                                   .select("video_id", "title", "channel_title", "views",
                                          "likes", "dislikes", "comment_count", "category_id", "tags") \
                                   .collect()

            stats = filtered_df.agg(
                count("*").alias("total_videos"),
                sum("views").alias("total_views"),
                avg("views").alias("avg_views"),
                max("views").alias("max_views"),
                sum("likes").alias("total_likes"),
                sum("comment_count").alias("total_comments")
            ).collect()[0]

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

        if results:
            self.db.trending_results.delete_many({})
            self.db.trending_results.insert_many(results)

        return results

    def generate_wordcloud_data(self, df):
        """Generate wordcloud data from video titles"""
        results = []

        countries_dates = df.select("country", "trending_date_parsed").distinct().collect()

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

            filtered_df = df.filter((col("country") == country) & (col("trending_date_parsed") == date))

            stop_words_list = list(stop_words)
            words_df = (
                filtered_df
                .select("title")
                .where(col("title").isNotNull())
                .select(regexp_replace(lower(col("title")), r"[^\p{L}0-9\s]", " ").alias("title_clean"))
                .select(regexp_replace(col("title_clean"), r"\s+", " ").alias("title_clean"))
                .select(split(col("title_clean"), r"\s").alias("words"))
                .select(explode(col("words")).alias("word"))
                .select(trim(col("word")).alias("word"))
                .filter((col("word") != "") & (length(col("word")) > 2))
                .filter(col("word").rlike(r"^[\p{L}0-9]+$"))
                .filter(~col("word").isin(stop_words_list))
            )

            counts_df = words_df.groupBy("word").count().orderBy(col("count").desc()).limit(50)
            top_words = [(r['word'], r['count']) for r in counts_df.collect()]

            wordcloud_data = {
                "country": country,
                "date": date.strftime("%Y-%m-%d") if date else None,
                "processed_at": datetime.now().isoformat(),
                "words": [{"text": word, "value": count} for word, count in top_words]
            }

            results.append(wordcloud_data)

        if results:
            self.db.wordcloud_data.delete_many({})
            self.db.wordcloud_data.insert_many(results)

        return results

    def create_ml_features(self, df):
        """Create ML features from raw data and save to MongoDB"""
        # Time-based Features
        df = df.withColumn("publish_hour", hour(to_timestamp(col("publish_time"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")))
        df = df.withColumn("publish_day_of_week", dayofweek(to_timestamp(col("publish_time"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")))
        
        # Calculate video age proxy based on trending date vs publish date
        df = df.withColumn("publish_date", to_date(to_timestamp(col("publish_time"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")))
        df = df.withColumn("video_age_days", datediff(col("trending_date_parsed"), col("publish_date")))
        df = df.withColumn("video_age_proxy", 
                          when(col("video_age_days") <= 1, 1)  # Very new (0-1 days)
                          .when(col("video_age_days") <= 7, 2)  # New (2-7 days)  
                          .when(col("video_age_days") <= 30, 3) # Recent (8-30 days)
                          .otherwise(4))  # Older (>30 days)
        
        # Engagement Features
        df = df.withColumn("like_ratio", when(col("views") > 0, col("likes") / col("views")).otherwise(0))
        df = df.withColumn("dislike_ratio", when(col("views") > 0, col("dislikes") / col("views")).otherwise(0))
        df = df.withColumn("comment_ratio", when(col("views") > 0, col("comment_count") / col("views")).otherwise(0))
        df = df.withColumn("engagement_score", when(col("views") > 0, (col("likes") + col("comment_count")) / col("views")).otherwise(0))

        # Content Features
        df = df.withColumn("title_length", length(col("title")))
        df = df.withColumn("has_caps", when(col("title").rlike("[A-Z]{3,}"), 1).otherwise(0))
        df = df.withColumn("tag_count", when(col("tags").isNotNull(), size(split(col("tags"), "\\|"))).otherwise(0))

        # Trending Target Variable
        window_spec = Window.partitionBy("country", "trending_date_parsed").orderBy(col("views").desc())
        df = df.withColumn("view_rank", row_number().over(window_spec))

        total_videos = df.groupBy("country", "trending_date_parsed").agg(count("*").alias("total_videos_per_day"))
        df = df.join(total_videos, ["country", "trending_date_parsed"], "left")
        df = df.withColumn("trending_threshold", greatest(lit(1), least(lit(20), (col("total_videos_per_day") * 0.2).cast("int"))))
        df = df.withColumn("is_trending", when(col("view_rank") <= col("trending_threshold"), 1).otherwise(0))

        # Views prediction target
        df = df.withColumn("log_views", log(col("views") + 1))

        # Select final feature columns
        feature_cols = [
            "video_id", "country", "trending_date_parsed",
            "views", "likes", "dislikes", "comment_count", "category_id",
            "like_ratio", "dislike_ratio", "comment_ratio", "engagement_score",
            "title_length", "has_caps", "tag_count",
            "publish_hour", "publish_day_of_week", "video_age_days", "video_age_proxy",
            "is_trending", "log_views"
        ]

        ml_df = df.select(feature_cols)
        pandas_df = ml_df.toPandas()

        self.db.ml_features.delete_many({})

        ml_features = []
        for _, row in pandas_df.iterrows():
            record = {
                "video_id": row["video_id"],
                "country": row["country"],
                "trending_date": row["trending_date_parsed"].strftime("%Y-%m-%d") if pd.notnull(row["trending_date_parsed"]) else None,
                "views": int(row["views"]) if pd.notnull(row["views"]) else 0,
                "likes": int(row["likes"]) if pd.notnull(row["likes"]) else 0,
                "dislikes": int(row["dislikes"]) if pd.notnull(row["dislikes"]) else 0,
                "comment_count": int(row["comment_count"]) if pd.notnull(row["comment_count"]) else 0,
                "category_id": int(row["category_id"]) if pd.notnull(row["category_id"]) else 0,
                "like_ratio": float(row["like_ratio"]) if pd.notnull(row["like_ratio"]) else 0.0,
                "dislike_ratio": float(row["dislike_ratio"]) if pd.notnull(row["dislike_ratio"]) else 0.0,
                "comment_ratio": float(row["comment_ratio"]) if pd.notnull(row["comment_ratio"]) else 0.0,
                "engagement_score": float(row["engagement_score"]) if pd.notnull(row["engagement_score"]) else 0.0,
                "title_length": int(row["title_length"]) if pd.notnull(row["title_length"]) else 0,
                "has_caps": int(row["has_caps"]) if pd.notnull(row["has_caps"]) else 0,
                "tag_count": int(row["tag_count"]) if pd.notnull(row["tag_count"]) else 0,
                "publish_hour": int(row["publish_hour"]) if pd.notnull(row["publish_hour"]) else 12,
                "publish_day_of_week": int(row["publish_day_of_week"]) if pd.notnull(row["publish_day_of_week"]) else 1,
                "video_age_days": int(row["video_age_days"]) if pd.notnull(row["video_age_days"]) else 7,
                "video_age_proxy": int(row["video_age_proxy"]) if pd.notnull(row["video_age_proxy"]) else 2,
                "is_trending": int(row["is_trending"]) if pd.notnull(row["is_trending"]) else 0,
                "log_views": float(row["log_views"]) if pd.notnull(row["log_views"]) else 0.0,
                "processed_at": datetime.now().isoformat()
            }
            ml_features.append(record)

        if ml_features:
            batch_size = 5000
            for i in range(0, len(ml_features), batch_size):
                batch = ml_features[i:i + batch_size]
                self.db.ml_features.insert_many(batch)

        return len(ml_features)

    def run_full_pipeline(self, data_path=None):
        """Run the complete data processing pipeline"""
        try:
            df = self.load_csv_data_from_hdfs()

            if df is None:
                if data_path and os.path.exists(data_path):
                    df = self.load_csv_data_from_local(data_path)
                else:
                    return False

            if df is None:
                return False

            self.save_raw_data_to_mongodb(df)
            ml_features_count = self.create_ml_features(df)
            trending_results = self.process_trending_analysis(df)
            wordcloud_results = self.generate_wordcloud_data(df)

            return True

        except Exception as e:
            return False
        finally:
            self.spark.stop()
            self.mongo_client.close()

def main():
    """Main execution function"""
    if len(sys.argv) > 2:
        sys.exit(1)

    data_path = sys.argv[1] if len(sys.argv) == 2 else "data"

    if not os.path.exists(data_path):
        sys.exit(1)

    processor = YouTubeTrendingProcessor()
    success = processor.run_full_pipeline(data_path)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
