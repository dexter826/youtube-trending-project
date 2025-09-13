"""
Data Processing Processor
"""

from datetime import datetime
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pandas as pd


class DataProcessor:
    def __init__(self, spark_session):
        self.spark = spark_session

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

        return results

    def create_ml_features(self, df):
        """Create ML features from raw data"""
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

        return ml_features