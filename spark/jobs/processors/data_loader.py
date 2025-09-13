"""
Data Loading Processor
"""

import sys
import os
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, BooleanType
from pyspark.sql.functions import lit, to_date, col

# Add spark directory to path
spark_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, spark_dir)

from core.spark_manager import get_spark_session, PRODUCTION_CONFIGS
from core.database_manager import get_database_connection


class DataLoader:
    def __init__(self, spark_session=None):
        self.spark = spark_session or get_spark_session("DataLoader", PRODUCTION_CONFIGS["data_processing"])
        self.db = get_database_connection()
        self.countries = ['US', 'CA', 'GB', 'DE', 'FR', 'IN', 'JP', 'KR', 'MX', 'RU']
        # Get the MongoDB client from the manager for proper cleanup
        from core.database_manager import _manager
        self.mongo_client = _manager._client
        self.hdfs_base_path = "hdfs://localhost:9000/youtube_trending"

    def get_schema(self):
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

    def load_csv_data_from_hdfs(self, hdfs_base_path="hdfs://localhost:9000/youtube_trending"):
        """Load CSV files from HDFS"""
        all_data = []

        for country in self.countries:
            hdfs_path = f"{hdfs_base_path}/raw_data/{country}/{country}videos.csv"

            try:
                df = self.spark.read.csv(
                    hdfs_path,
                    header=True,
                    schema=self.get_schema(),
                    timestampFormat="yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
                )

                df = self._process_dataframe(df, country)
                all_data.append(df)

            except Exception as e:
                continue

        return self._combine_dataframes(all_data)

    def load_csv_data_from_local(self, data_path):
        """Load CSV files from local directory"""
        all_data = []

        for country in self.countries:
            csv_path = os.path.join(data_path, f"{country}videos.csv")

            if os.path.exists(csv_path):
                try:
                    # Force local filesystem for this operation
                    local_spark = self.spark

                    df = local_spark.read.csv(
                        f"file:///{csv_path}",  # Explicit file:// protocol
                        header=True,
                        schema=self.get_schema(),
                        timestampFormat="yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
                    )

                    df = self._process_dataframe(df, country)
                    all_data.append(df)

                except Exception as e:
                    continue

        return self._combine_dataframes(all_data)

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