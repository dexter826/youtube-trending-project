"""
Data Validation Processor
"""

from pyspark.sql.functions import col, when, count, isnan, isnull
from pyspark.sql.types import StringType, IntegerType, FloatType, BooleanType


class DataValidator:
    def __init__(self, spark_session):
        self.spark = spark_session

    def validate_schema(self, df):
        """Validate dataframe schema and data types"""
        expected_schema = {
            'video_id': StringType(),
            'trending_date': StringType(),
            'title': StringType(),
            'channel_title': StringType(),
            'category_id': IntegerType(),
            'publish_time': StringType(),
            'tags': StringType(),
            'views': IntegerType(),
            'likes': IntegerType(),
            'dislikes': IntegerType(),
            'comment_count': IntegerType(),
            'thumbnail_link': StringType(),
            'comments_disabled': BooleanType(),
            'ratings_disabled': BooleanType(),
            'video_error_or_removed': BooleanType(),
            'description': StringType(),
            'country': StringType()
        }

        # Check if all expected columns exist
        missing_cols = set(expected_schema.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Validate data types
        for col_name, expected_type in expected_schema.items():
            if df.schema[col_name].dataType != expected_type:
                print(f"Column {col_name} has type {df.schema[col_name].dataType}, expected {expected_type}")

        return True

    def check_data_quality(self, df):
        """Check data quality metrics"""
        total_rows = df.count()

        # Check for null values in critical columns
        critical_cols = ['video_id', 'title', 'channel_title', 'views', 'likes', 'dislikes']
        null_counts = {}

        for col_name in critical_cols:
            null_count = df.filter(isnull(col(col_name))).count()
            null_counts[col_name] = null_count

        # Check for negative values in numeric columns
        numeric_cols = ['views', 'likes', 'dislikes', 'comment_count']
        negative_counts = {}

        for col_name in numeric_cols:
            neg_count = df.filter(col(col_name) < 0).count()
            negative_counts[col_name] = neg_count

        # Check for duplicate video_ids
        duplicate_count = df.groupBy('video_id').count().filter(col('count') > 1).count()

        quality_report = {
            'total_rows': total_rows,
            'null_counts': null_counts,
            'negative_counts': negative_counts,
            'duplicate_video_ids': duplicate_count
        }

        return quality_report

    def clean_data(self, df):
        """Clean and standardize data"""
        # Remove rows with null video_id or title
        df_clean = df.filter(col('video_id').isNotNull() & col('title').isNotNull())

        # Replace negative values with 0 for numeric columns
        numeric_cols = ['views', 'likes', 'dislikes', 'comment_count']
        for col_name in numeric_cols:
            df_clean = df_clean.withColumn(col_name, when(col(col_name) < 0, 0).otherwise(col(col_name)))

        # Fill null values in non-critical columns
        df_clean = df_clean.fillna({
            'description': '',
            'tags': '',
            'channel_title': 'Unknown'
        })

        return df_clean