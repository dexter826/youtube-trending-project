"""
Centralized Spark Session Management
"""

from pyspark.sql import SparkSession


class SparkManager:
    """Singleton Spark session manager"""

    _instance = None
    _spark = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_spark_session(self, app_name="YouTubeTrending", config=None):
        """Get or create Spark session with configuration"""
        if self._spark is None:
            builder = SparkSession.builder \
                .appName(app_name) \
                .master("local[*]")

            # Base configuration
            base_config = {
                "spark.driver.memory": "4g",
                "spark.executor.memory": "4g",
                "spark.driver.maxResultSize": "2g",
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                "spark.hadoop.fs.defaultFS": "hdfs://localhost:9000"
            }

            # Apply custom config if provided
            if config:
                base_config.update(config)

            # Configure Spark
            for key, value in base_config.items():
                builder = builder.config(key, value)

            self._spark = builder.getOrCreate()
            self._spark.sparkContext.setLogLevel("WARN")

        return self._spark

    def stop_session(self):
        """Stop Spark session"""
        if self._spark:
            self._spark.stop()
            self._spark = None


# Global instance
_manager = SparkManager()


def get_spark_session(app_name="YouTubeTrending", config=None):
    """Get Spark session"""
    return _manager.get_spark_session(app_name, config)


def stop_spark_session():
    """Stop Spark session"""
    _manager.stop_session()


# Production configurations
PRODUCTION_CONFIGS = {
    "data_processing": {
        "spark.driver.memory": "6g",
        "spark.executor.memory": "4g",
        "spark.driver.maxResultSize": "4g"
    },
    "ml_training": {
        "spark.driver.memory": "8g",
        "spark.executor.memory": "6g",
        "spark.driver.maxResultSize": "4g"
    },
    "ml_inference": {
        "spark.driver.memory": "2g",
        "spark.executor.memory": "2g",
        "spark.driver.maxResultSize": "1g"
    }
}