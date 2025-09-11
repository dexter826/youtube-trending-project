"""
Spark Performance Configuration
Description: Production-ready Spark configurations for optimal performance
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel

# Production Spark configuration
PRODUCTION_SPARK_CONFIG = {
    # Memory management
    "spark.driver.memory": "8g",
    "spark.executor.memory": "6g", 
    "spark.driver.maxResultSize": "4g",
    "spark.executor.memoryFraction": "0.8",
    "spark.executor.memoryStorageLevel": "MEMORY_AND_DISK_SER",
    
    # Core allocation
    "spark.executor.cores": "4",
    "spark.executor.instances": "2",
    "spark.default.parallelism": "8",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    
    # Adaptive Query Execution (AQE)
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.adaptive.localShuffleReader.enabled": "true",
    
    # Broadcast optimization
    "spark.sql.autoBroadcastJoinThreshold": "50MB",
    "spark.sql.broadcastTimeout": "600s",
    
    # Serialization
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.kryo.registrationRequired": "false",
    
    # File format optimization
    "spark.sql.parquet.compression.codec": "snappy",
    "spark.sql.adaptive.coalescePartitions.minPartitionNum": "1",
    
    # Dynamic allocation
    "spark.dynamicAllocation.enabled": "true",
    "spark.dynamicAllocation.minExecutors": "1",
    "spark.dynamicAllocation.maxExecutors": "4",
    
    # HDFS optimization
    "spark.hadoop.fs.defaultFS": "hdfs://namenode:9000",
    "spark.sql.warehouse.dir": "hdfs://namenode:9000/spark-warehouse"
}

class OptimizedSparkSession:
    def __init__(self, app_name="OptimizedYouTubeAnalytics"):
        """Create optimized Spark session for Big Data processing"""
        self.spark = self._create_optimized_session(app_name)
        print("[OK] Optimized Spark session created")

    def _create_optimized_session(self, app_name):
        """Create Spark session with all optimizations"""
        builder = SparkSession.builder.appName(app_name)
        
        # Apply all optimized configurations
        for key, value in OPTIMIZED_SPARK_CONFIG.items():
            builder = builder.config(key, value)
        
        # Enable Hive support for advanced SQL features
        spark = builder.enableHiveSupport().getOrCreate()
        
        # Set log level to reduce noise
        spark.sparkContext.setLogLevel("WARN")
        
        return spark

    def get_session(self):
        """Get the optimized Spark session"""
        return self.spark

class PartitionedDataLoader:
    def __init__(self, spark_session):
        """Initialize with optimized Spark session"""
        self.spark = spark_session
        
    def load_and_partition_youtube_data(self, hdfs_path="hdfs://namenode:9000/youtube_trending/raw_data"):
        """Load YouTube data with optimal partitioning strategy"""
        print("📁 Loading and partitioning YouTube data...")
        
        try:
            # Load all CSV files from HDFS
            df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(f"{hdfs_path}/*.csv")
            
            # Add partitioning columns
            df = df.withColumn("country_partition", col("country")) \
                   .withColumn("year_partition", year(to_timestamp(col("publish_time"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))) \
                   .withColumn("month_partition", month(to_timestamp(col("publish_time"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")))
            
            # Optimal partitioning by country and time
            partitioned_df = df.repartition(8, "country_partition", "year_partition")
            
            # Cache for reuse with optimized storage level
            partitioned_df.cache()
            
            print(f"[OK] Data loaded and partitioned: {partitioned_df.count()} records")
            print(f"[OK] Partitions: {partitioned_df.rdd.getNumPartitions()}")
            
            return partitioned_df
            
        except Exception as e:
            print(f"[ERROR] Failed to load partitioned data: {str(e)}")
            return None

    def save_partitioned_data(self, df, output_path="hdfs://namenode:9000/youtube_trending/partitioned_data"):
        """Save data with partitioning for optimal query performance"""
        try:
            # Write partitioned data to HDFS in Parquet format
            df.write.mode("overwrite") \
              .partitionBy("country_partition", "year_partition") \
              .parquet(output_path)
            
            print(f"[OK] Partitioned data saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save partitioned data: {str(e)}")
            return False

class OptimizedMLProcessor:
    def __init__(self, spark_session):
        """Initialize optimized ML processor"""
        self.spark = spark_session
        
    def process_without_collect(self, df):
        """Process data without using collect() - Big Data best practice"""
        print("🔧 Processing data without collect() operations...")
        
        try:
            # Use aggregations instead of collect()
            summary_stats = df.agg(
                count("*").alias("total_videos"),
                avg("views").alias("avg_views"),
                max("views").alias("max_views"),
                stddev("views").alias("stddev_views")
            )
            
            # Use window functions for rankings without collect()
            window_spec = Window.partitionBy("country").orderBy(desc("views"))
            ranked_df = df.withColumn("rank", row_number().over(window_spec))
            
            # Filter top videos per country without collect()
            top_videos = ranked_df.filter(col("rank") <= 10)
            
            # Use broadcast for small lookup tables
            category_mapping = self.spark.createDataFrame([
                (10, "Music"), (23, "Comedy"), (24, "Entertainment"),
                (25, "News"), (26, "Howto"), (28, "Science")
            ], ["category_id", "category_name"])
            
            # Broadcast join instead of collect + join
            enriched_df = top_videos.join(broadcast(category_mapping), "category_id", "left")
            
            # Write results directly to storage
            output_path = "hdfs://namenode:9000/youtube_trending/processed_results"
            enriched_df.write.mode("overwrite").parquet(output_path)
            
            # Return only summary statistics, not the full dataset
            return summary_stats.collect()[0].asDict()
            
        except Exception as e:
            print(f"[ERROR] Optimized processing failed: {str(e)}")
            return None

    def optimize_ml_pipeline(self, df):
        """Optimize ML pipeline performance"""
        print("🤖 Optimizing ML pipeline...")
        
        try:
            from pyspark.ml.feature import VectorAssembler, StandardScaler
            from pyspark.ml.clustering import KMeans
            from pyspark.ml import Pipeline
            
            # Select relevant features
            feature_cols = ["views", "likes", "comment_count", "title_length"]
            
            # Create ML pipeline
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
            scaler = StandardScaler(inputCol="raw_features", outputCol="features")
            kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=5, seed=42)
            
            # Create pipeline
            pipeline = Pipeline(stages=[assembler, scaler, kmeans])
            
            # Cache input data for ML training
            df.cache()
            
            # Train model
            model = pipeline.fit(df)
            
            # Get predictions without collect()
            predictions = model.transform(df)
            
            # Save model to HDFS
            model_path = "hdfs://namenode:9000/youtube_trending/optimized_models/kmeans_optimized"
            model.write().overwrite().save(model_path)
            
            # Return cluster statistics instead of full predictions
            cluster_stats = predictions.groupBy("cluster").agg(
                count("*").alias("cluster_size"),
                avg("views").alias("avg_views")
            ).collect()
            
            return [row.asDict() for row in cluster_stats]
            
        except Exception as e:
            print(f"[ERROR] ML pipeline optimization failed: {str(e)}")
            return None

class ClusterModeConfig:
    """Configuration for running Spark in cluster mode"""
    
    @staticmethod
    def get_cluster_config():
        """Get configuration for cluster deployment"""
        return {
            # Cluster mode settings
            "spark.master": "spark://spark-master:7077",
            "spark.submit.deployMode": "cluster",
            
            # Resource allocation for cluster
            "spark.executor.instances": "3",
            "spark.executor.cores": "2", 
            "spark.executor.memory": "4g",
            "spark.driver.memory": "2g",
            
            # Network optimization for cluster
            "spark.network.timeout": "800s",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes": "256MB",
            
            # Shuffle optimization
            "spark.sql.shuffle.partitions": "200",
            "spark.reducer.maxSizeInFlight": "96m",
            "spark.shuffle.io.maxRetries": "5",
            
            # HDFS optimization for cluster
            "spark.hadoop.dfs.client.read.shortcircuit": "true",
            "spark.hadoop.dfs.client.read.shortcircuit.skip.checksum": "true"
        }

def create_production_spark_session():
    """Create production-ready Spark session with all optimizations"""
    print("🚀 Creating production Spark session...")
    
    config = OPTIMIZED_SPARK_CONFIG.copy()
    
    # Add cluster config if running in cluster mode
    if os.getenv("SPARK_CLUSTER_MODE", "false").lower() == "true":
        config.update(ClusterModeConfig.get_cluster_config())
    
    builder = SparkSession.builder.appName("ProductionYouTubeAnalytics")
    
    # Apply all configurations
    for key, value in config.items():
        builder = builder.config(key, value)
    
    # Create session
    spark = builder.enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    print("[OK] Production Spark session created with optimizations")
    return spark

def main():
    """Test optimized Spark configurations"""
    # Create optimized session
    optimized_session = OptimizedSparkSession()
    spark = optimized_session.get_session()
    
    # Test partitioned data loading
    loader = PartitionedDataLoader(spark)
    df = loader.load_and_partition_youtube_data()
    
    if df is not None:
        # Test optimized processing
        processor = OptimizedMLProcessor(spark)
        stats = processor.process_without_collect(df)
        
        if stats:
            print(f"📊 Processing statistics: {stats}")
        
        # Test ML optimization
        ml_results = processor.optimize_ml_pipeline(df)
        if ml_results:
            print(f"🤖 ML clustering results: {ml_results}")
    
    print("✅ Spark optimization testing completed")

if __name__ == "__main__":
    main()
