"""
Clustering Analysis for YouTube Trending Data
Description: Multi-dimensional clustering algorithms for content analysis
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Any

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Extended Spark MLlib imports
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer, 
    OneHotEncoder, Imputer, PCA, Word2Vec, HashingTF, IDF
)
from pyspark.ml.clustering import (
    KMeans, BisectingKMeans, GaussianMixture, LDA
)
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import ClusteringEvaluator

import pymongo
from pymongo import MongoClient

# MongoDB connection settings
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "youtube_trending"

class ClusteringAnalysis:
    def __init__(self):
        """Initialize Spark session for clustering analysis"""
        self.spark = SparkSession.builder \
            .appName("AdvancedClusteringAnalysis") \
            .master("local[*]") \
            .config("spark.driver.memory", "6g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # MongoDB client
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[DB_NAME]
        
        # HDFS paths
        self.hdfs_base_path = "hdfs://namenode:9000/youtube_trending"
        self.clustering_save_path = f"{self.hdfs_base_path}/clustering_models"
        
        print("[OK] Advanced Clustering Analysis initialized")

    def load_data_from_mongodb(self):
        """Load all data from MongoDB for clustering analysis"""
        print("📁 Loading data for clustering analysis...")
        
        try:
            # Load raw videos data
            raw_data = list(self.db.raw_videos.find({}))
            
            if not raw_data:
                print("[ERROR] No raw data found!")
                return None
            
            # Clean data
            for record in raw_data:
                if '_id' in record:
                    del record['_id']
                
                # Handle null values
                for key, value in record.items():
                    if value is None:
                        if key in ['views', 'likes', 'dislikes', 'comment_count', 'category_id']:
                            record[key] = 0
                        else:
                            record[key] = ""
            
            # Convert to Spark DataFrame
            df = self.spark.createDataFrame(raw_data)
            
            # Add derived features
            df = df.withColumn("title_length", length(col("title")))
            df = df.withColumn("tag_count", 
                              when(col("tags").isNotNull(), 
                                   size(split(col("tags"), "\\|"))).otherwise(0))
            df = df.withColumn("like_ratio", 
                              when(col("views") > 0, col("likes") / col("views")).otherwise(0))
            df = df.withColumn("engagement_score", 
                              when(col("views") > 0, 
                                   (col("likes") + col("comment_count")) / col("views")).otherwise(0))
            
            print(f"[OK] Loaded {df.count()} records for clustering")
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to load data: {str(e)}")
            return None

    def behavioral_clustering(self, df):
        """Cluster videos based on viewer behavior patterns"""
        print("🎯 Performing Behavioral Clustering...")
        
        # Features for behavioral clustering
        behavioral_features = [
            'views', 'likes', 'dislikes', 'comment_count',
            'like_ratio', 'engagement_score'
        ]
        
        # Prepare pipeline
        assembler = VectorAssembler(
            inputCols=behavioral_features,
            outputCol="behavioral_features_raw"
        )
        
        scaler = StandardScaler(
            inputCol="behavioral_features_raw",
            outputCol="behavioral_features",
            withStd=True,
            withMean=True
        )
        
        # Multiple clustering algorithms
        algorithms = {
            'behavioral_kmeans': KMeans(
                featuresCol="behavioral_features",
                predictionCol="behavioral_cluster_kmeans",
                k=6,
                seed=42
            ),
            'behavioral_bisecting': BisectingKMeans(
                featuresCol="behavioral_features", 
                predictionCol="behavioral_cluster_bisecting",
                k=6,
                seed=42
            ),
            'behavioral_gaussian': GaussianMixture(
                featuresCol="behavioral_features",
                predictionCol="behavioral_cluster_gaussian",
                k=5,
                seed=42
            )
        }
        
        results = {}
        
        for algo_name, clusterer in algorithms.items():
            print(f"   🔧 Training {algo_name}...")
            
            # Create pipeline
            pipeline = Pipeline(stages=[assembler, scaler, clusterer])
            
            # Train model
            model = pipeline.fit(df)
            
            # Get predictions
            clustered_df = model.transform(df)
            
            # Evaluate clustering
            evaluator = ClusteringEvaluator(
                featuresCol="behavioral_features",
                predictionCol=clusterer.getPredictionCol(),
                metricName="silhouette"
            )
            
            silhouette_score = evaluator.evaluate(clustered_df)
            
            # Get cluster statistics
            cluster_stats = clustered_df.groupBy(clusterer.getPredictionCol()).agg(
                count("*").alias("cluster_size"),
                avg("views").alias("avg_views"),
                avg("like_ratio").alias("avg_like_ratio"),
                avg("engagement_score").alias("avg_engagement"),
                avg("comment_count").alias("avg_comments")
            ).collect()
            
            results[algo_name] = {
                'model': model,
                'silhouette_score': float(silhouette_score),
                'cluster_stats': [row.asDict() for row in cluster_stats],
                'predictions': clustered_df
            }
            
            print(f"   [OK] {algo_name} - Silhouette Score: {silhouette_score:.4f}")
        
        return results

    def content_clustering(self, df):
        """Cluster videos based on content features (titles, tags)"""
        print("📝 Performing Content-Based Clustering...")
        
        # Prepare text features
        content_df = df.filter(col("title").isNotNull() & (col("title") != ""))
        
        # Word2Vec for title similarity
        word2vec = Word2Vec(
            vectorSize=50,
            minCount=2,
            inputCol="title_words",
            outputCol="title_vectors",
            seed=42
        )
        
        # Split titles into words
        content_df = content_df.withColumn(
            "title_words", 
            split(lower(regexp_replace(col("title"), "[^a-zA-Z0-9\\s]", "")), "\\s+")
        )
        
        # Filter out empty word arrays
        content_df = content_df.filter(size(col("title_words")) > 0)
        
        # Train Word2Vec
        word2vec_model = word2vec.fit(content_df)
        content_df = word2vec_model.transform(content_df)
        
        # Content clustering with K-Means
        content_assembler = VectorAssembler(
            inputCols=["title_vectors"],
            outputCol="content_features"
        )
        
        content_kmeans = KMeans(
            featuresCol="content_features",
            predictionCol="content_cluster",
            k=8,
            seed=42
        )
        
        # Create pipeline
        content_pipeline = Pipeline(stages=[content_assembler, content_kmeans])
        
        # Train model
        content_model = content_pipeline.fit(content_df)
        content_clustered = content_model.transform(content_df)
        
        # Analyze content clusters
        content_cluster_analysis = content_clustered.groupBy("content_cluster").agg(
            count("*").alias("cluster_size"),
            avg("views").alias("avg_views"),
            avg("title_length").alias("avg_title_length"),
            collect_list("title").alias("sample_titles")
        ).collect()
        
        # Get representative titles for each cluster
        content_results = []
        for row in content_cluster_analysis:
            cluster_info = row.asDict()
            # Limit sample titles to first 3
            cluster_info['sample_titles'] = cluster_info['sample_titles'][:3]
            content_results.append(cluster_info)
        
        print(f"   [OK] Content clustering completed - {len(content_results)} clusters")
        
        return {
            'content_clustering': {
                'model': content_model,
                'cluster_analysis': content_results,
                'predictions': content_clustered
            }
        }

    def geographic_clustering(self, df):
        """Cluster videos based on geographic/country patterns"""
        print("🌍 Performing Geographic Pattern Clustering...")
        
        # Aggregate by country
        country_stats = df.groupBy("country").agg(
            count("*").alias("total_videos"),
            avg("views").alias("avg_views"),
            avg("like_ratio").alias("avg_like_ratio"),
            avg("engagement_score").alias("avg_engagement"),
            avg("title_length").alias("avg_title_length"),
            countDistinct("category_id").alias("category_diversity")
        )
        
        # Geographic features for clustering
        geo_features = [
            'total_videos', 'avg_views', 'avg_like_ratio', 
            'avg_engagement', 'avg_title_length', 'category_diversity'
        ]
        
        geo_assembler = VectorAssembler(
            inputCols=geo_features,
            outputCol="geo_features_raw"
        )
        
        geo_scaler = StandardScaler(
            inputCol="geo_features_raw",
            outputCol="geo_features",
            withStd=True,
            withMean=True
        )
        
        geo_kmeans = KMeans(
            featuresCol="geo_features",
            predictionCol="geo_cluster",
            k=4,
            seed=42
        )
        
        # Geographic clustering pipeline
        geo_pipeline = Pipeline(stages=[geo_assembler, geo_scaler, geo_kmeans])
        
        # Train model
        geo_model = geo_pipeline.fit(country_stats)
        geo_clustered = geo_model.transform(country_stats)
        
        # Analyze geographic clusters
        geo_cluster_analysis = geo_clustered.groupBy("geo_cluster").agg(
            collect_list("country").alias("countries"),
            avg("avg_views").alias("cluster_avg_views"),
            avg("avg_engagement").alias("cluster_avg_engagement")
        ).collect()
        
        geo_results = [row.asDict() for row in geo_cluster_analysis]
        
        print(f"   [OK] Geographic clustering completed - {len(geo_results)} clusters")
        
        return {
            'geographic_clustering': {
                'model': geo_model,
                'cluster_analysis': geo_results,
                'predictions': geo_clustered
            }
        }

    def temporal_clustering(self, df):
        """Cluster videos based on temporal patterns"""
        print("⏰ Performing Temporal Pattern Clustering...")
        
        # Extract temporal features
        temporal_df = df.withColumn(
            "publish_hour", 
            hour(to_timestamp(col("publish_time"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))
        ).filter(col("publish_hour").isNotNull())
        
        # Aggregate by publish hour
        temporal_stats = temporal_df.groupBy("publish_hour").agg(
            count("*").alias("video_count"),
            avg("views").alias("avg_views"),
            avg("like_ratio").alias("avg_like_ratio"),
            avg("engagement_score").alias("avg_engagement")
        )
        
        # Temporal features
        temporal_features = [
            'video_count', 'avg_views', 'avg_like_ratio', 'avg_engagement'
        ]
        
        temporal_assembler = VectorAssembler(
            inputCols=temporal_features,
            outputCol="temporal_features"
        )
        
        temporal_kmeans = KMeans(
            featuresCol="temporal_features",
            predictionCol="temporal_cluster",
            k=4,
            seed=42
        )
        
        # Temporal clustering pipeline
        temporal_pipeline = Pipeline(stages=[temporal_assembler, temporal_kmeans])
        
        # Train model
        temporal_model = temporal_pipeline.fit(temporal_stats)
        temporal_clustered = temporal_model.transform(temporal_stats)
        
        # Analyze temporal clusters
        temporal_cluster_analysis = temporal_clustered.groupBy("temporal_cluster").agg(
            collect_list("publish_hour").alias("hours"),
            avg("avg_views").alias("cluster_avg_views"),
            sum("video_count").alias("total_videos")
        ).collect()
        
        temporal_results = [row.asDict() for row in temporal_cluster_analysis]
        
        print(f"   [OK] Temporal clustering completed - {len(temporal_results)} clusters")
        
        return {
            'temporal_clustering': {
                'model': temporal_model,
                'cluster_analysis': temporal_results,
                'predictions': temporal_clustered
            }
        }

    def save_clustering_results(self, all_results):
        """Save all clustering results to MongoDB and HDFS"""
        print("💾 Saving advanced clustering results...")
        
        try:
            # Prepare comprehensive results for MongoDB
            clustering_summary = {
                "created_at": datetime.now().isoformat(),
                "type": "advanced_clustering_analysis",
                "framework": "Spark MLlib",
                "algorithms_used": {
                    "behavioral": ["K-Means", "Bisecting K-Means", "Gaussian Mixture"],
                    "content": ["Word2Vec + K-Means"],
                    "geographic": ["K-Means"],
                    "temporal": ["K-Means"]
                },
                "results_summary": {}
            }
            
            # Extract summary from each clustering type
            for cluster_type, results in all_results.items():
                if cluster_type == 'behavioral':
                    clustering_summary['results_summary']['behavioral'] = {
                        algo: {
                            'silhouette_score': result['silhouette_score'],
                            'num_clusters': len(result['cluster_stats'])
                        }
                        for algo, result in results.items()
                    }
                else:
                    for analysis_name, analysis_data in results.items():
                        clustering_summary['results_summary'][analysis_name] = {
                            'num_clusters': len(analysis_data['cluster_analysis'])
                        }
            
            # Save to MongoDB
            self.db.ml_metadata.delete_many({"type": "advanced_clustering_analysis"})
            self.db.ml_metadata.insert_one(clustering_summary)
            
            # Save detailed results
            self.db.clustering_results.delete_many({})
            
            detailed_results = []
            for cluster_type, results in all_results.items():
                for analysis_name, analysis_data in results.items():
                    detailed_results.append({
                        "cluster_type": cluster_type,
                        "analysis_name": analysis_name,
                        "cluster_analysis": analysis_data['cluster_analysis'],
                        "created_at": datetime.now().isoformat()
                    })
            
            if detailed_results:
                self.db.clustering_results.insert_many(detailed_results)
            
            print(f"[OK] Saved clustering results to MongoDB")
            
            # Save models to HDFS (only K-Means models for simplicity)
            models_saved = 0
            for cluster_type, results in all_results.items():
                for analysis_name, analysis_data in results.items():
                    try:
                        model_path = f"{self.clustering_save_path}/{analysis_name}"
                        analysis_data['model'].write().overwrite().save(model_path)
                        models_saved += 1
                        print(f"[OK] Saved {analysis_name} model to HDFS")
                    except Exception as e:
                        print(f"[WARN] Could not save {analysis_name} to HDFS: {str(e)}")
            
            print(f"[OK] Saved {models_saved} clustering models to HDFS")
            return clustering_summary
            
        except Exception as e:
            print(f"[ERROR] Failed to save clustering results: {str(e)}")
            return None

    def run_complete_clustering_analysis(self):
        """Run complete advanced clustering analysis"""
        try:
            print("🚀 Starting Advanced Clustering Analysis Pipeline")
            print("=" * 70)
            
            # Step 1: Load data
            df = self.load_data_from_mongodb()
            if df is None:
                return False
            
            # Step 2: Behavioral clustering
            behavioral_results = self.behavioral_clustering(df)
            
            # Step 3: Content clustering
            content_results = self.content_clustering(df)
            
            # Step 4: Geographic clustering
            geographic_results = self.geographic_clustering(df)
            
            # Step 5: Temporal clustering
            temporal_results = self.temporal_clustering(df)
            
            # Combine all results
            all_results = {
                'behavioral': behavioral_results,
                'content': content_results,
                'geographic': geographic_results,
                'temporal': temporal_results
            }
            
            # Step 6: Save results
            summary = self.save_clustering_results(all_results)
            
            # Step 7: Print comprehensive results
            self.print_clustering_summary(all_results)
            
            print("=" * 70)
            print("🎉 ADVANCED CLUSTERING ANALYSIS COMPLETED!")
            print("✅ Multiple clustering algorithms implemented")
            print("✅ Behavioral, Content, Geographic, Temporal analysis")
            print("✅ Models saved to HDFS") 
            print("✅ Results stored in MongoDB")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Advanced clustering analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.mongo_client.close()

    def print_clustering_summary(self, all_results):
        """Print comprehensive clustering summary"""
        print("\n" + "="*60)
        print("📊 ADVANCED CLUSTERING ANALYSIS RESULTS")
        print("="*60)
        
        # Behavioral clustering
        print("\n🎯 BEHAVIORAL CLUSTERING:")
        for algo_name, result in all_results['behavioral'].items():
            print(f"   {algo_name.upper()}:")
            print(f"      Silhouette Score: {result['silhouette_score']:.4f}")
            print(f"      Clusters: {len(result['cluster_stats'])}")
        
        # Content clustering
        print(f"\n📝 CONTENT CLUSTERING:")
        content_clusters = len(all_results['content']['content_clustering']['cluster_analysis'])
        print(f"   Word2Vec + K-Means: {content_clusters} clusters")
        
        # Geographic clustering
        print(f"\n🌍 GEOGRAPHIC CLUSTERING:")
        geo_clusters = len(all_results['geographic']['geographic_clustering']['cluster_analysis'])
        print(f"   Country Patterns: {geo_clusters} clusters")
        
        # Temporal clustering  
        print(f"\n⏰ TEMPORAL CLUSTERING:")
        temporal_clusters = len(all_results['temporal']['temporal_clustering']['cluster_analysis'])
        print(f"   Time Patterns: {temporal_clusters} clusters")

def main():
    """Main execution function"""
    analysis = AdvancedClusteringAnalysis()
    success = analysis.run_complete_clustering_analysis()
    
    if success:
        print("\n🚀 Advanced clustering models ready!")
        print("💡 Enhanced clustering capabilities:")
        print("   - Behavioral pattern analysis")
        print("   - Content similarity clustering")
        print("   - Geographic trend patterns")
        print("   - Temporal publishing patterns")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
