#!/bin/bash
# HDFS Initialization Script for YouTube Trending Project
# Author: BigData Expert

echo "🚀 Starting HDFS initialization for YouTube Trending Project..."

# Wait for NameNode to be ready
echo "⏳ Waiting for NameNode to be ready..."
sleep 30

# Create HDFS directories
echo "📁 Creating HDFS directory structure..."

# Create base directories
hdfs dfs -mkdir -p /youtube_trending
hdfs dfs -mkdir -p /youtube_trending/raw_data
hdfs dfs -mkdir -p /youtube_trending/processed_data
hdfs dfs -mkdir -p /youtube_trending/ml_features
hdfs dfs -mkdir -p /youtube_trending/models

# Create country-specific directories
echo "🌍 Creating country-specific directories..."
for country in US CA GB DE FR IN JP KR MX RU; do
    hdfs dfs -mkdir -p /youtube_trending/raw_data/${country}
done

# Upload data from local to HDFS
echo "📤 Uploading data files to HDFS..."

# Upload CSV files
for file in /input-data/*.csv; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        country="${filename:0:2}"
        echo "   Uploading $filename to HDFS..."
        hdfs dfs -put "$file" "/youtube_trending/raw_data/${country}/"
    fi
done

# Upload JSON category files
for file in /input-data/*_category_id.json; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        country="${filename:0:2}"
        echo "   Uploading $filename to HDFS..."
        hdfs dfs -put "$file" "/youtube_trending/raw_data/${country}/"
    fi
done

# Verify uploads
echo "✅ Verification - HDFS directory structure:"
hdfs dfs -ls -R /youtube_trending

echo "🎉 HDFS initialization completed successfully!"
echo "🔗 NameNode Web UI: http://localhost:9870"
echo "📊 Data location: hdfs://namenode:9000/youtube_trending/raw_data/"
