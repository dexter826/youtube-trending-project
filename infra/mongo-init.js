// MongoDB initialization script
db = db.getSiblingDB('youtube_trending');

// Create collections
db.createCollection('raw_videos');
db.createCollection('trending_results');
db.createCollection('wordcloud_data');

// Create indexes for better query performance
db.raw_videos.createIndex({ "country": 1, "trending_date": 1 });
db.raw_videos.createIndex({ "video_id": 1 });
db.raw_videos.createIndex({ "category_id": 1 });

db.trending_results.createIndex({ "country": 1, "date": 1 });
db.wordcloud_data.createIndex({ "country": 1, "date": 1 });

print('YouTube Trending database initialized successfully!');
