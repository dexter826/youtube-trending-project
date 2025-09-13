"""
Data Exporting Processor
"""

from datetime import datetime


class DataExporter:
    def __init__(self, db_connection):
        self.db = db_connection

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

    def save_trending_results(self, results):
        """Save trending analysis results to MongoDB"""
        if results:
            self.db.trending_results.delete_many({})
            self.db.trending_results.insert_many(results)

    def save_wordcloud_data(self, results):
        """Save wordcloud data to MongoDB"""
        if results:
            self.db.wordcloud_data.delete_many({})
            self.db.wordcloud_data.insert_many(results)

    def save_ml_features(self, ml_features):
        """Save ML features to MongoDB"""
        if ml_features:
            self.db.ml_features.delete_many({})

            batch_size = 5000
            for i in range(0, len(ml_features), batch_size):
                batch = ml_features[i:i + batch_size]
                self.db.ml_features.insert_many(batch)