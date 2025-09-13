"""
MongoDB Connection Manager
"""

import os
from pymongo import MongoClient


class DatabaseManager:
    """MongoDB connection manager singleton"""

    _instance = None
    _client = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_database(self, uri=None, db_name=None):
        """Establish MongoDB database connection"""
        if uri is None:
            uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

        if db_name is None:
            db_name = os.getenv("DB_NAME", "youtube_trending")

        if self._client is None or self._db is None:
            self._client = MongoClient(uri)
            self._db = self._client[db_name]

            # Verify connection
            self._client.admin.command('ping')

        return self._db

    def close_connection(self):
        """Terminate MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None

    def get_collection(self, name):
        """Retrieve collection from database"""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        return self._db[name]


# Global instance
_manager = DatabaseManager()


def get_database_connection(uri=None, db_name=None):
    """Get database connection"""
    return _manager.get_database(uri, db_name)


def get_collection(name):
    """Get collection"""
    return _manager.get_collection(name)


def close_database_connection():
    """Close database connection"""
    _manager.close_connection()


# Default configuration
DEFAULT_MONGO_URI = "mongodb://localhost:27017/"
DEFAULT_DB_NAME = "youtube_trending"