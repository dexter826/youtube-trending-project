"""
Response formatting utilities
"""

import json
from datetime import datetime, date
from bson import ObjectId

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB ObjectId and datetime"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

def format_success_response(data: dict, message: str = None) -> dict:
    """Format successful API response"""
    response = {
        "status": "success",
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    if message:
        response["message"] = message
    return response

def format_error_response(error: str, status_code: int = 500) -> dict:
    """Format error API response"""
    return {
        "status": "error",
        "error": error,
        "status_code": status_code,
        "timestamp": datetime.now().isoformat()
    }