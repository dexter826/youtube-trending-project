"""
Input validation utilities
"""

from typing import Optional
from fastapi import HTTPException

def validate_country(country: Optional[str]) -> Optional[str]:
    """Validate country parameter"""
    if country and len(country) != 2:
        raise HTTPException(status_code=400, detail="Country must be a 2-letter code")
    return country

def validate_limit(limit: int, max_limit: int = 1000) -> int:
    """Validate limit parameter"""
    if limit < 1 or limit > max_limit:
        raise HTTPException(status_code=400, detail=f"Limit must be between 1 and {max_limit}")
    return limit

def validate_category(category: Optional[str]) -> Optional[int]:
    """Validate category parameter"""
    if category:
        try:
            return int(category)
        except ValueError:
            raise HTTPException(status_code=400, detail="Category must be a valid integer")
    return None