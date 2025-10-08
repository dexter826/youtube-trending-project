"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel


class VideoMLInput(BaseModel):
    title: str
    views: int = 0
    likes: int = 0
    comment_count: int = 0
    category_id: int = 0
    tags: str = ""
    description: str = ""
    channel_title: str = ""
    duration: str = ""
    publish_hour: int = 12
    video_age_proxy: int = 2