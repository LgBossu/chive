"""To write and store models used by and for the fastapi's various endpoints."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class PlainResponse(BaseModel):
    """A simple response model for endpoints that return a message."""

    message: str


class JobStatus(str, Enum):
    RUNNING = "running"
    IDLE = "idle"
    STALLED = "stalled"
    COMPLETED = "completed"
    FAILED = "failed"


class CategorizerJobInfo(BaseModel):
    status: JobStatus
    total_messages: Optional[int] = None
    processed_messages: Optional[int] = None
    eta: Optional[str] = None  # e.g., "3h 27m"
    last_update: Optional[float] = None
    # datetime is not json serializable, so we use a float UNIX timestamp
    current_message_id: Optional[str] = None
    current_speed: Optional[float] = None  # messages per second


class UpdaterJobInfo(BaseModel):
    status: JobStatus
    updated_conversations: List[str] = []
