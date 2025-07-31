"""To write and store models used by and for the fastapi's various endpoints."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class PlainResponse(BaseModel):
    """A simple response model for endpoints that return a message."""

    message: str


class CommandValue(str, Enum):
    """Enum for command values used to instruct subprocesses to keep running or to stop."""

    DEFAULT = "##DEFAULT_STATE##"
    ABORT = "##ABORT##"


class CommandResponse(BaseModel):
    """A response model for commands sent to subprocesses."""

    command: CommandValue


class JobStatus(str, Enum):
    """Enum for long-running job statuses."""

    RUNNING = "Running"
    IDLE = "Idle"
    STALLED = "Stalled"
    COMPLETED = "Completed"
    FAILED = "Failed"
    ABORTED = "Aborted"


class CategorizerJobInfo(BaseModel):
    """When categorization is run, this model is used to track the job's status and progress."""

    status: JobStatus
    total_messages: Optional[int] = None
    processed_messages: Optional[int] = None
    eta: Optional[str] = None  # e.g., "3h 27m"
    last_update: Optional[float] = None
    # datetime is not json serializable, so we use a float UNIX timestamp
    current_message_id: Optional[str] = None
    current_speed: Optional[float] = None  # messages per second


class UpdaterJobInfo(BaseModel):
    """When updating conversations, this model is used to track the job's status and progress."""

    status: JobStatus
    updated_conversations: List[str] = []


class QueryDatabaseModel(BaseModel):
    """Model a complete request from frontend to query the database."""

    query_text: str  # May be empty string
    filtered_conversations: Optional[List[str]] = None
    # filtered_threads: Optional[List[str]] = None # Not yet implemented
    filtered_tags: Optional[List[str]] = None
    filtered_date_after: Optional[float] = None  # UNIX timestamp, filter messages after this date
    filtered_date_before: Optional[float] = None  # UNIX timestamp, filter messages before this date
