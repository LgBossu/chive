"""To write and store models used by and for the fastapi's various endpoints."""

from enum import Enum
from typing import List, Optional, TypedDict

import chromadb.api.types
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


class QueryDatabaseDict(TypedDict):
    """A dictionary representation of a query to the database."""

    query_texts: List[str]  # List of query texts
    n_results: int  # Number of results to return
    where: Optional[chromadb.Where]  # Filter conditions for the query
    where_document: Optional[chromadb.WhereDocument]  # Document-specific filter conditions
    include: chromadb.Include  # Fields to include in the results
    query_embeddings: Optional[chromadb.Embeddings]


class QueryDatabaseModel(BaseModel):
    """Model a complete request from frontend to query the database."""

    query_text: str | List[str]  # May be empty string
    num_results: int = 10  # Number of results to return
    filtered_conversations: Optional[List[str]] = None
    # filtered_threads: Optional[List[str]] = None # Not yet implemented
    filtered_tags: Optional[List[str]] = None
    filtered_date_after: Optional[float] = None  # UNIX timestamp, filter messages after this date
    filtered_date_before: Optional[float] = None  # UNIX timestamp, filter messages before this date
    include: chromadb.Include = [
        chromadb.api.types.IncludeEnum.documents,
        chromadb.api.types.IncludeEnum.metadatas,
    ]  # Fields to include in the results
    filter_empty: bool = True  # Filter out empty or non-text messages
    filter_non_text: bool = True  # Filter out messages that are not text content

    def _cast_dates_to_condition(
        self,
        date_after: Optional[float] = None,
        date_before: Optional[float] = None,
    ) -> Optional[chromadb.Where]:
        """
        Cast the date filters to a ChromaDB where condition.

        :param date_after: Optional UNIX timestamp for filtering messages after this date
        :param date_before: Optional UNIX timestamp for filtering messages before this date
        :return: A chromadb.Where condition or None if no dates are provided
        """
        if date_after is not None and date_before is not None:
            return {
                "$and": [{"timestamp": {"$gte": date_after}}, {"timestamp": {"$lte": date_before}}]
            }
        elif date_after is not None:
            return {"timestamp": {"$gte": date_after}}
        elif date_before is not None:
            return {"timestamp": {"$lte": date_before}}
        else:
            return None

    def cast_to_query_args(
        self,
    ) -> QueryDatabaseDict:
        query_dict = QueryDatabaseDict(
            query_texts=[self.query_text]
            if isinstance(self.query_text, str)
            else self.query_text,  # Ensure query_text is a list
            n_results=self.num_results,
            where=None,  # Will be set later
            where_document=None,  # Not used in this context
            include=self.include,
            query_embeddings=None,  # Not used in this context
        )

        dates_condition = self._cast_dates_to_condition(
            date_after=self.filtered_date_after,
            date_before=self.filtered_date_before,
        )
        query_dict["where"] = dates_condition

        return query_dict
