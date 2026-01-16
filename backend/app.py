import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing import Process

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.app_actions import app_categorize, search_database, update_db
from backend.models.app_models import (
    CategorizerJobInfo,
    CommandResponse,
    CommandValue,
    JobStatus,
    PlainResponse,
    QueryDatabaseModel,
    UpdaterJobInfo,
)
from backend.models.message_tree import MessageTree
from backend.utils.log_setup import LoggerSetup
from backend.utils.config_utils import load_config

# Set up user logging
logger_setup = LoggerSetup()
LOG_PATH = logger_setup.configure_logger()

API_CONFIG = load_config().API
CONFIG_ENDPOINTS = API_CONFIG.ENDPOINTS

# [AI GENERATED CODE]
# Filter to suppress logs for status endpoints
class StatusEndpointFilter(logging.Filter):
    def filter(self, record):
        # Only filter access logs (not error logs)
        msg = record.getMessage()
        # Suppress logs for status endpoints
        return not ("GET /update_db/status" in msg or "GET /categorizer/status" in msg)


# Add the filter to the logger
logging.getLogger("uvicorn.access").addFilter(StatusEndpointFilter())
# [END AI GENERATED CODE]


@dataclass
class Cache:
    """
    A simple dataclass to hold the in-memory cache.
    This is used to store job information and update status.
    """

    categorizer_cache: CategorizerJobInfo
    categorizer_command: CommandValue
    update_db_cache: UpdaterJobInfo
    update_db_command: CommandValue


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to set up shared state.
    This is called when the application starts and stops.
    """
    # Initialize the standard cached info
    categorizer_cache = CategorizerJobInfo(status=JobStatus.IDLE)
    update_db_cache = UpdaterJobInfo(status=JobStatus.IDLE, updated_conversations=list())

    # Store the caches in the app state
    app.state.cache = Cache(
        categorizer_cache=categorizer_cache,  # type: ignore
        categorizer_command=CommandValue.DEFAULT,  # type: ignore
        update_db_cache=update_db_cache,  # type: ignore
        update_db_command=CommandValue.DEFAULT,  # type: ignore
    )
    yield
    # Cleanup can be done here if needed


app = FastAPI(lifespan=lifespan)

# Mount static files from the frontend folder under /static
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path, html=True), name="static")


@app.get("/", response_model=PlainResponse)
async def read_root():
    logger.trace("Root endpoint accessed.")
    return PlainResponse(message="Welcome to the ChatGPT Post Processing API!")


@app.get("/favicon.ico", response_class=FileResponse)
async def favicon():
    return FileResponse("frontend/assets/icons/applogo.ico")


@app.post(CONFIG_ENDPOINTS.UPDATE_DB.RUN, response_model=PlainResponse)
async def update_db_run():
    """
    Endpoint to trigger the database update.
    This will set the job status to RUNNING and call the update_db function.
    It will also update the in-memory cache with the job status.
    If the update fails, it will set the status to FAILED.
    If the update is successful, it will set the status to COMPLETED.

    No ID is returned, as this is a single job that is expected to run only once
    (in current version, it always updates all the conversations).
    """
    logger.trace("Starting database update job.")
    cache: UpdaterJobInfo = app.state.cache.update_db_cache

    # Check if an update is already in progress
    if cache.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Update already in progress")

    # Reset the cache fields before starting a new update
    cache.status = JobStatus.RUNNING
    cache.updated_conversations = []

    p = Process(target=update_db, args=(LOG_PATH,))  # Pass the log path to the update_db function
    p.start()
    return PlainResponse(message="Database update started.")


@app.post(CONFIG_ENDPOINTS.UPDATE_DB.ABORT, response_model=PlainResponse)
async def update_db_abort():
    """
    Endpoint to abort the database update job.
    This will set the job's command to ABORT to cleanly end the subprocess.
    """
    logger.trace("Aborting database update job.")

    cache: UpdaterJobInfo = app.state.cache.update_db_cache

    # Check if an update is in progress
    if cache.status != JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="No update in progress")

    # Update the command to ABORT
    app.state.cache.update_db_command = CommandValue.ABORT

    # Set the status to ABORTED
    return PlainResponse(message="Database update command set to ABORT.")


@app.post(CONFIG_ENDPOINTS.UPDATE_DB.UPDATE, response_model=CommandResponse)
async def update_db_update(update: UpdaterJobInfo):
    """
    Endpoint to update the status of the database update job.
    This is used to update the in-memory cache with the latest job info.
    """
    logger.trace("Updating database job status with new information.")

    cache: UpdaterJobInfo = app.state.cache.update_db_cache

    if cache.status == JobStatus.IDLE:
        raise HTTPException(status_code=400, detail="No update in progress")

    if update.status == JobStatus.ABORTED:
        # JobStatus.ABORTED indicates the subprocess correctly aborted the job
        logger.info("Database update job was aborted.")
        cache.status = JobStatus.ABORTED
        cache.updated_conversations = update.updated_conversations
        # Reset command to default after abort
        app.state.cache.update_db_command = CommandValue.DEFAULT
    else:
        # Regular update
        cache.status = update.status
        for conversation in update.updated_conversations:
            if conversation not in cache.updated_conversations:
                cache.updated_conversations.append(conversation)

    command: CommandValue = app.state.cache.update_db_command
    return CommandResponse(command=command)


@app.get(CONFIG_ENDPOINTS.UPDATE_DB.STATUS, response_model=UpdaterJobInfo)
async def update_db_status():
    """
    Endpoint to get the current status of the database update.
    """
    # logger.trace("Fetching database update job status.")
    cache: UpdaterJobInfo = app.state.cache.update_db_cache
    return cache


@app.post(CONFIG_ENDPOINTS.CATEGORIZER.RUN, response_model=PlainResponse)
async def categorizer_start():
    """
    Endpoint to start a categorizer job.
    This will initialize the job info in the cache, start the job, and return the job ID.
    """
    logger.trace("Starting categorizer job.")

    cache: CategorizerJobInfo = app.state.cache.categorizer_cache

    # Check if a categorizer job is already running
    if cache.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Categorizer job already in progress")

    # Update the shared state via app.state
    cache.status = JobStatus.RUNNING
    cache.total_messages = None
    cache.processed_messages = 0
    cache.eta = None
    cache.last_update = None
    cache.current_message_id = None
    cache.current_speed = None

    p = Process(target=app_categorize, args=(LOG_PATH,))  # Pass the log path to the categorize function
    p.start()
    return PlainResponse(message="Categorizer job started.")


@app.post(CONFIG_ENDPOINTS.CATEGORIZER.ABORT, response_model=PlainResponse)
async def categorizer_abort():
    """
    Endpoint to abort the categorizer job.
    This will set the job's command to ABORT to cleanly end the subprocess.
    """
    logger.trace("Aborting categorizer job.")

    cache: CategorizerJobInfo = app.state.cache.categorizer_cache

    # Check if a categorizer job is running
    if cache.status != JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="No categorizer job in progress")

    # Update the command to ABORT
    app.state.cache.categorizer_command = CommandValue.ABORT

    # Set the status to ABORTED
    return PlainResponse(message="Categorizer job command set to ABORT.")


@app.post(CONFIG_ENDPOINTS.CATEGORIZER.UPDATE, response_model=CommandResponse)
async def categorizer_update(update: CategorizerJobInfo):
    """
    Endpoint to update the status of the categorizer job.
    This is used to update the in-memory cache with the latest job info.
    """
    logger.trace("Updating categorizer job status with new information.")

    cache: CategorizerJobInfo = app.state.cache.categorizer_cache

    # TODO : account for post-abort updates, the updating logic is different

    ### Update fields if applicable ###

    # Job status
    cache.status = update.status
    if update.status == JobStatus.ABORTED:
        # JobStatus.ABORTED indicates the subprocess correctly aborted the job
        logger.info("Categorizer job was aborted.")
        # Reset command to default after abort
        app.state.cache.categorizer_command = CommandValue.DEFAULT

    else:
        # Regular update

        # Total messages
        if update.total_messages is not None:
            cache.total_messages = max(
                update.total_messages,
                cache.total_messages if cache.total_messages is not None else 0,
            )  # We retain the maximum total messages logged so far
        # Processed messages
        if cache.processed_messages is None:
            cache.processed_messages = 0
        if update.processed_messages is not None:
            cache.processed_messages += update.processed_messages
        # ETA
        cache.eta = update.eta
        # Last update time
        if update.last_update is not None:
            cache.last_update = update.last_update
        # Current message ID
        if update.current_message_id is not None:
            cache.current_message_id = update.current_message_id
        # Current speed
        if update.current_speed is not None:
            cache.current_speed = update.current_speed

    # Command value
    command: CommandValue = app.state.cache.categorizer_command
    return CommandResponse(command=command)


@app.get(CONFIG_ENDPOINTS.CATEGORIZER.STATUS, response_model=CategorizerJobInfo)
async def categorizer_status():
    # logger.trace("Fetching categorizer job status.")

    cache: CategorizerJobInfo = app.state.cache.categorizer_cache
    return cache


@app.get(CONFIG_ENDPOINTS.SEARCH_ROOT, response_class=StreamingResponse)
async def search(query: QueryDatabaseModel):
    logger.trace(f"Searching for files with query: {query.query_text}")

    try:
        querier, sorted_nodes = search_database(query, log_path=LOG_PATH)
    except Exception as e:
        logger.error(f"Error searching database: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching database:\n{e}")

    def json_stream():
        yield "["
        first = True
        for messages in sorted_nodes.values():
            try:
                # For each conversation, create a MessageTree
                tree = MessageTree(
                    source=messages[0],
                    highlights=[node.id for node in messages],
                    querier=querier,
                )
            except Exception as e:
                logger.error(f"Error creating MessageTree: {e}")
                tree = {"error": str(e)}

            if not first:
                yield ","
            else:
                first = False

            yield json.dumps(tree if isinstance(tree, dict) else tree.to_dict())
        yield "]"

    return StreamingResponse(json_stream(), media_type="application/json")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host=API_CONFIG.HOST, port=API_CONFIG.PORT, reload=True)
