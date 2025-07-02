import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing import Process

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.app_actions import categorize, update_db
from backend.models.app_models import CategorizerJobInfo, JobStatus, PlainResponse, UpdaterJobInfo
from backend.utils.log_setup import LoggerSetup

# Set up logging
logger_setup = LoggerSetup()
LOG_PATH = logger_setup.configure_logger()


@dataclass
class Cache:
    """
    A simple dataclass to hold the in-memory cache.
    This is used to store job information and update status.
    """

    categorizer_cache: CategorizerJobInfo
    update_db_cache: UpdaterJobInfo


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
        categorizer_cache=categorizer_cache,
        update_db_cache=update_db_cache,
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


@app.post("/update_db/run", response_model=PlainResponse)
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


@app.post("/update_db/update", response_model=PlainResponse)
async def update_db_update(update: UpdaterJobInfo):
    """
    Endpoint to update the status of the database update job.
    This is used to update the in-memory cache with the latest job info.
    """
    logger.trace("Updating database job status with new information.")

    cache: UpdaterJobInfo = app.state.cache.update_db_cache
    if cache.status == JobStatus.IDLE:
        raise HTTPException(status_code=400, detail="No update in progress")
    # Update the shared state via app.state
    cache.status = update.status

    for conversation in update.updated_conversations:
        if conversation not in cache.updated_conversations:
            cache.updated_conversations.append(conversation)

    return PlainResponse(message="success")


@app.get("/update_db/status", response_model=UpdaterJobInfo)
async def update_db_status():
    """
    Endpoint to get the current status of the database update.
    """
    logger.trace("Fetching database update job status.")
    cache: UpdaterJobInfo = app.state.cache.update_db_cache
    return cache


@app.post("/categorizer/run", response_model=PlainResponse)
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

    p = Process(target=categorize, args=(LOG_PATH,))  # Pass the log path to the categorize function
    p.start()
    return PlainResponse(message="Categorizer job started.")


@app.post("/categorizer/update", response_model=PlainResponse)
async def categorizer_update(update: CategorizerJobInfo):
    """
    Endpoint to update the status of the categorizer job.
    This is used to update the in-memory cache with the latest job info.
    """
    logger.trace("Updating categorizer job status with new information.")

    cache: CategorizerJobInfo = app.state.cache.categorizer_cache

    ### Update fields if applicable ###

    # Job status
    cache.status = update.status
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

    return PlainResponse(message="success")


@app.get("/categorizer/status", response_model=CategorizerJobInfo)
async def categorizer_status():
    logger.trace("Fetching categorizer job status.")

    cache: CategorizerJobInfo = app.state.cache.categorizer_cache
    return cache


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
