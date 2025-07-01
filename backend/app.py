import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app_actions import categorize, update_db

# from pydantic import BaseModel, Field
from backend.models.app_models import CategorizerJobInfo, JobStatus, UpdaterJobInfo


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
    app.state.cache = Cache(
        categorizer_cache=CategorizerJobInfo(
            status=JobStatus.IDLE,  # Initialize the job status
        ),
        update_db_cache=UpdaterJobInfo(
            status=JobStatus.IDLE,  # Initialize the update status
            updated_conversations=list(),  # Initialize the list of updated conversations
        ),  # Initialize the update status
    )
    yield
    # Cleanup can be done here if needed


app = FastAPI(lifespan=lifespan)

# Mount static files from the frontend folder under /static
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path, html=True), name="static")


@app.get("/")
async def read_root():
    return {"message": "Welcome to your FastAPI app!"}


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("frontend/assets/icons/applogo.ico")


@app.post("/update_db_run")
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
    cache: UpdaterJobInfo = app.state.cache.update_db_cache
    # Update the shared state via app.state
    cache.status = JobStatus.RUNNING
    try:
        result = update_db()
        cache.status = JobStatus.COMPLETED
        return result
    except Exception as e:
        app.state.cache.update_db_cache = JobStatus.FAILED
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_db_update")
async def update_db_update(update: UpdaterJobInfo):
    """
    Endpoint to update the status of the database update job.
    This is used to update the in-memory cache with the latest job info.
    """
    cache: UpdaterJobInfo = app.state.cache.update_db_cache
    if cache.status == JobStatus.IDLE:
        raise HTTPException(status_code=400, detail="No update in progress")
    # Update the shared state via app.state
    cache.status = update.status
    cache.updated_conversations = update.updated_conversations
    return {"message": "Update status updated"}


@app.get("/update_db_status")
async def update_db_status():
    """
    Endpoint to get the current status of the database update.
    """
    status = app.state.cache.update_db_cache
    if status == JobStatus.IDLE:
        return {"status": "No update in progress"}
    return {"status": status.value}


@app.post("/categorizer_start")
async def categorizer_start():
    """Endpoint to start a categorizer job.
    This will initialize the job info in the cache, start the job, and return the job ID."""
    cache: CategorizerJobInfo = app.state.cache.job_info_cache
    # Update the shared state via app.state
    cache.status = JobStatus.RUNNING
    try:
        result = categorize()
        cache.status = JobStatus.COMPLETED
        return result
    except Exception as e:
        cache.status = JobStatus.FAILED
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/categorizer_update")
async def categorizer_update(update: CategorizerJobInfo):
    """
    Endpoint to update the status of the categorizer job.
    This is used to update the in-memory cache with the latest job info.
    """
    cache: CategorizerJobInfo = app.state.cache.categorizer_cache

    ### Update fields if applicable ###

    # Job status
    cache.status = update.status
    # Total messages
    if update.total_messages is not None:
        cache.total_messages = update.total_messages
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

    return {"message": "Job info updated"}


@app.get("/categorizer_status/{job_id}")
async def categorizer_status(job_id: str):
    cache = app.state.cache.job_info_cache
    if job_id not in cache:
        raise HTTPException(status_code=404, detail="Job not found")
    return cache[job_id].json()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
