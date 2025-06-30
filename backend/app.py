import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app_actions import update_db

# from pydantic import BaseModel, Field
from backend.models.app_models import CategorizerJobInfo, JobStatus


@dataclass
class Cache:
    """
    A simple dataclass to hold the in-memory cache.
    This is used to store job information and update status.
    """

    job_info_cache: Dict[str, CategorizerJobInfo]
    update_db_cache: JobStatus


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to set up shared state.
    This is called when the application starts and stops.
    """
    # Initialize the standard cached info
    app.state.cache = Cache(
        job_info_cache=dict(),  # Initialize an empty cache for job info
        update_db_cache=JobStatus.IDLE,  # Initialize the update status
    )
    yield
    # Cleanup can be done here if needed


app = FastAPI(lifespan=lifespan)

# Mount static files from the frontend folder under /static
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path, html=True), name="static")


def update_job_info_cache(job_id: str, job_info: CategorizerJobInfo):
    """
    Update the in-memory cache with the job information.
    """
    job_info_cache: Dict[str, CategorizerJobInfo] = app.state.cache.job_info_cache
    if job_id not in job_info_cache:
        if not all(
            [
                job_info.job_id,  # Required always
                job_info.status,  # Required always
                job_info.total_messages is not None,  # Required on start
                job_info.processed_messages == 0,  # Required, initializes
                # job_info.eta is not None, # Not required for initial job info, can be None
                job_info.last_update is not None,
                # job_info.current_message_id is not None, # Not required for initial job info
                # job_info.current_speed is not None, # Not required for initial job info
            ]
        ):
            raise ValueError("Job info must contain all required fields.")
        job_info_cache[job_id] = job_info
    else:
        existing_job_info = job_info_cache[job_id]
        existing_job_info.status = job_info.status
        # Update processed messages if applicable
        if job_info.processed_messages is not None:
            if existing_job_info.processed_messages is None:
                raise ValueError("Processed messages should be initialized as an int")
            existing_job_info.processed_messages += job_info.processed_messages
        existing_job_info.eta = job_info.eta
        if job_info.last_update is not None:
            existing_job_info.last_update = job_info.last_update
        if job_info.current_message_id is not None:
            existing_job_info.current_message_id = job_info.current_message_id
        if job_info.current_speed is not None:
            existing_job_info.current_speed = job_info.current_speed


@app.get("/")
async def read_root():
    return {"message": "Welcome to your FastAPI app!"}


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("frontend/assets/icons/applogo.ico")


@app.post("/update_db")
async def launch_update_db():
    """
    Endpoint to trigger the database update.
    This will run the ChromaUpserter to add any missing conversations
    to the chroma database.
    """
    # Update the shared state via app.state
    app.state.cache.update_db_cache = JobStatus.RUNNING
    try:
        result = update_db()
        app.state.cache.update_db_cache = JobStatus.COMPLETED
        return result
    except Exception as e:
        app.state.cache.update_db_cache = JobStatus.FAILED
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/update_db_status")
async def get_update_db_status():
    """
    Endpoint to get the current status of the database update.
    """
    status = app.state.cache.update_db_cache
    if status == JobStatus.IDLE:
        return {"status": "No update in progress"}
    return {"status": status.value}


@app.post("/update_jobinfo")
async def update_jobinfo(update: CategorizerJobInfo):
    update_job_info_cache(update.job_id, update)
    return {"message": "Job info updated"}


@app.get("/job_update/{job_id}")
async def job_update(job_id: str):
    cache = app.state.job_info_cache
    if job_id not in cache:
        raise HTTPException(status_code=404, detail="Job not found")
    return cache[job_id]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
