from typing import Dict

from fastapi import FastAPI, HTTPException

# from pydantic import BaseModel, Field
# from backend.loaders.chroma_endpoints import ChromaUpserter
from backend.models.app_models import CategorizerJobInfo

app = FastAPI()


# In-memory cache for job information
job_info_cache: Dict[str, CategorizerJobInfo] = {}


def update_job_info_cache(job_id: str, job_info: CategorizerJobInfo):
    """
    Update the in-memory cache with the job information.
    """
    if job_id not in job_info_cache:
        # If the job_id does not exist, we add it to the cache
        # We must check that the job_info holds the necessary fields
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
        # If it exists, we update the existing job info
        existing_job_info = job_info_cache[job_id]
        existing_job_info.status = job_info.status
        # Ignore total_messages -- it is given on startup.
        if job_info.processed_messages is not None:
            assert (
                existing_job_info.processed_messages is not None
            ), "Total messages should be initialized as an int"
            existing_job_info.processed_messages += job_info.processed_messages
            # TODO : determine an else clause ?
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


@app.post("/launch_job")
async def launch_job():
    pass  # TODO : implement the job launching logic (give an ID, start the task, etc.)
    # # Start background thread
    # job_id = start_job_thread()
    # return {"message": "Job started", "job_id": job_id}


@app.post("/update_jobinfo")
async def update_jobinfo(update: CategorizerJobInfo):
    update_job_info_cache(update.job_id, update)
    return {"message": "Job info updated"}


@app.get("/job_update/{job_id}")
async def job_update(job_id: str):
    if job_id not in job_info_cache:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_info_cache[job_id]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
