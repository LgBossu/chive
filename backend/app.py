from fastapi import FastAPI, HTTPException

# from pydantic import BaseModel, Field
# from backend.loaders.chroma_endpoints import ChromaUpserter
from backend.models.app_models import CategorizerJobInfo

app = FastAPI()


# In-memory cache for job information
job_info_cache = {}


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
    job_info_cache[update.job_id] = update
    # TODO: Figure out better logic for updating the job info, notably,
    # - incrementing the processed messages,
    # - keeping the original estimate of total messages,
    # etc.
    return {"message": "Job info updated"}


@app.get("/job_update/{job_id}")
async def job_update(job_id: str):
    if job_id not in job_info_cache:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_info_cache[job_id]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
