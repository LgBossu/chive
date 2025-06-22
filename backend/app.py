from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to your FastAPI app!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
