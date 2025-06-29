"""Main app starter script."""

import threading
import webbrowser

import uvicorn


def open_browser():
    # Open the frontend page; adjust if you want a different default file.
    webbrowser.open_new("http://127.0.0.1:8000/static/index.html")


if __name__ == "__main__":
    # Wait one second before opening to ensure the server is up
    threading.Timer(1, open_browser).start()
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
