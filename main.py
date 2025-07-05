"""Main app starter script."""

import threading
import webbrowser

import uvicorn

# from backend.utils.log_setup import LoggerSetup


def open_browser():
    # Open the frontend page; adjust if you want a different default file.
    webbrowser.open_new("http://127.0.0.1:8000/static/index.html")


if __name__ == "__main__":
    # # Set up logging
    # logger_setup = LoggerSetup()
    # logger_setup.configure_logger()

    # Wait one second before opening to ensure the server is up
    threading.Timer(1, open_browser).start()
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
