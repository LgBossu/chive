# from pathlib import Path
import os
import sys

import chromadb
from loguru import logger
from utils_legacy import load_paths as lp

# Initialize the paths
PATHS = lp.get_paths()
LOG_FILE = PATHS["log_file"]
CHROMA_DB_PATH = PATHS["chroma_db_path"]

# Set up the logger
logger.remove()
logger.add(
    sink=sys.stdout,  # Output to the console
    format="<level>{level:<10} | {message}</>",
    level="INFO",
    colorize=True,
)
logger.add(
    sink=LOG_FILE,  # Output to the log file
    format="{time} | {level:<10} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    backtrace=True,
    diagnose=True,
    colorize=True,
)


if os.path.exists(CHROMA_DB_PATH):
    logger.success("ChromaDB path already exists - Exiting...")
else:
    logger.info("Creating ChromaDB path")
    client = chromadb.PersistentClient(str(CHROMA_DB_PATH))
    logger.success("ChromaDB client instantiated")
    logger.info("Creating ChromaDB collections")
    conv_collection = client.create_collection("conversations")
    mess_collection = client.create_collection("messages")
    logger.success("ChromaDB collections created")
