import sys

from loguru import logger
from utils import load_paths as lp

logger.info("Setting up the logger")

# Initialize the paths
LOG_FILE = lp.get_paths()["log_file"]

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
    level="TRACE",
    backtrace=True,
    diagnose=True,
    colorize=True,
)


logger.info("Logger set up")
