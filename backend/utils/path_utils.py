import os
from dataclasses import dataclass
from pathlib import Path
from time import time

from dotenv import load_dotenv
from loguru import logger

logger.info("Initializing path environment variables")

# Load environment variables from the relative path.
# It is expected that the working directory is controlled externally.
paths_env = Path("env/paths.env")
env_loaded = load_dotenv(dotenv_path=paths_env)
logger.debug(f"load_dotenv output: {env_loaded}")

# Use a module-level constant for the session timestamp to ensure a static log file per execution.
SESSION_TIMESTAMP = time()


@dataclass(frozen=True)
class Paths:
    source_conversations_path: Path
    log_file: Path
    chroma_db_path: Path
    small_model_path: Path
    messages_categories: Path
    blacklist_categories: Path


def get_paths() -> Paths:
    logger.debug("Summoned get_paths")
    return Paths(
        source_conversations_path=Path(os.environ["SOURCE_JSON_PATH"]),
        # Use SESSION_TIMESTAMP so the filename is static during the session.
        log_file=Path(os.environ["LOG_PATH"].format(time=SESSION_TIMESTAMP)),
        chroma_db_path=Path(os.environ["PERSISTENT_CHROMADB_PATH"]),
        small_model_path=Path(os.environ["SMALL_LLM_MODEL_PATH"]),
        messages_categories=Path(os.environ["MESSAGES_CATEGORIES"]),
        blacklist_categories=Path(os.environ["BLACKLIST_CATEGORIES"]),
    )


if __name__ == "__main__":
    logger.debug("Summoned get_paths")
    paths = get_paths()
    # Loop through the dataclass fields for logging.
    for key, value in paths.__dict__.items():
        logger.debug(f"{key}: {value}")
    logger.debug("Finished get_paths")
