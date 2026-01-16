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
# Loaded at first import, then cached for the duration of the session.
SESSION_TIMESTAMP = time()


@dataclass(frozen=True)
class Paths:
    # Logging file path
    log_file: Path
    # Current source, db, metafiles
    source_conversations_path: Path
    chroma_db_path: Path
    metafiles_dir: Path
    # Legacy paths
    legacy_chroma_dir: Path
    metafiles_legacy_dir: Path
    # Metafiles directory navigation
    dynamic_tags_db: Path
    # LLM model path
    small_model_path: Path


# --- Singleton Pattern for Shared Paths ---
_paths_instance: Paths | None = None

def load_paths() -> Paths:
    """Initialize and cache the Paths instance from environment variables."""
    global _paths_instance
    logger.debug("Initializing and caching Paths instance")
    _paths_instance = Paths(
        log_file=Path(os.environ["LOG_PATH"].format(time=SESSION_TIMESTAMP)),

        source_conversations_path=Path(os.environ["SOURCE_JSON_PATH"]),
        chroma_db_path=Path(os.environ["PERSISTENT_CHROMADB_PATH"]),
        metafiles_dir=Path(os.environ["METAFILES_DIR_PATH"]),

        metafiles_legacy_dir=Path(os.environ["METAFILES_LEGACY_DIR"]),
        legacy_chroma_dir=Path(os.environ["LEGACY_CHROMA_DIR"]),

        dynamic_tags_db=(
            Path(os.environ["METAFILES_DIR_PATH"]) / Path(os.environ["DYNAMIC_TAGS_DB"])
        ),

        small_model_path=Path(os.environ["SMALL_LLM_MODEL_PATH"]),
    )
    return _paths_instance

def get_paths() -> Paths:
    """Get the cached Paths instance, initializing if necessary."""
    global _paths_instance
    if _paths_instance is None:
        logger.debug("Paths instance not initialized, calling load_paths()")
        return load_paths()
    logger.trace("Returning cached Paths instance")
    return _paths_instance


if __name__ == "__main__":
    logger.debug("Running path_utils.py")
    paths = get_paths()
    # Loop through the dataclass fields for logging.
    for key, value in paths.__dict__.items():
        logger.debug(f"{key}: {value}")
    logger.debug("Finished get_paths")
