import os
from pathlib import Path
from time import time

from dotenv import load_dotenv
from loguru import logger

logger.info("Initializing path environment variables")

paths_env = Path("env/paths.env")
out = load_dotenv(dotenv_path=paths_env)

logger.debug(f"load_dotenv output: {out}")


def get_paths():
    logger.debug("Summoned get_paths")
    return {
        "source_conversations_path": Path(os.environ["SOURCE_JSON_PATH"]),
        "log_file": Path(os.environ["LOG_PATH"].format(time=time())),
        "chroma_db_path": Path(os.environ["PERSISTENT_CHROMADB_PATH"]),
        "small_model_path": Path(os.environ["SMALL_LLM_MODEL_PATH"]),
        "messages_categories": Path(os.environ["MESSAGES_CATEGORIES"]),
        "blacklist_categories": Path(os.environ["BLACKLIST_CATEGORIES"]),
    }


if __name__ == "__main__":
    logger.debug("Summoned get_paths")
    paths = get_paths()
    for key, value in paths.items():
        logger.debug(f"{key}: {value}")
    logger.debug("Finished get_paths")
