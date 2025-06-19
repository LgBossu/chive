from multiprocessing import Process, Queue
from time import time
from typing import Dict, List, Optional, Tuple

from loguru import logger

from backend.loaders.chroma_endpoints import ChromaQuerier
from backend.loaders.metafiles_handlers import (
    BLACKLIST_TAG,
    EMPTY_TAG,
    NO_TAGS_TAG,
    MetafileQuerier,
    MetafileWriter,
)
from backend.models.categorizer_model import AvailableCategorizers, CategorizerModel


def display_time(seconds: float, tz: int = 1, duration: bool = False) -> str:
    _, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    if not duration:
        hours = (hours + tz) % 24  # We are dealing with a date and account for timezone
    minutes, seconds = divmod(seconds, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    seconds = int(seconds)
    return f"{int(hours)}h {int(minutes)}m {seconds}s {int(milliseconds)}ms"


class CategorizerEngine:
    pass


class AutoCategorizerEngine(CategorizerEngine):
    pass


if __name__ == "__main__":
    from backend.utils.log_setup import LoggerSetup

    LoggerSetup.configure_logger(console_level="DEBUG")

    logger.info("Starting Categorizer process...")
    logger.warning("""You are running the categorizing handling script directly.
                   This is intended for debugging purposes only.

                   The script IS SUSCPTIBLE to write, read, and modify actual databases.
                   It is not recommended to run this script in production environments.

                   This script is not intended for production use.
                   Users stay advised.""")

    # Debug run
    # TODO : at some point, ask for user input to proceed OR remove the debug run script.

    # RUN AUTOMATIC CATEGORIZER PIPELINE
    logger.info("Running automatic categorizer pipeline...")
    raise NotImplementedError("Refactoring to come")

    # # Initialize the categorizer engine with the model and metafile handlers
    # categorizer = AutoCategorizerEngine(
    #     categorizer_model=Categorizer0(),
    #     metafile_writer=MetafileWriter(),
    #     metafile_querier=MetafileQuerier(),
    #     chroma_querier=ChromaQuerier(),
    # )
    # # Run the categorization pipeline
    # categorizer.pipeline()
