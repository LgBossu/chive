import re
import signal
import sys
from multiprocessing import Process, get_start_method, set_start_method
from pathlib import Path
from time import sleep, time
from typing import Optional, Tuple, TypedDict, Union


from loguru import logger
from requests import ConnectionError, HTTPError, get, post

from backend.loaders.chroma_endpoints import ChromaQuerier
from backend.loaders.metafiles_handlers import (
    BLACKLIST_TAG,
    EMPTY_TAG,
    MetafileQuerier,
    MetafileWriter,
)
from backend.models.app_models import CategorizerJobInfo, JobStatus
from backend.models.categorizer_model import Categorizer0, CategorizerModel
from backend.utils.log_setup import LoggerSetup


# TODO : maybe shard the databases and categorization process
# to avoid memory issues when and if the project scales up.

# TODO : factorize this module to separate logic and utils from runtime.

NO_STALLING_ID = "[NotAnId]"

class DatabaseCounts(TypedDict):
    total_uncategorized: int
    nonempty_uncategorized: int
    empty_uncategorized: int
    first_uncategorized_offset: Optional[int] # Let's us skip to the first uncategorized message directly

def display_time(seconds: float, tz: int = 1, duration: bool = False) -> str:
    _, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    if not duration:
        hours = (hours + tz) % 24  # We are dealing with a date and account for timezone
    minutes, seconds = divmod(seconds, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    seconds = int(seconds)
    return f"{int(hours)}h {int(minutes)}m {seconds}s {int(milliseconds)}ms"


class WorkerConfig(TypedDict):
    pass

class Worker:
    class Subclass:
        pass

    def __init__(self, config:WorkerConfig) -> None:
        pass

    def run(self):
        pass

def subprocess(worker_cls: type[Worker], config: WorkerConfig):
    worker = worker_cls(config)
    worker.run()

class Engine:

    def __init__(self) -> None:
        pass
    
    def start(self):
        p = Process(target=subprocess, args=(Worker, WorkerConfig())) # spawned, not forked
        p.start()
        p.join()
