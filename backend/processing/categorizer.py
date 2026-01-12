import re
import signal
import sys
from multiprocessing import Process, get_start_method, set_start_method
from pathlib import Path
from time import sleep, time
from typing import List, Optional, Tuple, TypedDict, Union


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

# TODO : make a list and scheme somewhere of WHO owns WHAT objects, to ensure proper and consistent resource freeing

NO_STALLING_ID = "[NotAnId]"

class DatabaseCounts(TypedDict):
    total_uncategorized: int
    nonempty_uncategorized: int
    empty_uncategorized: int
    first_uncategorized_offset: Optional[int] # Lets us skip to the first uncategorized message directly

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
    # Logging and API
    log_file: Optional[Path]
    api_endpoint: str

    # Model type and wrapper classes
    categorizer_model_type: type[CategorizerModel]
    metafile_writer_type: type[MetafileWriter]
    metafile_querier_type: type[MetafileQuerier]
    chroma_querier_type: type[ChromaQuerier]

    # Categorization loop data
    database_counts: DatabaseCounts


class Worker:
    class SignalHandler:
        """
        Handles system signals for subprocess termination.
        """
        @staticmethod
        def sigterm_handler(signum, frame):
            logger.warning("Received SIGTERM. Exiting subprocess.")
            sys.exit(143)  # 143 = 128 + 15 (SIGTERM)

        @staticmethod
        def sigint_handler(signum, frame):
            logger.warning("Received SIGINT. Exiting subprocess.")
            sys.exit(130)  # 130 = 128 + 2 (SIGINT)

        @staticmethod
        def sigabrt_handler(signum, frame):
            logger.warning("Received SIGABRT. Exiting subprocess.")
            sys.exit(134)  # 134 = 128 + 6 (SIGABRT)

        def register_signal_handlers(self):
            signal.signal(signal.SIGTERM, self.sigterm_handler)
            signal.signal(signal.SIGINT, self.sigint_handler)
            signal.signal(signal.SIGABRT, self.sigabrt_handler)
        
        def abort_exit(self):
            logger.info("Exiting subprocess under abort signal.")
            sys.exit(46)  # 46 is a custom exit code for abort (leet speak "Ab = 46")
        
        def finished_exit(self):
            logger.info("Exiting subprocess after successful completion.")
            sys.exit(0)

        def __init__(self) -> None:
            self.register_signal_handlers()

    class APIMessenger:
        """
        Handles API communication for job status updates.
        """
        def __init__(self, api_endpoint: str) -> None:
            self.api_endpoint = api_endpoint
            self.last_update: Union[float, None] = None

        def post_status_after_message(
            self,
            eta: Optional[str] = None,
            current_message_id: Optional[str] = None,
            current_speed: Optional[float] = None,
            # last_update: Optional[float] = None,
            processed_messages: int = 1,  # Increment processed messages by 1
        ) -> bool:
            # TODO : add abort logic to the subprocess
            # TODO : refactor the code below to use the API after every finished message.
            # This will allow to update the job status and progress in real-time,
            
            if self.last_update is None:
                self.last_update = time()

            job_info = CategorizerJobInfo(
                status=JobStatus.RUNNING,
                total_messages=None,  # This was sent at the beginning of the job
                processed_messages=processed_messages,
                eta=eta,
                last_update=self.last_update,
                current_message_id=current_message_id,
                current_speed=current_speed,
            )

            try:
                response = post(self.api_endpoint, json=job_info.model_dump())
                abort_signal = response.json()["command"]
                return abort_signal == "##ABORT##"  # TODO : do NOT hardcode
            except HTTPError as e:
                logger.error(f"Failed to post job progress: {e}")
                logger.warning("The job cannot be aborted through the API. Be advised.")
            except ConnectionError as e:
                logger.error(
                    f"Failed to connect to the job progress endpoint: {e}. Is the server running?"
                )
                logger.warning("The job cannot be aborted through the API. Be advised.")
            
            return False  # Do not abort the job if the connection fails
        
    class CategorizerModelWrapper:
        """
        Wraps the categorizer model for easy instantiation and usage.
        """
        def __init__(self, model_type: type[CategorizerModel]) -> None:
            self.model_type = model_type
        
        def load_model(self) -> None:
            self.model = self.model_type()
        
        def close_model(self) -> None:
            self.model.close()

        def categorize(self, message_id: str, message_content: str) -> List[str]:
            try:
                return self.model.categorize(message_content)
            except self.model.ExceedingSafetyLimitError:
                logger.warning(f"Message [{message_id}] exceeded safety limit. Assigning BLACKLIST_TAG.")
                return [BLACKLIST_TAG]

    def __init__(self, config:WorkerConfig) -> None:
        LoggerSetup.configure_logger(force_log_file=config["log_file"])
        
        self.signal_handler = self.SignalHandler()

        self.api_endpoint = config["api_endpoint"]
        self.api_messenger = self.APIMessenger(api_endpoint=self.api_endpoint)
        
        self.categorizer_model_type = config["categorizer_model_type"]
        
        self.metafile_writer_type = config["metafile_writer_type"]
        self.metafile_querier_type = config["metafile_querier_type"]
        self.chroma_querier_type = config["chroma_querier_type"]

        self.database_counts = config["database_counts"]

        self.first_message: bool = True

    def open_connections(self):
        """Instantiate connections to the databases"""
        self.metafile_writer: MetafileWriter = self.metafile_writer_type()
        self.metafile_querier: MetafileQuerier = self.metafile_querier_type()
        self.chroma_querier: ChromaQuerier = self.chroma_querier_type()
    
    def close_connections(self):
        """
        Close connections to the databases, and delete the attributes
        (connections need to be reinstantiated to be reopened).
        """
        self.metafile_writer.close()
        self.metafile_querier.close()
        self.chroma_querier.close()
        del self.metafile_writer
        del self.metafile_querier
        del self.chroma_querier

    def load_categorizer_model(self):
        """Instantiate the categorizer model."""
        self.categorizer_model_wrapper = self.CategorizerModelWrapper(
            model_type=self.categorizer_model_type,
        )
        self.categorizer_model_wrapper.load_model()
    
    def close_categorizer_model(self):
        """Close the categorizer model and free resources."""
        self.categorizer_model_wrapper.close_model()
        del self.categorizer_model_wrapper

    def identify_batch(self, message_ids: List[str]) -> List[int]:
        """Given a batch of messages, identify which ones need categorization.
        
        :param message_ids: List of message IDs in the batch.
        :return: A tuple containing a list of uncategorized message IDs and their indices in the batch.
        """
        assert self.metafile_querier is not None, "MetafileQuerier connection is not open."
        
        categorized_ids = set(self.metafile_querier.match_ids(message_ids))
        uncategorized_idx = [idx for idx, msg_id in enumerate(message_ids) if msg_id not in categorized_ids]
        return uncategorized_idx

    def categorize_message_batch(self, messages: List[Tuple[str, str, bool]]):
        """Given a batch's subset of uncategorized messages, categorize them."""
        for message_id, message_content, is_empty in messages:
            if is_empty:
                categories = [EMPTY_TAG]
            else:
                categories = self.categorizer_model_wrapper.categorize(message_id, message_content)
            # TODO : write tagline to database and continue code


    def run(self):
        """Main categorization loop."""
        self.open_connections()

        for n, batch in enumerate(self.chroma_querier.stream_messages(include_content=True, include_metadata=True)): # TODO : after merge, account for offset
            logger.debug(f"Processing batch (number {n})...")
            assert batch["documents"] is not None, "Documents weren't fetched." # Linter enforcement
            assert batch["metadatas"] is not None, "Metadatas weren't fetched." # Linter enforcement
            uncategorized_idx = self.identify_batch(batch["ids"])

            uncategorized_messages: List[Tuple[str, str, bool]] = []
            for idx in uncategorized_idx:
                message_id, message_content = batch["ids"][idx], batch["documents"][idx]
                is_empty = batch["metadatas"][idx].get("empty_or_non_text", None)
                assert is_empty is not None and isinstance(is_empty, bool), "Metadata missing 'empty_or_non_text' flag."
                uncategorized_messages.append((message_id, message_content, is_empty))

            self.categorize_message_batch(uncategorized_messages)

        self.close_connections()


def subprocess(worker_cls: type[Worker], config: WorkerConfig):
    worker = worker_cls(config)
    worker.run()

class CategorizerEngine:
    def set_io(self, api_endpoint: str) -> None:
        self.api_url = api_endpoint
        self.api_post_status = f"{self.api_url}/update"
        self.api_get_status = f"{self.api_url}/status"
    
    def set_categorizer_model(
        self,
        categorizer_model: Optional[type[CategorizerModel]],
    ) -> None:
        if categorizer_model is not None:
            self.categorizer_model_type = categorizer_model
        else:
            self.categorizer_model_type = Categorizer0
        logger.info(f"Setting categorizer model: {self.categorizer_model_type.__name__}")

    def set_wrapper_classes(self) -> None:
        self.metafile_writer_type = MetafileWriter
        self.metafile_querier_type = MetafileQuerier
        self.chroma_querier_type = ChromaQuerier
    
    def set_subprocess_start_method(self) -> None:
        try:
            set_start_method("spawn", force=True)
        except RuntimeError:
            logger.warning(
                "Failed to set start method to 'spawn'. "
                "This is likely because the start method has already been set."
            )
            assert (get_start_method() == "spawn"), ("The start method is not 'spawn'. Check configuration.")  # Debug this issue if it ever occurs
        finally:
            logger.info("Subprocess start method set to 'spawn'.")


    def __init__(
        self,
        api_endpoint: str,
        override_categorizer_model: Optional[type[CategorizerModel]] = None,
        stalling_timeout: int = 45,
        LOG_FILE: Optional[Path] = None,
        CONSOLE_LOG_LEVEL: Optional[str] = "DEBUG",
    ) -> None:
        """
        Initializes the categorizer engine.
        This is a wrapper for the categorization loop.
        """
        self.set_io(api_endpoint)
        self.ongoing_log_file = LoggerSetup.configure_logger(force_log_file=LOG_FILE, console_level=CONSOLE_LOG_LEVEL)
        self.set_categorizer_model(override_categorizer_model)
        self.set_wrapper_classes()
        self.stalling_timeout = stalling_timeout
        self.set_subprocess_start_method()


    def open_connections(self) -> None:
        """Instantiate connections to the databases"""
        self.metafile_writer: MetafileWriter = self.metafile_writer_type()
        self.metafile_querier: MetafileQuerier = self.metafile_querier_type()
        self.chroma_querier: ChromaQuerier = self.chroma_querier_type()

    def close_connections(self):
        """
        Close connections to the databases, and delete the attributes
        (connections need to be reinstantiated to be reopened).
        """
        self.metafile_writer.close()
        self.metafile_querier.close()
        self.chroma_querier.close()
        del self.metafile_writer
        del self.metafile_querier
        del self.chroma_querier

    class Planner:
        """
        Establishes statistics, batch estimations, and other planning utilities for the categorization process.
        """
        def __init__(self,
                     connection_wrappers: Tuple[ChromaQuerier, MetafileQuerier],
                     ) -> None:
            self.chroma_querier, self.metafile_querier = connection_wrappers
        
        def count_uncategorized_messages(self) -> DatabaseCounts:
            """
            Counts the number of uncategorized messages in the database.
            Returns a tuple of (total count,non-empty uncategorized count, empty uncategorized count).
            """
            start_time = time()
            logger.debug("Counting uncategorized messages in the database...")

            total_uncategorized = 0
            nonempty_uncategorized = 0
            empty_uncategorized = 0

            first_uncategorized_offset: Optional[int] = None

            for batch in self.chroma_querier.stream_messages(include_metadata=True):
                assert batch["metadatas"] is not None, "Metadatas weren't fetched." # Linter enforcement

                categorized_ids = set(self.metafile_querier.match_ids(batch['ids']))
                uncategorized_ids = [msg_id for msg_id in batch['ids'] if msg_id not in categorized_ids]

                if first_uncategorized_offset is None and len(uncategorized_ids) > 0:
                    first_uncategorized_offset = batch['ids'].index(uncategorized_ids[0])
                
                total_uncategorized += len(uncategorized_ids)
                for msg_idx, msg_id in enumerate(batch['ids']):
                    if msg_id in uncategorized_ids:
                        is_empty = batch['metadatas'][msg_idx]['empty_or_non_text']
                        if is_empty:
                            empty_uncategorized += 1
                        else:
                            nonempty_uncategorized += 1
            
            elapsed = time() - start_time
            logger.debug(f"Counted {total_uncategorized} uncategorized messages "
                         f"({nonempty_uncategorized} non-empty, {empty_uncategorized} empty) in {display_time(elapsed, duration=True)}.")

            return DatabaseCounts(
                total_uncategorized = total_uncategorized,
                nonempty_uncategorized = nonempty_uncategorized,
                empty_uncategorized = empty_uncategorized,
                first_uncategorized_offset = first_uncategorized_offset,
            )

    # class Supervisor:
    #     """
    #     Supervises the categorization subprocess, monitoring its status and handling stalling detection.
    #     """
    #     pass


    def configure_worker(
            self,
            database_counts:DatabaseCounts,
        ) -> WorkerConfig:
        worker_config: WorkerConfig = {
            "log_file": self.ongoing_log_file,
            "api_endpoint": self.api_url,
            "categorizer_model_type": self.categorizer_model_type,
            "metafile_writer_type": self.metafile_writer_type,
            "metafile_querier_type": self.metafile_querier_type,
            "chroma_querier_type": self.chroma_querier_type,
            "database_counts": database_counts,
        }
        return worker_config

    def setup_loop(self) -> None:
        self.open_connections()
        planner = self.Planner(
            connection_wrappers=(
                self.chroma_querier,
                self.metafile_querier,
            )
        )
        self.database_counts = planner.count_uncategorized_messages()
        self.close_connections()

    # def start(self):
    #     worker_config = self.configure_worker(database_counts=self.database_counts)
    #     p = Process(target=subprocess, args=(Worker, worker_config)) # spawned, not forked
    #     p.start()
    #     p.join()
