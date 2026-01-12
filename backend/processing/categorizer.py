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

from chromadb.api.types import GetResult


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

        def post_wrapper(
            self,
            eta: Optional[str] = None,
            current_message_id: Optional[str] = None,
            current_speed: Optional[float] = None,
            processed_messages: int = 1,  # Increment processed messages by 1
        ) -> bool:
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
    
    class ConnectionsWrapper:
        """
        Wraps the database connection classes for easy instantiation.
        """
        def __init__(
            self,
            metafile_writer_type: type[MetafileWriter],
            metafile_querier_type: type[MetafileQuerier],
            chroma_querier_type: type[ChromaQuerier],
        ) -> None:
            self.metafile_writer_type = metafile_writer_type
            self.metafile_querier_type = metafile_querier_type
            self.chroma_querier_type = chroma_querier_type
        
        def open_connections(self):
            self.metafile_writer: MetafileWriter = self.metafile_writer_type()
            self.metafile_querier: MetafileQuerier = self.metafile_querier_type()
            self.chroma_querier: ChromaQuerier = self.chroma_querier_type()
        
        def close_connections(self):
            self.metafile_writer.close()
            self.metafile_querier.close()
            self.chroma_querier.close()
            del self.metafile_writer
            del self.metafile_querier
            del self.chroma_querier

        def stream_messages(
            self,
            include_content: bool = True,
            include_metadata: bool = True,
            offset: int = 0,
        ):
            return self.chroma_querier.stream_messages(
                include_content=include_content,
                include_metadata=include_metadata,
                offset=offset,
            )

        def identify_batch(self, message_ids: List[str]) -> List[int]:
            """Given a batch of messages, identify which ones need categorization.
            
            :param message_ids: List of message IDs in the batch.
            :return: A tuple containing a list of uncategorized message IDs and their indices in the batch.
            """
            assert self.metafile_querier is not None, "MetafileQuerier connection is not open."
            
            categorized_ids = set(self.metafile_querier.match_ids(message_ids))
            uncategorized_idx = [idx for idx, msg_id in enumerate(message_ids) if msg_id not in categorized_ids]
            return uncategorized_idx


        def write_tagline(
            self,
            message_id: str,
            categories: List[str],
        ) -> None:
            try:
                self.metafile_writer.write_single_tagline(
                    message_id=message_id,
                    tags=categories,
                    check_for_duplicates=True, # Should still not happen. Big red flag if it does.
                )
                logger.debug(f"Tags written for message {message_id}: {categories}")
            except RuntimeError as e:
                logger.critical(f"Failed to write tags for message {message_id}: {e}")
                

    def __init__(self, config:WorkerConfig) -> None:
        LoggerSetup.configure_logger(force_log_file=config["log_file"])
        
        self.signal_handler = self.SignalHandler()

        self.api_endpoint = config["api_endpoint"]
        self.api_messenger = self.APIMessenger(api_endpoint=self.api_endpoint)
        
        self.categorizer_model_wrapper = self.CategorizerModelWrapper(model_type=config["categorizer_model_type"])
        self.connections_wrapper = self.ConnectionsWrapper(
            metafile_writer_type=config["metafile_writer_type"],
            metafile_querier_type=config["metafile_querier_type"],
            chroma_querier_type=config["chroma_querier_type"],
        )
        
        self.database_counts = config["database_counts"]

        self.first_message: bool = True


    def open_connections(self):
        self.connections_wrapper.open_connections()
    
    def close_connections(self):
        self.connections_wrapper.close_connections()
        del self.connections_wrapper

    def load_categorizer_model(self):
        """Instantiate the categorizer model."""
        self.categorizer_model_wrapper.load_model()
    
    def close_categorizer_model(self):
        """Close the categorizer model and free resources."""
        self.categorizer_model_wrapper.close_model()
        del self.categorizer_model_wrapper


    def initialize_loops(self):
        logger.info("Initializing categorization loops and opening connections...")
        
        self.open_connections()
        self.load_categorizer_model()

        self.starting_time = time()
        self.last_checked_time = self.starting_time
        self.processing_times = []
        self.total_processed = 0
        self.llm_processed = 0

        logger.info("Initialization complete. Resources opened.")
    
    def end_loops(self):
        logger.info("Closing categorization loops and freeing resources...")
        self.close_categorizer_model()
        self.close_connections()
        logger.info("Resources closed.")

    def console_log_progress(self):
        cur_time = time()
        average_processing_time = (
            sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        )  # noqa: E501
        average_processing_time = int(average_processing_time * 100) / 100
        logger.info(
            f"Time since start: {display_time(cur_time - self.starting_time, duration=True)}"  # noqa: E501
        )
        logger.info(
            f"Examined  : {self.total_processed} messages out of {self.database_counts['total_uncategorized']} ({int(self.total_processed/self.database_counts['total_uncategorized']*10000)/100}%)"  # noqa: E501
        )
        logger.info(
            f"Processed : {self.llm_processed} messages"  # noqa: E501
        )
        logger.info(
            f"Average processing time: {average_processing_time} seconds"  # noqa: E501
        )
        logger.info(
            f"ETA: {display_time(average_processing_time * (self.database_counts["nonempty_uncategorized"] - self.llm_processed), duration=True)}"  # noqa: E501
        )
        self.last_checked_time = cur_time


    def categorize_message_batch(self, messages: List[Tuple[str, str, bool]]) -> bool:
        """Given a batch's subset of uncategorized messages, categorize them.
        
        :return: True if the process was aborted, False otherwise.
        """
        abort = False

        for message_id, message_content, is_empty in messages:
            if self.first_message:
                abort = self.api_messenger.post_wrapper(processed_messages=0, current_message_id=message_id)
                self.first_message = False
            else:
                average_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0
                abort = self.api_messenger.post_wrapper(
                    eta=display_time(
                        average_processing_time * (self.database_counts["total_uncategorized"] - self.total_processed),
                        duration=True,
                    ),
                    current_message_id=message_id,
                    current_speed=average_processing_time,
                )

            if abort:
                break

            if is_empty:
                categories = [EMPTY_TAG]
                self.total_processed += 1
            else:
                start_time = time()
                categories = self.categorizer_model_wrapper.categorize(message_id, message_content)
                end_time = time()
                self.processing_times.append(end_time - start_time)
                self.total_processed += 1
                self.llm_processed += 1
                # Keep processing times list to last 100 entries to avoid memory issues
                if len(self.processing_times) > 100:
                    self.processing_times = self.processing_times[-100:]
            # TODO : write tagline to database and continue code

            self.connections_wrapper.write_tagline(
                message_id=message_id,
                categories=categories,
            )
            if time() - self.last_checked_time > 60*3:
                # Console log every 3 minutes
                self.console_log_progress()

        return abort

    def process_batch(self, batch: GetResult) -> bool:
        """Process a batch of messages from the database.
        
        :return: True if the process was aborted, False otherwise.
        """
        assert batch["documents"] is not None, "Documents weren't fetched." # Linter enforcement
        assert batch["metadatas"] is not None, "Metadatas weren't fetched." # Linter enforcement
        uncategorized_idx = self.connections_wrapper.identify_batch(batch["ids"])

        uncategorized_messages: List[Tuple[str, str, bool]] = []
        for idx in uncategorized_idx:
            message_id, message_content = batch["ids"][idx], batch["documents"][idx]
            is_empty = batch["metadatas"][idx].get("empty_or_non_text", None)
            assert is_empty is not None and isinstance(is_empty, bool), "Metadata missing 'empty_or_non_text' flag."
            uncategorized_messages.append((message_id, message_content, is_empty))

        abort = self.categorize_message_batch(uncategorized_messages)
        return abort

    def run(self) -> bool:
        """Main categorization loop.
        
        :return: True if the process was aborted, False otherwise.
        """
        if self.database_counts["total_uncategorized"] is None:
            # TODO : handle differently ?
            logger.info("No uncategorized messages found. Exiting categorization loop.")
            self.close_connections()
            return False
        assert self.database_counts["first_uncategorized_offset"] is not None # For mypy linter

        self.initialize_loops()

        abort = False
        for n, batch in enumerate(
            self.connections_wrapper.stream_messages(
                include_content=True,
                include_metadata=True,
                offset=self.database_counts["first_uncategorized_offset"],
                )
            ):
            logger.debug(f"Processing batch (number {n})...")
            abort = self.process_batch(batch)
            if abort:
                break

        if abort:
            logger.warning("Abort signal received from API. Exiting categorization process. Closing resources...")
        
        self.end_loops()
        logger.info("Resources closed. Exiting categorization process.")
        return abort

def subprocess(worker_cls: type[Worker], config: WorkerConfig):
    worker = worker_cls(config)
    abort = worker.run()

    # TODO :resource freeing necessary ?

    if abort:
        worker.signal_handler.abort_exit()
    else:
        worker.signal_handler.finished_exit()

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
