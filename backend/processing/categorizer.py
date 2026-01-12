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
            logger.success("Exiting subprocess after successful completion.")
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
            
            self.last_update = time()
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
        self.api_messenger = self.APIMessenger(api_endpoint=config["api_endpoint"])        
        self.categorizer_model_wrapper = self.CategorizerModelWrapper(model_type=config["categorizer_model_type"])
        self.connections_wrapper = self.ConnectionsWrapper(
            metafile_writer_type=config["metafile_writer_type"],
            metafile_querier_type=config["metafile_querier_type"],
            chroma_querier_type=config["chroma_querier_type"],
        )

        self.database_counts = config["database_counts"]


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
        self.first_message: bool = True

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

                self.api_messenger.post_wrapper(processed_messages=0, current_message_id=None)
                # Notify no message is being processed, do not increment processed count

                self.total_processed += 1
                self.llm_processed += 1
                
                self.processing_times.append(end_time - start_time)
                # Keep processing times list to last 100 entries to avoid memory issues
                if len(self.processing_times) > 100:
                    self.processing_times = self.processing_times[-100:]

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
        else:
            logger.success("Categorization process completed successfully. Closing resources...")
            logger.info(f"Total messages examined: {self.total_processed}")
            logger.info(f"Total messages processed by LLM: {self.llm_processed}")
            logger.info(f"Total time elapsed: {display_time(time() - self.starting_time, duration=True)}")
        
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
    """
    A master class in charge of aggregating data, and spawning and supervising the categorization subprocess.
    """
    def set_io(self, api_endpoint: str) -> None:
        self.api_url = api_endpoint
        self.api_post_status = f"{self.api_url}/update"
        self.api_get_status = f"{self.api_url}/status"
    
    def set_categorizer_model(
        self,
        categorizer_model: Optional[type[CategorizerModel]],
    ) -> None:
        if categorizer_model is not None:
            self.categorizer_model_type: type[CategorizerModel] = categorizer_model
        else:
            self.categorizer_model_type: type[CategorizerModel] = Categorizer0
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

        :param api_endpoint: The API endpoint for job status updates.
        :param override_categorizer_model: An optional categorizer model class to override the default.
        :param stalling_timeout: Timeout in seconds to detect stalling in the subprocess.
        :param LOG_FILE: Optional path to the log file.
        :param CONSOLE_LOG_LEVEL: Optional console log level.
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

    def blacklist(self, faulty_id: str) -> None:
        """
        Blacklist a message that caused stalling by tagging it accordingly in the metafile database.
        """
        try:
            self.metafile_writer.write_single_tagline(
                message_id=faulty_id,
                tags=[BLACKLIST_TAG],
                check_for_duplicates=True,  # Should not happen. # Erratum : it happens.
                # TODO : check how and why it happens.
            )
        except RuntimeError as e:
            logger.error(
                f"Collided with a duplicate when blacklisting {faulty_id}: {e}"
            )
            # Ignore the blacklisted message even on collision, I don't know how the collision happened,
            # TODO : for now i'll ignore it, but investigate logs of the 04/07/25 to
            # try to figure it out.

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


    class Supervisor:
        """
        Supervises the categorization subprocess, monitoring its status and handling stalling detection.
        """
        def __init__(
                self,
                api_post_status: str,
                api_get_status: str,
                stalling_timeout: int,
            ) -> None:
            self.api_post_status = api_post_status
            self.api_get_status = api_get_status
            self.stalling_timeout = stalling_timeout

        def post_status(
                self,
                total_messages: Optional[int],
                status: JobStatus,
        ) -> None:
            """
            Post the current progress and status of the categorization job to the appropriate API endpoint.
            This is used to update the job status and progress in the system, for UI and data tracking.
            """
            if status == JobStatus.ABORTED:
                job_info = CategorizerJobInfo(
                    status=status,
                )
            else:
                job_info = CategorizerJobInfo(
                    status=status,
                    total_messages=total_messages,
                    processed_messages=int(status == JobStatus.STALLED),
                    # A stalling message gets blacklisted, and is thus technically getting processed
                    last_update=time(),
                )
            
            try:
                post(self.api_post_status, json=job_info.model_dump())
            except HTTPError as e:
                logger.error(f"Failed to post job status: {e}")
            except ConnectionError as e:
                logger.error(f"Failed to connect to the job status endpoint: {e}. Is the server running?")

        def check_for_timeouts(self) -> Union[str, None]:
            """
            Checks to detect subprocess stalling,
            and if so returns the faulty message's id

            """
            response = get(self.api_get_status)

            if response.status_code != 200:
                logger.error(
                    f"Failed to get job status from API: [{response.status_code} - {response.text}]"
                )
                return None
            job_info = CategorizerJobInfo.model_validate(response.json())

            if job_info.current_message_id is None:
                logger.debug("Job current message ID is None. No stalling to check.")
                return None
            elif job_info.last_update is None:
                logger.debug("Job last update is None. Cannot check for timeouts.")
                return None

            else:
                elapsed_time = time() - job_info.last_update
                if elapsed_time > self.stalling_timeout:
                    logger.warning(f"Subprocess stalled for {elapsed_time} seconds.")
                    faulty_id = job_info.current_message_id
                    logger.info(f"Faulty message ID: {faulty_id}")
                    return faulty_id
                else:
                    logger.debug(f"Subprocess is running fine. Elapsed time: {elapsed_time} seconds.")
                    return None
    
        def kill_on_timeout(self, process: Process) -> None:
            """
            Kills the subprocess if it has stalled beyond the allowed timeout.

            :param process: The monitored subprocess.
            """
            process.terminate()  # We rely on SIGTERM handlers, Unix-only.
            # Post stalling status for user information
            self.post_status(
                total_messages=None,
                status=JobStatus.STALLED,
            )
            process.join(timeout=5)
            
            # If the subprocess is still alive, we forcefully kill it
            if process.is_alive():
                logger.warning(
                    f"Subprocess {process.pid} is still alive after termination. Force killing."
                )
                process.kill()

        def parse_exit_code(self, exit_code: Union[int, None]) -> Tuple[bool, bool]:
            """
            Parses the exit code of the subprocess to determine if it was aborted.

            Returns a tuple indicating whether the subprocess has finished,
            and whether it ended normally or encountered an unhandled crash.

            :param exit_code: The exit code of the subprocess.
            :return: A tuple (finished: bool, normal_end: bool).
            """
            if exit_code == 0:
                logger.success("Subprocess completed successfully.")
                self.post_status(total_messages=None, status=JobStatus.COMPLETED)
                return True, True
            elif exit_code == 46:
                logger.info("Subprocess exited with abort code.")
                self.post_status(total_messages=None, status=JobStatus.ABORTED)
                return True, True
            elif exit_code in (143, -15):  # SIGTERM (compliant or forced on C extensions)
                logger.info("Subprocess was terminated by SIGTERM. Reloading.")
                return False, True
            elif exit_code == -signal.SIGKILL:  # SIGKILL
                logger.error("Subprocess was killed by SIGKILL. Assuming it stalled in low-level code, and restarting.")
                return False, True
            elif exit_code is None: # This happens when the process is killed by an unhandled signal
                logger.error("Subprocess crashed by unhandled signal.")
                return False, True  # Restart the subprocess
            else:
                logger.error(f"Subprocess exited with unknown code {exit_code}. Assuming critical crash.")
                self.post_status(total_messages=None, status=JobStatus.FAILED)
                return True, False


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

    def prepare_for_start(self):
        # TODO : dosctring
        self.setup_loop()
        
        self.supervisor = self.Supervisor(
            api_post_status=self.api_post_status,
            api_get_status=self.api_get_status,
            stalling_timeout=self.stalling_timeout,
        )
        self.supervisor.post_status(
            total_messages=self.database_counts["total_uncategorized"],
            status=JobStatus.RUNNING,
        )

    def start(self):
        """
        Starts the categorization subprocess.
        Requires prior setup_loop() call to establish database counts.
        """
        worker_config = self.configure_worker(database_counts=self.database_counts)
        self.process = Process(target=subprocess, args=(Worker, worker_config)) # spawned, not forked
        self.process.start()
    
    def run_categorization(self):
        """
        Runs the categorization process, supervising the subprocess and handling stalling detection.
        """
        self.prepare_for_start()

        finished, normal_end = False, False
        while not finished:
            self.start()
            logger.info(f"Subprocess started with PID {self.process.pid}. Listening for stalling.")

            while self.process.is_alive():
                sleep(self.stalling_timeout // 4)
                faulty_id = self.supervisor.check_for_timeouts()

                if faulty_id is not None:
                    self.blacklist(faulty_id)
                    self.supervisor.kill_on_timeout(self.process)
                    break

            exit_code = self.process.exitcode
            finished, normal_end = self.supervisor.parse_exit_code(exit_code)
        
        if not normal_end:
            logger.critical("Categorization subprocess ended abnormally. Please check the logs for details.")
        logger.info("Categorization engine shutting down.")
        # TODO : check if any remaining resources need to be freed
        