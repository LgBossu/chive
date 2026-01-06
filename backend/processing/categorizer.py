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


class CategorizerEngine:
    @staticmethod
    def categorization_loop(
        LOG_FILE: Path,
        logger_setup: LoggerSetup,
        categorizer_model_type: type[CategorizerModel],  # Lets the user specify
        # the categorizer model to use. We can try different models and implementations.
        api_endpoint: Optional[str] = None,
    ):
        # Set up exit codes
        import signal
        import sys

        def sigterm_handler(signum, frame):
            logger.warning("Received SIGTERM. Exiting subprocess.")
            sys.exit(143)  # 143 = 128 + 15 (SIGTERM)

        def sigint_handler(signum, frame):
            logger.warning("Received SIGINT. Exiting subprocess.")
            sys.exit(130)  # 130 = 128 + 2 (SIGINT)

        def sigabrt_handler(signum, frame):
            logger.warning("Received SIGABRT. Exiting subprocess.")
            sys.exit(134)  # 134 = 128 + 6 (SIGABRT)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, sigterm_handler)
        signal.signal(signal.SIGINT, sigint_handler)
        signal.signal(signal.SIGABRT, sigabrt_handler)

        # Set up proper exiting function
        def abort_exit():
            """
            Exit the subprocess after abort signal.
            """
            logger.info("Exiting subprocess under abort signal.")
            sys.exit(46)  # 46 is a custom exit code for abort (leet speak "Ab = 46")

        def finished_exit():
            """
            Exit the subprocess after successful completion.
            """
            logger.info("Exiting subprocess after successful completion.")
            sys.exit(0)

        # Set up the logger for the subprocess
        logger_setup.configure_logger(
            # console_level="DEBUG",
            force_log_file=LOG_FILE,
        )
        logger.info("Subprocess logger set up")

        logger.info("Setting up utils")

        # Define post utility
        from requests import ConnectionError, HTTPError, post
        from backend.models.app_models import CategorizerJobInfo, JobStatus

        def post_status_after_message(
            api_endpoint: str,
            eta: Optional[str] = None,
            current_message_id: Optional[str] = None,
            current_speed: Optional[float] = None,
            last_update: Optional[float] = None,
            processed_messages: int = 1,  # Increment processed messages by 1
        ) -> bool:
            # TODO : add abort logic to the subprocess
            """
            Posts the job info to the specified URL.
            This is used to update the job status and progress in the metafile.

            A priori, this function is called after every processed message, no more, no less.

            It declares runtime info, and the successor message's ID (so the current message ID).
            """
            # TODO : refactor the code below to use the API after every finished message.
            # This will allow to update the job status and progress in real-time,
            if last_update is None:
                last_update = time()
            job_info = CategorizerJobInfo(
                status=JobStatus.RUNNING,
                total_messages=None,  # This was set at the beginning of the job
                processed_messages=processed_messages,
                eta=eta,
                last_update=last_update,
                current_message_id=current_message_id,
                current_speed=current_speed,
            )
            try:
                response = post(api_endpoint, json=job_info.model_dump())
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

        # Define time display utility
        def display_time(seconds: float, tz: int = 1, duration: bool = False) -> str:
            if not duration:
                _, seconds = divmod(seconds, 86400)
                hours, seconds = divmod(seconds, 3600)
                hours = (hours + tz) % 24  # We are dealing with a date and account for timezone
            else:
                hours, seconds = divmod(seconds, 3600)
            minutes, seconds = divmod(seconds, 60)
            milliseconds = (seconds - int(seconds)) * 1000
            seconds = int(seconds)
            return f"{int(hours)}h {int(minutes)}m {seconds}s {int(milliseconds)}ms"

        # Instantiate the categorizer and latent models within the subprocess
        categorizer_model: CategorizerModel = categorizer_model_type()

        # Instantiate queriers and metafile writer
        metafile_writer: MetafileWriter = MetafileWriter()
        metafile_querier: MetafileQuerier = MetafileQuerier()
        chroma_querier: ChromaQuerier = ChromaQuerier()

        # Get the tagged list
        logger.debug("Retrieving tagged list from metafile")
        already_tagged_messages = set(metafile_querier.get_all_tagged_ids()) # Horrifying
        logger.info(f"Already tagged messages: {len(already_tagged_messages)}")
        logger.debug(f"Already tagged messages: {list(already_tagged_messages)[:10]}...")

        # Get the messages to categorize
        logger.debug("Retrieving messages to categorize")
        nonempty_array = chroma_querier.get_all_nonempty_messages()
        nonempty_uncategorized = [
            (message_id, message_content)
            for message_id, message_content in nonempty_array
            if message_id not in already_tagged_messages
        ]
        logger.info(
            f"Found {len(nonempty_uncategorized)} uncategorized non-empty messages to process"
        )
        empty_array = chroma_querier.get_all_empty_messages()
        empty_uncategorized = list(set(list(empty_array)) - set(already_tagged_messages))
        logger.info(f"Found {len(empty_uncategorized)} uncategorized empty messages to process")

        # Categorize empty messages as EMPTY_TAG
        if empty_uncategorized:
            logger.info("Categorizing empty messages")
            for message_id in empty_uncategorized:
                try:
                    metafile_writer.write_single_tagline(
                        message_id=message_id,
                        tags=[EMPTY_TAG],
                        check_for_duplicates=True,  # Should not happen.
                        # TODO : check that it is not needed and deprecate. It is marginally costly.
                    )
                    logger.debug(f"Tags written for empty message {message_id}: [{EMPTY_TAG}]")
                except RuntimeError as e:
                    logger.warning(f"Failed to write tags for empty message {message_id}: {e}")

        # Start the subprocess categorization loop
        logger.info("Categorizing non-empty messages")
        starting_time = time()
        last_checked_time = starting_time
        processing_times = []
        total_processed = 0
        llm_processed = 0

        logger.info(f"Starting categorization process at {display_time(starting_time)}")
        first_message = True

        for message_id, message_content in nonempty_uncategorized:
            message_start_time = time()
            total_processed += 1

            if api_endpoint is None:
                logger.warning(
                    "No API endpoint provided for job status updates. "
                    "Job progress will not be posted."
                )
                abort = False  # No API endpoint, no abort signal
            else:
                if first_message:
                    abort = post_status_after_message(
                        api_endpoint=api_endpoint,
                        processed_messages=0,  # No messages processed yet
                        current_message_id=message_id,
                    )
                    first_message = False
                else:
                    average_processing_time = (
                        sum(processing_times) / len(processing_times) if processing_times else 0
                    )
                    abort = post_status_after_message(
                        api_endpoint=api_endpoint,
                        current_message_id=message_id,
                        current_speed=average_processing_time,  # Average processing time  # noqa: E501
                        last_update=message_start_time,
                        eta=display_time(
                            average_processing_time
                            * (len(nonempty_uncategorized) - total_processed),
                            duration=True,
                        ),
                    )
            if abort:
                logger.warning("Abord signal received. Exiting categorization loop process.")
                abort_exit()

            if message_id in already_tagged_messages:
                logger.trace(f"Message {message_id} already seen, skipping.")
                continue

            llm_processed += 1
            logger.debug(f"Categorizing message {message_id}")
            try:
                categories = categorizer_model.categorize(
                    message_content,
                    max_output_length=25,
                    safety_input_length=2048,
                )  # Do NOT FORGET TO SET THE MAXIMUM OUTPUT LENGTH
                # This omission can lead to excessive processing times and memory usage.
                # With 5 beams full search, this can be VERY, VERY expensive, real fast.
                logger.debug(f"Categories for message {message_id}: {categories}")
            except categorizer_model.ExceedingSafetyLimitError as e:
                categories = [BLACKLIST_TAG]
                logger.warning(
                    f"Message {message_id} exceeds safety limit: {e}. "
                    f"Categories for message {message_id}: {categories}"
                )

            try:
                metafile_writer.write_single_tagline(
                    message_id=message_id,
                    tags=categories,
                    check_for_duplicates=True,  # Should not happen.
                    # TODO : check that it is not needed and deprecate
                )
                logger.debug(f"Tags written for message {message_id}: {categories}")
            except RuntimeError as e:
                logger.critical(f"Failed to write tags for message {message_id}: {e}")

            message_end_time = time()
            processing_time = message_end_time - message_start_time
            processing_times.append(processing_time)

            if message_end_time - last_checked_time > 60 * 3:
                # Console log every 3 minutes
                average_processing_time = (
                    sum(processing_times) / len(processing_times) if processing_times else 0
                )  # noqa: E501
                average_processing_time = int(average_processing_time * 100) / 100
                logger.info(
                    f"Time since start: {display_time(message_end_time - starting_time, duration=True)}"  # noqa: E501
                )
                logger.info(
                    f"Examined  : {total_processed} messages out of {len(nonempty_uncategorized)} ({int(total_processed/len(nonempty_uncategorized)*10000)/100}%)"  # noqa: E501
                )
                logger.info(
                    f"Processed : {llm_processed} messages"  # noqa: E501
                )
                # logger.info(
                #     f"Jettisoned: {llm_jettisoned} messages."  # noqa: E501
                # )
                logger.info(
                    f"Average processing time: {average_processing_time} seconds"  # noqa: E501
                )
                logger.info(
                    f"ETA: {display_time(average_processing_time * (len(nonempty_uncategorized) - total_processed), duration=True)}"  # noqa: E501
                )
                last_checked_time = message_end_time

        # TODO : identify the ending reason, discriminate between success, failure, and abort.
        logger.success("Categorization process completed.")
        logger.info(f"Total messages examined    : {total_processed}")
        logger.info(f"Total messages categorized : {llm_processed}")
        # logger.info(f"Total messages jettisoned  : {llm_jettisoned}")
        logger.info(
            f"Total processing time      : {display_time(time() - starting_time, duration=True)}"  # noqa: E501
        )
        logger.info(
            f"Average processing time    : {sum(processing_times) / len(processing_times) if processing_times else 0:.2f} seconds"  # noqa: E501
        )
        del categorizer_model
        del metafile_writer
        del metafile_querier
        logger.success("Shutting down.")
        finished_exit()  # Exit the subprocess after successful completion

    # #####################################################################################
    # #####################################################################################
    # #####################################################################################

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
        # Set the API endpoint for job status updates
        self.api_url = api_endpoint
        self.api_post_status = f"{self.api_url}/update"
        self.api_get_status = f"{self.api_url}/status"

        # Set up the logger
        self.ongoing_log_file = LoggerSetup.configure_logger(
            force_log_file=LOG_FILE,
            console_level=CONSOLE_LOG_LEVEL,
        )

        # Set the categorizer model
        if override_categorizer_model is not None:
            self.categorizer_model_type = override_categorizer_model
        else:
            self.categorizer_model_type = Categorizer0
        logger.info(f"Setting categorizer model: {self.categorizer_model_type.__name__}")

        # Set the metafile writer and querier
        self.metafile_writer_type = MetafileWriter
        self.metafile_querier_type = MetafileQuerier
        # Set the chroma querier
        self.chroma_querier_type = ChromaQuerier

        # Set the stalling timeout
        self.stalling_timeout = stalling_timeout

        # Set the subprocess start method
        try:
            set_start_method("spawn", force=True)
        except RuntimeError:
            logger.warning(
                "Failed to set start method to 'spawn'. "
                "This is likely because the start method has already been set."
            )
            assert (
                get_start_method() == "spawn"
            ), (
                "The start method is not 'spawn'. Check configuration."
            )  # Debug this issue if it ever occurs
        finally:
            logger.info("Subprocess start method set to 'spawn'.")
    
    def _set_helpers(self):
        # Instantiate connections to the databases
        self.metafile_writer: MetafileWriter = self.metafile_writer_type()
        self.metafile_querier: MetafileQuerier = self.metafile_querier_type()
        self.chroma_querier: ChromaQuerier = self.chroma_querier_type()


    class Planner:
        """
        Establishes statistics, batch estimations, and other planning utilities for the categorization process.
        """
        def __init__(self,
                     connection_wrappers: Tuple[MetafileQuerier, ChromaQuerier],
                    ) -> None:
            self.metafile_querier, self.chroma_querier = connection_wrappers

        def count_uncategorized_messages(self) -> DatabaseCounts:
            """
            Counts the number of uncategorized messages in the database.
            Returns a tuple of (total count,non-empty uncategorized count, empty uncategorized count).
            """
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
            
            return DatabaseCounts(
                total_uncategorized = total_uncategorized,
                nonempty_uncategorized = nonempty_uncategorized,
                empty_uncategorized = empty_uncategorized,
                first_uncategorized_offset = first_uncategorized_offset,
            )

        def __del__(self) -> None:
            del self.metafile_querier
            del self.chroma_querier



    class Supervisor:
        """
        Supervises the categorization subprocess, monitoring its status and handling stalling detection.
        """
        pass

    class Worker:
        """
        Worker class to handle subprocess catorization tasks.
        """
        # TODO : add destructors to all helper classes, and Worker
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
            
            def display_time(self, seconds: float, tz: int = 1, duration: bool = False) -> str:
                if not duration:
                    _, seconds = divmod(seconds, 86400)
                    hours, seconds = divmod(seconds, 3600)
                    hours = (hours + tz) % 24  # We are dealing with a date and account for timezone
                else:
                    hours, seconds = divmod(seconds, 3600)

                minutes, seconds = divmod(seconds, 60)
                milliseconds = (seconds - int(seconds)) * 1000
                seconds = int(seconds)

                return f"{int(hours)}h {int(minutes)}m {seconds}s {int(milliseconds)}ms"

        def __init__(self,
                     LOG_FILE: Path,
                    #  logger_setup: LoggerSetup,
                     categorizer_model_type: type[CategorizerModel],
                     connection_wrappers: Tuple[MetafileWriter, MetafileQuerier, ChromaQuerier],
                     api_endpoint: Optional[str] = None
            ):
            # TODO : does the worker need a reference to the parent class ?
            # Set up exit codes
            LoggerSetup.configure_logger(
                force_log_file=LOG_FILE,
            )
            self.categorizer_model_type: type[CategorizerModel] = categorizer_model_type
            self.api_endpoint: Optional[str] = api_endpoint     

        def set_helpers(self):
            self.signal_handler = self.SignalHandler()
            
            if self.api_endpoint is not None:
                self.api_messenger = self.APIMessenger(api_endpoint=self.api_endpoint)
            else:
                self.api_messenger = None
                logger.warning(
                    "No API endpoint provided for job status updates. "
                    "Job progress will not be posted."
                )

            # DO NOT instantiate the model here, wait for the run method
            # self.categorizer_model: CategorizerModel = self.categorizer_model_type()




    @property
    def log_line_regex(self):
        """
        Regex to match the log lines produced by the categorizer.
        This is used to monitor the subprocess' activity, and watch for stalling.
        """
        log_line_regex = r"\d{4}-\d{2}-\d{2}T(\d{2}:\d{2}:\d{2}\.\d{6})\+\d{4}\s.\s([A-Z]+)\s*.\s[\w.]*:[^:]*:\d*\s-\s(.*)"  # noqa: E501
        return log_line_regex

    @property
    def log_line_hunt_id(self):
        """
        Regex to match and retrieve the ID responsible for subprocess stalling.
        """
        log_line_hunt_id = r"Categorizing message ([0-9a-f]{64})"
        return log_line_hunt_id

    # Posting info on the running job
    def post_progress(
        self,
        total_messages: Optional[int],
        status: JobStatus,  # Whether the job was aborted
    ) -> None:
        """
        Posts the progress of the categorization job to the appropriate API endpoint.
        This is used to update the job status and progress in the metafile.
        """
        if status == JobStatus.ABORTED:
            job_info = CategorizerJobInfo(
                status=JobStatus.ABORTED,
            )
        else:
            job_info = CategorizerJobInfo(
                status=status,
                total_messages=total_messages,  # Log the total number of messages to categorize
                processed_messages=int(status == JobStatus.STALLED),
                # If we stalled, we count the stalling message as processed,
                # since it's getting blacklisted
                last_update=time(),
                current_message_id=None,  # We are not currently processing any message
            )

        try:
            post(
                self.api_post_status,
                json=job_info.model_dump(),
            )
        except HTTPError as e:
            logger.error(f"Failed to post job progress: {e}")
        except ConnectionError as e:
            logger.error(
                f"Failed to connect to the job progress endpoint: {e}. Is the server running?"
            )

    # Util functions to factorize the code
    def parse_log_line(self, line: str) -> tuple[str, str]:
        """Parses a line from the log file"""
        parsed = re.match(self.log_line_regex, line)
        if parsed is None:
            logger.error(f"Failed to parse log line: {line}")
            raise ValueError(f"Failed to parse log line: {line}")
        return parsed.groups()[0], parsed.groups()[1]

    def check_for_timeouts(self, timeout: int) -> Union[str, None]:
        """
        Checks to detect subprocess stalling,
        and if so returns the faulty message's id

        """
        response = get(self.api_get_status)

        if response.status_code != 200:
            logger.error(
                f"Failed to get job status from API: {response.status_code} - {response.text}"
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
            if elapsed_time > timeout:
                logger.warning(f"Subprocess stalled for {elapsed_time} seconds.")
                faulty_id = job_info.current_message_id
                logger.info(f"Faulty message ID: {faulty_id}")
                return faulty_id
            else:
                logger.debug(f"Subprocess is running fine. Elapsed time: {elapsed_time} seconds.")
                return None

    # Launching and running the categorization process
    def launch_categorization(self) -> Process:
        logger.info(f"Starting subprocess - logging to {self.ongoing_log_file}")

        categorization_process = Process(
            target=self.categorization_loop,
            args=(
                self.ongoing_log_file,
                LoggerSetup,
                self.categorizer_model_type,
                self.api_post_status,  # Pass the API endpoint for job status updates
            ),
        )
        categorization_process.start()
        logger.info(f"Subprocess started with PID {categorization_process.pid}")
        return categorization_process

    def run_categorization(self) -> None:
        # Get initial job information
        logger.info("Retrieving initial job information")
        already_tagged_messages = set(MetafileQuerier().get_all_tagged_ids())
        nonempty_array = ChromaQuerier().get_all_nonempty_messages()
        total_nonempty_uncategorized = len(set(nonempty_array[:, 0]) - already_tagged_messages)
        self.post_progress(total_messages=total_nonempty_uncategorized, status=JobStatus.RUNNING)

        logger.info("Setting up persistent categorization")
        finished = False
        while not finished:
            subprocess = self.launch_categorization()
            logger.info("Subprocess up and running. Watching for stalling.")
            while subprocess.is_alive():
                sleep(self.stalling_timeout // 4)
                faulty_id = self.check_for_timeouts(timeout=self.stalling_timeout)

                if faulty_id is None:
                    continue

                else:
                    logger.error(f"Subprocess stalled on message {faulty_id}. Killing.")
                    logger.info(f"Blacklisting message {faulty_id}")
                    # Blacklist the faulty message
                    try:
                        self.metafile_writer_type().write_single_tagline(
                            message_id=faulty_id,
                            tags=[BLACKLIST_TAG],
                            check_for_duplicates=True,  # Should not happen. # Erratum : it happens.
                            # TODO : check how and why it happens.
                        )
                    except RuntimeError as e:
                        logger.error(
                            f"Collided with a duplicate when blacklisting {faulty_id}: {e}"
                        )
                        # Ignore the blacklisted message, I don't know how the collision happened,
                        # TODO : for now i'll ignore it, but investigate logs of the 04/07/25 to
                        # try to figure it out.
                    # Terminate the stalling subprocess
                    subprocess.terminate()  # We rely on SIGTERM handlers, Unix-only.
                    # Post stalling status for user information
                    self.post_progress(
                        total_messages=None,
                        status=JobStatus.STALLED,
                    )
                    # Wait for the subprocess to terminate so it can set its exit code
                    subprocess.join(timeout=5)
                    # If the subprocess is still alive, we forcefully kill it
                    if subprocess.is_alive():
                        logger.warning(
                            f"Subprocess {subprocess.pid} is still alive after termination. Force killing."  # noqa: E501
                        )
                        subprocess.kill()
                    break

            logger.info("Subprocess terminated. Checking exit code.")
            if subprocess.exitcode == 0:
                logger.success("Subprocess completed normally.")
                self.post_progress(total_messages=None, status=JobStatus.COMPLETED)
                finished = True
            elif subprocess.exitcode in (143, -15):  # SIGTERM (compliant or forced on C extensions)
                logger.info("Subprocess was terminated by SIGTERM. Reloading.")
                # This is a normal exit, we can reload the subprocess
                continue
            elif subprocess.exitcode == -signal.SIGKILL:  # SIGKILL
                logger.error(
                    "Subprocess was killed by SIGKILL. Assuming it stalled in low-level code, and restarting."  # noqa: E501
                )
                # This is a normal exit, we can reload the subprocess
                continue
            elif subprocess.exitcode is None:
                logger.error(
                    "Subprocess crashed with no exit code. This happens when the process is killed by an unhandled signal."  # noqa: E501
                )
                continue  # Restart the subprocess (generally, this is a SIGKILL)
            elif subprocess.exitcode == 46:  # Custom exit code for abort
                logger.warning(
                    "Subprocess was aborted. Exiting categorization loop and attesting reception of the signal."  # noqa: E501
                )
                self.post_progress(total_messages=None, status=JobStatus.ABORTED)  # Reset progress
                finished = True
            else:
                logger.error(
                    f"Subprocess crashed with an unknown error or exit code {subprocess.exitcode}."
                )  # noqa: E501
                self.post_progress(total_messages=None, status=JobStatus.FAILED)
                break

        logger.info(f"Shutting down. Process finished under normal status : {finished}.")
        if not finished:
            logger.error(
                "Categorization process did not finish successfully. Check logs for details."
            )


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

    # # RUN AUTOMATIC CATEGORIZER PIPELINE
    # logger.info("Running automatic categorizer pipeline...")
    # categorizer_engine = CategorizerEngine(api_endpoint="http://127.0.0.1:8000/categorizer")
    # categorizer_engine.run_categorization()
