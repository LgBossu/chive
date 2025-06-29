import datetime
import re
from multiprocessing import Process, get_start_method, set_start_method
from pathlib import Path
from time import sleep, time
from typing import Optional

from loguru import logger
from requests import ConnectionError, HTTPError, post

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

NO_STALLING_ID = "[NotAnId]"


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
        job_id: str,
        LOG_FILE: Path,
        logger_setup: LoggerSetup,
        categorizer_model_type: type[CategorizerModel],  # Lets the user specify
        # the categorizer model to use. We can try different models and implementations.
    ):
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
            eta: Optional[str] = None,
            current_message_id: Optional[str] = None,
            current_speed: Optional[float] = None,
            last_update: Optional[float] = None,
            processed_messages: int = 1,  # Increment processed messages by 1
        ) -> None:
            """
            Posts the job info to the specified URL.
            This is used to update the job status and progress in the metafile.

            A priori, this function is called after every processed message, no more, no less.
            """
            # TODO : refactor the code below to use the API after every finished message.
            # This will allow to update the job status and progress in real-time,
            url: str = (
                "http://localhost:8000/update_jobinfo"  # TODO : do not hardcode the actual job URL
            )
            if last_update is None:
                last_update = time()
            job_info = CategorizerJobInfo(
                job_id=job_id,
                status=JobStatus.RUNNING,
                total_messages=None,  # This was set at the beginning of the job
                processed_messages=processed_messages,
                eta=eta,
                last_update=last_update,
                current_message_id=current_message_id,
                current_speed=current_speed,
            )
            try:
                post(url, json=job_info.model_dump())
            except HTTPError as e:
                logger.error(f"Failed to post job progress: {e}")
            except ConnectionError as e:
                logger.error(
                    f"Failed to connect to the job progress endpoint: {e}. Is the server running?"
                )

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
        already_tagged_messages = set(metafile_querier.get_all_tagged_ids())
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

            if first_message:
                post_status_after_message(
                    processed_messages=0,  # No messages processed yet
                )
                first_message = False
            else:
                average_processing_time = (
                    sum(processing_times) / len(processing_times) if processing_times else 0
                )
                post_status_after_message(
                    current_message_id=message_id,
                    current_speed=average_processing_time,  # Average processing time  # noqa: E501
                    last_update=message_start_time,
                    eta=display_time(
                        average_processing_time * (len(nonempty_uncategorized) - total_processed),
                        duration=True,
                    ),
                )
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
                # With 5 beams full search, this can be VERY, VERY expensive real fast.
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

    def __init__(
        self,
        job_id: str,
        override_categorizer_model: Optional[type[CategorizerModel]] = None,
        stalling_timeout: int = 30,
        non_faulty_stalls_max: int = 4,
    ) -> None:
        """
        Initializes the categorizer engine.
        This is a wrapper for the categorization loop.
        """
        # Set the job ID
        self.job_id = job_id

        # Set up the logger
        self.ongoing_log_file = LoggerSetup.configure_logger()

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
        self.chroma_querier = ChromaQuerier

        # Set the stalling timeout and non-faulty stalls max
        self.stalling_timeout = stalling_timeout
        self.non_faulty_stalls_max = non_faulty_stalls_max

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
            )  # Debug this issue if it occurs
        finally:
            logger.info("Subprocess start method set to 'spawn'.")

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
        total_messages: int,
        processed_messages: int = 0,
        eta: Optional[str] = None,
    ) -> None:
        """
        Posts the progress of the categorization job to the appropriate API endpoint.
        This is used to update the job status and progress in the metafile.
        """
        job_info = CategorizerJobInfo(
            job_id=self.job_id,
            status=JobStatus.RUNNING,
            total_messages=total_messages,  # This will be set later
            processed_messages=processed_messages,  # This will be set later
            eta=eta,  # This will be set later
            last_update=time(),
            current_message_id=None,  # This will be set later
            current_speed=None,  # This will be set later
        )
        try:
            post(
                "http://localhost:8000/update_jobinfo",  # TODO : do not hardcode the actual job URL
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

    def check_for_timeouts(self, offset: int) -> None | str:
        """
        Reads the log files to detect subprocess stalling,
        and if so returns the faulty message's id

        :param LOG_FILE: Log file to search for stalling in
        :param timeout: Timeout in seconds to consider a subprocess stalled
        :return: The ID of the message that caused the stalling, or None if no stalling is detected.
        """
        # TODO : refactor absolutely this function to use the API instead of text logs.
        with open(self.ongoing_log_file, "r") as f:
            log_content = f.readlines()
        last_index = (
            -(offset * 3) - 1
        )  # Ignore the last `3*offset` lines, which are info logs on the ongoing stall
        try:
            self.parse_log_line(log_content[last_index])
        except ValueError:
            # If the line cannot be parsed, it is not a valid log line
            last_index -= 1
        except IndexError:
            # If the index is out of range, we have reached the beginning of the file
            # It is likely too early to check for stalling
            return None

        # We have a valid log line, now we can check the last time it was logged
        str_last_time = self.parse_log_line(log_content[last_index])[0]

        date_last_time = datetime.datetime.strptime(str_last_time, "%H:%M:%S.%f")
        if (datetime.datetime.now() - date_last_time).seconds > self.stalling_timeout:
            # The last log entry is older than the timeout
            id_matches = [re.search(self.log_line_hunt_id, log_content[-i]) for i in range(1, 12)]
            # Search for the first match in the last 12 lines
            id_match = next((match for match in id_matches if match), None)
            if id_match:
                return id_match.groups()[0]
            else:
                logger.warning(
                    "Did not identify the faulty message from logs. Please check formatting"
                )
                logger.warning("Program will assume non-fatal stalling.")
                return NO_STALLING_ID
        else:
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
        self.post_progress(total_messages=total_nonempty_uncategorized)

        logger.info("Setting up persistent categorization")
        while True:
            non_faulty_stalls = 0
            # TODO : check where ELSE non_faulty_stalls needs to be reset, if at all.
            subprocess = self.launch_categorization()
            logger.info("Subprocess up and running. Watching for stalling.")
            terminated = False
            while subprocess.is_alive():
                sleep(10)
                faulty_id = self.check_for_timeouts(offset=non_faulty_stalls)
                if faulty_id is None:
                    continue
                elif faulty_id == NO_STALLING_ID:
                    non_faulty_stalls += 1
                    if non_faulty_stalls > self.non_faulty_stalls_max:
                        logger.error(
                            "Subprocess stalled multiple times without identifying the faulty message. Killing."  # noqa: E501
                        )
                        subprocess.terminate()
                        terminated = True
                        break
                    else:
                        logger.warning(
                            f"Subprocess stalled {non_faulty_stalls} times without identifying the faulty message. Waiting."  # noqa: E501
                        )
                        continue
                else:
                    logger.error(f"Subprocess stalled on message {faulty_id}. Killing.")
                    logger.info(f"Blacklisting message {faulty_id}")
                    self.metafile_writer_type().write_single_tagline(
                        message_id=faulty_id,
                        tags=[BLACKLIST_TAG],
                        check_for_duplicates=True,  # Should not happen.
                        # TODO : check that it is not needed and deprecate. It is marginally costly.
                    )
                    subprocess.terminate()
                    terminated = True
                    break
            if not terminated:
                logger.info("Subprocess terminated by itself. Checking for completion.")
                if subprocess.exitcode == 0:
                    logger.success("Subprocess completed successfully.")
                    break
                else:
                    logger.error("Subprocess terminated with an error")
                    break

        logger.info("Shutting down.")


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
    categorizer_engine = CategorizerEngine(job_id="DEBUG")
    categorizer_engine.run_categorization()
