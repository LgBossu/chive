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
    """
    A wrapper class to run categorization in a separate process,
    to control stalling on difficult messages.
    """

    def __init__(
        self,
        categorizer_model: AvailableCategorizers,
        metafile_writer: MetafileWriter,
    ) -> None:
        """
        Initializes the CategorizerEngine with a given categorizer model.

        :param categorizer_model: An instance of a subclass of CategorizerModel.
        """
        self.categorizer_model = categorizer_model
        self.metafile_writer = metafile_writer

    def _worker_categorize(self, queue: Queue, message: str, max_output_length: int) -> None:
        """
        Worker function to categorize a message in a separate process.

        :param queue: The queue to put the result into.
        :param message: The message to categorize.
        :param max_output_length: The maximum length of the output categories.
        """
        try:
            categories = self.categorizer_model.categorize(
                message, max_output_length=max_output_length
            )
            queue.put(categories)
        except Exception as e:
            queue.put(e)

    def get_categories(
        self,
        message: str,
        message_id: str,
        max_output_length_override: Optional[int] = None,
        timeout_override: Optional[int] = None,
    ) -> List[str] | None:
        """
        Categorizes a message using the categorizer model in a separate process.

        :param message: The message to categorize.
        :param message_id: The ID of the message being categorized.
        :param max_output_length: The maximum length of the output categories.
        :return: A list of categories for the message.
        """
        if timeout_override is not None:
            timeout = timeout_override
        else:
            timeout = self.categorizer_model.timeout

        if max_output_length_override is not None:
            max_output_length = max_output_length_override
        else:
            max_output_length = self.categorizer_model.recommended_output_length

        queue = Queue()
        process = Process(target=self._worker_categorize, args=(queue, message, max_output_length))

        process.start()
        process.join(timeout)

        if process.is_alive():
            logger.warning(
                f"Message {message_id} has timed out after {timeout}s and will be blacklisted"  # noqa: E501
            )
            process.terminate()
            process.join()
            return [BLACKLIST_TAG]

        if queue.empty():
            logger.error(f"Subprocess finished but returned nothing for message {message_id}.")
            return None

        result = queue.get()

        if isinstance(result, Exception):
            logger.error(f"Error during category generation for message {message_id}: {result}")
            return None

        return result

    def categorize_single(
        self,
        message: str,
        message_id: str,
        max_output_length_override: Optional[int] = None,
        timeout_override: Optional[int] = None,
    ) -> None:
        """
        Categorizes a single message using the categorizer model.

        This method writes the categories to the database directly after inference.

        :param message: The message to categorize.
        :param message_id: The ID of the message being categorized.
        :param max_output_length_override: Optional maximum length of the output categories.
        :param timeout_override: Optional timeout for the categorization process.
        """
        categories = self.get_categories(
            message,
            message_id,
            max_output_length_override=max_output_length_override,
            timeout_override=timeout_override,
        )

        if categories is None:
            logger.error(f"Failed to categorize message {message_id}.")
            raise RuntimeError(f"Failed to categorize message {message_id}.")

        if not categories:
            logger.warning(f"No categories returned for message {message_id}.")
            categories = [NO_TAGS_TAG]

        try:
            self.metafile_writer.write_single_tagline(
                message_id, categories, check_for_duplicates=True
            )
        except RuntimeError as e:
            logger.error(f"Failed to write categories for message {message_id}: {e}")
            raise RuntimeError(f"Failed to write categories for message {message_id}.") from e

        logger.debug(f"Message {message_id} categorized and written with: {categories}")

    def categorize_batch(
        self,
        messages: List[Tuple[str, str]],
        max_output_length_override: Optional[int] = None,
        timeout_override: Optional[int] = None,
    ) -> None:
        """
        Categorizes a batch of messages using the categorizer model.

        Because categorization is costly and sets of messages can be vast,
        this method writes categories to database as it goes,
        rather than returning them all at once.

        The method also uses a default maximum output length for categories,
        set when initializing the CategorizerEngine.

        :param messages: A list of tuples containing message ID and message text.
        :param max_output_length_override: Optional maximum length of the output categories.
        :param timeout_override: Optional timeout for the categorization process.
        """
        # TODO : absolutely refactor this method. The wrong script was refactored from legacy,
        # script logic is faulty due to abusive passing of XPU-bound objects to multiprocessing.

        logger.info(f"Batch categorizing {len(messages)} messages...")

        # Logging-relevant variables
        starting_time = time()
        last_checked_time = starting_time
        processing_times = []
        total_processed = 0
        llm_processed = 0
        uncategorized = 0

        # Get to work !
        logger.info(f"Starting categorization process at {display_time(starting_time)}")
        for message_id, message in messages:
            message_start_time = time()
            total_processed += 1
            logger.trace(
                f"Processing message {message_id}, content: {message[:20].replace('\n','')}..., timestamp: {message_start_time}"  # noqa: E501
            )

            try:
                self.categorize_single(
                    message,
                    message_id,
                    max_output_length_override=max_output_length_override,
                    timeout_override=timeout_override,
                )
                llm_processed += 1

            except RuntimeError as e:
                logger.error(f"Error categorizing message {message_id}: {e}")
                uncategorized += 1

            finally:
                message_end_time = time()
                processing_time = message_end_time - message_start_time
                processing_times.append(processing_time)

                if message_end_time - last_checked_time > 60 * 3:
                    logger.info(f"Since start: {total_processed} messages examined")
                    logger.info(
                        f"Processed {llm_processed}, failed {uncategorized} messages in {display_time(message_end_time - starting_time, duration=True)}"  # noqa: E501
                    )
                    logger.info(
                        f"Average processing time: {sum(processing_times) / len(processing_times) if processing_time else 0:.2f} seconds"  # noqa: E501
                    )
                    last_checked_time = message_end_time

        # Final summary
        end_time = time()
        logger.success(f"Batch categorization completed at {display_time(end_time)}")
        logger.info(
            f"Total messages processed: {total_processed}, successful: {llm_processed}, failed: {uncategorized}"  # noqa: E501
        )
        logger.info(
            f"Total processing time: {display_time(end_time - starting_time, duration=True)}"
        )
        logger.info(
            f"Average processing time per message: {sum(processing_times) / len(processing_times) if processing_times else 0:.2f} seconds"  # noqa: E501
        )
        # logger.info(
        #     f"Average processing time per message (excluding failed): {sum(processing_times) / llm_processed if llm_processed else 0:.2f} seconds"  # noqa: E501
        # )

    def categorize_empty_batch(self, empty_messages: List[str]) -> None:
        """
        Batch categorize messages that are empty content.

        This method DOESN'T CHECK that messages are actually empty. Message IDs provided MUST be checked for emptiness beforehand.
        """  # TODO : complete docstring  # noqa: E501
        empty_messages_dict: Dict[str, List[str]] = dict()

        for empty_message_id in empty_messages:
            empty_messages_dict[empty_message_id] = [EMPTY_TAG]

        self.metafile_writer.write_tags_dict(empty_messages_dict)


class AutoCategorizerEngine(CategorizerEngine):
    """
    A subclass of CategorizerEngine that automatically categorizes messages
    using a predefined categorizer model.
    """

    def __init__(
        self,
        categorizer_model: CategorizerModel,
        metafile_writer: MetafileWriter,
        metafile_querier: MetafileQuerier,
        chroma_querier: ChromaQuerier,
    ) -> None:
        """
        Initializes the AutoCategorizerEngine with a given categorizer model.

        :param categorizer_model: An instance of a subclass of CategorizerModel.
        :param metafile_writer: An instance of MetafileWriter to write categories.
        :param metafile_querier: An instance of MetafileQuerier to query existing categories.
        :param chroma_querier: An instance of ChromaQuerier to query existing messages.
        """
        super().__init__(categorizer_model, metafile_writer)
        self.metafile_querier = metafile_querier
        self.chroma_querier = chroma_querier

    def autodetemine_uncategorized(self) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Determines uncategorized messages by cross-checking the metafile and Chroma database.

        :return: A tuple containing a list of uncategorized messages (message ID, message text)
                 and a list of message IDs for uncategorized empty messages.
        """
        logger.info("Determining uncategorized messages...")

        # Get all messages from Chroma
        nonempty = self.chroma_querier.get_all_nonempty_messages()
        empty = self.chroma_querier.get_all_empty_messages()

        logger.info(
            f"Found {len(nonempty)} non-empty messages and {len(empty)} empty messages in Chroma."
        )  # noqa: E501

        # Get all categorized messages from metafile
        categorized = self.metafile_querier.get_all_tagged_ids()

        logger.info(f"Found {len(categorized)} categorized messages in metafile.")

        # Determine uncategorized messages
        nonempty_id_set = set(nonempty[:, 0])  # Extract message IDs from nonempty messages
        empty_id_set = set(empty)

        categorized_id_set = set(categorized)

        uncategorized_nonempty_set = nonempty_id_set - categorized_id_set
        uncategorized_empty_set = empty_id_set - categorized_id_set

        # Recast to proper types
        uncategorized_nonempty = [
            (message_id, message_text)
            for message_id, message_text in nonempty
            if message_id in uncategorized_nonempty_set
        ]
        uncategorized_empty = list(uncategorized_empty_set)

        logger.info(
            f"Found {len(uncategorized_nonempty)} uncategorized non-empty messages and {len(uncategorized_empty)} uncategorized empty messages."  # noqa: E501
        )

        return uncategorized_nonempty, uncategorized_empty

    def pipeline(self):
        """
        Runs the entire categorization pipeline:
        1. Determines uncategorized messages.
        2. Categorizes empty messages.
        3. Categorizes non-empty messages.
        """
        logger.info("Starting categorization pipeline...")

        # Step 1: Determine uncategorized messages
        uncategorized_nonempty, uncategorized_empty = self.autodetemine_uncategorized()

        # Step 2: Categorize empty messages
        if uncategorized_empty:
            self.categorize_empty_batch(uncategorized_empty)

        # Step 3: Categorize non-empty messages
        raise NotImplementedError(
            "Categorization of non-empty messages is not implemented yet. "
            "Please implement the categorization logic in the subclass."
        )
        if uncategorized_nonempty:
            self.categorize_batch(uncategorized_nonempty)

        logger.success("Categorization pipeline completed successfully.")


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
    # Initialize the categorizer engine with the model and metafile handlers
    categorizer = AutoCategorizerEngine(
        categorizer_model=Categorizer0(),
        metafile_writer=MetafileWriter(),
        metafile_querier=MetafileQuerier(),
        chroma_querier=ChromaQuerier(),
    )
    # Run the categorization pipeline
    categorizer.pipeline()
