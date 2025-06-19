from multiprocessing import Process, Queue
from time import time
from typing import List, Optional, Tuple

from loguru import logger

from backend.loaders.metafiles_handlers import BLACKLIST_TAG, NO_TAGS_TAG, MetafileWriter
from backend.models.categorizer_model import CategorizerModel


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

    def __init__(self, categorizer_model: CategorizerModel, metafile_writer: MetafileWriter):
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
