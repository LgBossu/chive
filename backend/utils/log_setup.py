import sys
from typing import Optional

from loguru import logger

from backend.utils.path_utils import get_paths


class LoggerSetup:
    # Retrieve the log file path once to be reused.
    LOG_FILE = get_paths().log_file

    @staticmethod
    def configure_logger(console_level: Optional[str] = None) -> None:
        """
        Set up loguru logger with a console and a file sink.
        This should be called once in the application's lifetime.
        """
        if console_level not in ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            console_level = "INFO"
        # Remove any default loggers to prevent duplicate logs.
        logger.remove()

        # Set up logging to the console.
        logger.add(
            sink=sys.stdout,
            format="<level>{level:<10} | {message}</>",
            level=console_level,
            colorize=True,
        )

        # Set up logging to a file.
        logger.add(
            sink=LoggerSetup.LOG_FILE,
            format="{time} | {level:<10} | {name}:{function}:{line} - {message}",
            level="TRACE",
            backtrace=True,
            diagnose=True,
            colorize=True,
        )

        logger.info("Logger set up")


if __name__ == "__main__":
    LoggerSetup.configure_logger()
