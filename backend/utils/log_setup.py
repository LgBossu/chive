import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.utils.path_utils import get_paths


class LoggerSetup:
    """
    A utility class to set up logging for the application using loguru.
    This class provides a static method `configure_logger` to configure logging to both the console
    and a log file.
    """
    # Retrieve the log file path once to be reused.

    @staticmethod
    def configure_logger(
        console_level: Optional[str] = None,
        force_log_file: Optional[Path] = None,
    ) -> Path:
        """
        Set up loguru logger with a console and a file sink.
        This should be called once in the application's lifetime.

        Args:
            console_level (Optional[str]): The logging level for console output. 
                                           Defaults to "INFO" if not provided or invalid.
                                           Options are: "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
            force_log_file (Optional[Path]): An optional path to force the log file location. 
                                             If not provided, the default application log file path is used.
        """
        # TODO : check when and how it is called multiple times, and if it is legitimate or if we can pass the logsetup by reference
        if force_log_file is not None:
            LOG_FILE = force_log_file
        else:
            LOG_FILE = get_paths().log_file

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
            sink=LOG_FILE,
            format="{time} | {level:<10} | {name}:{function}:{line} - {message}",
            level="TRACE",
            backtrace=True,
            diagnose=True,
            colorize=True,
        )

        logger.info("Logger set up")

        # Return the log file path for reference.
        return LOG_FILE


if __name__ == "__main__":
    LoggerSetup.configure_logger()
