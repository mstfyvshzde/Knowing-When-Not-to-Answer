"""
To create one logger that shows program messages in the terminal and, optionally, saves the same messages in a log file.
"""

import logging  # Logging records what the program is doing, including progress, warnings, and errors. It helps us debug experiments and keep a history of what happened.
from pathlib import (
    Path,
)  #  # Create and manage file paths safely across operating systems.


# To create one logger that shows program messages in the terminal and, optionally, saves the same messages in a log file.
def setup_logger(
    name: str = "research",  # Gives the logger an identity, such as "research" or "training".
    log_file: (
        str | Path | None
    ) = None,  # Sets the file path where logs should be saved; None means terminal only.
    level: str = "INFO",  # Sets the minimum importance level to record, such as "INFO", "WARNING", or "ERROR". (INFO -> Everything is working normally. (WARNING -> Something might become a problem. (ERROR -> Something went wrong.)))
) -> logging.Logger:
    # It creates or retrieves a logger with the given name. That logger is the object you later use with logger.info(), logger.warning(), and logger.error().
    logger = logging.getLogger(name)

    # It converts the text level, such as "info", into logging’s numeric constant, such as logging.INFO.
    # Because the logging system uses numeric levels internally, not plain text like "INFO". This line translates the user’s string into the value that logger.setLevel() understands.
    numeric_level = getattr(logging, level.upper(), None)

    if not isinstance(numeric_level, int):
        raise TypeError(f"Invalid logging level: {level}")

    # tells the logger which messages it should record.
    logger.setLevel(numeric_level)

    # stops the log message from being passed to the root logger. Without it, the same message may appear twice.
    logger.propagate = False

    # This checks whether the logger already has handlers attached.
    # If it does, the function returns the existing logger immediately so it does not add duplicate terminal or file handlers and print the same message twice.
    if logger.handlers:
        return logger

    # This defines how every log message will look.
    # Example output:
    # 2026-07-30 14:05:12 | INFO | research | Training started
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # This creates a handler that sends log messages to the terminal (console).
    # Without it, logger.info(), logger.warning(), and logger.error() would not be displayed in the terminal.
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)

        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")

        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    logger = setup_logger(
        name="logger_test",
        log_file="outputs/logs/logger_test.log",
        level="INFO",
    )

    logger.info("Logger initialized successfully.")

    logger.warning("This is a test warning.")

    logger.error("This is a test error.")
