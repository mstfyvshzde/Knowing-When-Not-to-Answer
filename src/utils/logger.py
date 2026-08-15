"""
Create a reusable logger for experiment progress, warnings, and errors.

Logging (kayıt tutma), deney sırasında ne olduğunu terminalde göstermeye ve
istenirse aynı mesajları bir dosyada saklamaya yarar. Bu özellikle uzun
deneylerde hata ayıklama (debugging) ve hangi ayarlarla ne olduğunu takip etme
açısından önemlidir.

The logger writes messages to the terminal by default and can optionally add
a UTF-8 file handler when log_file is provided.
"""

import logging
from pathlib import Path


def setup_logger(
    name: str = "research",
    log_file: str | Path | None = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Create or update a named logger with console and optional file output.

    name identifies the logger so different parts of the project can keep
    separate logging channels.

    level is the minimum severity (önem seviyesi) that will be recorded.
    Common levels are INFO, WARNING, and ERROR.

    log_file enables persistent logging (mesajları dosyada kalıcı saklama)
    in addition to terminal output.
    """

    # It creates or retrieves a logger with the given name. That logger is the object you later use with logger.info(), logger.warning(), and logger.error().
    logger = logging.getLogger(name)

    # Python logging levels are represented internally as integers.
    # Convert a readable value such as "INFO" into logging.INFO.
    numeric_level = getattr(logging, level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}")  # noqa: TRY004

    # Ignore messages below the requested severity level.
    logger.setLevel(numeric_level)

    # Prevent messages from propagating (üst/root logger'a aktarılma) and being
    # printed a second time by another logger configuration.
    logger.propagate = False


    # Use the same timestamped format for both terminal and file logs.
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add one console handler only. Repeated setup_logger() calls should not
    # duplicate the same message in the terminal.
    has_console_handler = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )

    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # A file handler (dosya kayıt yöneticisi) writes the same experiment
        # messages permanently to disk. Avoid adding the same file twice when
        # setup_logger() is called repeatedly.
        resolved_log_path = log_path.resolve()

        has_file_handler = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == resolved_log_path
            for handler in logger.handlers
        )

        if not has_file_handler:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
