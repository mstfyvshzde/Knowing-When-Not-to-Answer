"""
Bu dosya, deneylerde veya projelerde oluşan mesajları düzenli şekilde kaydetmek için bir logger oluşturur.
Mesajları terminalde gösterir.
log_file verilirse aynı mesajları dosyaya da kaydeder.
INFO, DEBUG, WARNING, ERROR gibi log seviyelerini destekler.
Log satırına tarih, seviye, logger adı ve mesaj ekler.
Log klasörü yoksa otomatik oluşturur.
Aynı logger tekrar çağrılırsa handler’ları yeniden eklemez.
"""

import logging
from pathlib import Path


def setup_logger(
    name: str = "research",
    log_file: str | Path | None = None,
    level: str = "INFO",
) -> logging.Logger:
    
    logger = logging.getLogger(name)

    numeric_level = getattr(logging, level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}")

    logger.setLevel(numeric_level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    test_logger = setup_logger(
        name="test",
        log_file="outputs/logs/test.log",
    )

    test_logger.info("Logger initialized successfully.")
    test_logger.warning("This is a test warning.")