"""
Logging utilities for research experiments.

Bu dosyanın görevi:
- Deney ilerlemesini Terminal'de göstermek
- Aynı mesajları log dosyasına kaydetmek
- Hata, uyarı ve bilgi mesajlarını düzenli tutmak
"""

import logging
from pathlib import Path


def setup_logger(
    name: str = "research",
    log_file: str | Path | None = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name:
            Logger'ın adı.
            Örnek: "raw_baseline"

        log_file:
            Logların kaydedileceği dosya yolu.
            None verilirse yalnızca Terminal'e yazar.

        level:
            Hangi önem seviyesinden itibaren mesajların
            gösterileceğini belirler.

            Yaygın seviyeler:
            - DEBUG
            - INFO
            - WARNING
            - ERROR

    Returns:
        Ayarlanmış logging.Logger nesnesi.
    """

    # Verilen isimle bir logger oluşturuyoruz.
    logger = logging.getLogger(name)

    # "INFO" gibi metin değerini logging.INFO gibi
    # sayısal bir değere çeviriyoruz.
    numeric_level = getattr(
        logging,
        level.upper(),
        None,
    )

    # Geçersiz bir logging seviyesi verilmişse hata veriyoruz.
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid logging level: {level}"
        )

    # Logger'ın minimum mesaj seviyesini belirliyoruz.
    logger.setLevel(numeric_level)

    # Mesajların ana/root logger'a tekrar gönderilmesini kapatıyoruz.
    # Aksi hâlde aynı mesaj iki kez görünebilir.
    logger.propagate = False

    # Logger daha önce ayarlanmışsa tekrar handler eklemiyoruz.
    # Bu da duplicate log mesajlarını önler.
    if logger.handlers:
        return logger

    # Her log mesajının nasıl görüneceğini belirliyoruz.
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Terminal'e yazacak handler.
    console_handler = logging.StreamHandler()

    # Terminal mesajlarına belirlediğimiz formatı uygula.
    console_handler.setFormatter(formatter)

    # Terminal handler'ını logger'a ekle.
    logger.addHandler(console_handler)

    # Kullanıcı bir log dosyası verdiyse
    # mesajları dosyaya da kaydediyoruz.
    if log_file is not None:

        # String yolu Path nesnesine çeviriyoruz.
        log_path = Path(log_file)

        # outputs/logs gibi üst klasörler yoksa oluşturuyoruz.
        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Log dosyasını UTF-8 biçiminde açacak handler.
        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8",
        )

        # Dosyadaki mesajlara aynı formatı uygula.
        file_handler.setFormatter(formatter)

        # Dosya handler'ını logger'a ekle.
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    """
    Küçük test bölümü.

    Terminal komutu:
        python -m src.utils.logger
    """

    logger = setup_logger(
        name="logger_test",
        log_file="outputs/logs/logger_test.log",
        level="INFO",
    )

    # Normal bilgi mesajı.
    logger.info(
        "Logger initialized successfully."
    )

    # Bir uyarı mesajı.
    logger.warning(
        "This is a test warning."
    )

    # Örnek hata mesajı.
    logger.error(
        "This is a test error."
    )