"""
YAML configuration loading utilities.

Bu dosyanın görevi:
- YAML config dosyalarını okumak
- Birden fazla config dosyasını birleştirmek
- Deney ayarlarını Python dictionary olarak döndürmek
"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a single YAML configuration file.

    Örnek:
        config = load_yaml("configs/base.yaml")

    Dönen değer:
        YAML içeriğini Python dictionary olarak döndürür.
    """

    # Gelen string yolu Path nesnesine çeviriyoruz.
    # Path, dosya yollarını daha güvenli yönetmemizi sağlar.
    config_path = Path(path)

    # Dosya gerçekten var mı kontrol ediyoruz.
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Yanlışlıkla JSON veya TXT dosyası verilmesini engelliyoruz.
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Expected a YAML file, received: {config_path}")

    # YAML dosyasını okuma modunda açıyoruz.
    with config_path.open("r", encoding="utf-8") as file:
        # safe_load YAML içeriğini Python dictionary'ye çevirir.
        # safe_load kullanmak, yaml.load kullanmaktan daha güvenlidir.
        config = yaml.safe_load(file)

    # YAML dosyası tamamen boşsa safe_load None döndürür.
    # Bu durumda boş dictionary döndürüyoruz.
    if config is None:
        return {}

    # Config'in dictionary olması gerekir.
    # Örneğin dosyanın tamamı sadece bir listeyse hata veririz.
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must contain a dictionary: {config_path}")

    return config


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    base:
        Ana ayarlar.

    override:
        Ana ayarların üzerine yazılacak yeni ayarlar.

    Örnek:

        base = {
            "model": {
                "name": "roberta",
                "batch_size": 16
            }
        }

        override = {
            "model": {
                "batch_size": 32
            }
        }

        Sonuç:

        {
            "model": {
                "name": "roberta",
                "batch_size": 32
            }
        }
    """

    # Orijinal base dictionary'yi değiştirmemek için kopyalıyoruz.
    merged = base.copy()

    # Override içindeki her key-value çiftini inceliyoruz.
    for key, value in override.items():
        # Aynı key iki dictionary içinde de varsa
        # iç içe dictionary'leri recursive(bir fonksiyonun problemi çözmek için kendi kendini çağırması) olarak birleştiriyoruz.
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(
                merged[key],
                value,
            )

        else:
            # Key yeni ise eklenir.
            # Key zaten varsa override değeri base değerinin üzerine yazılır.
            merged[key] = value

    return merged


def load_config(
    config_paths: list[str | Path],
) -> dict[str, Any]:
    """
    Load and merge multiple YAML configuration files.

    Örnek:

        config = load_config(
            [
                "configs/base.yaml",
                "configs/dataset.yaml",
                "configs/model.yaml",
            ]
        )

    Dosyalar verilen sırayla birleştirilir.

    Daha sonra gelen config dosyası,
    önceki config içindeki aynı değerin üzerine yazabilir.
    """

    # En az bir config dosyası verilmesini zorunlu tutuyoruz.
    if not config_paths:
        raise ValueError("At least one configuration file is required.")

    # Bütün config dosyaları bunun içinde birleşecek.
    combined_config: dict[str, Any] = {}

    # Dosyaları sırayla okuyoruz.
    for config_path in config_paths:
        # Tek bir YAML dosyasını dictionary olarak yükle.
        current_config = load_yaml(config_path)

        # Yeni config'i önceki config'lerle birleştir.
        combined_config = deep_merge(
            combined_config,
            current_config,
        )

    return combined_config


if __name__ == "__main__":
    """
    Bu blok yalnızca dosya doğrudan çalıştırıldığında çalışır.

    Terminal komutu:

        python -m src.utils.config

    Başka dosyadan import edildiğinde bu bölüm çalışmaz.
    """

    config = load_config(
        [
            "configs/base.yaml",
            "configs/dataset.yaml",
            "configs/model.yaml",
            "configs/verification.yaml",
            "configs/evaluation.yaml",
        ]
    )

    # Birleşmiş config'i okunabilir YAML formatında terminale yazdır.
    print(
        yaml.safe_dump(
            config,
            sort_keys=False,
        )
    )
