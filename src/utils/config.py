from pathlib import Path 
from typing import Any

import yaml

# load_yaml() -> Tek bir YAML dosyasını okur ve dict olarak döndürür.
def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f'configuration file not found')
    
    if config_path.suffix not in {'.yaml', '.yml'}:
        raise ValueError('expected a YAML file, received')
    
    with config_path.open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}
    
    if not isinstance(config, dict):
        raise ValueError(
            f'configuration must contain a dictionary'
        )
    
    return config



# deep_merge() -> İki sözlüğü iç içe birleştirir; yeni değerler eskilerin üstüne yazar.
def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any]
) -> dict[str, Any]:
    merged = base.copy()

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)

        else:
            merged[key] = value

    return merged



# load_config() -> Birden fazla YAML dosyasını sırayla yükleyip tek config yapar.
def load_config(
    config_paths: list[str | Path]
) -> dict[str, Any]:
    if not config_paths:
        raise ValueError('at leaset one configuration file is required')
    
    combined_comfig: dict[str, Any] = {}

    for config_path in config_paths:
        current_config = load_yaml(config_path)
        combined_comfig = deep_merge(combined_comfig, current_config)

    return combined_comfig




# if __name__ == "__main__": → Dosya doğrudan çalıştırılınca configleri yükler ve ekrana yazdırır.
if __name__ == "__main__":
    config = load_config(
        [
            "configs/base.yaml",
            "configs/dataset.yaml",
            "configs/model.yaml",
            "configs/verification.yaml",
            "configs/evaluation.yaml",
        ]
    )

    print(yaml.safe_dump(config, sort_keys=False))