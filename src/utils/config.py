"""
To load YAML configuration files, validate them, and merge several configuration dictionaries into one final configuration.
"""

from pathlib import Path # Create and manage file paths safely across operating systems.
from typing import Any # Allow a function to accept values of any data type.

import yaml # YAML is a human-readable file format used to store configuration data.


# To safely read one YAML configuration file and return its contents as a Python dictionary.
def load_yaml(
    path: str | Path
) -> dict[str, Any]:
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Accept only YAML file extensions.
    if config_path.suffix not in {".yaml", '.yml'}:
        raise ValueError(f"Expected a YAML file, received: {config_path}")

    with config_path.open(
        'r',
        encoding='utf-8'
    ) as file:
        # reads YAML content and converts it into a Python object, usually a dictionary.
        config = yaml.safe_load(file)

        if config is None:
            return {}

        if not isinstance(config, dict):
            raise TypeError(
                f'Configuration must contain a dictionary: {config_path}'
            )

        return config



# To combine two dictionaries recursively, while letting values from override replace matching values in base.
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
            merged[key] = deep_merge(
                merged[key],
                value
            )

        else:
            merged[key] = value

    return merged



# Loads multiple YAML configuration files and merges them into one Python dictionary.
def load_config(
    config_paths: list[str | Path]
) -> dict[str, Any]:
    
    if not config_paths:
        raise ValueError(
            'At least one configuration file is required.'
        )

    combined_config: dict[str, Any] = {}

    for config_path in config_paths:
        current_config = load_yaml(config_path)

        combined_config = deep_merge(
            combined_config,
            current_config
        )

    return combined_config




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

    print(
        yaml.safe_dump(
            config,
            sort_keys=False,
        )
    )
