"""
Load and combine YAML configuration files used by the project.

Configuration files separate experiment settings from implementation code.
For example, model, dataset, verification, and evaluation parameters can be
changed without editing Python files directly.

When multiple configuration files are loaded, they are merged in order.
Later files override earlier values when the same key appears.

Nested dictionaries are merged recursively (iç içe sözlükler alt seviyelerine
kadar birleştirilir), while non-dictionary values are replaced directly.
"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load one YAML configuration file as a Python dictionary.

    YAML (insan tarafından okunabilir yapılandırma formatı) is used to store
    experiment settings outside the Python source code.

    Empty YAML files are interpreted as empty configurations.
    """

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Reject unrelated file formats so configuration-loading mistakes are detected
    # before an experiment starts.
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Expected a YAML file, received: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        # safe_load parses YAML without constructing arbitrary Python objects,
        # which is safer for configuration files than unrestricted YAML loading.
        config = yaml.safe_load(file)

        if config is None:
            return {}

        if not isinstance(config, dict):
            raise TypeError(f"Configuration must contain a dictionary: {config_path}")

        return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge an override configuration into a base configuration.

    Override (üzerine yazan ayar) values take priority over base values.

    If both values are dictionaries, their nested keys are merged recursively.
    Other values such as strings, numbers, booleans, or lists are replaced
    completely rather than combined.
    """
    merged = base.copy()

    for key, value in override.items():
    # Merge nested configuration sections instead of replacing the entire
    # section when both sides contain dictionaries.
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)

        else:
            merged[key] = value

    return merged


def load_config(config_paths: list[str | Path]) -> dict[str, Any]:
    """
    Load multiple YAML files and combine them into one final configuration.

    Files are processed from left to right. Therefore, settings from a later
    file override matching settings from files loaded earlier.

    Example:
        base.yaml -> model.yaml -> evaluation.yaml

    If evaluation.yaml defines a key that already exists in base.yaml,
    the evaluation.yaml value becomes the final value.
    """

    if not config_paths:
        raise ValueError("At least one configuration file is required.")


    combined_config: dict[str, Any] = {}

    for config_path in config_paths:
        current_config = load_yaml(config_path)

        # Merge in order so later configuration files can intentionally override
        # defaults defined earlier.
        combined_config = deep_merge(combined_config, current_config)

    return combined_config
