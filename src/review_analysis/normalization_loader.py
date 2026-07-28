import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_REQUIRED_TOP_LEVEL_KEYS = {
    "modalities",
    "diagnosis",
    "architectures",
    "datasets",
    "metrics",
}


def _normalization_yaml_path() -> Path:
    return Path(__file__).with_name("normalization.yaml")


def _validate_normalization_config(config: dict[str, Any]) -> None:
    missing = sorted(_REQUIRED_TOP_LEVEL_KEYS - set(config.keys()))
    if missing:
        raise ValueError(
            "Missing required normalization keys in normalization.yaml: "
            + ", ".join(missing)
        )

    if (
        not isinstance(config.get("datasets"), dict)
        or "neuro" not in config["datasets"]
    ):
        raise ValueError("'datasets.neuro' is required in normalization.yaml")


@lru_cache(maxsize=1)
def get_normalization_config() -> dict[str, Any]:
    config_path = _normalization_yaml_path()
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError("normalization.yaml must define a top-level mapping")

    _validate_normalization_config(loaded)
    logger.debug("Loaded normalization config from %s", config_path)
    return loaded
