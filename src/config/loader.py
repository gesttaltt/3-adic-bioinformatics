# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Configuration loader with YAML and environment variable support.

This module provides a unified configuration loading system that:
1. Loads defaults from schema
2. Overrides with YAML file (if provided)
3. Overrides with environment variables (TVAE_* prefix)
4. Overrides with explicit arguments

Priority (highest to lowest):
    explicit overrides > environment variables > YAML file > defaults

Usage:
    from src.config import load_config

    # Load with all defaults
    config = load_config()

    # Load from YAML
    config = load_config("experiments/config.yaml")

    # Load with overrides
    config = load_config("config.yaml", overrides={"epochs": 500})

    # Environment variables work automatically:
    # export TVAE_EPOCHS=1000
    # export TVAE_LEARNING_RATE=0.0001
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .schema import ConfigValidationError, TrainingConfig

logger = logging.getLogger(__name__)

# Environment variable prefix
ENV_PREFIX = "TVAE_"


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    validate: bool = True,
) -> TrainingConfig:
    """Load configuration with priority: overrides > env > yaml > defaults.

    Args:
        config_path: Path to YAML configuration file (optional)
        overrides: Dictionary of explicit overrides (optional)
        validate: Whether to validate the configuration (default True)

    Returns:
        Validated TrainingConfig instance

    Raises:
        ConfigValidationError: If validation fails
        FileNotFoundError: If config_path doesn't exist
        yaml.YAMLError: If YAML parsing fails

    Example:
        # Load defaults only
        config = load_config()

        # Load from file
        config = load_config("config.yaml")

        # Load with overrides
        config = load_config(overrides={"epochs": 500, "batch_size": 128})
    """
    config_dict: dict[str, Any] = {}

    # 1. Load from YAML if provided
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config_dict = _load_yaml(config_path)
        logger.info(f"Loaded config from {config_path}")

    # 2. Apply environment variables
    env_overrides = _load_env_vars()
    if env_overrides:
        config_dict = _deep_merge(config_dict, env_overrides)
        logger.debug(f"Applied {len(env_overrides)} environment variable overrides")

    # 3. Apply explicit overrides
    if overrides:
        config_dict = _deep_merge(config_dict, overrides)
        logger.debug(f"Applied {len(overrides)} explicit overrides")

    # 4. Create and validate config
    try:
        config = TrainingConfig.from_dict(config_dict)
    except (TypeError, ValueError) as e:
        raise ConfigValidationError(f"Invalid configuration: {e}") from e

    if validate:
        _validate_config(config)

    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file and return dictionary."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config loading. Install with: pip install pyyaml")

    with open(path) as f:
        data = yaml.safe_load(f)

    return data if data else {}


def _load_env_vars() -> dict[str, Any]:
    """Load configuration from environment variables with TVAE_ prefix.

    Environment variable mapping:
        TVAE_EPOCHS -> epochs
        TVAE_BATCH_SIZE -> batch_size
        TVAE_LEARNING_RATE -> optimizer.learning_rate
        TVAE_GEOMETRY_CURVATURE -> geometry.curvature

    Returns:
        Dictionary of overrides from environment
    """
    overrides: dict[str, Any] = {}

    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue

        # Remove prefix and convert to lowercase
        config_key = key[len(ENV_PREFIX) :].lower()

        # Parse value
        parsed_value = _parse_env_value(value)

        # Handle nested keys (TVAE_GEOMETRY_CURVATURE -> geometry.curvature)
        if "_" in config_key:
            parts = config_key.split("_", 1)
            if parts[0] in ("geometry", "optimizer", "ranking", "loss", "vae"):
                # Nested config
                nested_key = parts[0]
                inner_key = parts[1]
                if nested_key not in overrides:
                    overrides[nested_key] = {}
                overrides[nested_key][inner_key] = parsed_value
                continue

        # Top-level key
        overrides[config_key] = parsed_value

    return overrides


def _parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate Python type.

    Handles: bool, int, float, str
    """
    # Boolean
    if value.lower() in ("true", "yes", "1", "on"):
        return True
    if value.lower() in ("false", "no", "0", "off"):
        return False

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # String (default)
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _validate_config(config: TrainingConfig) -> None:
    """Additional validation beyond schema validation.

    Checks cross-field constraints and semantic validity.
    """
    # Check paths are writable (if they exist)
    for path_attr in ["checkpoint_dir", "log_dir", "tensorboard_dir"]:
        path = Path(getattr(config, path_attr))
        if path.exists() and not os.access(path, os.W_OK):
            logger.warning(f"{path_attr} exists but is not writable: {path}")

    # Check CUDA availability if device is cuda
    if config.device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("device='cuda' but CUDA is not available, falling back to CPU")
        except ImportError:
            pass

    # Warn about potentially problematic configurations
    if config.batch_size > 1024:
        logger.warning(f"Large batch_size={config.batch_size} may cause memory issues")

    if config.geometry.latent_dim > 64:
        logger.warning(f"Large latent_dim={config.geometry.latent_dim} may slow training")


def save_config(config: TrainingConfig, path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save
        path: Output file path
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {path}")


__all__ = [
    "load_config",
    "save_config",
    "ConfigValidationError",
]
