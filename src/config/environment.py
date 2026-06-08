# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Environment-aware configuration loader.

This module provides environment-based configuration that allows seamless
switching between development, testing, and production environments without
code changes.

Usage:
    from src.config.environment import get_env_config, Environment

    config = get_env_config()
    print(f"Running in {config.env.value} mode")
    print(f"Checkpoints: {config.checkpoint_dir}")

Environment Variables:
    TERNARY_VAE_ENV: "development" | "test" | "production" (default: development)
    CHECKPOINT_DIR: Override checkpoint directory
    TENSORBOARD_DIR: Override TensorBoard directory
    LOG_DIR: Override log directory
    LOG_LEVEL: "DEBUG" | "INFO" | "WARNING" | "ERROR" (default: INFO)
    CUDA_VISIBLE_DEVICES: GPU selection
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .constants import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_TENSORBOARD_DIR,
)

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


@dataclass
class EnvConfig:
    """Environment-aware configuration.

    Attributes:
        env: Current environment (development/test/production)
        checkpoint_dir: Directory for model checkpoints
        tensorboard_dir: Directory for TensorBoard logs
        log_dir: Directory for file logs
        log_level: Logging level string
        device: Compute device ("cuda", "cpu", or specific GPU)
        debug_mode: Enable debug features (extra logging, assertions)
        profile_mode: Enable profiling instrumentation
    """

    env: Environment
    checkpoint_dir: Path
    tensorboard_dir: Path
    log_dir: Path
    log_level: str
    device: str
    debug_mode: bool = False
    profile_mode: bool = False

    # Derived paths (created in __post_init__)
    _experiment_dir: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate and create directories if needed."""
        # Ensure directories exist
        for dir_path in [self.checkpoint_dir, self.tensorboard_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Set debug mode based on environment
        if self.env == Environment.DEVELOPMENT:
            self.debug_mode = True

    @classmethod
    def from_env(cls) -> EnvConfig:
        """Create configuration from environment variables.

        Returns:
            EnvConfig instance configured from environment
        """
        # Determine environment
        env_name = os.getenv("TERNARY_VAE_ENV", "development").lower()
        try:
            env = Environment(env_name)
        except ValueError:
            logger.warning(f"Unknown environment '{env_name}', defaulting to development")
            env = Environment.DEVELOPMENT

        # Determine device
        cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            device = "cuda"
        else:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get log level
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            log_level = "INFO"

        # Get paths with environment-specific subdirectories
        base_checkpoint = Path(os.getenv("CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))
        base_tensorboard = Path(os.getenv("TENSORBOARD_DIR", DEFAULT_TENSORBOARD_DIR))
        base_log = Path(os.getenv("LOG_DIR", DEFAULT_LOG_DIR))

        # Add environment suffix for isolation
        if env != Environment.PRODUCTION:
            suffix = f"_{env.value}"
            checkpoint_dir = base_checkpoint.parent / f"{base_checkpoint.name}{suffix}"
            tensorboard_dir = base_tensorboard.parent / f"{base_tensorboard.name}{suffix}"
            log_dir = base_log.parent / f"{base_log.name}{suffix}"
        else:
            checkpoint_dir = base_checkpoint
            tensorboard_dir = base_tensorboard
            log_dir = base_log

        # Profile mode
        profile_mode = os.getenv("PROFILE_MODE", "0").lower() in ("1", "true", "yes")

        config = cls(
            env=env,
            checkpoint_dir=checkpoint_dir,
            tensorboard_dir=tensorboard_dir,
            log_dir=log_dir,
            log_level=log_level,
            device=device,
            profile_mode=profile_mode,
        )

        logger.info(f"Environment config loaded: {env.value} on {device}")
        return config

    def get_experiment_dir(self, experiment_name: str) -> Path:
        """Get directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to experiment directory (created if needed)
        """
        exp_dir = self.checkpoint_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def configure_logging(self) -> None:
        """Configure logging based on environment settings."""
        level = getattr(logging, self.log_level, logging.INFO)

        # Configure root logger
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler for persistent logs
        if self.env != Environment.TEST:
            log_file = self.log_dir / "ternary_vae.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logging.getLogger().addHandler(file_handler)

        # Reduce noise from third-party libraries
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.env == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env == Environment.PRODUCTION

    @property
    def is_test(self) -> bool:
        """Check if running in test mode."""
        return self.env == Environment.TEST


# Module-level singleton
_env_config: EnvConfig | None = None


def get_env_config() -> EnvConfig:
    """Get the environment configuration singleton.

    Returns:
        EnvConfig instance (created on first call)
    """
    global _env_config
    if _env_config is None:
        _env_config = EnvConfig.from_env()
    return _env_config


def reset_env_config() -> None:
    """Reset the environment configuration singleton.

    Useful for testing or when environment variables change.
    """
    global _env_config
    _env_config = None


__all__ = [
    "Environment",
    "EnvConfig",
    "get_env_config",
    "reset_env_config",
]
