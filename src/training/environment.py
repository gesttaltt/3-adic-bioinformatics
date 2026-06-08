# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Pre-training environment validation.

This module validates the training environment before starting:
- CUDA availability and device info
- Disk space for checkpoints
- Directory write permissions
- TensorBoard availability
- Python/PyTorch version compatibility

Single responsibility: Environment validation only.
"""

import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import torch

from src.config.paths import CHECKPOINTS_DIR

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .monitor import TrainingMonitor


@dataclass
class EnvironmentStatus:
    """Environment validation results."""

    # Hardware
    cuda_available: bool = False
    cuda_device_name: str = ""
    cuda_memory_gb: float = 0.0

    # Storage
    disk_space_gb: float = 0.0
    log_dir_writable: bool = False
    checkpoint_dir_writable: bool = False

    # Dependencies
    tensorboard_available: bool = False
    pytorch_version: str = ""
    python_version: str = ""

    # Issues
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if environment is valid for training."""
        return len(self.errors) == 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Environment Status:",
            f"  Python: {self.python_version}",
            f"  PyTorch: {self.pytorch_version}",
            f"  CUDA: {'Available' if self.cuda_available else 'Not available'}",
        ]

        if self.cuda_available:
            lines.append(f"    Device: {self.cuda_device_name}")
            lines.append(f"    Memory: {self.cuda_memory_gb:.1f} GB")

        lines.extend(
            [
                f"  Disk space: {self.disk_space_gb:.1f} GB",
                f"  TensorBoard: {'Available' if self.tensorboard_available else 'Not installed'}",
            ]
        )

        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")

        return "\n".join(lines)


def validate_environment(
    config: dict[str, Any] | Any,
    monitor: Optional["TrainingMonitor"] = None,
    strict: bool = False,
) -> EnvironmentStatus:
    """Validate training environment before starting.

    Checks:
    - CUDA availability (warning if not available, error if explicitly required)
    - Disk space for checkpoints (>1GB warning, >100MB error)
    - Directory write permissions (log_dir, checkpoint_dir)
    - TensorBoard availability (warning if not installed)
    - Python/PyTorch versions

    Args:
        config: Training configuration dict (or TrainingConfig dataclass)
        monitor: Optional TrainingMonitor for logging (uses print if None)
        strict: If True, treat warnings as errors

    Returns:
        EnvironmentStatus with validation results
    """
    status = EnvironmentStatus()

    def log(msg: str) -> None:
        if monitor:
            monitor._log(msg)
        else:
            logger.info(msg)

    # Version info
    status.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    status.pytorch_version = torch.__version__

    # CUDA check
    status.cuda_available = torch.cuda.is_available()
    if status.cuda_available:
        status.cuda_device_name = torch.cuda.get_device_name(0)
        status.cuda_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log(f"CUDA available: {status.cuda_device_name} ({status.cuda_memory_gb:.1f} GB)")
    else:
        log("CUDA not available, using CPU")
        status.warnings.append("CUDA not available - training will be slow")

    # Extract paths from config (handle both dict and dataclass)
    if hasattr(config, "checkpoint_dir"):
        checkpoint_dir = config.checkpoint_dir
        log_dir = config.log_dir
        tensorboard_dir = getattr(config, "tensorboard_dir", "runs")
    else:
        checkpoint_dir = config.get("checkpoint_dir", str(CHECKPOINTS_DIR / "v5_10"))
        log_dir = config.get("log_dir", "logs")
        tensorboard_dir = config.get("tensorboard_dir", "runs")

    # Disk space check (on checkpoint directory)
    checkpoint_path = Path(checkpoint_dir)
    try:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        disk_usage = shutil.disk_usage(checkpoint_path)
        status.disk_space_gb = disk_usage.free / (1024**3)

        if status.disk_space_gb < 0.1:
            status.errors.append(f"Critically low disk space: {status.disk_space_gb:.2f} GB")
        elif status.disk_space_gb < 1.0:
            status.warnings.append(f"Low disk space: {status.disk_space_gb:.1f} GB")

        log(f"Disk space available: {status.disk_space_gb:.1f} GB")
    except Exception as e:
        status.errors.append(f"Cannot access checkpoint directory '{checkpoint_dir}': {e}")

    # Directory write permissions
    for dir_name, dir_path in [
        ("log_dir", log_dir),
        ("checkpoint_dir", checkpoint_dir),
        ("tensorboard_dir", tensorboard_dir),
    ]:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()

            if dir_name == "log_dir":
                status.log_dir_writable = True
            elif dir_name == "checkpoint_dir":
                status.checkpoint_dir_writable = True

        except PermissionError:
            status.errors.append(f"No write permission for {dir_name}: {dir_path}")
        except Exception as e:
            status.errors.append(f"Cannot access {dir_name} '{dir_path}': {e}")

    # TensorBoard check
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401

        status.tensorboard_available = True
    except ImportError:
        status.tensorboard_available = False
        status.warnings.append("TensorBoard not installed - metrics won't be visualized")

    # PyTorch version check (2.0+ recommended for torch.compile)
    major_version = int(torch.__version__.split(".")[0])
    if major_version < 2:
        status.warnings.append(f"PyTorch {torch.__version__} detected - 2.0+ recommended for torch.compile")

    # Strict mode: treat warnings as errors
    if strict and status.warnings:
        status.errors.extend([f"[strict] {w}" for w in status.warnings])

    # Log summary
    if status.warnings:
        for w in status.warnings:
            log(f"WARNING: {w}")
    if status.errors:
        for err in status.errors:
            log(f"ERROR: {err}")

    if status.is_valid:
        log("Environment validation: PASSED")
    else:
        log("Environment validation: FAILED")

    return status


def require_valid_environment(
    config: dict[str, Any] | Any, monitor: Optional["TrainingMonitor"] = None
) -> EnvironmentStatus:
    """Validate environment and raise if invalid.

    Convenience wrapper that raises ConfigValidationError if
    environment validation fails.

    Args:
        config: Training configuration
        monitor: Optional TrainingMonitor for logging

    Returns:
        EnvironmentStatus if valid

    Raises:
        RuntimeError: If environment validation fails
    """
    status = validate_environment(config, monitor)

    if not status.is_valid:
        error_msg = "Environment validation failed:\n  " + "\n  ".join(status.errors)
        raise RuntimeError(error_msg)

    return status


__all__ = [
    "EnvironmentStatus",
    "validate_environment",
    "require_valid_environment",
]
