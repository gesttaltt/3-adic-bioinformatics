# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Observability utilities (logging, metrics, coverage)."""

from src.utils.observability.async_writer import AsyncTensorBoardWriter, NullWriter
from src.utils.observability.coverage import CoverageEvaluator, CoverageStats
from src.utils.observability.logging import get_logger, setup_logging
from src.utils.observability.metrics_buffer import MetricsBuffer
from src.utils.observability.training_history import TrainingHistory

# Backward compatibility aliases
AsyncWriter = AsyncTensorBoardWriter
CoverageTracker = CoverageEvaluator

__all__ = [
    "AsyncWriter",
    "AsyncTensorBoardWriter",
    "NullWriter",
    "CoverageTracker",
    "CoverageEvaluator",
    "CoverageStats",
    "setup_logging",
    "get_logger",
    "MetricsBuffer",
    "TrainingHistory",
]
