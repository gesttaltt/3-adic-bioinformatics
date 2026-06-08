# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""In-memory metrics buffer - Zero I/O during training.

This module provides a lightweight metrics buffer that accumulates
metrics during training without any I/O. Writers can drain the
buffer periodically for async logging.

Architecture:
    Training loop -> MetricsBuffer (fast, in-memory)
                           |
                           v (drain periodically)
                     AsyncWriter (async I/O)

Usage:
    buffer = MetricsBuffer()

    # During training (fast, no I/O)
    buffer.record('loss', 0.5, step=100)
    buffer.record('coverage_A', 85.2, step=100)

    # Periodically drain for logging (can be async)
    metrics = buffer.drain()
    async_writer.write(metrics)
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricRecord:
    """Single metric record."""

    name: str
    value: float
    step: int
    tags: dict[str, str] = field(default_factory=dict)


class MetricsBuffer:
    """Thread-safe in-memory metrics buffer.

    Accumulates metrics during training without I/O.
    Designed to be drained periodically by an async writer.
    """

    def __init__(self, max_size: int = 10000):
        """Initialize metrics buffer.

        Args:
            max_size: Maximum records before auto-drain warning
        """
        self._records: list[MetricRecord] = []
        self._lock = threading.Lock()
        self._max_size = max_size
        self._overflow_warned = False

        # Aggregation for epoch-level metrics
        self._epoch_accumulators: dict[str, list[float]] = defaultdict(list)

    def record(
        self,
        name: str,
        value: float,
        step: int,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record a metric value.

        This is O(1) and does no I/O.

        Args:
            name: Metric name (e.g., 'loss', 'coverage_A')
            value: Metric value
            step: Global step or epoch number
            tags: Optional tags for grouping (e.g., {'vae': 'A'})
        """
        record = MetricRecord(name=name, value=value, step=step, tags=tags or {})

        with self._lock:
            self._records.append(record)

            if len(self._records) > self._max_size and not self._overflow_warned:
                logger.warning(f"MetricsBuffer has {len(self._records)} records. Consider draining more frequently.")
                self._overflow_warned = True

    def record_batch(
        self,
        metrics: dict[str, float],
        step: int,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record multiple metrics at once.

        Args:
            metrics: Dict of metric_name -> value
            step: Global step or epoch number
            tags: Optional tags applied to all metrics
        """
        for name, value in metrics.items():
            self.record(name, value, step, tags)

    def accumulate(self, name: str, value: float) -> None:
        """Accumulate a value for epoch-level averaging.

        Use this for batch-level metrics that should be averaged.

        Args:
            name: Metric name
            value: Value to accumulate
        """
        with self._lock:
            self._epoch_accumulators[name].append(value)

    def get_accumulated_mean(self, name: str) -> float | None:
        """Get mean of accumulated values.

        Args:
            name: Metric name

        Returns:
            Mean value, or None if no values accumulated
        """
        with self._lock:
            values = self._epoch_accumulators.get(name, [])
            if not values:
                return None
            return sum(values) / len(values)

    def clear_accumulators(self) -> dict[str, float]:
        """Clear accumulators and return means.

        Returns:
            Dict of metric_name -> mean_value
        """
        with self._lock:
            means = {}
            for name, values in self._epoch_accumulators.items():
                if values:
                    means[name] = sum(values) / len(values)
            self._epoch_accumulators.clear()
            return means

    def drain(self) -> list[MetricRecord]:
        """Drain all records from buffer.

        This is the only method that should be called by the writer.

        Returns:
            List of all accumulated records (buffer is cleared)
        """
        with self._lock:
            records = self._records
            self._records = []
            self._overflow_warned = False
            return records

    def size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._records)

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.size() == 0


class ScopedMetrics:
    """Context manager for scoped metric collection.

    Usage:
        with ScopedMetrics(buffer, step=100) as m:
            m.record('loss', 0.5)
            m.record('accuracy', 0.95)
    """

    def __init__(
        self,
        buffer: MetricsBuffer,
        step: int,
        tags: dict[str, str] | None = None,
    ):
        self._buffer = buffer
        self._step = step
        self._tags = tags or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def record(self, name: str, value: float) -> None:
        """Record a metric in this scope."""
        self._buffer.record(name, value, self._step, self._tags)


__all__ = ["MetricsBuffer", "MetricRecord", "ScopedMetrics"]
