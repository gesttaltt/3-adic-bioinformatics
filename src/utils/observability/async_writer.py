# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Async TensorBoard writer - Non-blocking I/O.

This module provides an async writer that:
1. Accepts metrics from MetricsBuffer
2. Batches writes to reduce I/O overhead
3. Flushes asynchronously to avoid blocking training

Architecture:
    MetricsBuffer.drain() -> AsyncTensorBoardWriter.write()
                                      |
                                      v (background thread)
                                TensorBoard files

Usage:
    writer = AsyncTensorBoardWriter('runs/experiment')

    # Non-blocking write
    records = buffer.drain()
    writer.write(records)

    # Single flush per epoch (async)
    writer.flush_async()

    # Cleanup
    writer.close()
"""

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

from .metrics_buffer import MetricRecord

logger = logging.getLogger(__name__)

# TensorBoard integration (optional)
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class AsyncTensorBoardWriter:
    """Async TensorBoard writer with batched I/O.

    Writes happen in a background thread to avoid blocking training.
    Flushes are batched to reduce I/O overhead.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str | None = None,
        flush_interval: float = 5.0,  # Seconds between auto-flushes
        queue_size: int = 10000,
    ):
        """Initialize async writer.

        Args:
            log_dir: Base directory for TensorBoard logs
            experiment_name: Name for this experiment (auto-generated if None)
            flush_interval: Seconds between automatic flushes
            queue_size: Maximum queue size before blocking
        """
        self._enabled = TENSORBOARD_AVAILABLE

        if not self._enabled:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            return

        # Generate experiment name if not provided
        if experiment_name is None:
            import datetime

            experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        log_path = Path(log_dir) / f"ternary_vae_{experiment_name}"
        self._writer = SummaryWriter(str(log_path))
        self._log_path = log_path

        # Async infrastructure
        self._queue: queue.Queue[list[MetricRecord] | tuple[str, None]] = queue.Queue(maxsize=queue_size)
        self._flush_interval = flush_interval
        self._running = True
        self._last_flush = time.time()

        # Start background writer thread
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

        # Stats
        self._records_written = 0
        self._flushes = 0

        logger.info(f"AsyncTensorBoardWriter: logging to {log_path}")

    def write(self, records: list[MetricRecord]) -> None:
        """Queue records for async writing.

        Non-blocking unless queue is full.

        Args:
            records: List of metric records to write
        """
        if not self._enabled or not records:
            return

        try:
            self._queue.put_nowait(records)
        except queue.Full:
            logger.warning(f"TensorBoard write queue full, dropping {len(records)} records")

    def write_scalar(self, name: str, value: float, step: int) -> None:
        """Write a single scalar.

        Convenience method for immediate writes.

        Args:
            name: Metric name
            value: Metric value
            step: Global step
        """
        record = MetricRecord(name=name, value=value, step=step)
        self.write([record])

    def write_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Write multiple scalars.

        Args:
            metrics: Dict of name -> value
            step: Global step
        """
        records = [MetricRecord(name=k, value=v, step=step) for k, v in metrics.items()]
        self.write(records)

    def flush_async(self) -> None:
        """Request an async flush.

        The flush will happen in the background thread.
        """
        if self._enabled:
            self._queue.put(("FLUSH", None))

    def _writer_loop(self) -> None:
        """Background thread that processes the write queue."""
        while self._running:
            try:
                # Block for a bit, then check for auto-flush
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    # Check if we need auto-flush
                    if time.time() - self._last_flush > self._flush_interval:
                        self._do_flush()
                    continue

                if item == ("FLUSH", None):
                    self._do_flush()
                elif isinstance(item, list):
                    self._write_records(item)

            except Exception as e:
                logger.error(f"AsyncTensorBoardWriter error: {e}")

    def _write_records(self, records: list[MetricRecord]) -> None:
        """Write records to TensorBoard (called in background thread)."""
        for record in records:
            # Handle grouped scalars (metrics with tags)
            if record.tags:
                tag_group = "/".join(f"{k}_{v}" for k, v in record.tags.items())
                name = f"{record.name}/{tag_group}"
            else:
                name = record.name

            self._writer.add_scalar(name, record.value, record.step)
            self._records_written += 1

    def _do_flush(self) -> None:
        """Execute a flush (called in background thread)."""
        self._writer.flush()
        self._last_flush = time.time()
        self._flushes += 1

    def get_stats(self) -> dict[str, Any]:
        """Get writer statistics."""
        return {
            "records_written": self._records_written,
            "flushes": self._flushes,
            "queue_size": self._queue.qsize() if self._enabled else 0,
            "log_path": str(self._log_path) if self._enabled else None,
        }

    def close(self) -> None:
        """Close the writer and wait for pending writes."""
        if not self._enabled:
            return

        # Signal thread to stop
        self._running = False

        # Wait for thread to finish
        self._thread.join(timeout=5.0)

        # Final flush
        self._do_flush()

        # Close TensorBoard writer
        self._writer.close()

        logger.info(f"AsyncTensorBoardWriter closed: {self._records_written} records, {self._flushes} flushes")


class NullWriter:
    """Null writer for when TensorBoard is disabled.

    Implements the same interface but does nothing.
    """

    def write(self, records: list[MetricRecord]) -> None:
        pass

    def write_scalar(self, name: str, value: float, step: int) -> None:
        pass

    def write_scalars(self, metrics: dict[str, float], step: int) -> None:
        pass

    def flush_async(self) -> None:
        pass

    def get_stats(self) -> dict[str, Any]:
        return {"enabled": False}

    def close(self) -> None:
        pass


def create_writer(log_dir: str | None, experiment_name: str | None = None) -> "AsyncTensorBoardWriter | NullWriter":
    """Factory to create appropriate writer.

    Args:
        log_dir: Log directory, or None to disable
        experiment_name: Experiment name

    Returns:
        AsyncTensorBoardWriter if log_dir provided, NullWriter otherwise
    """
    if log_dir is None:
        return NullWriter()
    return AsyncTensorBoardWriter(log_dir, experiment_name)


__all__ = ["AsyncTensorBoardWriter", "NullWriter", "create_writer"]
