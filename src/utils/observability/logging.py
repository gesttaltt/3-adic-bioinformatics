# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Structured logging configuration for the Ternary VAE project.

This module provides centralized logging setup that:
- Configures consistent log formatting across all modules
- Supports JSON structured logging for production
- Provides different verbosity levels
- Integrates with file and console handlers

Usage:
    from src.utils.observability.logging import setup_logging, get_logger

    # At application startup
    setup_logging(level="INFO", log_file="training.log")

    # In any module
    logger = get_logger(__name__)
    logger.info("Training started", extra={"epoch": 1, "lr": 0.001})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs.

    Ideal for production environments where logs need to be parsed.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add file/line info for errors
        if record.levelno >= logging.ERROR:
            log_entry["file"] = record.filename
            log_entry["line"] = record.lineno

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "event"):
            log_entry["event"] = record.event
        if hasattr(record, "metrics"):
            log_entry["metrics"] = record.metrics

        # Add any other extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "event",
                "metrics",
            ) and not key.startswith("_"):
                log_entry[key] = value

        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Formatter with colored output for console.

    Makes logs easier to read in development.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str | None = None, use_colors: bool = True):
        """Initialize colored formatter.

        Args:
            fmt: Log format string
            use_colors: Whether to use ANSI colors
        """
        super().__init__(fmt or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format with optional colors."""
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"

        return super().format(record)


def setup_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    json_format: bool = False,
    console: bool = True,
    colored: bool = True,
) -> None:
    """Configure logging for the application.

    Call this once at application startup to configure all loggers.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_format: Use JSON structured format (for production)
        console: Whether to output to console
        colored: Use colored console output

    Example:
        # Development
        setup_logging(level="DEBUG", colored=True)

        # Production
        setup_logging(
            level="INFO",
            log_file="logs/training.log",
            json_format=True,
            colored=False
        )
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if json_format:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(ColoredFormatter(use_colors=colored))

        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)

        if json_format:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))

        root_logger.addHandler(file_handler)

    # Set level for third-party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    This is the preferred way to get a logger. It ensures consistent
    naming and configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing batch", extra={"batch_idx": 42})
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding context to logs.

    Adds extra fields to all logs within the context.

    Example:
        with LogContext(epoch=5, phase="training"):
            logger.info("Processing...")  # Includes epoch and phase
    """

    _context: dict[str, Any] = {}

    def __init__(self, **kwargs):
        """Initialize log context with extra fields."""
        self.fields = kwargs
        self._old_factory = None

    def __enter__(self):
        """Enter context and install custom record factory."""
        LogContext._context.update(self.fields)

        old_factory = logging.getLogRecordFactory()
        self._old_factory = old_factory

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in LogContext._context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original factory."""
        for key in self.fields:
            LogContext._context.pop(key, None)

        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)


__all__ = [
    "setup_logging",
    "get_logger",
    "LogContext",
    "StructuredFormatter",
    "ColoredFormatter",
]
