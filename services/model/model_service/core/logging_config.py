"""
Structured logging configuration for Model Service module.

This module provides centralized logging configuration with correlation IDs,
structured output, and performance metrics.
"""

import contextvars
import logging
import time
from typing import Any, Optional
import uuid


# Context variable for correlation ID tracking
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationIdFilter(logging.Filter):
    """Logging filter that adds correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add correlation ID to log record.

        Args:
            record: Log record to filter

        Returns:
            True (always pass the record through)
        """
        record.correlation_id = correlation_id_var.get() or "N/A"
        return True


class StructuredFormatter(logging.Formatter):
    """Structured log formatter with consistent field ordering."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with structured fields.

        Args:
            record: Log record to format

        Returns:
            Formatted log message
        """
        # Build structured log message
        parts = [
            f"[{self.formatTime(record, self.datefmt)}]",
            f"[{record.levelname}]",
            f"[{getattr(record, 'correlation_id', 'N/A')}]",
            f"[{record.name}]",
            record.getMessage(),
        ]

        # Add exception info if present
        if record.exc_info:
            parts.append("\n" + self.formatException(record.exc_info))

        return " ".join(parts)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert log level string to constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create console handler with structured formatter
    handler = logging.StreamHandler()
    handler.setLevel(numeric_level)

    # Apply structured formatter
    formatter = StructuredFormatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    # Add correlation ID filter
    handler.addFilter(CorrelationIdFilter())

    # Configure root logger (모든 모듈에 적용)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers from root
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    root_logger.addHandler(handler)

    # Also configure "model" logger for backward compatibility
    model_logger = logging.getLogger("model")
    model_logger.setLevel(numeric_level)
    model_logger.propagate = True  # root로 전파

    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"model.{name}")


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for the current context.

    Args:
        correlation_id: Correlation ID to set (generates UUID if None)

    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID.

    Returns:
        Current correlation ID or None if not set
    """
    return correlation_id_var.get()


def clear_correlation_id() -> None:
    """Clear the current correlation ID."""
    correlation_id_var.set(None)


class PerformanceLogger:
    """Context manager for logging operation performance."""

    def __init__(self, logger: logging.Logger, operation: str, **context: Any) -> None:
        """
        Initialize performance logger.

        Args:
            logger: Logger instance to use
            operation: Name of the operation being measured
            **context: Additional context to log
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time: Optional[float] = None

    def __enter__(self) -> "PerformanceLogger":
        """Start performance measurement."""
        self.start_time = time.perf_counter()
        context_str = " ".join(f"{k}={v}" for k, v in self.context.items())
        self.logger.debug(f"Starting {self.operation} {context_str}".strip())
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        End performance measurement and log results.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        if self.start_time is None:
            return

        duration_ms = (time.perf_counter() - self.start_time) * 1000
        context_str = " ".join(f"{k}={v}" for k, v in self.context.items())

        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation} in {duration_ms:.2f}ms {context_str}".strip()
            )
        else:
            self.logger.error(
                f"Failed {self.operation} after {duration_ms:.2f}ms {context_str} "
                f"error={exc_type.__name__}".strip()
            )


def log_payload(logger: logging.Logger, direction: str, data: bytes, label: str = "") -> None:
    """
    Log binary payload data in hex format.

    Args:
        logger: Logger instance to use
        direction: Direction indicator (e.g., "TX", "RX")
        data: Binary data to log
        label: Optional label for the payload
    """
    hex_data = data.hex().upper()
    # Format as space-separated hex pairs
    formatted = " ".join(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
    label_str = f" {label}" if label else ""
    logger.debug(f"{direction}{label_str}: {formatted} ({len(data)} bytes)")
