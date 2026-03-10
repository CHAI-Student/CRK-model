"""Core module for Model service."""

from model_service.core.exceptions import (
    FFmpegError,
    InvalidLoadcellValueError,
    LoadcellDataError,
    ModelServiceError,
    SessionError,
    SessionExpiredError,
    SessionNotFoundError,
    VideoCorruptedError,
    VideoFileNotFoundError,
    VideoProcessingError,
    YOLOGPUError,
    YOLOInferenceError,
    YOLOModelNotLoadedError,
)
from model_service.core.config import (
    Settings,
    config,
)
from model_service.core.logging_config import (
    PerformanceLogger,
    clear_correlation_id,
    get_correlation_id,
    get_logger,
    log_payload,
    set_correlation_id,
    setup_logging,
)

__all__ = [
    # Exceptions
    "ModelServiceError",
    "VideoProcessingError",
    "VideoFileNotFoundError",
    "VideoCorruptedError",
    "FFmpegError",
    "YOLOInferenceError",
    "YOLOModelNotLoadedError",
    "YOLOGPUError",
    "SessionError",
    "SessionNotFoundError",
    "SessionExpiredError",
    "LoadcellDataError",
    "InvalidLoadcellValueError",
    # Config
    "Settings",
    "config",
    # Logging
    "PerformanceLogger",
    "clear_correlation_id",
    "get_correlation_id",
    "get_logger",
    "log_payload",
    "set_correlation_id",
    "setup_logging",
]
