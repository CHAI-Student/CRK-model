"""
Custom Exceptions for Model Service.

에러 타입별 분류를 위한 커스텀 예외 클래스.
"""


class ModelServiceError(Exception):
    """Model 서비스 기본 예외."""

    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class VideoProcessingError(ModelServiceError):
    """비디오 처리 관련 예외."""

    def __init__(self, message: str, video_path: str = None):
        self.video_path = video_path
        super().__init__(message, error_code="VIDEO_PROCESSING_ERROR")


class VideoFileNotFoundError(VideoProcessingError):
    """비디오 파일 없음 예외."""

    def __init__(self, video_path: str):
        super().__init__(f"Video file not found: {video_path}", video_path=video_path)
        self.error_code = "VIDEO_FILE_NOT_FOUND"


class VideoCorruptedError(VideoProcessingError):
    """손상된 비디오 파일 예외."""

    def __init__(self, video_path: str, reason: str = None):
        msg = f"Corrupted video file: {video_path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg, video_path=video_path)
        self.error_code = "VIDEO_CORRUPTED"


class FFmpegError(VideoProcessingError):
    """FFmpeg 처리 에러."""

    def __init__(self, message: str, video_path: str = None, ffmpeg_output: str = None):
        self.ffmpeg_output = ffmpeg_output
        super().__init__(message, video_path=video_path)
        self.error_code = "FFMPEG_ERROR"


class YOLOInferenceError(ModelServiceError):
    """YOLO 추론 관련 예외."""

    def __init__(self, message: str):
        super().__init__(message, error_code="YOLO_INFERENCE_ERROR")


class YOLOModelNotLoadedError(YOLOInferenceError):
    """YOLO 모델 미로드 예외."""

    def __init__(self):
        super().__init__("YOLO model not loaded")
        self.error_code = "YOLO_MODEL_NOT_LOADED"


class YOLOGPUError(YOLOInferenceError):
    """YOLO GPU 에러 (CUDA/TensorRT)."""

    def __init__(self, message: str):
        super().__init__(f"GPU error during inference: {message}")
        self.error_code = "YOLO_GPU_ERROR"


class SessionError(ModelServiceError):
    """세션 관련 예외."""

    def __init__(self, message: str, session_id: str = None):
        self.session_id = session_id
        super().__init__(message, error_code="SESSION_ERROR")


class SessionNotFoundError(SessionError):
    """세션 없음 예외."""

    def __init__(self, session_id: str):
        super().__init__(f"Session not found: {session_id}", session_id=session_id)
        self.error_code = "SESSION_NOT_FOUND"


class SessionExpiredError(SessionError):
    """세션 만료 예외."""

    def __init__(self, session_id: str):
        super().__init__(f"Session expired: {session_id}", session_id=session_id)
        self.error_code = "SESSION_EXPIRED"


class LoadcellDataError(ModelServiceError):
    """로드셀 데이터 관련 예외."""

    def __init__(self, message: str):
        super().__init__(message, error_code="LOADCELL_DATA_ERROR")


class InvalidLoadcellValueError(LoadcellDataError):
    """잘못된 로드셀 값 예외."""

    def __init__(self, value: str):
        super().__init__(f"Invalid loadcell value: {value}")
        self.error_code = "INVALID_LOADCELL_VALUE"
