"""
Model Service Configuration.

Pydantic BaseSettings 기반 환경변수 설정.
Jetson Orin Nano TensorRT 전용.
"""

import logging

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Settings Classes
# =============================================================================


class APIModel(BaseModel):
    """API server configuration settings."""

    host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    port: int = Field(
        default=8002,
        description="API server port",
    )
    log_level: str = Field(
        default="info",
        description="API log level",
    )
    timeout_graceful_shutdown: int = Field(
        default=10,
        description="Graceful shutdown timeout in seconds",
    )

    @field_validator("port", mode="after")
    def validate_port(cls, value: int) -> int:
        if value <= 0 or value > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {value}")
        return value

    @field_validator("log_level", mode="after")
    def validate_log_level(cls, value: str) -> str:
        valid_levels = [
            "critical",
            "error",
            "warning",
            "info",
            "debug",
            "trace",
        ]
        if value.lower() not in valid_levels:
            raise ValueError(f"Invalid log level: {value}")
        return value.lower()


class VisionModel(BaseModel):
    """Vision configuration settings (Jetson Orin Nano TensorRT only)."""

    yolo_model_path: str = Field(
        default="models/siyeon_best.engine",
        description="YOLO TensorRT engine path (.engine only). Jetson Orin Nano required.",
    )
    hand_class_id: int = Field(
        default=0,
        description="Hand class ID in YOLO model",
    )
    max_distance_px: float = Field(
        default=150.0,
        description="Max distance in pixels for hand-product proximity",
    )
    top_k: int = Field(
        default=1,
        description="Top-K candidates to extract",
    )

    # Ensemble settings
    top_weight: float = Field(
        default=0.5,
        description="Top camera weight in ensemble",
    )
    side_weight: float = Field(
        default=0.5,
        description="Side camera weight in ensemble",
    )
    common_class_bonus: float = Field(
        default=0.2,
        description="Bonus for common classes between cameras",
    )
    top_only_weight: float = Field(
        default=0.6,
        description="Top camera only weight (단방향 감지 시)",
    )
    side_only_weight: float = Field(
        default=0.5,
        description="Side camera only weight (단방향 감지 시)",
    )

    # FFmpeg 영상 보정 필터 - Top 카메라 (v4.6)
    ffmpeg_top_gamma: float = Field(
        default=1.5,
        description="Top 카메라 gamma correction (1.0=원본, >1=밝게)",
    )
    ffmpeg_top_contrast: float = Field(
        default=1.3,
        description="Top 카메라 contrast (1.0=원본, >1=높게)",
    )

    # FFmpeg 영상 보정 필터 - Side 카메라 (v4.6)
    ffmpeg_side_gamma: float = Field(
        default=1.5,
        description="Side 카메라 gamma correction (1.0=원본, >1=밝게)",
    )
    ffmpeg_side_contrast: float = Field(
        default=1.3,
        description="Side 카메라 contrast (1.0=원본, >1=높게)",
    )


class WeightModel(BaseModel):
    """Weight verification configuration settings."""

    tolerance_percent: float = Field(
        default=0.08,
        description="Weight tolerance percentage (0.08 = 8%)",
    )
    tolerance_grams: float = Field(
        default=3.0,
        description="Fixed weight tolerance in grams (고정 허용 오차, v5.1: 5g→3g)",
    )
    min_weight_change: float = Field(
        default=5.0,
        description="Minimum weight change in grams",
    )
    max_combination_size: int = Field(
        default=2,
        description="Maximum combination size for weight matching",
    )
    strict_mode: bool = Field(
        default=True,
        description="엄격 무게 검증 모드 (v5.1: 무게로 설명 불가 시 NO_DETECTION)",
    )
    max_combination_items: int = Field(
        default=5,
        description="조합 검색 시 최대 상품 종류 수 (v5.1)",
    )
    # v5.2: StrictWeightMatcher 추가 설정
    max_count_per_item: int = Field(
        default=10,
        description="상품당 최대 개수 (v5.2)",
    )
    max_combinations: int = Field(
        default=100,
        description="최대 조합 수 (성능 제한, v5.2)",
    )
    strict_mode_fallback: bool = Field(
        default=False,
        description="strict 모드 실패 시 기존 로직 폴백 여부 (v5.2)",
    )


class TriggerModel(BaseModel):
    """Trigger service configuration settings (v5.2)."""

    dedup_ttl_seconds: float = Field(
        default=5.0,
        description="Idempotency key 중복 체크 TTL (초)",
    )
    dedup_max_size: int = Field(
        default=1000,
        description="Deduplication 캐시 최대 크기 (메모리 누수 방지)",
    )
    queue_max_size: int = Field(
        default=20,
        description="Trigger 큐 최대 크기 (v4.10)",
    )
    min_weight_change_grams: float = Field(
        default=5.0,
        description="최소 무게 변화량 (이하면 비디오 처리 스킵)",
    )


class AsyncStreamingModel(BaseModel):
    """Async streaming video processing configuration (v5.3)."""

    enabled: bool = Field(
        default=True,
        description="Async streaming 비디오 처리 활성화 여부",
    )
    frame_queue_size: int = Field(
        default=10,
        description="프레임 큐 최대 크기 (Top/Side 인터리빙용)",
    )
    early_termination_enabled: bool = Field(
        default=False,
        description="조기 종료 기능 활성화 여부 (미래 확장용)",
    )
    early_termination_vote_threshold: int = Field(
        default=50,
        description="조기 종료를 위한 최소 투표 수 (미래 확장용)",
    )


class BufferModel(BaseModel):
    """Session store configuration settings (v4.2)."""

    ttl_seconds: float = Field(
        default=300.0,
        description="Session TTL in seconds (default: 5 minutes)",
    )
    max_sessions: int = Field(
        default=100,
        description="Maximum concurrent sessions",
    )
    cleanup_interval_seconds: float = Field(
        default=60.0,
        description="Background cleanup interval in seconds (v4.2)",
    )


class DoorSessionModel(BaseModel):
    """Door Session configuration settings (v4.2)."""

    enabled: bool = Field(
        default=True,
        description="Door Session 기능 활성화 여부",
    )
    yaml_dir: str = Field(
        default="data/sessions",
        description="YAML 저장 디렉토리",
    )
    session_timeout_seconds: float = Field(
        default=30.0,
        description="마지막 trigger 후 타임아웃 (초)",
    )
    weight_tolerance_grams: float = Field(
        default=5.0,
        description="반환 매칭 무게 허용 오차 (g)",
    )
    max_duration_seconds: float = Field(
        default=600.0,
        description="최대 세션 지속 시간 (초, 10분)",
    )
    yaml_retention_days: int = Field(
        default=7,
        description="완료된 YAML 세션 파일 보관 기간 (일, v4.2)",
    )


class Settings(BaseSettings):
    """
    Global application settings.

    Environment Variables (with MODEL__ prefix):
        MODEL__API__HOST: API server host (default: 0.0.0.0)
        MODEL__API__PORT: API server port (default: 8002)
        MODEL__API__LOG_LEVEL: Log level (default: info)
        MODEL__VISION__YOLO_MODEL_PATH: YOLO model path
        MODEL__NODEJS_URL: Node.js orchestrator URL
    """

    model_config = SettingsConfigDict(
        env_prefix="MODEL__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api: APIModel = APIModel()
    vision: VisionModel = VisionModel()
    weight: WeightModel = WeightModel()
    buffer: BufferModel = BufferModel()
    door_session: DoorSessionModel = DoorSessionModel()
    trigger: TriggerModel = TriggerModel()  # v5.2
    async_streaming: AsyncStreamingModel = AsyncStreamingModel()  # v5.3

    # Node.js Orchestrator settings
    nodejs_url: str = Field(
        default="http://localhost:8888",
        description="Node.js orchestrator URL",
    )
    nodejs_judgment_endpoint: str = Field(
        default="/api/sensor/judgment",
        description="Node.js judgment endpoint",
    )

    # Convenience properties for commonly used settings
    @property
    def host(self) -> str:
        return self.api.host

    @property
    def port(self) -> int:
        return self.api.port

    @property
    def log_level(self) -> str:
        return self.api.log_level

    @property
    def yolo_model_path(self) -> str:
        return self.vision.yolo_model_path

    @property
    def top_weight(self) -> float:
        return self.vision.top_weight

    @property
    def side_weight(self) -> float:
        return self.vision.side_weight

    @property
    def common_class_bonus(self) -> float:
        return self.vision.common_class_bonus

    @property
    def top_only_weight(self) -> float:
        return self.vision.top_only_weight

    @property
    def side_only_weight(self) -> float:
        return self.vision.side_only_weight


# Global config instance
config = Settings()


if __name__ == "__main__":
    settings = Settings()
    print(settings.model_dump_json(indent=4))
