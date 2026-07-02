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


class MachineModel(BaseModel):
    """Machine-level deployment configuration."""

    cabinet_type: str = Field(
        default="refrigerated",
        description="Cabinet type: refrigerated or freezer",
    )

    @field_validator("cabinet_type", mode="after")
    def validate_cabinet_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        valid = {"refrigerated", "freezer"}
        if normalized not in valid:
            raise ValueError(f"Invalid cabinet type: {value}")
        return normalized


class VisionModel(BaseModel):
    """Vision configuration settings (Jetson Orin Nano TensorRT only)."""

    yolo_model_path: str = Field(
        default="models/0204_morning.engine",
        description="YOLO TensorRT engine path (.engine only). Jetson Orin Nano required.",
    )
    yolo_internal_conf_threshold: float = Field(
        default=0.01,
        description="Internal YOLO confidence floor before service-side thresholds",
    )
    log_engine_classes: bool = Field(
        default=False,
        description="Log loaded YOLO engine class id/name pairs at startup",
    )
    hand_class_id: int = Field(
        default=0,
        description="Hand class ID in YOLO model",
    )
    hand_confidence_threshold: float = Field(
        default=0.40,
        description="Minimum confidence for hand detections used by hand tracking",
    )
    max_distance_px: float = Field(
        default=150.0,
        description="Max distance in pixels for hand-product proximity",
    )
    top_k: int = Field(
        default=10,
        description="Top-K candidates to extract",
    )
    top_confidence_threshold: float = Field(
        default=0.70,
        description="Minimum confidence for top camera detections",
    )
    side_confidence_threshold: float = Field(
        default=0.70,
        description="Minimum confidence for side camera detections",
    )
    camera_layout: str = Field(
        default="legacy_top_side",
        description="Physical camera layout: legacy_top_side or dual_top_proxy",
    )

    min_vote_ratio: float = Field(
        default=0.05,
        description="Minimum frame vote ratio for a class to remain a candidate",
    )
    min_vote_count: int = Field(
        default=2,
        description="Minimum absolute vote count for short trigger videos",
    )
    motion_min_displacement_px: float = Field(
        default=10.0,
        description="Minimum bbox-center movement for motion filtering",
    )
    freezer_min_vote_ratio: float = Field(
        default=0.08,
        description="Freezer-only minimum frame vote ratio for final candidates",
    )
    freezer_min_vote_count: int = Field(
        default=3,
        description="Freezer-only minimum absolute vote count for final candidates",
    )
    freezer_motion_min_displacement_px: float = Field(
        default=12.0,
        description="Freezer-only minimum bbox-center movement",
    )
    freezer_roi_vertical_region: str = Field(
        default="upper",
        description="Freezer dual-top vertical ROI: upper or lower",
    )
    freezer_roi_y_split: float | None = Field(
        default=None,
        description="Freezer dual-top bbox center y split. Falls back to legacy lower split when unset.",
    )
    freezer_lower_roi_y_split: float = Field(
        default=240.0,
        description="Deprecated fallback for freezer dual-top ROI split",
    )
    freezer_min_exit_path_votes: int = Field(
        default=3,
        description="Minimum freezer ROI-passed votes treated as handled exit-path evidence",
    )
    side_roi_x_max: float = Field(
        default=400.0,
        description="Maximum side-camera bbox center x accepted before ROI filtering",
    )
    side_roi_soft_margin_px: float = Field(
        default=5.0,
        description="Conditional side-camera soft ROI margin for threshold-passed detections",
    )
    roi_rescue_require_motion: bool = Field(
        default=True,
        description="Require side-camera motion before ROI-filtered detections can be rescued",
    )
    roi_rescue_max_over_limit_px: float = Field(
        default=0.0,
        description="Maximum average center_x over side ROI limit allowed for ROI rescue",
    )
    top_roi_enabled: bool = Field(
        default=True,
        description="Enable top-camera vertical ROI filtering",
    )
    top_roi_y_split: float = Field(
        default=240.0,
        description="Top-camera bbox center y split; top of image is 0",
    )
    crop_width: int = Field(
        default=480,
        description="Width used by crop policies before YOLO inference",
    )
    top_crop_policy: str = Field(
        default="left",
        description="Top camera crop policy: left, center, right, offset, none, or letterbox",
    )
    side_crop_policy: str = Field(
        default="left",
        description="Side camera crop policy: left, center, right, offset, none, or letterbox",
    )
    top_crop_x_offset: int | None = Field(
        default=None,
        description="Top camera x offset when crop policy is offset",
    )
    side_crop_x_offset: int | None = Field(
        default=None,
        description="Side camera x offset when crop policy is offset",
    )
    diagnostic_all_class_trace: bool = Field(
        default=False,
        description="Run a limited all-class diagnostic pass for trigger traces only",
    )
    diagnostic_trace_max_frames: int = Field(
        default=3,
        description="Maximum frames per camera for all-class diagnostic trace",
    )
    threshold_rescue_enabled: bool = Field(
        default=True,
        description="Allow low-confidence moving detections to be weight-gated rescue candidates",
    )
    threshold_rescue_confidence_cap: float = Field(
        default=0.18,
        description="Maximum confidence assigned to threshold rescue candidates",
    )
    threshold_rescue_require_motion: bool = Field(
        default=True,
        description="Require bbox movement before a threshold-filtered class can be rescued",
    )
    threshold_rescue_max_candidates: int = Field(
        default=20,
        description="Maximum low-confidence rescue diagnostics retained per trigger",
    )
    weight_rescue_no_motion_enabled: bool = Field(
        default=True,
        description="Allow tight weight-gated rescue for low-confidence detections without bbox motion",
    )
    weight_rescue_no_motion_min_raw_votes: int = Field(
        default=8,
        description="Minimum raw low-confidence votes for no-motion weight rescue",
    )
    weight_rescue_no_motion_max_residual_grams: float = Field(
        default=2.0,
        description="Maximum residual for no-motion low-confidence weight rescue",
    )

    # Ensemble settings
    top_weight: float = Field(
        default=0.60,
        description="Top camera weight in ensemble",
    )
    side_weight: float = Field(
        default=0.40,
        description="Side camera weight in ensemble",
    )
    common_class_bonus: float = Field(
        default=0.2,
        description="Bonus for common classes between cameras",
    )
    top_only_weight: float = Field(
        default=0.60,
        description="Top camera only weight (단방향 감지 시)",
    )
    side_only_weight: float = Field(
        default=0.40,
        description="Side camera only weight (단방향 감지 시)",
    )

    # FFmpeg 영상 보정 필터 - Top 카메라 (v4.6)
    ffmpeg_top_gamma: float = Field(
        default=1.0,
        description="Top 카메라 gamma correction (1.0=원본, >1=밝게)",
    )
    ffmpeg_top_contrast: float = Field(
        default=1.0,
        description="Top 카메라 contrast (1.0=원본, >1=높게)",
    )

    # FFmpeg 영상 보정 필터 - Side 카메라 (v4.6)
    ffmpeg_side_gamma: float = Field(
        default=1.0,
        description="Side 카메라 gamma correction (1.0=원본, >1=밝게)",
    )
    ffmpeg_side_contrast: float = Field(
        default=1.0,
        description="Side 카메라 contrast (1.0=원본, >1=높게)",
    )
    @field_validator("top_crop_policy", "side_crop_policy", mode="after")
    def validate_crop_policy(cls, value: str) -> str:
        normalized = value.strip().lower()
        valid = {"left", "center", "right", "offset", "none", "letterbox"}
        if normalized not in valid:
            raise ValueError(f"Invalid crop policy: {value}")
        return normalized

    @field_validator("camera_layout", mode="after")
    def validate_camera_layout(cls, value: str) -> str:
        normalized = value.strip().lower()
        valid = {"legacy_top_side", "dual_top_proxy"}
        if normalized not in valid:
            raise ValueError(f"Invalid camera layout: {value}")
        return normalized

    @field_validator("freezer_roi_vertical_region", mode="after")
    def validate_freezer_roi_vertical_region(cls, value: str) -> str:
        normalized = value.strip().lower()
        valid = {"upper", "lower"}
        if normalized not in valid:
            raise ValueError(f"Invalid freezer ROI vertical region: {value}")
        return normalized

    @field_validator(
        "yolo_internal_conf_threshold",
        "hand_confidence_threshold",
        "top_confidence_threshold",
        "side_confidence_threshold",
        "threshold_rescue_confidence_cap",
        mode="after",
    )
    def validate_confidence_threshold(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return value

    @field_validator(
        "crop_width",
        "min_vote_count",
        "freezer_min_vote_count",
        "diagnostic_trace_max_frames",
        "threshold_rescue_max_candidates",
        "weight_rescue_no_motion_min_raw_votes",
        mode="after",
    )
    def validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Value must be >= 0")
        return value


class WeightModel(BaseModel):
    """Weight verification configuration settings."""

    identity_policy: str = Field(
        default="vision_first",
        description="Product identity policy: vision_first or weight_aware",
    )
    fusion_vision_weight: float = Field(
        default=0.65,
        description="Vision confidence weight used for fused decision confidence",
    )
    fusion_loadcell_weight: float = Field(
        default=0.25,
        description="Loadcell match weight used for fused decision confidence",
    )
    fusion_count_weight: float = Field(
        default=0.10,
        description="Count plausibility weight used for fused decision confidence",
    )
    # Percent-based tolerance remains useful for relaxed matching where the
    # unit count is inferred from the delta and product weight.
    tolerance_percent: float = Field(
        default=0.08,
        description="Weight tolerance percentage (0.08 = 8%)",
    )
    # Strict matching uses a fixed gram window so operators can reason about
    # loadcell error budgets directly in the field.
    tolerance_grams: float = Field(
        default=5.0,
        description="Fixed strict weight tolerance in grams",
    )
    multi_kind_min_confidence: float = Field(
        default=0.18,
        description="Minimum vision confidence for each item in multi-kind combinations",
    )
    rescue_tolerance_grams: float = Field(
        default=5.0,
        description="Vision-rescue-only fixed tolerance in grams",
    )
    same_product_count_tolerance_grams: float = Field(
        default=5.0,
        description="Per-item tolerance for repeated same-product count matching",
    )
    same_product_max_count: int = Field(
        default=8,
        description="Maximum repeated count accepted for one detected product",
    )
    max_items_per_segment: int = Field(
        default=3,
        description="Maximum product count allowed for one detected loadcell segment",
    )
    detected_single_fallback_enabled: bool = Field(
        default=True,
        description="Allow detected single-item nearest-weight fallback after strict/relaxed misses",
    )
    detected_single_fallback_tolerance_grams: float = Field(
        default=8.0,
        description="Maximum residual for detected one-item nearest fallback",
    )
    detected_single_fallback_min_votes: int = Field(
        default=4,
        description="Minimum stage/diagnostic votes for detected single-item fallback",
    )
    freezer_confidence_tie_band: float = Field(
        default=0.08,
        description="Freezer-only confidence band where weight residual can break ties",
    )
    freezer_multi_min_confidence: float = Field(
        default=0.45,
        description="Freezer-only confidence threshold for multi-kind vision decisions",
    )
    freezer_weight_tolerance_grams: float = Field(
        default=15.0,
        description="Freezer-only gram tolerance for reliable loadcell weight matches",
    )
    freezer_vision_multi_without_weight_enabled: bool = Field(
        default=True,
        description=(
            "Legacy freezer multi-kind vision flag. Nonzero freezer deltas still "
            "require combined weight to fit freezer_weight_tolerance_grams."
        ),
    )
    freezer_distinct_mixed_preference_enabled: bool = Field(
        default=True,
        description=(
            "Prefer all-single mixed freezer baskets over same-product repeats "
            "when both explain the same item count within tolerance."
        ),
    )
    freezer_distinct_mixed_max_extra_residual_grams: float = Field(
        default=5.0,
        description=(
            "Maximum extra residual allowed when preferring a distinct mixed "
            "freezer basket over a same-product repeat."
        ),
    )
    freezer_prior_trigger_dedupe_enabled: bool = Field(
        default=True,
        description=(
            "Exclude products already selected by earlier freezer removal "
            "triggers in the same global door session."
        ),
    )
    min_weight_change: float = Field(
        default=5.0,
        description="Minimum weight change in grams",
    )
    max_combination_size: int = Field(
        default=5,
        description="Maximum relaxed combination unit count for weight matching",
    )
    # Strict mode is the weight-first branch. `strict_mode_fallback` controls
    # whether a strict miss degrades to relaxed matching or hard-fails.
    strict_mode: bool = Field(
        default=True,
        description="엄격 무게 검증 모드 (v5.1: 무게로 설명 불가 시 NO_DETECTION)",
    )
    max_combination_items: int = Field(
        default=5,
        description="Maximum total units searched by strict combination matching",
    )
    # v5.2: StrictWeightMatcher 추가 설정
    # Strict matcher search limits bound runtime on noisy deltas and large
    # catalogs without changing the matching policy itself.
    max_combination_kinds: int = Field(
        default=3,
        description="Maximum distinct product kinds searched by strict combination matching",
    )
    max_count_per_item: int = Field(
        default=10,
        description="상품당 최대 개수 (v5.2)",
    )
    max_combinations: int = Field(
        default=100,
        description="최대 조합 수 (성능 제한, v5.2)",
    )
    # The operational default is to recover into the relaxed path so minor
    # weight mismatches do not immediately collapse to NO_DETECTION.
    strict_mode_fallback: bool = Field(
        default=True,
        description="strict 모드 실패 시 기존 로직 폴백 여부 (v5.2)",
    )
    nearest_single_margin_grams: float = Field(
        default=10.0,
        description="Minimum gap to second-nearest product for loadcell-only nearest fallback",
    )

    @field_validator("identity_policy", mode="after")
    def validate_identity_policy(cls, value: str) -> str:
        normalized = value.strip().lower()
        valid = {"vision_first", "weight_aware"}
        if normalized not in valid:
            raise ValueError(f"Invalid identity policy: {value}")
        return normalized


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
    low_weight_vision_fallback: bool = Field(
        default=True,
        description=(
            "Analyze recorded video in vision-only mode when loadcell delta is "
            "below the low-weight skip threshold"
        ),
    )
    return_video_skip_enabled: bool = Field(
        default=True,
        description="Skip YOLO for positive return deltas and process them as loadcell-only returns",
    )
    return_stabilization_wait_seconds: float = Field(
        default=1.0,
        description="Seconds to wait before committing a positive return delta",
    )
    return_stabilization_require_stable_regions: bool = Field(
        default=True,
        description="Require confirmed stable start/end regions before loadcell-only return commit",
    )
    balanced_event_cancel_enabled: bool = Field(
        default=True,
        description="Cancel queued removal triggers when a later return delta balances them out",
    )
    cooperative_cancel_enabled: bool = Field(
        default=True,
        description="Check queued trigger cancellation before starting expensive video work",
    )
    rapid_same_zone_window_seconds: float = Field(
        default=3.0,
        description="Seconds of same-zone loadcell history exposed to trigger traces",
    )


class LoadcellModel(BaseModel):
    """Loadcell delta analysis settings."""

    stable_window_size: int = Field(
        default=5,
        description="Stable window size for trigger weight delta analysis",
    )
    stability_threshold_grams: float = Field(
        default=15.0,
        description="Maximum per-window standard deviation for a stable loadcell region",
    )
    freezer_endpoint_fallback_enabled: bool = Field(
        default=True,
        description="Use conservative first/last loadcell fallback for freezer removals",
    )
    endpoint_fallback_min_samples: int = Field(
        default=10,
        description="Minimum parsed samples required for endpoint loadcell fallback",
    )
    endpoint_fallback_min_span_seconds: float = Field(
        default=2.0,
        description="Minimum sample span required for endpoint loadcell fallback",
    )


class VideoModel(BaseModel):
    """Video readiness and decode configuration."""

    ready_max_wait_seconds: float = Field(
        default=2.0,
        description="Maximum time to poll ffprobe while an AVI is still being finalized",
    )
    ready_poll_interval_seconds: float = Field(
        default=0.2,
        description="Polling interval for AVI readiness checks",
    )

    @field_validator("ready_max_wait_seconds", "ready_poll_interval_seconds", mode="after")
    def validate_non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Value must be >= 0")
        return value


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
    frame_stride: int = Field(
        default=1,
        description=(
            "Async streaming frame stride. Use 1 for all frames or 2 for "
            "latency rollback."
        ),
    )
    early_termination_enabled: bool = Field(
        default=False,
        description="조기 종료 기능 활성화 여부 (미래 확장용)",
    )
    early_termination_vote_threshold: int = Field(
        default=50,
        description="조기 종료를 위한 최소 투표 수 (미래 확장용)",
    )


    zero_frame_retry_enabled: bool = Field(
        default=True,
        description="Retry a sync ffmpeg decode when async extraction yields zero frames",
    )

    @field_validator("frame_queue_size", mode="after")
    def validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Value must be >= 1")
        return value

    @field_validator("frame_stride", mode="after")
    def validate_frame_stride(cls, value: int) -> int:
        normalized = int(value)
        if normalized not in (1, 2):
            raise ValueError("frame_stride must be 1 or 2")
        return normalized


class TraceModel(BaseModel):
    """Frame split trace logging configuration."""

    sample_export_enabled: bool = Field(
        default=False,
        description="샘플 split 프레임 JPEG export 활성화 여부",
    )
    sample_count_per_camera: int = Field(
        default=3,
        description="카메라별 최대 샘플 프레임 수",
    )
    sample_export_dir: str = Field(
        default="logs/frame_samples",
        description="샘플 프레임 export 경로 (services/model 기준 상대경로 허용)",
    )

    @field_validator("sample_count_per_camera", mode="after")
    def validate_sample_count_per_camera(cls, value: int) -> int:
        if value < 0:
            raise ValueError("sample_count_per_camera must be >= 0")
        return value


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
    close_initial_wait_seconds: float = Field(
        default=3.0,
        description="CLOSE debounce when no trigger has arrived yet",
    )
    close_subsequent_wait_seconds: float = Field(
        default=1.0,
        description="CLOSE debounce after the latest trigger has been processed",
    )
    yaml_retention_days: int = Field(
        default=7,
        description="완료된 YAML 세션 파일 보관 기간 (일, v4.2)",
    )


    @field_validator(
        "session_timeout_seconds",
        "weight_tolerance_grams",
        "max_duration_seconds",
        "close_initial_wait_seconds",
        "close_subsequent_wait_seconds",
        mode="after",
    )
    def validate_non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Value must be >= 0")
        return value


class CatalogModel(BaseModel):
    """Runtime product catalog policy settings."""

    source_policy: str = Field(
        default="node_first",
        description="Product catalog source policy: node_first or static_mapping_compat",
    )
    static_validation_enabled: bool = Field(
        default=False,
        description="Enable advisory dataset/yolo_product_mapping.json validation at startup",
    )
    product_name_fallback_enabled: bool = Field(
        default=True,
        description=(
            "Legacy compatibility flag retained for env stability. Runtime "
            "active-product class identity is resolved from product_eng_name "
            "matched against loaded YOLO engine class names; temporary "
            "name/product_name engine-name fallbacks are also supported."
        ),
    )

    @field_validator("source_policy", mode="after")
    def validate_source_policy(cls, value: str) -> str:
        normalized = value.strip().lower()
        valid = {"node_first", "static_mapping_compat"}
        if normalized not in valid:
            raise ValueError(f"Invalid catalog source policy: {value}")
        return normalized


class Settings(BaseSettings):
    """
    Global application settings.

    Environment Variables (with MODEL__ prefix):
        MODEL__API__HOST: API server host (default: 0.0.0.0)
        MODEL__API__PORT: API server port (default: 8002)
        MODEL__API__LOG_LEVEL: Log level (default: info)
        MODEL__MACHINE__CABINET_TYPE: refrigerated or freezer
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
    machine: MachineModel = MachineModel()
    vision: VisionModel = VisionModel()
    weight: WeightModel = WeightModel()
    loadcell: LoadcellModel = LoadcellModel()
    video: VideoModel = VideoModel()
    buffer: BufferModel = BufferModel()
    door_session: DoorSessionModel = DoorSessionModel()
    trigger: TriggerModel = TriggerModel()  # v5.2
    async_streaming: AsyncStreamingModel = AsyncStreamingModel()  # v5.3
    trace: TraceModel = TraceModel()
    catalog: CatalogModel = CatalogModel()

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
