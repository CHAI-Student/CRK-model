"""
YOLO Wrapper for Product Detection (TensorRT Only, v4.5).

Jetson Orin Nano (JetPack 6.2) 전용 TensorRT 엔진 래퍼.
.engine 파일만 지원하며, CUDA가 필수입니다.

v4.5 변경사항:
- 100회 추론마다 GPU 캐시 자동 정리
- clear_gpu_cache() 메서드 추가

실제 YOLO 출력 형식:
    det[0] xyxy=[258.72, 47.65, 315.12, 113.97] conf=0.788 cls=0 name=hand
    det[1] xyxy=[257.67, 75.54, 284.33, 110.22] conf=0.492 cls=109 name=BAG_DALGWANG_DONUT_CHOCO_45G

파싱하여 YOLODetection 객체 리스트로 변환.

요구사항:
    - Jetson Orin Nano + JetPack 6.2 (Ubuntu 22.04)
    - CUDA, cuDNN, TensorRT (사전 설치됨)
    - .engine 파일 (Jetson에서 직접 변환 필요)

사용 예시:
    wrapper = YOLOWrapper(model_path="models/0204_morning.engine")
    detections = wrapper.detect(image)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from model_service.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class YOLODetection:
    """
    YOLO 감지 결과.

    실제 YOLO 출력 형식과 1:1 매핑.

    Attributes:
        xyxy: Bounding box [x1, y1, x2, y2] (픽셀)
        conf: Confidence (0.0 ~ 1.0)
        cls: Class ID (0=hand, 1+=products)
        name: Class name (예: "hand", "chickenmayo_rice")
    """
    xyxy: Tuple[float, float, float, float]  # x1, y1, x2, y2
    conf: float
    cls: int
    name: str

    @property
    def x1(self) -> float:
        return self.xyxy[0]

    @property
    def y1(self) -> float:
        return self.xyxy[1]

    @property
    def x2(self) -> float:
        return self.xyxy[2]

    @property
    def y2(self) -> float:
        return self.xyxy[3]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        """Bounding box 중심점."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        """Bounding box 면적."""
        return self.width * self.height

    @property
    def is_hand(self) -> bool:
        """손인지 여부 (cls == 0)."""
        return self.cls == 0

    @property
    def is_product(self) -> bool:
        """상품인지 여부 (cls > 0)."""
        return self.cls > 0

    def distance_to(self, other: "YOLODetection") -> float:
        """다른 Detection과의 중심점 거리 (픽셀)."""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    def iou(self, other: "YOLODetection") -> float:
        """IoU (Intersection over Union) 계산."""
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        return {
            "xyxy": list(self.xyxy),
            "conf": round(self.conf, 4),
            "cls": self.cls,
            "name": self.name,
            "center": list(self.center),
            "area": round(self.area, 2),
            "is_hand": self.is_hand,
        }


class YOLOWrapper:
    """
    YOLO TensorRT 모델 래퍼 (Jetson Orin Nano 전용, v4.5).

    YOLO 추론 결과를 YOLODetection 리스트로 변환.
    TensorRT (.engine) 모델만 지원하며, CUDA가 필수입니다.

    v4.5: 100회 추론마다 GPU 캐시 자동 정리

    Attributes:
        model: YOLO 모델 (ultralytics)
        conf_threshold: 최소 confidence (기본값 0.01, 매우 낮게)
        device: 추론 디바이스 (항상 "cuda")
        is_tensorrt: TensorRT 모델 여부 (항상 True)
    """

    HAND_CLASS_ID = 0  # 손 클래스 ID

    # Jetson Orin Nano 4GB 최적화 상수
    INPUT_SIZE = 480  # 480x480 입력 크기
    CROP_WIDTH = 480  # 640x480 AVI에서 오른쪽 160px 제거
    MAX_DETECTIONS = 20  # 최대 탐지 개수 제한

    # v4.5: GPU 캐시 정리 주기
    CACHE_CLEANUP_INTERVAL = 100  # 100회 추론마다 정리

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        device: str = '0',  # Jetson: GPU 0 (문자열)
    ):
        """
        YOLO TensorRT 래퍼 초기화 (Jetson Orin Nano 전용).

        Args:
            model_path: YOLO TensorRT 엔진 경로 (.engine 파일)
            conf_threshold: 최소 confidence (기본값 0.01)
            device: 추론 디바이스 (Jetson에서는 항상 cuda)
        """
        self.model = None
        self.model_path = model_path or config.yolo_model_path
        self.conf_threshold = (
            float(config.vision.yolo_internal_conf_threshold)
            if conf_threshold is None
            else float(conf_threshold)
        )
        self.device = device
        self.class_names: dict = {}
        self._loaded = False
        self.is_tensorrt = True  # Always TensorRT
        self._cuda_available = False  # load() 시 검증됨
        self._inference_count = 0  # v4.5: 추론 횟수 카운터
        self.last_error: Optional[str] = None
        self.last_preprocess: dict[str, Any] = {}

    def _set_last_error(self, message: str) -> None:
        """Persist the latest load failure reason for startup diagnostics."""
        self.last_error = message

    def load(self) -> bool:
        """
        YOLO TensorRT 엔진 로드 (Jetson Orin Nano 전용).

        .engine (TensorRT) 파일만 지원합니다.
        CUDA가 없으면 서비스 시작이 실패합니다 (의도된 동작).

        Returns:
            성공 여부
        """
        if self._loaded:
            return True

        self.last_error = None

        try:
            from ultralytics import YOLO

            # 1. CUDA 환경 필수 검증
            self._cuda_available = self._verify_cuda()
            if not self._cuda_available:
                if self.last_error is None:
                    self._set_last_error("CUDA is required for TensorRT inference on Jetson.")
                logger.error(self.last_error)
                return False

            # 2. 모델 경로 해석
            resolved_path = self._resolve_model_path()

            # 3. .engine 파일 확인
            if not resolved_path.endswith(".engine"):
                message = (
                    f"Only .engine files are supported. Got: {resolved_path}\n"
                    "Convert your model on Jetson: yolo export model=best.pt format=engine device=0 half=True"
                )
                self._set_last_error(message)
                logger.error(message)
                return False

            # 4. 파일 존재 확인
            if not os.path.exists(resolved_path):
                message = (
                    f"TensorRT engine not found: {resolved_path}\n"
                    "Generate the engine on Jetson: yolo export model=best.pt format=engine device=0 half=True"
                )
                self._set_last_error(message)
                logger.error(message)
                return False

            # 5. 모델 로드
            self.device = '0'  # Jetson: GPU 0 (문자열)
            logger.info(f"Loading TensorRT engine: {resolved_path}")
            self.model = YOLO(resolved_path)

            self.class_names = self.model.names
            self._loaded = True

            logger.info(
                f"YOLO TensorRT loaded: {len(self.class_names)} classes, device={self.device}"
            )

            # GPU 워밍업 - 첫 추론 지연 방지
            self._warmup()

            return True
        except ImportError:
            message = "ultralytics not installed. Run: pip install ultralytics"
            self._set_last_error(message)
            logger.error(message)
            return False
        except Exception as e:
            message = f"Failed to load TensorRT engine: {e}"
            self._set_last_error(message)
            logger.error(message, exc_info=True)
            return False

    def _verify_cuda(self) -> bool:
        """
        CUDA 환경 검증 (Jetson Orin Nano 전용).

        Jetson 환경에서 CUDA, TensorRT가 정상인지 확인합니다.

        Returns:
            CUDA 사용 가능 여부
        """
        try:
            import torch

            if not torch.cuda.is_available():
                cuda_version = getattr(torch.version, "cuda", None)
                if cuda_version is None:
                    message = (
                        "PyTorch was installed without CUDA support. "
                        "The default PyPI CPU wheel is active instead of a Jetson-compatible GPU build."
                    )
                else:
                    message = (
                        "CUDA not available. Jetson Orin Nano requires CUDA for TensorRT.\n"
                        "Ensure JetPack 6.2 is installed and CUDA is configured."
                    )
                self._set_last_error(message)
                logger.error(message)
                return False

            try:
                import tensorrt as trt
            except ImportError:
                message = "TensorRT Python package not found. Engine loading may fail."
                self._set_last_error(message)
                logger.warning(message)
                return False

            # CUDA 디바이스 정보 확인 (torch.cuda.init() 제거 - Jetson에서 문제 발생 가능)
            try:
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
            except Exception as e:
                logger.warning(f"Could not get CUDA device info: {e}")
                device_name = "Unknown GPU"
                cuda_version = "Unknown"

            # Jetson 환경 확인
            is_jetson = "orin" in device_name.lower() or "tegra" in device_name.lower()
            if is_jetson:
                logger.info(f"Jetson detected: {device_name} (CUDA {cuda_version})")
            else:
                logger.info(f"CUDA initialized: {device_name} (CUDA {cuda_version})")

            logger.info(f"TensorRT version: {trt.__version__}")

            return True
        except ImportError:
            message = (
                "PyTorch is not installed. Install a Jetson-compatible PyTorch build before starting the service."
            )
            self._set_last_error(message)
            logger.error(message)
            return False
        except Exception as e:
            message = f"CUDA verification failed: {e}"
            self._set_last_error(message)
            logger.error(message)
            return False

    def _warmup(self) -> None:
        """
        GPU 워밍업 (Jetson Orin Nano).

        더미 추론으로 JIT 컴파일 완료 및 CUDA 컨텍스트 초기화.
        첫 실제 추론 시 지연을 방지합니다.
        """
        try:
            import torch

            # 480x480 warmup image for CUDA context and TensorRT cache setup.
            warmup_image = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=np.uint8)

            logger.info("GPU warmup: running warmup inference...")

            # Warmup inference runs twice: first for JIT, second for cache activation.
            for i in range(2):
                _ = self.model.predict(
                    warmup_image,
                    conf=0.5,
                    verbose=False,
                    imgsz=self.INPUT_SIZE,
                    half=True,
                    max_det=self.MAX_DETECTIONS,
                    device=self.device,  # 문자열 '0'
                )

            # GPU 메모리 정리
            torch.cuda.empty_cache()

            logger.info("GPU warmup complete")

        except Exception as e:
            logger.warning(f"GPU warmup failed (non-critical): {e}")

    def _resolve_model_path(self) -> str:
        """
        모델 경로 해석 (TensorRT 전용, fallback 없음).

        Returns:
            해석된 모델 경로
        """
        original_path = self.model_path

        # 서비스 디렉토리 기준 (services/model/src/vision -> services/model)
        service_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # 프로젝트 루트 (services/model -> Edge_Environment)
        project_root = os.path.dirname(os.path.dirname(service_dir))

        # 1. 절대 경로이면 그대로 반환
        if os.path.isabs(original_path):
            logger.debug(f"Using absolute path: {original_path}")
            return original_path

        # 2. 상대 경로 → 프로젝트 루트 기준으로 변환
        full_path = os.path.join(project_root, original_path)
        logger.debug(f"Resolved relative path: {original_path} -> {full_path}")
        return full_path

    def _crop_policy_for_camera(self, camera_type: Optional[str]) -> tuple[str, Optional[int]]:
        camera = (camera_type or "top").lower()
        if camera == "side":
            return config.vision.side_crop_policy, config.vision.side_crop_x_offset
        return config.vision.top_crop_policy, config.vision.top_crop_x_offset

    @staticmethod
    def _clamp_offset(offset: int, width: int, crop_width: int) -> int:
        return max(0, min(offset, max(0, width - crop_width)))

    def _preprocess_image(
        self,
        image: np.ndarray,
        camera_type: Optional[str],
    ) -> np.ndarray:
        original_height = int(image.shape[0])
        original_width = int(image.shape[1])
        crop_width = int(config.vision.crop_width or self.CROP_WIDTH)
        policy, configured_offset = self._crop_policy_for_camera(camera_type)
        policy = policy.lower()
        processed = image
        x1 = 0
        x2 = original_width

        if original_width > crop_width and policy not in {"none", "letterbox"}:
            if policy == "center":
                x1 = (original_width - crop_width) // 2
            elif policy == "right":
                x1 = original_width - crop_width
            elif policy == "offset":
                x1 = self._clamp_offset(configured_offset or 0, original_width, crop_width)
            else:
                policy = "left"
                x1 = 0
            x2 = x1 + crop_width
            processed = image[:, x1:x2]

        self.last_preprocess = {
            "camera_type": camera_type or "unknown",
            "original_width": original_width,
            "original_height": original_height,
            "processed_width": int(processed.shape[1]),
            "processed_height": int(processed.shape[0]),
            "crop_policy": policy,
            "crop_box": {
                "x1": int(x1),
                "y1": 0,
                "x2": int(x2),
                "y2": original_height,
            },
        }

        return processed

    def detect(
        self,
        image: np.ndarray,
        allowed_class_ids: Optional[List[int]] = None,
        camera_type: Optional[str] = None,
    ) -> List[YOLODetection]:
        """
        이미지에서 객체 감지.

        Jetson Orin Nano 4GB 최적화:
        - 입력: 640x480 → 480x480 (오른쪽 160px 크롭)
        - FP16 추론 (half=True)
        - 최대 탐지 개수 제한 (max_det=20)

        Args:
            image: numpy array (BGR), 640x480 또는 480x480
            allowed_class_ids: 허용된 클래스 ID 리스트 (v4.4)
                               None이면 모든 클래스 탐지
                               빈 리스트면 탐지 안함 (빈 결과 반환)
                               리스트가 있으면 해당 클래스만 탐지

        Returns:
            YOLODetection 리스트
        """
        if not self._loaded:
            if not self.load():
                return []

        if self.model is None:
            logger.error("YOLO model not loaded")
            return []

        # v4.4: 빈 allowed_class_ids면 탐지 안함
        if allowed_class_ids is not None and len(allowed_class_ids) == 0:
            logger.debug("[YOLO] allowed_class_ids is empty, skipping detection")
            return []

        try:
            image = self._preprocess_image(image, camera_type)

            # v4.4: classes 파라미터로 사전 필터링
            predict_kwargs = {
                "conf": self.conf_threshold,
                "verbose": False,
                "imgsz": self.INPUT_SIZE,  # 480x480 입력
                "half": True,  # FP16 추론
                "max_det": self.MAX_DETECTIONS,  # 최대 20개 탐지
                "device": self.device,  # 문자열 '0' (Jetson GPU)
            }

            if allowed_class_ids is not None:
                predict_kwargs["classes"] = allowed_class_ids
                logger.debug(
                    f"[YOLO] Filtered detect: {len(allowed_class_ids)} classes allowed"
                )

            results = self.model.predict(image, **predict_kwargs)
            detections = self.parse_results(results[0], self.class_names)

            # v4.5: 추론 횟수 증가 및 주기적 캐시 정리
            self._inference_count += 1
            if self._inference_count >= self.CACHE_CLEANUP_INTERVAL:
                self._periodic_cache_cleanup()

            # 탐지 결과 로깅 (상위 3개만)
            if detections:
                logger.debug(f"[YOLO] 탐지: {len(detections)}개")
                for det in detections[:3]:
                    logger.debug(
                        f"  - {det.name}: conf={det.conf:.3f}, is_hand={det.is_hand}"
                    )

            return detections
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def _periodic_cache_cleanup(self) -> None:
        """
        주기적 GPU 캐시 정리 (v4.5).

        CACHE_CLEANUP_INTERVAL마다 호출됩니다.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(
                    f"[YOLO] GPU cache cleaned after {self._inference_count} inferences"
                )
        except Exception as e:
            logger.warning(f"[YOLO] GPU cache cleanup failed: {e}")
        finally:
            self._inference_count = 0

    def clear_gpu_cache(self) -> bool:
        """
        GPU 캐시 수동 정리 (v4.5).

        외부에서 명시적으로 GPU 메모리를 해제할 때 사용.

        Returns:
            성공 여부
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("[YOLO] GPU cache manually cleared")
                return True
            return False
        except Exception as e:
            logger.error(f"[YOLO] GPU cache clear failed: {e}")
            return False

    @staticmethod
    def parse_results(
        result: Any,
        class_names: Optional[dict] = None,
    ) -> List[YOLODetection]:
        """
        YOLO Results 객체 파싱.

        Args:
            result: YOLO Results 객체 (results[0])
            class_names: {cls_id: name} 매핑

        Returns:
            YOLODetection 리스트
        """
        detections = []

        if not hasattr(result, 'boxes') or result.boxes is None:
            return detections

        boxes = result.boxes
        names = class_names or getattr(result, 'names', {})

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist() if hasattr(boxes.xyxy[i], 'tolist') else list(boxes.xyxy[i])
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            name = names.get(cls_id, f"class_{cls_id}")

            det = YOLODetection(
                xyxy=tuple(xyxy),
                conf=conf,
                cls=cls_id,
                name=name,
            )
            detections.append(det)

        return detections

    @staticmethod
    def parse_detection_list(
        detection_data: List[dict],
    ) -> List[YOLODetection]:
        """
        딕셔너리 리스트에서 YOLODetection 파싱.

        테스트용 또는 외부 API에서 받은 데이터 파싱.

        Args:
            detection_data: [{"xyxy": [...], "conf": ..., "cls": ..., "name": ...}, ...]

        Returns:
            YOLODetection 리스트
        """
        detections = []

        for d in detection_data:
            det = YOLODetection(
                xyxy=tuple(d["xyxy"]),
                conf=float(d["conf"]),
                cls=int(d["cls"]),
                name=str(d["name"]),
            )
            detections.append(det)

        return detections

    @property
    def is_loaded(self) -> bool:
        """모델 로드 상태."""
        return self._loaded
