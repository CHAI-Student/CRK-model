import numpy as np


class _FakeResult:
    boxes = None


class _FakeModel:
    names = {0: "hand", 1: "PRODUCT_A"}

    def __init__(self):
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image.copy(), dict(kwargs)))
        return [_FakeResult()]


def test_yolo_wrapper_default_left_crop_matches_480_engine(monkeypatch):
    from model_service.core.config import config
    from model_service.vision.yolo_wrapper import YOLOWrapper

    monkeypatch.setattr(config.vision, "top_crop_policy", "left", raising=False)
    monkeypatch.setattr(config.vision, "side_crop_policy", "left", raising=False)
    monkeypatch.setattr(config.vision, "crop_width", 480, raising=False)

    model = _FakeModel()
    wrapper = YOLOWrapper()
    wrapper.model = model
    wrapper.class_names = model.names
    wrapper._loaded = True

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    wrapper.detect(image, camera_type="side")

    predicted_image, _ = model.calls[0]
    assert predicted_image.shape == (480, 480, 3)
    assert wrapper.last_preprocess == {
        "camera_type": "side",
        "original_width": 640,
        "original_height": 480,
        "processed_width": 480,
        "processed_height": 480,
        "crop_policy": "left",
        "crop_box": {"x1": 0, "y1": 0, "x2": 480, "y2": 480},
    }


def test_yolo_wrapper_uses_configurable_center_crop(monkeypatch):
    from model_service.core.config import config
    from model_service.vision.yolo_wrapper import YOLOWrapper

    monkeypatch.setattr(config.vision, "top_crop_policy", "center", raising=False)
    monkeypatch.setattr(config.vision, "crop_width", 480, raising=False)

    model = _FakeModel()
    wrapper = YOLOWrapper()
    wrapper.model = model
    wrapper.class_names = model.names
    wrapper._loaded = True

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    wrapper.detect(image, camera_type="top")

    predicted_image, _ = model.calls[0]
    assert predicted_image.shape == (480, 480, 3)
    assert wrapper.last_preprocess == {
        "camera_type": "top",
        "original_width": 640,
        "original_height": 480,
        "processed_width": 480,
        "processed_height": 480,
        "crop_policy": "center",
        "crop_box": {"x1": 80, "y1": 0, "x2": 560, "y2": 480},
    }


def test_yolo_wrapper_letterbox_policy_preserves_full_frame(monkeypatch):
    from model_service.core.config import config
    from model_service.vision.yolo_wrapper import YOLOWrapper

    monkeypatch.setattr(config.vision, "side_crop_policy", "letterbox", raising=False)
    monkeypatch.setattr(config.vision, "crop_width", 480, raising=False)

    model = _FakeModel()
    wrapper = YOLOWrapper()
    wrapper.model = model
    wrapper.class_names = model.names
    wrapper._loaded = True

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    wrapper.detect(image, camera_type="side")

    predicted_image, _ = model.calls[0]
    assert predicted_image.shape == (480, 640, 3)
    assert wrapper.last_preprocess["crop_policy"] == "letterbox"
    assert wrapper.last_preprocess["crop_box"] == {
        "x1": 0,
        "y1": 0,
        "x2": 640,
        "y2": 480,
    }
