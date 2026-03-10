"""
Model Service - AI Smart Vending Machine Product Judgment.

v5.4 - Project Structure Migration (src/ -> model_service/)

Features:
- POST /trigger: AVI 트리거 (Camera → Model)
- POST /api/judge/multi-zone: 상품 판단 폴링 (Node.js → Model)
- YOLOv8 TensorRT 추론 (480x480, FP16, max_det=20)
- VotingEnsemble 기반 다중 프레임 집계
- Weight-based count validation
- Async streaming video processing (v5.3)
- Entry point: uv run model-service
"""

__version__ = "5.4.0"
