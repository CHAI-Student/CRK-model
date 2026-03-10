# TensorRT Convert Dockerfile for Jetson Orin Nano 4GB
#
# .pt → .engine (TensorRT FP16) 변환 전용 컨테이너
# NumPy >= 2.0 환경 (학습 환경에서 직렬화된 .pt 파일 호환)
#
# Build (from Edge_Environment directory):
#   docker compose --profile convert build convert
#
# Run:
#   docker compose --profile convert run --rm convert
#
# Custom model:
#   PT_FILE=new_model.pt docker compose --profile convert run --rm convert

# ==============================================================================
# Base: L4T PyTorch (JetPack 6.2, CUDA 12.2, TensorRT 포함)
# ==============================================================================
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# uv 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# ==============================================================================
# Venv 생성 + 패키지 설치
# ==============================================================================

# 시스템 패키지 상속 (CUDA, TensorRT, PyTorch)
RUN uv venv --system-site-packages --python python3.10 .venv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# NumPy >= 2.0 (학습 환경 호환 — .pt 파일이 NumPy 2.x로 직렬화됨)
# ultralytics: YOLO export 기능
# albumentations, onnxslim: ultralytics export 의존성
RUN uv pip install --no-cache \
    "numpy>=2.0.0" \
    "ultralytics>=8.1.0" \
    "albumentations>=1.3.0" \
    "onnxslim>=0.1.0"

# TensorRT 시스템 패키지를 venv으로 심볼릭 링크
RUN ln -sf /usr/lib/python3.10/dist-packages/tensorrt \
        $VIRTUAL_ENV/lib/python3.10/site-packages/tensorrt \
    && ln -sf /usr/lib/python3.10/dist-packages/tensorrt_libs \
        $VIRTUAL_ENV/lib/python3.10/site-packages/tensorrt_libs \
    && ln -sf /usr/lib/python3.10/dist-packages/tensorrt_bindings \
        $VIRTUAL_ENV/lib/python3.10/site-packages/tensorrt_bindings

# ==============================================================================
# 변환 스크립트 복사
# ==============================================================================
COPY scripts/convert_engine.sh /app/convert_engine.sh
RUN chmod +x /app/convert_engine.sh

# 모델 디렉토리 마운트 포인트
VOLUME ["/models"]

ENTRYPOINT ["/app/convert_engine.sh"]
