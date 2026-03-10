# Model Service Dockerfile for Jetson Orin Nano 4GB (v5.4)
#
# Jetson Orin Nano (JetPack 6.2) TensorRT 전용
# uv + pyproject.toml 기반 패키지 설치 (CUDA/TensorRT 시스템 패키지 상속)
#
# Build (from Edge_Environment directory):
#   docker build -t model-service:latest -f services/model/Dockerfile .
#
# Run:
#   docker run --runtime nvidia -p 8002:8002 model-service:latest

# ==============================================================================
# Stage 1: Base image with CUDA and TensorRT
# ==============================================================================
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3 AS base

# Jetson 환경 변수
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 시스템 패키지 설치 + uv + curl (healthcheck 용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# uv를 PATH에 추가
ENV PATH="/root/.local/bin:$PATH"

# Python 패키지 디렉토리
WORKDIR /app

# ==============================================================================
# Stage 2: Dependencies with uv (pyproject.toml 기반)
# ==============================================================================
FROM base AS deps

# pyproject.toml 복사 (캐시 활용 — 소스 변경 시에도 deps 레이어 재사용)
COPY pyproject.toml README.md ./

# dummy 패키지 구조 생성 (pyproject.toml의 패키지 경로 충족)
RUN mkdir -p model_service && echo "" > model_service/__init__.py

# uv로 가상환경 생성 (시스템 패키지 상속 - CUDA/TensorRT 필수!)
RUN uv venv --system-site-packages --python python3.10 .venv

# 가상환경 활성화
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# pyproject.toml 기반 의존성 설치 + Docker용 opencv-headless 추가
RUN uv pip install --no-cache . \
    && uv pip install --no-cache opencv-python-headless>=4.8.0

# ==============================================================================
# Stage 3: Application
# ==============================================================================
FROM deps AS app

# 패키지 소스 코드 복사 (model_service 패키지)
COPY services/model/model_service/ ./model_service/

# pyproject.toml 다시 복사 (패키지 설치에 필요)
COPY pyproject.toml README.md ./

# 패키지 설치 (--no-deps: 의존성은 이미 설치됨)
RUN uv pip install --no-cache --no-deps .

# 데이터 디렉토리 생성
RUN mkdir -p /data/models /data/sessions /app/logs

# 헬스체크는 docker-compose.yml에서 정의 (단일 소스 원칙)

# 포트 노출
EXPOSE 8002

# 환경 변수 (런타임 설정 가능)
ENV MODEL__API__HOST=0.0.0.0
ENV MODEL__API__PORT=8002
ENV MODEL__API__LOG_LEVEL=info
ENV MODEL__VISION__YOLO_MODEL_PATH=/data/models/siyeon_best.engine
ENV MODEL__BUFFER__TTL_SECONDS=300
ENV MODEL__BUFFER__MAX_SESSIONS=100
ENV MODEL__DOOR_SESSION__ENABLED=true
ENV MODEL__DOOR_SESSION__SESSION_TIMEOUT_SECONDS=30.0
ENV MODEL__DOOR_SESSION__WEIGHT_TOLERANCE_GRAMS=5.0
ENV MODEL__DOOR_SESSION__MAX_DURATION_SECONDS=600.0
ENV MODEL__DOOR_SESSION__YAML_DIR=/data/sessions

# 볼륨 마운트 포인트
VOLUME ["/data/models", "/data/sessions", "/app/logs"]

# 실행 (가상환경 PATH에 있으므로 직접 실행)
CMD ["python", "-m", "model_service"]
