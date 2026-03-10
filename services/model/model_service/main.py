"""
Model Service - FastAPI Entry Point.

AI 상품 판단 서비스 (v5.4 - Project Structure Migration).
Jetson Orin Nano TensorRT 전용.

실행 방법:
    uv run model-service

기능:
    - POST /trigger: Camera에서 AVI 녹화 완료 시 YOLO 추론
    - POST /api/judge/multi-zone: Node.js 폴링용 결과 조회

Note:
    Camera에서 AVI 파일 녹화 후 /trigger 호출.
    Node.js는 /api/judge/multi-zone으로 결과를 폴링합니다.
"""

import argparse
import asyncio
import sys

from model_service.api.manager import serve_api
from model_service.core.config import Settings
from model_service.core.logging_config import get_logger, setup_logging


def main():
    """메인 진입점."""
    # Load settings
    settings = Settings()

    # Setup logging
    setup_logging(settings.log_level.upper())
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(description="Model Service (v5.4)")
    parser.add_argument(
        "--host",
        type=str,
        default=settings.host,
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="Server port (default: 8002)",
    )

    args = parser.parse_args()

    # Update settings from CLI args
    if args.host != settings.host:
        settings.api.host = args.host
    if args.port != settings.port:
        settings.api.port = args.port

    # 서버 시작
    logger.info(f"Starting Model Service v5.4 on {settings.host}:{settings.port}")
    asyncio.run(serve_api(settings))


def run():
    """Run the Model service (entry point for uv run / PM2)."""
    try:
        main()
    except KeyboardInterrupt:
        print("Model service stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Model service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
