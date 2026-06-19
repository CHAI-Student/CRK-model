"""Model Service entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys

from model_service.core.runtime_env import bootstrap_runtime_environment
from model_service.core.terminal import install_terminal_restore, restore_terminal_state


def _parse_args(default_host: str, default_port: int) -> argparse.Namespace:
    """Parse CLI arguments using runtime defaults from settings."""
    parser = argparse.ArgumentParser(description="Model Service (v5.4)")
    parser.add_argument(
        "--host",
        type=str,
        default=default_host,
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="Server port (default: 8002)",
    )
    return parser.parse_args()


def main() -> None:
    """Load settings, configure logging, and start the API server."""
    # Re-exec early on Jetson when a fresh shell forgot to restore the CUDA /
    # TensorRT linker paths. This must happen before importing the FastAPI
    # stack because those imports eventually touch torch / TensorRT.
    bootstrap_runtime_environment()

    from model_service.api.manager import serve_api
    from model_service.core.config import Settings
    from model_service.core.logging_config import get_logger, setup_logging

    settings = Settings()
    setup_logging(settings.log_level.upper())
    logger = get_logger(__name__)

    args = _parse_args(settings.host, settings.port)

    if args.host != settings.host:
        settings.api.host = args.host
    if args.port != settings.port:
        settings.api.port = args.port

    logger.info(f"Starting Model Service v5.4 on {args.host}:{args.port}")
    asyncio.run(serve_api(settings))


def run() -> None:
    """Console entry point used by `model-service` and `python -m model_service`."""
    terminal_state = install_terminal_restore()
    try:
        main()
    except KeyboardInterrupt:
        print("Model service stopped by user")
        sys.exit(0)
    except Exception as exc:  # pragma: no cover - exercised in integration runs.
        print(f"Model service failed: {exc}")
        sys.exit(1)
    finally:
        restore_terminal_state(terminal_state)


if __name__ == "__main__":
    run()
