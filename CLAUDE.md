# Edge Environment - Model Service Guide

## Purpose
AI smart vending machine model service. Jetson Orin Nano 4GB, TensorRT-only inference.

## Project Structure
- `services/model/model_service/` - Source (FastAPI, Python 3.10)
- `services/model/model_service/api/routes/` - API route handlers
- `services/model/model_service/session/` - DoorSessionStore, SessionStore
- `services/model/model_service/video/` - AVI processing, async streaming
- `services/model/model_service/vision/` - YOLO TensorRT wrapper
- `services/model/model_service/engine/` - Decision engine
- `services/model/tests/` - Tests (225+)
- `data/sessions/` - Door session YAML persistence
- `models/` - TensorRT .engine files
- `scripts/` - Jetson setup and utility scripts

## Always Apply
- Run: `model-service` (preferred) or `uv run --no-sync model-service`
- Test: `pytest services/model/tests -q` (preferred) or `uv run --no-sync pytest services/model/tests -q`
- NumPy must be < 2.0. FP16, 480x480 input.
- Entry point: `services/model/model_service/main.py`
- `.env` is auto-loaded by `model_service.core.config.Settings`.
- Default engine path: `models/siyeon_best.engine`

## Guides (Read only when needed)
IMPORTANT: Check relevant guides before starting work. Read only what you need.
- `docs/agent-guides/architecture.md` - Data flow, APIs, service ports
- `docs/agent-guides/conventions.md` - Code patterns, env vars, project rules
- `docs/agent-guides/build-test.md` - pytest patterns, fixtures, CI
- `docs/agent-guides/jetson-setup.md` - JetPack, CUDA, NumPy, uv setup
- `docs/agent-guides/agent-orchestration.md` - Agent execution order and roles
