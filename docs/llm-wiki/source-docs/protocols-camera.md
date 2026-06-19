# Source Summary: protocols/camera.md

Source: [docs/protocols/camera.md](../../protocols/camera.md)
Status: current

## Use This When

Use this for Camera-to-Model and Node-to-Camera contracts.

## Key Facts

- Direct interfaces:
  Node `GET /health`, Node `POST /recording/start`, Node
  `POST /recording/stop`, and Camera `POST /trigger` to the model.
- `/recording/start` creates archival and trigger-inference output paths and
  starts loadcell SSE subscription.
- Camera sends file paths, not video binaries, to the model.
- Camera and model must share the same host or filesystem namespace for AVI
  paths.
- `/trigger` payload includes `zone`, `loadcells`, and `videos.top/side`.
- Observed trigger statuses include `queued`, `complete`, `skipped`, and
  `duplicate`.
- Trigger frame traces are persisted under
  `services/model/logs/frame_split_YYYYMMDD.jsonl`.
- Optional sample frame export is controlled by `MODEL__TRACE__*` env vars.

## Related Code

- `CRK-CAMERA/src/api/v1/routers/management.py`
- `CRK-CAMERA/src/api/v1/routers/recording.py`
- `CRK-CAMERA/src/services/loadcell.py`
- `services/model/model_service/api/routes/trigger.py`

## Caveats

- This page references sibling repositories through relative paths that only
  resolve in the expected CRK workspace layout.

## Related Wiki Pages

- [Protocol contracts](../synthesis/protocol-contracts.md)
- [System map](../synthesis/system-map.md)
