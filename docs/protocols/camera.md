# Camera Protocol

## Scope

- Camera service repository: [CRK-CAMERA](../../../../CRK-CAMERA)
- Model service repository: [current model repo](../..)

This document describes the contracts between the Camera service and the model service, plus the way Node controls the Camera service.

## Direct Interfaces

| Direction | Endpoint | Purpose |
| --- | --- | --- |
| Node -> Camera | `GET /health` | Camera connectivity check |
| Node -> Camera | `POST /recording/start` | Start recording and initialize output paths |
| Node -> Camera | `POST /recording/stop` | Stop recording |
| Camera -> Model | `POST /trigger` | Submit zone inference request |

## Node -> Camera

### `GET /health`

- Source: [management.py](../../../../CRK-CAMERA/src/api/v1/routers/management.py)
- Success response:

```json
{ "status": "HEALTHY" }
```

- Unhealthy response:

```json
{
  "status": "UNHEALTHY",
  "missing_cameras": [
    { "serial": "1234567890", "index": 1 }
  ]
}
```

### `POST /recording/start`

- Source: [recording.py](../../../../CRK-CAMERA/src/api/v1/routers/recording.py)
- Request:

```json
{
  "save_path": "/some/base/path"
}
```

- Side effects:
  - archival output: `save_path/archival/cam_{key}`
  - trigger inference output: `save_path/inference/zone_{key}`
  - loadcell SSE subscription starts

### `POST /recording/stop`

- Source: [recording.py](../../../../CRK-CAMERA/src/api/v1/routers/recording.py)
- No request body
- Stops recording and loadcell collection

## Camera -> Model

### `POST http://localhost:8002/trigger`

- Camera source: [loadcell.py](../../../../CRK-CAMERA/src/services/loadcell.py)
- Model source: [trigger.py](../../services/model/model_service/api/routes/trigger.py)

Request body:

```json
{
  "zone": 1,
  "loadcells": [
    {
      "timestamp": "2026-03-10T12:00:00.000Z",
      "raw_value": ["+5000", "+4998"],
      "filtered_value": ["+4999", "+4999"],
      "filter_method": "exponential"
    }
  ],
  "videos": {
    "top": "/shared/path/inference/zone_1/top.avi",
    "side": "/shared/path/inference/zone_1/side.avi"
  }
}
```

Success response shape:

```json
{
  "success": true,
  "session_id": "zone_1_...",
  "door_session_id": "door_zone_1_...",
  "message": "Inference complete",
  "status": "queued",
  "waiting_for": null
}
```

Observed status values:

- `queued`
- `complete`
- `skipped`
- `duplicate`

Error cases:

- `400`: video path validation failure, queue full, trigger validation failure
- `500`: inference or video-processing internal error
- `503`: YOLO model not loaded

## Operational Notes

- Camera does not upload video binaries. It sends file paths only.
- Camera and model must share the same host or filesystem namespace.
- Camera-side [loadcell.py](../../../../CRK-CAMERA/src/services/loadcell.py) logs `/trigger` failures but does not retry them.
- Camera runtime currently assumes Python 3.14 based on [.python-version](../../../../CRK-CAMERA/.python-version).

## Source Of Truth

- [CRK-CAMERA/src/api/v1/routers/management.py](../../../../CRK-CAMERA/src/api/v1/routers/management.py)
- [CRK-CAMERA/src/api/v1/routers/recording.py](../../../../CRK-CAMERA/src/api/v1/routers/recording.py)
- [CRK-CAMERA/src/main.py](../../../../CRK-CAMERA/src/main.py)
- [CRK-CAMERA/src/services/loadcell.py](../../../../CRK-CAMERA/src/services/loadcell.py)
- [model_service/api/routes/trigger.py](../../services/model/model_service/api/routes/trigger.py)
