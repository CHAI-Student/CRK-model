# API Reference

Last reviewed: 2026-03-09

Base URL: `http://<host>:8002`

## Health

### `GET /api/health`

Example response:

```json
{
  "model": "HEALTHY",
  "status": "ok",
  "yolo_loaded": true,
  "session_store_ready": true,
  "timestamp": 1741500000.0
}
```

### `GET /api/health/detailed`

Example response:

```json
{
  "service": "model",
  "version": "5.4.0",
  "status": "ok",
  "dependencies": {
    "initialized": true,
    "session_store": true,
    "yolo": true,
    "yolo_loaded": true,
    "engine": true,
    "video_processor": true,
    "door_session_store": true,
    "active_product_store": true,
    "trigger_service": true
  },
  "config": {
    "host": "0.0.0.0",
    "port": 8002,
    "yolo_model_path": "models/siyeon_best.engine"
  },
  "timestamp": 1741500000.0
}
```

`config` now reflects the runtime app settings, not stale import-time defaults.

## Trigger

### `POST /trigger`

Request:

```json
{
  "zone": 1,
  "loadcells": [
    {
      "timestamp": "2026-03-09T12:00:00.000Z",
      "raw_value": ["+5000", "+5000"],
      "filtered_value": ["+5000", "+5000"],
      "filter_method": "none"
    }
  ],
  "videos": {
    "top": "/data/videos/top.avi",
    "side": "/data/videos/side.avi"
  }
}
```

Success response:

```json
{
  "success": true,
  "session_id": "zone_1_260309_120000_123456",
  "door_session_id": "door_zone_1_260309_120000_654321",
  "message": "Trigger accepted",
  "status": "queued",
  "waiting_for": null
}
```

Error responses use FastAPI `detail` payloads.

### `GET /trigger/stats`

Returns in-memory trigger worker statistics from the optional `TriggerService`.

## Multi-Zone Judge

### `POST /api/judge/multi-zone`

This endpoint supports both:

- object payloads with `session_id`, `zone`, and `products`
- array payloads where Node.js sends `products` directly

Processing response:

```json
{
  "status": "processing",
  "message": "YOLO processing in progress"
}
```

Complete response:

```json
{
  "status": "complete",
  "zone": 1,
  "products": [
    {
      "productIdx": "26",
      "productId": 26,
      "name": "Example Product",
      "count": 1,
      "price": 3500,
      "confidence": 0.92
    }
  ],
  "productCount": 1,
  "totalPrice": 3500,
  "confidence": 0.92,
  "weightInfo": {
    "delta": -365.0,
    "isRemoval": true
  },
  "stats": {
    "topFrames": 150,
    "sideFrames": 150,
    "processingTimeMs": 2500.3
  }
}
```

### `GET /api/judge/session/{session_id}`

Returns whether a session exists and, if so, its stored session payload.

### `GET /api/judge/sessions/stats`

Returns aggregate SessionStore metrics.

### `GET /api/judge/door-sessions/stats`

Returns aggregate DoorSessionStore metrics when the feature is enabled.

### `GET /api/judge/door-session/{zone}`

Returns the current in-progress door session for a zone when present.

### `POST /api/judge/door-session/{zone}/finalize`

Forces finalization of the active door session for a zone.

## Common Error Codes

| Code | Meaning |
|------|---------|
| `VIDEO_FILE_NOT_FOUND` | a requested AVI path does not exist |
| `VALIDATION_ERROR` | request payload is invalid |
| `VIDEO_CORRUPTED` | FFmpeg or video parsing failed |
| `FFMPEG_ERROR` | FFmpeg processing failed |
| `YOLO_GPU_ERROR` | GPU inference failed |
| `YOLO_MODEL_NOT_LOADED` | TensorRT model was not loaded |
