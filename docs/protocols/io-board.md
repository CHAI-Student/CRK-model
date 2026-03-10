# IO Board Protocol

## Scope

- IO Board repository: [CRK-IO-BOARD](../../../../CRK-IO-BOARD)
- Model service repository: [current model repo](../..)

This document explains how the IO Board relates to the model service. In the current architecture, the model does not call the IO Board directly.

## Directness

- Model -> IO Board: none
- IO Board -> Model: none
- Actual path:
  - Node -> IO Board
  - Camera -> IO Board SSE
  - Camera -> Model `/trigger`

From the model point of view, loadcell data arrives after Camera has already transformed it.

## Interfaces Used Elsewhere In The System

| Direction | Endpoint | Purpose |
| --- | --- | --- |
| Node -> IO Board | `GET /health` | Deadbolt and loadcell health check |
| Node -> IO Board | `POST /recording/start` | Start loadcell recording |
| Node -> IO Board | `POST /recording/stop` | Stop loadcell recording |
| Node -> IO Board | `GET /recording/data` | Fetch recorded loadcell data |
| Camera -> IO Board | `GET /sse?...` | Subscribe to loadcell and door events |

## Camera-Relevant SSE Contract

- Camera source: [main.py](../../../../CRK-CAMERA/src/main.py)
- IO Board source: [sse.py](../../../../CRK-IO-BOARD/src/api/v1/routers/sse.py)

Camera currently subscribes with this URL.

```text
http://localhost:8000/sse?streams=loadcells&filter_method=exponential&filter_alpha=0.8&threshold=2
```

Important SSE event types:

- `loadcell.update`
- `loadcell.change`
- `loadcell.uncertainty`
- `door.update`
- `error`

Event payload summary:

- `loadcell.update`: `timestamp`, `raw_values[10]`, `filtered_values[10]`, `filter_method`
- `loadcell.change`: `timestamp`, `changed_indices`, `old_values`, `new_values`, `deltas`
- `loadcell.uncertainty`: uncertainty or sensor-error state
- `door.update`: `timestamp`, `door`, `deadbolt`

## Model-Relevant Implication

The model does not parse raw IO Board APIs directly. It only receives Camera-packaged payloads that match `/trigger`.

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
  ]
}
```

If the IO Board protocol changes, check Camera's [loadcell.py](../../../../CRK-CAMERA/src/services/loadcell.py) before changing the model.

## Source Of Truth

- [CRK-IO-BOARD/src/api/v1/routers/machine.py](../../../../CRK-IO-BOARD/src/api/v1/routers/machine.py)
- [CRK-IO-BOARD/src/api/v1/routers/recording.py](../../../../CRK-IO-BOARD/src/api/v1/routers/recording.py)
- [CRK-IO-BOARD/src/api/v1/routers/sse.py](../../../../CRK-IO-BOARD/src/api/v1/routers/sse.py)
- [CRK-IO-BOARD/docs/API.md](../../../../CRK-IO-BOARD/docs/API.md)
- [CRK-CAMERA/src/main.py](../../../../CRK-CAMERA/src/main.py)
- [CRK-CAMERA/src/services/loadcell.py](../../../../CRK-CAMERA/src/services/loadcell.py)
