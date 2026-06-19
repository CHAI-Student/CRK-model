# System Map

## Current Thesis

CRK-model is the Jetson-side inference service. It is not the system
orchestrator, camera controller, IO Board client, or payment client. Its direct
inputs are Camera `/trigger` payloads and Node `/api/judge/multi-zone` calls.
Within this Python repo, README now describes the service as the
legacy/reference TensorRT path; fresh clone-based operation is expected to use
`CRK-model-go` unless a task targets this Python service directly.

## Service Boundaries

| Component | Repo | Model relationship |
| --- | --- | --- |
| Node / Edge runtime | `Edge_Environment` | Starts door sessions, polls model, sends `OPEN`/`CLOSE`, provides `active_products`, combines inference with payment. |
| Camera service | `CRK-CAMERA` | Records top/side AVI, subscribes to loadcell SSE, sends file paths and loadcell samples to model `/trigger`. |
| IO Board | `CRK-IO-BOARD` | Supplies loadcell and door/deadbolt events to Camera/Node. Model does not call it directly. |
| Payment | `CRK-PAYMENT` | Approves/cancels payment through Node. Model results are inserted by Node into downstream payment/PNT payloads. |
| Model service | `CRK-model` | Runs trigger queue, video/YOLO processing, decision engine, and door-session aggregation. |

## Primary Data Paths

```text
Node -> Camera /recording/start
Camera -> IO Board SSE for loadcell/door events
Camera -> Model /trigger with AVI paths and loadcell samples
Node -> Model /api/judge/multi-zone with OPEN, polling, CLOSE, active_products
Node -> Payment/PNT using final model basket
```

## Operational Rules

- Door unlock should happen only after Camera has armed capture and returned
  `ready=true`.
- Model receives video paths, not video binaries. Camera and Model must share a
  filesystem namespace.
- Model receives Camera-packaged loadcell history, not raw IO Board SSE.
- Payment incidents often require Node/PNT inspection first because Payment and
  Model do not talk directly.
- Node health MQTT is not proof that `/trigger` or `/api/judge/multi-zone`
  works.
- Known sibling-repo risks are outside this CRK-model-only change: hardcoded
  Edge/Camera/IO URLs, missing Camera retry for model
  `waiting_for=stable_loadcell`, and stale IO Board docs.

## Evidence

- [Repo overview](../source-code/repo-overview.md)
- [File inventory](../source-code/file-inventory.md)
- [Camera protocol](../source-docs/protocols-camera.md)
- [IO Board protocol](../source-docs/protocols-io-board.md)
- [Node protocol](../source-docs/protocols-node.md)
- [Payment protocol](../source-docs/protocols-payment.md)
- [Trigger capture debugging](../source-docs/trigger-capture-debugging.md)
