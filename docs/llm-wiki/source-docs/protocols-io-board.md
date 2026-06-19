# Source Summary: protocols/io-board.md

Source: [docs/protocols/io-board.md](../../protocols/io-board.md)
Status: current

## Use This When

Use this to understand how IO Board data reaches the model indirectly.

## Key Facts

- The model does not call IO Board directly.
- Actual path is Node to IO Board, Camera to IO Board SSE, then Camera to Model
  `/trigger`.
- IO Board interfaces used elsewhere include health, recording start/stop,
  recording data, and SSE streams.
- Camera subscribes to an SSE URL with `streams=loadcells`,
  `filter_method=exponential`, `filter_alpha=0.8`, and `threshold=2`.
- Important SSE event types include `loadcell.update`, `loadcell.change`,
  `loadcell.uncertainty`, `door.update`, and `error`.
- The model only receives Camera-packaged loadcell payloads matching `/trigger`.
- If IO Board protocol changes, check Camera `loadcell.py` before changing the
  model.

## Related Code

- `CRK-IO-BOARD/src/api/v1/routers/sse.py`
- `CRK-IO-BOARD/src/api/v1/routers/recording.py`
- `CRK-CAMERA/src/main.py`
- `CRK-CAMERA/src/services/loadcell.py`
- `services/model/model_service/api/routes/trigger.py`

## Caveats

- This document is a boundary guide. Model-side payload validation still lives
  in the model repo.

## Related Wiki Pages

- [Protocol contracts](../synthesis/protocol-contracts.md)
- [System map](../synthesis/system-map.md)
