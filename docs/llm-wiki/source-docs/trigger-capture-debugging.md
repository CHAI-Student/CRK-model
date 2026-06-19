# Source Summary: TRIGGER_CAPTURE_DEBUGGING.md

Source: [docs/TRIGGER_CAPTURE_DEBUGGING.md](../../TRIGGER_CAPTURE_DEBUGGING.md)
Status: current with caveats

## Use This When

Use this for field debugging of trigger capture timing across Edge, Camera, and
Model.

## Key Facts

- Expected open sequence:
  Edge creates inference folder, calls Camera `/recording/start`, Camera arms
  archival/trigger/loadcell capture, Camera returns `ready=true`, Edge starts
  model polling, then Edge unlocks the door.
- Door unlock should not happen if Camera does not return `ready=true`.
- Camera keeps trigger capture alive until the minimum capture window and stable
  hold condition are satisfied, or until `max_capture_seconds`.
- `/trigger` timing metadata can include capture, loadcell, trigger start/end,
  and `trigger_end_reason`.
- Model-side logs should reveal whether ffprobe saw frames, async decode
  returned zero frames, sync/raw retry recovered frames, loadcell history was
  truncated, and why the engine fell back.
- Key extractor diagnostics include `expected_frames`, `decoded_frames`,
  `bytes_read`, `partial_reads`, `stderr_tail`, and `final_branch`.
- Common failures include decode-path failure despite positive ffprobe count,
  max capture timeout, and loadcell payloads ending before the stable tail.

## Related Code

- `services/model/model_service/video/frame_extractor.py`
- `services/model/model_service/service/trigger_service.py`
- `services/model/model_service/api/routes/trigger.py`

## Caveats

- This is a debugging contract, not an API reference. Pair it with protocol
  summaries for payload shape.

## Related Wiki Pages

- [Runtime flow](../synthesis/runtime-flow.md)
- [Protocol contracts](../synthesis/protocol-contracts.md)
- [Latency and frame stride](../synthesis/latency-and-frame-stride.md)
