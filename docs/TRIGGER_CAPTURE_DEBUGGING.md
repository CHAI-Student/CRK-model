# Trigger Capture Debugging

This note documents the field-debugging contract across `Edge_Environment`,
`CRK-CAMERA`, and `CRK-model`.

## Expected Open Sequence

1. `Edge_Environment` creates the inference folder.
2. `Edge_Environment` calls `CRK-CAMERA /recording/start`.
3. `CRK-CAMERA` arms archival recording, trigger recording, and the loadcell
   SSE stream.
4. `CRK-CAMERA` returns `ready=true` plus the timing configuration.
5. `Edge_Environment` starts model polling.
6. `Edge_Environment` unlocks the door.

If step 4 does not return `ready=true`, the door must remain locked.

## Camera Trigger Contract

`CRK-CAMERA` now keeps the capture alive until one of these conditions is met:

- the minimum capture window is satisfied and the loadcell has been stable for
  `stable_hold_seconds`, or
- the capture hits `max_capture_seconds`.

The submitted `/trigger` payload may include:

- `capture_started_at`
- `capture_ended_at`
- `loadcell_started_at`
- `loadcell_ended_at`
- `trigger_started_at`
- `trigger_end_reason`

## Model-Side Debug Signals

`CRK-model` should log enough context to answer these questions without opening
code:

- Did `ffprobe` detect frames?
- Did async decode return `0` frames?
- Did sync/raw retry recover frames?
- Was the loadcell history truncated before the stable tail arrived?
- Did the engine fall back because vision had no candidates or because strict
  matching failed?

The key extractor fields are:

- `expected_frames`
- `decoded_frames`
- `bytes_read`
- `partial_reads`
- `stderr_tail`
- `final_branch`

The key loadcell analysis fields are:

- sample count
- sample time span
- stable start/end index
- baseline window
- final window
- computed delta

## Common Failure Patterns

- `ffprobe > 0` and `decoded_frames = 0`
  This points to decode-path failure, not an empty AVI. The extractor should
  retry via the sync/raw path before loadcell-only fallback.
- `trigger_end_reason=max_capture_timeout`
  The customer interaction lasted longer than the stable hold window. Review
  whether `max_capture_seconds` is too small.
- Expected delta is roughly half the true product weight
  This usually means the payload ended before the post-event stable tail was
  captured. Check `loadcell_started_at`, `loadcell_ended_at`, and the stable
  window logs.
