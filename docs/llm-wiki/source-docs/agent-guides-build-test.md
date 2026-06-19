# Source Summary: agent-guides/build-test.md

Source: [docs/agent-guides/build-test.md](../../agent-guides/build-test.md)
Status: current

## Use This When

Use this to choose local verification commands and focused regression suites.

## Key Facts

- Preferred full local gate after venv activation:
  `pytest services/model/tests -q`.
- Acceptable uv fallback:
  `uv run --no-sync pytest services/model/tests -q`.
- Unit tests should not require YOLO, TensorRT, or GPU startup.
- `test_runtime_env.py` covers Jetson startup bootstrap and should be rerun
  when `main.py`, `runtime_env.py`, or Jetson startup scripts change.
- Trigger inference changes should rerun decision engine and return-recovery
  suites because fallback and session repair interact.
- AVI decode or loadcell delta changes should rerun
  `test_trigger_pipeline_regressions.py`.
- Cross-repo timing/capture regressions exist in CRK-CAMERA and
  Edge_Environment, but they are outside this repo's default local gate.

## Related Code

- `services/model/tests/`
- `services/model/tests/test_runtime_env.py`
- `services/model/tests/test_trigger_pipeline_regressions.py`
- `services/model/tests/test_decision_engine.py`
- `services/model/tests/test_product_aggregator.py`
- `services/model/tests/test_cross_zone_return.py`

## Caveats

- Passing local tests is not Jetson runtime proof. It is the safe local
  regression signal before Jetson verification.

## Related Wiki Pages

- [Jetson and testing](../synthesis/jetson-and-testing.md)
- [Runtime flow](../synthesis/runtime-flow.md)
