# Source Summary: IMPLEMENTATION_ROADMAP.md

Source: [docs/IMPLEMENTATION_ROADMAP.md](../../IMPLEMENTATION_ROADMAP.md)
Status: historical

## Use This When

Use this to understand the scenario-driven implementation history and test
coverage expectations.

## Key Facts

- The roadmap organizes work into scenario phases:
  data quality foundation, simultaneous returns, camera interference, loadcell
  interference, complex scenario verification, and two-handed extraction.
- Phase 0 covered loadcell multi-channel averaging and multi-item return count
  deduction.
- Phase 1 covered simultaneous multi-item return processing with return
  combination search.
- Phase 2 covered camera interference and top-only/side-only voting weights.
- Phase 3 covered loadcell interference and peak filtering.
- Phase 4 covered cross-zone returns and product-switch scenario verification.
- Phase 5 covered two-handed extraction and multi-zone aggregation.
- The checklist marks all phases complete, but the document still recommends
  sequential execution as a next step because it is historical.

## Related Code

- `services/model/model_service/api/routes/trigger.py`
- `services/model/model_service/service/trigger_service.py`
- `services/model/model_service/session/product_aggregator.py`
- `services/model/model_service/weight/strict_weight_matcher.py`
- `services/model/model_service/video/voting_ensemble.py`
- `services/model/tests/test_scenarios_phase4.py`
- `services/model/tests/test_scenarios_phase5.py`

## Caveats

- Treat completion status as historical context. For current behavior, prefer
  code and the agent guides.
- Phase 0's "multi-channel averaging" language is historical; current
  README/tests describe summed zone channel totals.

## Related Wiki Pages

- [Historical risk and fixes](../synthesis/historical-risk-and-fixes.md)
- [Product detection pipeline](../synthesis/product-detection-pipeline.md)
