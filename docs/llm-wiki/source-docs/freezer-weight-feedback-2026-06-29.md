# Freezer Weight Feedback 2026-06-29

## Source

- Type: operator feedback in Codex thread
- Date: 2026-06-29
- Scope: CRK-model freezer loadcell and candidate-selection behavior

## Key Claims

- The freezer loadcell has started capturing weight reliably again.
- Freezer weight error is larger than refrigerator error but is usually within
  about `5g` and can reach roughly `10g-15g`.
- A reported failure removed only `178g`, but the system selected the first,
  second, and third candidates together instead of one nearby `170g` product
  that was already present in the candidate list.

## Implementation Implications

- Freezer can use measured loadcell delta as a hard gate for multi-kind output
  with a freezer-specific default tolerance of `15g`.
- Ordinary nonzero freezer removals should not return multiple strong
  candidates only because dual-camera exit-path evidence is strong. Multi-kind
  freezer output must fit segment or combined candidate weight.
- The expected fallback for mismatched freezer multi evidence is single handled
  candidate selection, with diagnostics preserving rejected candidates for
  operations review.

## Related Pages

- [decision-and-weight](../source-code/decision-and-weight.md)
- [video-and-vision](../source-code/video-and-vision.md)
- [product-detection-pipeline](../synthesis/product-detection-pipeline.md)
