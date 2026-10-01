# Optimizer diagnosis: fixed-model development comparison

Recorded on 2026-10-01 UTC. This fresh panel used `gemini-3.5-flash-lite`
for all four cases: one defective and one repaired candidate at each of
training seeds 7 and 2026. All cases reuse completed Camelyon17/ResNet18
training evidence; this evaluation performed no new training.

The Agent and deterministic rules inspect the same recorded optimizer audit
and parameter-group telemetry. Agent outputs are revalidated against the
saved citations and current artifact hashes.

| Measure | Rules | Agent |
|---|---:|---:|
| Correct cases | 4/4 | 4/4 |
| Correct implementation classification | 4/4 | 4/4 |
| Correct performance classification | 4/4 | 4/4 |
| Correct fault top-1 diagnosis | 2/2 | 2/2 |
| Correct no-action decision on repaired controls | 2/2 | 2/2 |
| False-positive binding defects on repaired controls | 0/2 | 0/2 |
| First diagnostic response passed validation | Not applicable | 3/4 |

Each case used one Agent attempt. Three attempts passed diagnosis validation
immediately; one required the existing single bounded correction. All four
attempts completed without an API-error report. No case-level recovery
attempt was needed.

Per-case limits were six model calls, five tool calls, and 3,000 output tokens
per model call. Recorded usage for this new panel alone was nine model calls,
eight tool calls, 39,505 input tokens, and 6,101 output tokens. Usage reporting
was complete. Historical attempts are excluded from these totals and remain
preserved separately.

## Re-evaluate saved reports

```cmd
python -m runsleuth.evaluate_optimizer --manifest evaluations/optimizer_method_comparison_lite.json
```

This command reads local artifacts and makes no API or training calls.
The manifest requires the original run artifacts, which are excluded from Git.

Source evaluation report:
`artifacts/optimizer_reruns/20261001T051003637586Z/evaluation_report.json`.

Source execution plan and generated manifest are in the same directory.
The earlier mixed-model attempts remain in
`evaluations/optimizer_method_comparison.json` as a separate historical panel.

## Interpretation

Rules and Agent matched on this small development panel. The audit directly
exposes parameter-membership mismatches, and the Agent validator already
enforces related audit and performance rules. These results establish no
diagnostic-accuracy advantage for the LLM.

The four cases cover two seeds and one injected defect mechanism. Repaired
controls use matched initialization and training settings. These results do
not establish broad healthy-run robustness, statistical superiority, or
generalization to unseen failure mechanisms. Citation validation checks
structured values and classifications, not every sentence of an explanation.
