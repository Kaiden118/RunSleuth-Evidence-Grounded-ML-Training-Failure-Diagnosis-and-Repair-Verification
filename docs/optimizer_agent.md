# Agent diagnosis of recorded optimizer bindings

This optional diagnostic profile separates an implementation defect from observed
validation performance. It uses the completed paired Camelyon17 experiment;
it does not retrain models or execute a repair.

## Integration

Copy the new package files into the repository, preserving their directories.
Apply `patches/agent_optimizer.patch` to the three existing modules supplied for
this change (`agent.py`, `diagnostic_tools.py`, `agent_diagnose.py`).

```cmd
git apply --check patches/agent_optimizer.patch && git apply patches/agent_optimizer.patch
```

Apply the patch only once. A failed check changes none of those three files;
retain the error output and reconcile it against the actual local source.
The default training profile retains its original tools, schema and validation.

## Check and run

```cmd
python scripts\check_optimizer_agent.py --pair-run artifacts/optimizer_experiments/pair-20260927T012822181931Z --diagnose
```

The script runs formatting, tests, lint and whitespace checks, followed by local
artifact validation. With `--diagnose`, it then calls the configured LLM twice:
once for the stale-head candidate and once with clean as both reference and candidate.
The latter is a self-comparison control, not an independent healthy training run.
Omit `--diagnose` to run without an API call. No training, checkpoint loading or
data downloads occur. Existing LLM environment variables must already be set.

Each diagnosis is bounded by 6 model calls, 5 tool calls and 3,000 output tokens
per model request. Failed requests may consume provider quota; usage accounting
retains the existing `usage_complete` semantics. This script stops on first failure.

A direct diagnosis command is:

```cmd
python -m runsleuth.agent_diagnose --profile optimizer_binding --reference-run artifacts/optimizer_experiments/pair-20260927T012822181931Z/clean --candidate-run artifacts/optimizer_experiments/pair-20260927T012822181931Z/stale_head --max-output-tokens 3000
```

## Evidence and interpretation

`inspect_optimizer_run` reads six fixed JSON/JSONL artifacts within the project
boundary. It validates completed status, epoch ordering, finite telemetry, report
consistency, selected sample hashes and preprocessing identity. It returns compact
audit and per-epoch evidence, omitting injected variant labels and config paths.
Requested run paths remain visible to the model but are not grounds for diagnosis.

All cited scalar values are checked against successful calls and exact task paths.
The implementation classification must agree with the recorded membership audit.
Performance is classified independently using final ID/OOD accuracy and loss:
`no_observed_degradation`, `degraded`, `mixed`, or `inconclusive` for incomparable runs.
These describe signed differences, not significance or a repair acceptance gate.
Citation validation does not prove every natural-language explanation is entailed.

For the observed seed-42 experiment, the expected interpretation is:
- stale candidate: `optimizer_binding_defect` + `no_observed_degradation`;
- clean self-comparison: `no_optimizer_binding_defect` + `no_observed_degradation`.

The audit is recorded evidence, not a current in-memory optimizer inspection.
Artifact consistency checks do not authenticate who produced the files.
The supported cause is `optimizer_parameter_mismatch`; initialization order is
not proven by membership counts alone. Partial optimization can be intentional
outside this controlled full-finetuning experiment.

The output has `profile: optimizer_binding` and `proposed_patch: null`.
Do not feed it to the existing FashionMNIST configuration repair command.
Source-level repair and a separately specified verification policy are future work.

## Validated integration cases

Validated against the recorded three-epoch Camelyon17 experiment
`pair-20260927T012822181931Z`.

| Case | First-pass valid | Corrections | Final result |
|---|---|---:|---|
| Stale classifier head | No | 1 | Binding defect; no observed performance degradation |
| Same-run clean control | Yes | 0 | No binding defect; no change recommended |

The defect diagnosis passed after validator-assisted correction.
The clean control compared the reference run with itself.
These are integration checks, not a general diagnostic accuracy benchmark.
Both reports proposed no patch; no optimizer repair was executed.

Reports:
- `artifacts/agent_runs/20260927T045102117145Z/agent_report.json`
- `artifacts/agent_runs/20260927T045113830418Z/agent_report.json`

## Bounded repair handoff

A saved, evidence-validated Agent diagnosis can now enter a controlled
repair workflow through `python -m runsleuth.optimizer_agent_repair`.

The default mode validates the evidence and saves a repair plan.
Explicit `--execute` runs bounded training and verification.

The controller selects the allowlisted action
`rebuild_optimizer_before_first_step`; the Agent's `proposed_patch`
remains null. Training starts from the saved reference initialization,
not from a failed checkpoint.

### Recorded execution: 2026-09-27

Three variants ran for three epochs each: clean, stale head, and repaired
stale head. This execution made no new LLM calls.

| Check | Observed | Requirement | Result |
|---|---:|---:|---|
| Structural verification | All checks passed | All must pass | Pass |
| OOD accuracy drop versus stale head | 1.40 percentage points | At most 1.00 | Fail |
| OOD loss ratio versus stale head | 1.1831 | At most 1.10 | Fail |

Execution status: `completed`. Acceptance decision: `rejected`.

The optimizer binding was repaired, but the performance non-regression
requirements were not met. Thresholds were preserved. This single-seed
development experiment does not establish general performance effects.

Local execution record:
`artifacts/optimizer_agent_repairs/20260927T071333035701Z/optimizer_agent_repair_report.json`