# Evaluation and reproducibility

## Fixed evaluation files

Each `dataset/dataset_<domain>/` directory has seven JSON arrays with 100 records each:

| Filename | Evaluation |
|---|---|
| `rq1_main.json` | Simulation of the specified action sequence |
| `rq2_hard_deadlock.json` | Foresight from an unsolvable state |
| `rq2_hard_solvable.json` | Foresight from a solvable state |
| `rq2_hard_ablation_deadlock.json` | Explicit solvability question, unsolvable state |
| `rq2_hard_ablation_solvable.json` | Explicit solvability question, solvable state |
| `rq3_hard.json` | Identify the first action making completion impossible |
| `rq3_hard_ablation_final.json` | Causal attribution with the final-state image |

Every record includes `eval_id`, `puzzle_dir`, `rq`, `variant`, `image_path`, `prompt`, `ground_truth` and `metadata`. Image paths are relative to the corresponding domain directory. RQ3 ground truth includes both a zero-indexed `fatal_step` and one-indexed `fatal_step_1indexed`; the judge scores the latter.

## Recommended runner

Use `scripts/model_runs/run_openai_compatible.py`. The endpoint is supplied through `OPENAI_BASE_URL` or `--base-url`, and credentials through `OPENAI_API_KEY` or `--api-key`. Environment variables are preferred so keys do not appear in shell history. The runner accepts JSON arrays or JSONL and supports images, streaming, concurrent requests, token accounting, resumable output and disk caching.

The default output directory is `responses/<dataset_parent>/`, e.g. `responses/dataset_mosaic_grid/`. The default filename is `<task>__openai_compatible__<sanitized_model>.jsonl`. Supplying `--output` gives a predictable name. Use that exact path with the judge's `--response-file` option. The multi-model runner evaluates requested models sequentially, with concurrency within each model.

An output file is specific to one experimental configuration. Resuming skips successful IDs; it does not compare the old output's parameters to a new invocation. Change output paths or use `--overwrite` when changing generation settings or inputs. Cache entries include generation settings. Review failed requests before comparing results: incomplete response sets must not be treated as complete benchmark runs.

## Scoring

`judge.py` performs deterministic parsing for RQ1, RQ2 ablations and RQ3. RQ2 hard uses an external judge; `--llm-fallback` also invokes an external model for extraction failures. Record the judge model and endpoint along with inference parameters. Provider behavior can change; model IDs alone do not guarantee exact reproducibility.

`judge_all.py` discovers response files by filename and model name. For precise individual experiments, prefer `judge.py --response-file PATH`. `generate_plots.py` is an analysis utility for accumulated result directories; it does not ship published experimental scores.

## Optional Google runners

```bash
uv sync --locked --extra google
uv run python scripts/run_eval.py --help
uv run python scripts/run_eval_batch.py --help
```

Set `GOOGLE_API_KEY` to use these scripts. The direct runner can use Redis at `localhost:6379` for caching and falls back if the server is unavailable. For the batch runner, explicitly pass `--dataset dataset_mosaic_grid` (or the other domain name): its legacy default refers to a `subset_50` directory that is not part of this release. Check current model availability with your provider.

## Validation boundaries

Offline tests exercise key puzzle mechanics, deterministic answer scoring, archive integrity, launcher paths, cache parameter separation and non-destructive dry runs. `validate_dataset.py` checks every original file hash, record counts, per-task identifiers, and image decoding. These checks do not constitute a fresh scientific replication of all oracle labels or published model results. No model inference is run by CI.
