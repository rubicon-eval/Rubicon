# Mosaic Grid — Dataset Generation

End-to-end pipeline for generating evaluation-ready puzzle instances.

---

## Quick Start

```bash
# End-to-end: generate puzzles + build eval dataset in one command
python -m puzzles.mosaic_grid full \
  --templates "barrier,spiral,cross" \
  --grid-sizes "5,6" \
  --seeds-per-config 10 \
  --output test_dataset/

# Or run each step separately:

# Step 1: Generate raw puzzle instances → JSONL
python -m puzzles.mosaic_grid generate \
  --templates "barrier,spiral,cross" \
  --grid-sizes "5,6" \
  --seeds-per-config 10 \
  --output test.jsonl

# Step 2: Build eval dataset from JSONL → per-puzzle dirs + eval index
python -m puzzles.mosaic_grid build \
  --input test.jsonl \
  --output test_dataset/
```

---

## Pipeline Overview

The pipeline has two stages:

```
Stage 1: generate
  CLI flags → task list (template × grid_size × doors × seed)
    → multiprocessing pool
    → 8 validation filters per candidate grid
    → fingerprint dedup
    → raw JSONL (one line per instance)

Stage 2: build
  raw JSONL → per-puzzle directories
    → render initial.png, solution_trace.png
    → generate RQ1/RQ2/RQ3 eval items (prompts + ground truth)
    → render reference images (rq2_state, rq3_trace)
    → write eval_index.jsonl + build_stats.json
```

The `full` command chains both stages: generate writes to `{output}/raw.jsonl`, then build reads that JSONL and writes eval artifacts into the same output directory.

---

## What to Expect: Puzzle and Eval Distribution

This section explains exactly what comes out of the pipeline so there are no surprises.

### Puzzle Distribution

The generate step attempts `templates × grid_sizes × door_counts × seeds_per_config` total seeds. The actual unique puzzle count is significantly lower because of **fingerprint deduplication** — different seeds routinely produce structurally identical grids.

**Benchmarked yield rates:**

| Config | Seeds Attempted | Unique Puzzles | Yield | Dedup Rate |
|:-------|:----------------|:---------------|:------|:-----------|
| 12 templates × 2 sizes × 5 seeds | 120 | 33 | 28% | 72% |
| 12 templates × 2 sizes × 10 seeds | 240 | 44 | 18% | 82% |
| 12 templates × 2 sizes × 15 seeds | 360 | 49 | 14% | 86% |

Yield **decreases** as you add more seeds per config — the first few seeds for a template/size combo find the unique layouts; additional seeds mostly produce duplicates. Template diversity is more efficient than more seeds.

**Puzzle distribution by template is uneven.** Some templates (lshape, barrier) produce more structurally distinct grids than others (corridor, pinch, moat). From the 49-puzzle benchmark:

| Template | Unique Puzzles | % of Total |
|:---------|:---------------|:-----------|
| lshape | 8 | 16% |
| barrier, bottleneck, spiral, peninsula | 5 each | 10% each |
| horizontal, island, zigzag | 4 each | 8% each |
| cross | 3 | 6% |
| corridor, moat, pinch | 2 each | 4% each |

**Grid size split** depends on your `--grid-sizes` flag. With `5,6`, the 6x6 puzzles produce more eval items per puzzle because they have longer solutions and richer deadlock catalogs, so the eval set skews toward 6x6 (~67% of eval items from 6x6 vs ~33% from 5x5 in our benchmark, despite roughly equal puzzle counts).

### Puzzle Filtering (Paired Baseline Requirement)

Only puzzles that produce **both** at least 1 valid RQ2 pair **and** at least 1 valid RQ3 trace are kept. Puzzles missing either are dropped — they can't support the full paired evaluation design.

This means:
- ~65% of puzzles are dropped (no eligible RQ2 deadlock pairs)
- ~2% are dropped (no eligible RQ3 traces)
- Only ~35% of the raw puzzle pool makes it into the eval set

The `build_stats.json` file reports `puzzles_dropped_missing_rq` so you can see exactly how many were dropped. To compensate for the reduction, generate a larger raw pool (`--seeds-per-config 50+`).

### What You Get Per Puzzle

Every puzzle that makes it into the eval set produces **images** (saved to its directory) and **eval items** (rows in `eval_index.jsonl`). Here's the full breakdown.

#### Images

Every eval item — across all three RQs — shows the model the same image: `initial.png`. That's the grid at its starting state (walls, agent, plates, doors, unpainted cells). The model gets this image plus a text prompt, and has to reason from there.

The other images in the puzzle directory are rendered for human inspection when reviewing results. They're never sent to the model.

| Image | Always present? | Purpose |
|:------|:----------------|:--------|
| `initial.png` | Yes | The puzzle state shown to the model — every eval item references this image |
| `solution_trace.png` | Yes | Multi-panel oracle solution (reference only, never shown to model) |
| `rq2_state_e{i}.png` | 1 per RQ2 pair (1–3) | Deadlock state after K actions (reference only) |
| `rq3_trace_e{i}.png` | 1 per RQ3 trace (1–2) | Full trace with fatal step highlighted (reference only) |
| `rq3_final_e{i}.png` | 1 per RQ3 trace (1–2) | Final state after full trace — shown to model for `ablation_final` |
| `rq3_comparison_e{i}.png` | 1 per RQ3 trace (1–2) | Side-by-side initial + final — shown to model for `ablation_comparison` |

Total: **6–11 images** per puzzle directory.

#### Eval items

Every puzzle produces items for all three RQs. The number depends on how many RQ2 deadlock pairs and RQ3 traces the puzzle's deadlock catalog supports (determined by the catalog's k-horizons and fatal steps).

**RQ1 — Simulation Fidelity (18–28 items)**

RQ1 asks: "Can the model simulate what happens after a sequence of actions?" It has four sub-types (stored in `metadata.sub_type`), plus an ablation.

**Trajectory items (3–4).** These use prefixes of the oracle solution. The code computes which lengths to test:

```python
lengths = sorted(set(min(n, solution_length) for n in [5, 10, 15]) | {solution_length})
```

So if the solution is 14 moves long: `{min(5,14), min(10,14), min(15,14)} ∪ {14}` = `{5, 10, 14}` → 3 items. If the solution is 20 moves: `{5, 10, 15, 20}` → 4 items. The set deduplicates, so a 10-move solution gives `{5, 10}` → 2 items (rare — most solutions are 8–30 moves).

For each length N, the prompt says: "Here's the initial grid. After these N actions [UP, RIGHT, DOWN, ...], predict the agent's position, which cells are painted, which plates are activated, and which doors are open." Ground truth is the fully simulated state.

**Ablation items (4 × len(lengths) = 12–16).** For each trajectory length N (same lengths as the trajectory items above), and for each direction (UP, DOWN, LEFT, RIGHT), the prompt gives the same N actions, then asks: "After all N actions, is moving [DIRECTION] legal?" Answer is A (legal) or B (illegal). This requires the model to simulate the full trajectory (tracking paint, plates, doors) before answering a simple binary question — same simulation demand as the main task, but with a simpler answer format.

**RQ2 baseline items (2 per RQ2 pair → 2–6).** For each successful RQ2 pair (see RQ2 below), two baseline items are created:

1. Take the **deadlock** action sequence from that pair, ask "after these actions, predict the state" — standard RQ1 simulation.
2. Take the **solvable** action sequence from that pair, same question.

These use the exact same action sequences as the corresponding RQ2 items. The point: if the model fails RQ2 ("is this solvable after K moves?"), we check whether it also failed the RQ1 baseline for those same K moves. If it can't even simulate the actions correctly, the RQ2 failure is explained by a simulation failure, not a foresight failure.

**RQ3 baseline items (1 per RQ3 trace → 1–2).** Same idea. For each successful RQ3 trace, take the full action sequence and create an RQ1 simulation item. If the model can't simulate the trace end-to-end, its RQ3 failure ("which step was fatal?") isn't about causal reasoning — it just couldn't track the state.

**RQ2 — Foresight (4–12 items)**

RQ2 asks: "After K actions, can the model tell whether the puzzle is still solvable — and if so, what's the best next move?"

The build step scans the puzzle's deadlock catalog for entries with `k_horizon >= 3` and `fatal_step >= 2`, takes up to 3. For each, it calls `generate_rq2_pair()`, which tries to construct a **matched pair**: a deadlock path and a solvable path, both of length K from the initial state. K = `fatal_step + 1` (the number of actions that lead into the deadlock).

Each successful pair produces **4 eval items**:

| Item | Prompt | Ground truth |
|:-----|:-------|:-------------|
| main deadlock | "After these K actions, what's the best next move?" | UNSOLVABLE |
| main solvable | Same question, different K actions | e.g. RIGHT (oracle's best move) |
| ablation deadlock | "After these K actions, is the puzzle still solvable?" | NO |
| ablation solvable | Same question, different K actions | YES |

The matched-pair design is key: both paths are the same length, both start from the same initial state, both are plausible-looking action sequences. The only difference is that one leads to a deadlock and the other doesn't. The model can't shortcut by counting moves or noticing something superficial — it has to actually reason about what each path does to the game state.

Up to 3 pairs × 4 items = 12 items. Minimum 1 pair (4 items) — puzzles with 0 successful pairs are dropped entirely.

**RQ3 — Causal Attribution (3–6 items)**

RQ3 asks: "Here's a full trace that ends in an unsolvable state. Which step was the fatal one?"

The build step filters for deadlock catalog entries with `k_horizon >= 5` (needs a long enough "Walking Dead" phase — moves that are legal but can't lead to a solution). Takes up to 2. For each, `generate_rq3_instance()` constructs the full trace: the actions leading up to and including the fatal step, followed by continuation moves (random legal actions from the deadlock state). It requires a cause-symptom gap of at least 3 — meaning the puzzle stays "alive" (legal moves exist) for at least 3 steps after the fatal one, so the model can't just pick the last move.

Each successful trace produces **3 eval items** — 1 main + 2 ablations:

| Item | Image shown | What it tests |
|:-----|:------------|:--------------|
| main | `initial.png` | Full task: simulate trace from initial state, then identify fatal step |
| ablation_final | `rq3_final_e{i}.png` (final state only) | Causal reasoning without simulation — model sees end result |
| ablation_comparison | `rq3_comparison_e{i}.png` (initial + final side by side) | Causal reasoning with both states visible — model can compare |

The two ablation variants remove the simulation burden by showing the model the final state directly. Both ask the same question as main ("which step was fatal?") and share the same ground truth. The difference isolates whether failures on the main task come from simulation difficulty or from causal reasoning itself.

Up to 2 traces × 3 items = 6 items. Minimum 1 trace (3 items) — puzzles with 0 successful traces are dropped.

#### Totals per puzzle

| Scenario | RQ1 | RQ2 | RQ3 | Total eval items | Images |
|:---------|:----|:----|:----|:-----------------|:-------|
| 1 RQ2 pair + 1 RQ3 trace (typical) | 18–23 | 4 | 3 | **25–30** | 6 |
| 2 RQ2 pairs + 2 RQ3 traces | 21–26 | 8 | 6 | **35–40** | 11 |
| 3 RQ2 pairs + 2 RQ3 traces (max) | 23–28 | 12 | 6 | **41–46** | 11 |

So for a 30-puzzle eval set, expect roughly **750–950 eval items** in `eval_index.jsonl`. The model answers each item independently — it sees one image and one prompt per eval item, no memory across items.

### Why Paired Baselines?

If a model fails RQ2 (foresight) or RQ3 (causal attribution), the paired baselines let us distinguish two failure modes:
- **Can't simulate state:** The model also fails the RQ1 baseline for the same action sequence → the higher-order failure is explained by a simulation failure.
- **Can simulate, but can't reason:** The model passes the RQ1 baseline but fails RQ2/RQ3 → the failure is genuinely about higher-order reasoning (foresight or causal attribution).

This is why every puzzle must have both RQ2 and RQ3 entries — puzzles missing either are dropped so that the paired design holds across the entire eval set.

### Recipes for Specific Puzzle Counts

**Exactly 30 puzzles** (target-driven, auto-loops until met):
```bash
python -m puzzles.mosaic_grid full \
  --grid-sizes "5,6" \
  --seeds-per-config 10 \
  --puzzle-target 30 \
  --output dataset/
```

**Exactly 100 puzzles** (target-driven, all sizes):
```bash
python -m puzzles.mosaic_grid full \
  --grid-sizes "5,6,7,8" \
  --seeds-per-config 10 \
  --puzzle-target 100 \
  --output dataset/
```

**~50 puzzles** (single pass, no target):
```bash
python -m puzzles.mosaic_grid full \
  --grid-sizes "5,6" \
  --seeds-per-config 10 \
  --output dataset/
```

**~200 puzzles** (single pass, 5x5 + 6x6 only):
```bash
python -m puzzles.mosaic_grid full \
  --grid-sizes "5,6" \
  --seeds-per-config 50 \
  --output dataset/
```

**~200 puzzles** (single pass, all sizes — 7x7/8x8 are slow):
```bash
python -m puzzles.mosaic_grid full \
  --grid-sizes "5,6,7,8" \
  --seeds-per-config 20 \
  --output dataset/
```

### Timing

| Grid Size | Avg per seed | Notes |
|:----------|:-------------|:------|
| 5x5 | < 0.01s | All templates fast |
| 6x6 | 0.01–1.3s | Most templates fast, zigzag slowest |
| 7x7 | 0.1–25s | Wide variance by template |
| 8x8 | 0.3–100s | Island, peninsula, zigzag can take minutes |

**Practical estimates (4 workers, all 12 templates):**

| Config | Generate | Build | Total |
|:-------|:---------|:------|:------|
| 5x5 + 6x6, 10 seeds | ~2.5s | ~11s | ~14s |
| 5x5 + 6x6, 50 seeds | ~10s | ~50s | ~1 min |
| 5x5 through 8x8, 20 seeds | ~20 min | ~2 min | ~22 min |
| 5x5 through 8x8, 50 seeds | ~1 hr | ~5 min | ~1 hr |

Generation dominates total time. The build step scales linearly at ~0.25s/puzzle. The 8x8 templates (island, peninsula, bottleneck, zigzag) are the main time sink — deadlock discovery via BFS + A* is expensive at that grid size.

---

## Output Structure

### `full` command output:
```
dataset/
  raw.jsonl                     # intermediate JSONL (kept for reproducibility)
  raw_stats.json                # generation aggregate stats
  raw_failures.jsonl            # failed seeds
  eval_index.jsonl              # one line per eval item (prompt + image path + ground truth)
  build_stats.json              # build aggregate stats
  barrier_6x6_1d_seed42/
    metadata.json               # puzzle metadata
    initial.png                 # initial state — shown to model for all RQs
    solution_trace.png          # multi-panel oracle solution (reference only)
    rq2_state_e0.png            # deadlock state after k actions (reference only)
    rq3_trace_e0.png            # full trace with fatal step highlighted (reference only)
  ...
```

### `eval_index.jsonl` schema

Each line is one complete eval item:

```jsonc
{
  "eval_id": "barrier_6x6_1d_seed42__rq1_main__n5",
  "puzzle_dir": "barrier_6x6_1d_seed42",
  "rq": "rq1",
  "variant": "main",
  "image_path": "barrier_6x6_1d_seed42/initial.png",
  "prompt": "You are playing a grid-based puzzle...",
  "ground_truth": {
    "agent_pos": [3, 2],
    "painted": [[0, 0], [1, 0], ...],
    "activated_plates": ["P1"],
    "open_doors": []
  },
  "metadata": {
    "template_name": "barrier",
    "grid_size": "6x6",
    "num_doors": 1,
    "solution_length": 14,
    "trajectory_length": 5,
    "seed": 42,
    "fingerprint": "a3f8c91e02b74d16"
  }
}
```

Prompt text is stored **inline** — one JSONL line = one complete eval item. No separate `.txt` files to chase.

---

## CLI Reference

### `generate` — Create raw puzzle instances

```
python -m puzzles.mosaic_grid generate [flags]
```

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--templates` | all 12 | Comma-separated template names |
| `--grid-sizes` | `5,6,7,8` | Comma-separated grid sizes |
| `--doors` | `1` | Comma-separated door counts |
| `--seeds-per-config` | `100` | Seeds per (template, grid, doors) config |
| `--max-retries` | `400` | Max attempts per seed before declaring failure |
| `--workers` | CPU count | Parallel worker processes |
| `--output` | `mosaic_dataset.jsonl` | Output JSONL path |
| `--resume` | off | Resume from existing output (append mode) |
| `--seed-start` | `0` | Starting seed value |

### `build` — Convert raw JSONL to eval dataset

```
python -m puzzles.mosaic_grid build [flags]
```

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--input` | `mosaic_dataset.jsonl` | Input JSONL from generate step |
| `--output` | `dataset` | Output directory root |
| `--workers` | CPU count | Parallel workers |
| `--rq1-lengths` | `5,10,15` | Comma-separated RQ1 trajectory lengths |
| `--overwrite` | off | Overwrite existing puzzle directories |
| `--puzzle-target` | `0` (no limit) | Max puzzles in final eval index (stratified by template, grid_size, num_doors) |
| `--sample-seed` | `42` | Random seed for reproducible sampling |

### `full` — End-to-end pipeline (generate → build)

```
python -m puzzles.mosaic_grid full [flags]
```

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--templates` | all 12 | Comma-separated template names |
| `--grid-sizes` | `5,6,7,8` | Comma-separated grid sizes |
| `--doors` | `1` | Comma-separated door counts |
| `--seeds-per-config` | `100` | Seeds per (template, grid, doors) config |
| `--max-retries` | `400` | Max retries per seed during generation |
| `--rq1-lengths` | `5,10,15` | Comma-separated RQ1 trajectory lengths |
| `--workers` | CPU count | Parallel workers (shared by both steps) |
| `--output` | `dataset` | Output directory root |
| `--seed-start` | `0` | Starting seed value |
| `--overwrite` | off | Overwrite existing puzzle dirs in build step |
| `--puzzle-target` | `0` (no target) | Target puzzle count: loops generate+build until met (0 = single pass) |
| `--sample-seed` | `42` | Random seed for reproducible sampling |
| `--stall-rounds` | `3` | Bail out after N consecutive rounds with no new puzzles |
| `--max-rounds` | `50` | Hard ceiling on generation rounds |

### Templates

`barrier`, `bottleneck`, `horizontal`, `spiral`, `cross`, `peninsula`, `lshape`, `island`, `corridor`, `moat`, `pinch`, `zigzag`

See `docs/TEMPLATES.md` for wall patterns and door mechanisms.

---

## Puzzle-Level Sampling (`--puzzle-target N`)

The natural unit of control is the **puzzle**. Each kept puzzle contributes all its eval items — RQ1 (trajectory + ablation + baselines), RQ2 (matched pairs), and RQ3 (causal traces) — so paired integrity is preserved by design.

### The problem

Not every generated puzzle makes it into the eval set. Between fingerprint deduplication (~70-85% of seeds produce duplicate grids) and the RQ2/RQ3 filtering (~65% of unique puzzles lack the required deadlock pairs), only ~10-15% of attempted seeds survive. If you want exactly 30 eval-eligible puzzles, you'd have to guess the right `--seeds-per-config` — and guess wrong and you either overshoot or come up short.

### Two modes

| Command | `--puzzle-target N` behavior |
|:--------|:-----------------------------|
| `build` | **Cap** — sample at most N puzzles from whatever the input JSONL already contains. No generation, no loop. |
| `full`  | **Target-driven loop** — keep generating new seeds in rounds until N eval-eligible puzzles exist, then sample down to exactly N if overshot. |

### How the `full` loop works

Say you run:
```bash
python -m puzzles.mosaic_grid full \
  --puzzle-target 10 --seeds-per-config 5 --grid-sizes "5,6" \
  --output dataset/
```

Here's what happens:

**Round 1** — The pipeline truncates `raw.jsonl` (clean start), then generates with seeds 0–4 across all 12 templates × 2 grid sizes = 120 tasks. Fingerprint dedup removes duplicates; say ~35 unique instances land in `raw.jsonl`. Next, `build_eval_dataset` processes all 35 — drops any puzzle missing RQ2 pairs or RQ3 traces — say 12 survive. The loop checks: `12 >= 10`? Yes → **target reached**, break.

**Final sampling** — We overshot (12 > 10), so one last build pass runs with `puzzle_target=10`. This stratified-samples 10 puzzles across `(template, grid_size, num_doors)` buckets (roughly equal per bucket, redistributing surplus from buckets with fewer puzzles). This ensures balanced representation across all three axes — plates follow from grid size automatically. The final `eval_index.jsonl` has exactly 10 puzzles' worth of eval items, and `build_stats.json` shows `"puzzles_sampled": "12 -> 10"`.

**What if Round 1 isn't enough?** Say only 7 puzzles survive Round 1:

- **Round 2**: generates with seeds 5–9. `resume=True` loads existing fingerprints from `raw.jsonl`, so duplicates across rounds are still caught. New unique instances are appended. Build again → say 11 puzzles now. `gained=4`, `11 >= 10` → target reached. Final sample: 11 → 10.

**What if the config is exhausted?** Say the template/size combo can't produce more unique puzzles:

- Round 1: 7 puzzles (+7)
- Round 2: 7 puzzles (+0) — stall 1/3
- Round 3: 7 puzzles (+0) — stall 2/3
- Round 4: 7 puzzles (+0) — stall 3/3
- **Bail out**: "3 consecutive rounds with no new puzzles. Built 7 of 10."
- No final sampling (7 < 10). You get 7 puzzles — the best the pipeline could produce.

### Why `overwrite=True` every round

The build step's `overwrite=False` mode skips existing puzzle dirs and returns empty entries for them — they don't appear in `eval_index.jsonl`. So every round must rebuild all puzzles (`overwrite=True`) to get an accurate `puzzles_built` count. This is cheap: generation dominates runtime (~0.25s/puzzle for build).

### Why no sampling until the end

Mid-loop build calls use `puzzle_target=0` (keep everything). If we sampled during the loop, `puzzles_built` would be capped to the target and we couldn't tell whether we had exactly 10 puzzles or 50 that got sampled to 10. The true count drives the loop; sampling only happens once at the end.

### Examples

```bash
# full: generate until we have 30 puzzles, 10 seeds per round
python -m puzzles.mosaic_grid full \
  --seeds-per-config 10 --output dataset/ \
  --puzzle-target 30

# full: same but bail out faster if stuck
python -m puzzles.mosaic_grid full \
  --seeds-per-config 10 --output dataset/ \
  --puzzle-target 30 --stall-rounds 2 --max-rounds 20

# full: single pass (no target, same as before)
python -m puzzles.mosaic_grid full \
  --seeds-per-config 20 --output dataset/

# build: cap to 10 puzzles from existing JSONL (no generation)
python -m puzzles.mosaic_grid build --input raw.jsonl --output dataset/ \
  --puzzle-target 10
```

### Reproducibility

Same `--sample-seed` (default 42) + same input pool = same output every time. Change the seed to get a different random subset.

---

## Validation Filters

Every candidate grid must pass all 8 filters in order:

1. **Grid generation** — template function produced a valid grid
2. **Door count** — grid has the requested number of doors
3. **Solvable** — oracle A* finds a solution from the initial state
4. **Doors required** — every door lies on the mandatory start→exit path (BFS check)
5. **Solution range** — solution length within configured bounds
6. **Greedy-proof** — all 5 greedy strategies fail (3 manhattan + 2 BFS with 8 tiebreaking orders)
7. **Deadlock-rich** — sufficient deadlocks with k-horizon >= 3
8. **Tier-2 deadlocks** — sufficient deadlocks with k-horizon >= 5

### Auto-Scaling Defaults

When not overridden, plates and solution ranges scale by grid size:

| Grid | Plates | Solution range | Rationale |
|:-----|:-------|:---------------|:----------|
| 5x5  | 2      | 6–20           | Baseline |
| 6x6  | 3      | 8–30           | Extra plate adds collection detour |
| 7x7  | 4      | 10–40          | More plates constrain open space |
| 8x8  | 6      | 10–40          | Maximum plates, dense paint patterns |

Multi-door instances auto-raise plate minimum to `max(3, num_doors)`.

---

## Resume & Deduplication

**Resume:** If generation is interrupted, re-run with `--resume`. The pipeline reads the existing JSONL, rebuilds the set of completed task keys (`{template}_{W}x{H}_{D}d_s{seed}`), and skips them. New results are appended.

**Deduplication:** Each instance gets a SHA-256 fingerprint of its canonical grid layout (walls, start, exit, plate positions, door positions). If two different seeds produce structurally identical grids, the duplicate is silently skipped. Duplicates are counted in the stats file.

---

## Raw JSONL Format

Each line in the raw JSONL contains:

```jsonc
{
  "task": {                          // generation config
    "template": "barrier",
    "width": 6, "height": 6,
    "num_doors": 1,
    "seed": 42,
    "max_retries": 400
  },
  "success": true,
  "total_attempts": 23,
  "elapsed_seconds": 0.87,
  "fingerprint": "a3f8c91e02b74d16",
  "instance": {
    "grid": { ... },                 // full grid layout
    "initial_state": { ... },
    "solution": ["DOWN", "RIGHT", ...],
    "deadlock_catalog": [ ... ],
    "seed": 42,
    "template_name": "barrier",
    "grid_width": 6, "grid_height": 6,
    "num_plates": 3, "num_doors": 1,
    "solution_length": 14,
    "tier1_deadlocks": 5,
    "tier2_deadlocks": 2,
    "max_k_horizon": 12,
    "generation_attempts": 23,
    "fingerprint": "a3f8c91e02b74d16"
  }
}
```

The `instance` block is a full round-trip serialization — `dict_to_instance()` in `dataset.py` reconstructs the `GeneratedInstance` object.

---

## Prompt Templates

Six prompt builders in `prompts.py`, each taking a grid and an action list:

| Builder | RQ | Task |
|:--------|:---|:-----|
| `build_rq1_prompt` | RQ1 (Simulation) | Given N actions, predict final state |
| `build_rq1_ablation_prompt` | RQ1 ablation | After N actions, is moving [DIRECTION] legal? |
| `build_rq2_main_prompt` | RQ2 (Foresight) | After K actions, best next move or UNSOLVABLE? |
| `build_rq2_ablation_prompt` | RQ2 ablation | After K actions, is the puzzle still solvable? YES/NO |
| `build_rq3_prompt` | RQ3 (Causal) | Full trace is unsolvable — which step was fatal? |
| `build_rq3_ablation_prompt` | RQ3 ablation | Fatal step with final state given (removes simulation burden) |

All prompts include the shared game-rules preamble (`build_rules_prompt`) which describes the grid layout, plate-door linkages, and paint mechanics for that specific instance.

See `docs/PROMPTS.md` for full prompt text and expected output formats.

---

## Project Structure

```
puzzles/mosaic_grid/
  __main__.py          # CLI (demo, generate, build, full)
  engine.py            # Grid, state, movement, plates, doors
  oracle.py            # A* solver, solvability, fatal step detection
  generator.py         # 12 template generators, validation, deadlock discovery
  dataset.py           # Serialization, fingerprinting, parallel orchestration
  build_eval.py        # Raw JSONL → eval dataset (per-puzzle dirs + eval index)
  renderer.py          # PNG rendering of states and solution traces
  prompts.py           # RQ1/RQ2/RQ3 prompt builders
  demo.py              # Demo script (renders examples)
  docs/
    DESIGN.md          # Game rules, state representation, action space
    TEMPLATES.md       # 12 template specifications and analysis
    PROMPTS.md         # Full prompt text and expected output formats
    IMPROVEMENTS.md    # Project status and roadmap
  demo_output/         # Pre-rendered demo images
```

---

## Further Reading

- `docs/DESIGN.md` — game mechanics, state space, deadlock theory
- `docs/TEMPLATES.md` — per-template wall patterns, door mechanisms, distinctiveness
- `docs/PROMPTS.md` — exact prompt text sent to models
- `docs/IMPROVEMENTS.md` — what's done and what's next
