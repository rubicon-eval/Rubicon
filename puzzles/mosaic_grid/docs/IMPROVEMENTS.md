# Mosaic Grid — Improvements Plan

Tracks what's done, what's next, and why each item matters for reviewer confidence.

---

## Status

| Component | State |
|:---|:---|
| Engine (paint, plates, doors) | Done |
| Oracle (A*, solvability, fatal step) | Done |
| Renderer | Done |
| Prompts (RQ1/RQ2/RQ3 + ablations) | Done |
| 12 templates (barrier through zigzag) | Done |
| Exit pocket / chokepoint on all templates | Done |
| Plate spacing + distinct-distance constraints | Done |
| Greedy filter (5 strategies incl. BFS, enforced) | Done |
| Deadlock discovery + k-horizon filtering | Done |
| Orientation randomization (all templates) | Done |
| Multi-template demo (RQ1/RQ2/RQ3 per template) | Done |
| Template documentation (`TEMPLATES.md`) | Done |
| Grid size variation (5x5 through 8x8) | Done |
| Auto-scaled plates (2→3→4→6 by grid size) | Done |
| Multi-door support (N doors, all 12 templates) | Done |
| Generation engine robustness (metadata, stats, serialization) | Done |
| Parallel generation + dataset CLI | Done |
| Door-required filter (`_all_doors_required`) | Done |
| Stochastic template scaling (5 templates) | Done |
| Eval build pipeline (`build_eval.py`) | Done |
| Per-puzzle directories + metadata | Done |
| Prompt generation + trajectory variation | Done |
| Per-RQ reference images | Done |
| `build` and `full` CLI subcommands | Done |

**Generation health:**

| Metric | Value |
|:---|:---|
| Gen success rate | **100%** (all 12 templates, all grid sizes 5x5–8x8) |
| Greedy failure rate | **100%** (all 5 strategies fail) |
| RQ3 trace success | **100%** (k-horizon filtering + seed retry) |
| Templates | **12** |
| Grid sizes | **4** (5x5, 6x6, 7x7, 8x8) |
| Greedy strategies | **5** (3 manhattan + 2 BFS with 8 tiebreaking orders) |
| Multi-door (2-door) | **12/12** templates at 6x6 and 7x7; all tested at 8x8 |
| Multi-door (3-door) | **12/12** templates at 8x8 |
| Door-required filter | **100%** — every door must be on mandatory start→exit path |
| Demo RQ coverage | **12/12** templates produce RQ1+RQ2+RQ3 |

---

## What Already Works (no changes needed)

| Component | Why it already works |
|:---|:---|
| **Engine** (`engine.py`) | `PlateInfo` supports arbitrary linked doors, `DoorInfo` supports arbitrary required plates. Multiple plates, multiple doors, multi-color linkage all work. Paint mechanics are fully general. |
| **Oracle** (`oracle.py`) | A* search and `is_solvable` handle arbitrary painted-tile sets. `find_fatal_step` scans any trace. |
| **Renderer** (`renderer.py`) | Renders plates (any count/position), doors (any count), painted tiles. Uses distinct colors per door. No hardcoded positions. |
| **Prompts** (`prompts.py`) | `generate_rules_text` iterates all plates/doors dynamically. Rules describe paint mechanics generically. |
| **Deadlock discovery** (`generator.py`) | Uses monotonicity (painted tiles only grow → once unsolvable, always unsolvable). Works with any template. |

---

## Current Template Landscape

See `docs/TEMPLATES.md` for full details. Key findings from the template review:

**5 clearly distinct templates at 5x5:** Barrier, L-Shape, Spiral, Cross, Island.
**3 moderately distinct:** Horizontal, Pinch, Moat.
**4 with low distinctiveness at 5x5:** Bottleneck, Peninsula, Corridor, Zigzag.

The low-distinctiveness templates converge toward "a few random walls + exit pocket" on a 5x5 grid. All 12 are retained because they differentiate at larger grid sizes (especially with stochastic scaling), and they all pass the full validation pipeline.

---

## Completed Improvements

### Greedy filter enforcement — DONE

Fixed the bug where greedy filter was computed but never enforced. `all_greedy_strategies_fail()` now gates instance acceptance.

### 5 greedy strategies — DONE

3 manhattan-based (nearest_plate_first, toward_exit, visit_all_then_exit) + 2 BFS-based (with 8 tiebreaking orders). All enforced.

### k-horizon deadlock filtering — DONE

Two-tier filtering: k >= 3 for RQ2 pairs (min 2 entries), k >= 5 for RQ3 traces (min 1 entry).

### Hand-crafted fallback removed — DONE

Demo uses only generated instances.

### 12 templates with structural review — DONE

Each template reviewed for: (1) door structurally required (exit pocket or wall-gap or chokepoint), (2) plates not in dead-ends where entering traps the agent, (3) wall patterns create non-trivial routing on 5x5.

### Plate spacing + distance constraints — DONE

`_select_plate_positions()` enforces manhattan distance >= 2 between plates (hard) and tries for distinct manhattan distances from start (soft fallback to spacing only).

### Grid size variation (5x5 through 8x8) — DONE

All 12 templates now generate valid instances at 5x5, 6x6, 7x7, and 8x8. Auto-scaled solution ranges per grid size (5x5: 6–20, 6x6: 8–25, 7x7+: 8–30). Added early-stop to deadlock discovery and lazy k-horizon computation.

**Generation timing (avg per instance, 5 seeds, 2 plates, max_retries=800):**

| Template | 5x5 | 6x6 | 7x7 | 8x8 |
|:---|:---|:---|:---|:---|
| barrier | <0.01s | 0.01s | 0.10s | 2.6s |
| bottleneck | <0.01s | 0.15s | 5.9s | 36.0s |
| horizontal | <0.01s | 0.03s | 0.29s | 0.35s |
| spiral | 0.05s | 0.02s | 0.15s | 13.5s |
| cross | 0.01s | 0.04s | 5.8s | 41.6s |
| peninsula | 0.01s | 0.02s | 0.61s | 70.9s |
| lshape | 0.01s | 0.01s | 0.09s | 4.7s |
| island | 0.03s | 0.04s | 0.84s | 100.2s |
| corridor | 0.18s | 0.06s | 0.34s | 0.71s |
| moat | 0.01s | 0.04s | 0.24s | 23.0s |
| pinch | 0.18s | 0.09s | 0.11s | 0.48s |
| zigzag | 0.01s | 1.34s | 24.9s | 40.9s |

**Bottleneck analysis:** Deadlock discovery (BFS + A* solvability checks) dominates at 8x8. The early-stop optimization helps but templates with low greedy-filter hit rates (zigzag, peninsula) still require many retries. Generation is a one-time cost — these times are acceptable for dataset building.

**Per-grid-size summary:**

| Grid | Success rate | Avg time (fastest) | Avg time (slowest) | Total for 12 templates × 1 seed |
|:---|:---|:---|:---|:---|
| 5x5 | 12/12 | <0.01s | 0.18s | ~0.5s |
| 6x6 | 12/12 | 0.01s | 1.34s | ~2s |
| 7x7 | 12/12 | 0.09s | 24.9s | ~45s |
| 8x8 | 12/12 | 0.35s | 100.2s | ~6 min |

### Auto-scaled plates — DONE

`generate_validated_instance` now auto-scales plate count by grid size when `num_plates=None` (the default):

| Grid size | Plates | Visit orderings | Rationale |
|:---|:---|:---|:---|
| 5x5 | 2 | 2 | Baseline. Works well at this size. |
| 6x6 | 3 | 6 | Extra plate adds collection detour → more paint → richer deadlocks. |
| 7x7 | 4 | 24 | Larger grid needs more plates to constrain the open space. |
| 8x8 | 6 | 720 | Maximum plates for maximum difficulty. Dense paint patterns, abundant deadlocks. |

All 12 templates validated at all grid sizes with 100% success. Solution ranges auto-adjusted: 6x6 → (8, 30), 7x7+ → (10, 40).

### Multi-door support (N doors, all 12 templates) — DONE

All 12 templates support N-door generation via `num_doors` parameter. D1 guards the exit pocket; additional doors placed by `_add_secondary_wall` (spanning wall segments with one gap). Plates split across door groups by proximity (`_split_plates_n_way`, >= 1 per group). `_all_doors_required` BFS filter rejects any instance where a door can be bypassed. Door count filter rejects instances that didn't achieve the requested count. Plate auto-minimum: `max(3, num_doors)`. Pinch and Corridor converted from chokepoint-based to exit-pocket door placement (chokepoint was broken at 7x7+).

**Coverage:** 2-door 12/12 at 6x6, 7x7, and 8x8. 3-door 12/12 at 8x8. All 1-door unchanged.

### Generation engine robustness + parallel dataset CLI — DONE

Added infrastructure for building large-scale datasets:

**Metadata on `GeneratedInstance`:** `grid_width`, `grid_height`, `num_plates`, `num_doors`, `solution_length`, `tier1_deadlocks`, `tier2_deadlocks`, `max_k_horizon`, `generation_attempts`, `fingerprint`. All have defaults so existing callers are unaffected.

**Filter rejection tracking:** `_generate_validated_inner()` tracks 8 rejection counters (`grid_gen_failed`, `door_count`, `unsolvable`, `doors_not_required`, `solution_range`, `greedy_passes`, `insufficient_deadlocks`, `insufficient_tier2`). `generate_validated_instance_with_stats()` returns a `GenerationResult` with instance, stats, total attempts, and elapsed time.

**Serialization (`dataset.py`):** `instance_to_dict()` / `dict_to_instance()` for full round-trip JSON serialization. `Pos` → `[r, c]`, `frozenset[Pos]` → sorted list, `Direction` → `.name`, all engine types → dicts.

**Fingerprinting:** SHA-256 of canonical grid layout (walls as sorted ints, start, exit, plate positions, door positions), truncated to 16 hex chars. Used for deduplication.

**Parallel orchestrator (`generate_dataset`):** Builds task list from all `(template, grid_size, num_doors, seed)` combos. `multiprocessing.Pool` with `imap_unordered`, tqdm progress bar. JSONL output with per-line `flush()` (the JSONL IS the checkpoint). `--resume` reloads existing output and skips done seeds. Duplicates detected by fingerprint set. Failures written to `_failures.jsonl`. Aggregate stats written to `_stats.json`.

**CLI (`__main__.py`):** `python -m puzzles.mosaic_grid demo` (existing demo) and `python -m puzzles.mosaic_grid generate` with flags: `--templates`, `--grid-sizes`, `--doors`, `--seeds-per-config`, `--max-retries`, `--workers`, `--output`, `--resume`, `--seed-start`.

### Stochastic template-specific scaling (5 templates) — DONE

At `grid_size >= 6` (or `>= 7` for Spiral), each generation attempt flips a coin (`rng.random() < 0.5`) to choose between the original parameters and enhanced ones. At 5x5, behavior is always original — Python `and` short-circuits before calling `rng.random()`, so the rng sequence is unchanged. The `rng` is seeded per attempt, so the enhanced/original choice is deterministic per seed and reproducible.

**Design pattern** (same in all 5 templates):
```python
grid_size = max(width, height)
enhanced = grid_size >= 6 and rng.random() < 0.5
```

**Per-template enhancements:**

| Template | Original | Enhanced | Trigger |
|:---|:---|:---|:---|
| **Peninsula** | `pen_len = 2–3`, 1-tile corridor, no plates in peninsula | `pen_len = 3–min(5, grid_size-2)`, 2-tile-wide corridor, plates allowed at tip | `grid_size >= 6` |
| **Zigzag** | `min_stub = 1`, `stub_len = 1–width//2`, extra walls 0–1 | `min_stub = 2`, `stub_len = 2–width*2//3`, extra walls `1–(grid_size-4)` | `grid_size >= 6` |
| **Bottleneck** | Default extra walls budget | Extra walls budget `max(2, floor_count // 6)` | `grid_size >= 6` |
| **Corridor** | `n_segments = 3–5`, `seg_len = 2–3`, extra walls 0–2 | `n_segments = 3+gs//3 – 5+gs//3`, `seg_len = 2–min(4, gs//2)`, extra walls `1–(gs-3)` | `grid_size >= 6` |
| **Spiral** | `min_sep = 2`, 2-tile gaps, extra walls 0–2 | `min_sep = 3`, 3-tile gaps, extra walls `1–(grid_size-4)` | `grid_size >= 7`, `width >= 6` |

**What it buys:** The dataset naturally contains a mix of original and enhanced layouts at larger grid sizes (~50/50 split). Enhanced instances have denser walls and more constrained corridors, producing harder puzzles with richer deadlock patterns. The 4 previously low-distinctiveness templates (Bottleneck, Peninsula, Corridor, Zigzag) now produce structurally varied grids at 6x6+.

---

## Completed: Evaluation Pipeline (P1 + P2 + P3)

### P1: Per-puzzle metadata + directory structure — DONE

Each generated puzzle instance gets its own directory:
```
dataset/{template}_{grid}_{doors}d_seed{N}/
  metadata.json        # puzzle metadata (template, grid size, plates, doors, solution length, deadlocks, fingerprint, etc.)
  initial.png          # initial state image (provided to model)
  solution_trace.png   # multi-panel oracle solution (for reference, not shown to model)
```

Plus `eval_index.jsonl` — one line per eval item with prompt text, image path, and ground truth inline. No separate prompt `.txt` files; one JSONL line = one complete eval item for bulk evaluation.

### P2: Prompt generation with trajectory variation — DONE

Prompt text generated per RQ per puzzle via existing `prompts.py` builders: `build_rq1_prompt`, `build_rq1_ablation_prompt`, `build_rq2_main_prompt`, `build_rq2_ablation_prompt`, `build_rq3_prompt`. Prompts stored inline in `eval_index.jsonl`.

**RQ1 trajectory variation:** For each N in `{3, 5, 8}` (configurable via `--rq1-lengths`) plus `solution_length` itself, capped at solution_length. Typically 3–4 entries per puzzle. Plus 4 ablation entries (one per direction from initial position).

**RQ2 matched pairs:** For each deadlock catalog entry with `k_horizon >= 3` (up to 3), generates 4 eval items: main deadlock (→ UNSOLVABLE), main solvable (→ best next move), ablation deadlock (→ NO), ablation solvable (→ YES).

**RQ3 causal attribution:** For each deadlock catalog entry with `k_horizon >= 5` (up to 2), generates 1 eval item with full trace and fatal step ground truth.

### P3: Per-RQ reference images — DONE

All three RQs give the model the same image: the **initial state** (`initial.png`). Additional reference images generated for spot-checking:
- `rq2_state_eN.png` — board after k deadlock actions.
- `rq3_trace_eN.png` — full trace with fatal step highlighted.
- `solution_trace.png` — multi-panel oracle solution.

### Build pipeline CLI — DONE

**`build` command:** Converts raw JSONL → eval dataset. Flags: `--input`, `--output`, `--workers`, `--rq1-lengths`, `--overwrite`.

**`full` command:** End-to-end pipeline (generate → build). Runs `generate_dataset()` writing to `{output}/raw.jsonl`, then `build_eval_dataset()`. Intermediate JSONL kept for reproducibility.

Implementation in `build_eval.py`: parallel workers via `multiprocessing.Pool`, per-instance `_build_worker()` that creates puzzle directories, renders images, generates all RQ eval entries. Outputs `eval_index.jsonl` and `build_stats.json`.

---

## What We Don't Need

- **Dynamic plates** (plates that move or deactivate): The mechanic is paint permanence, not plate complexity.
- **Grids > 8x8**: State space grows exponentially with painted tiles. Oracle A* becomes impractical.
- **Erasable paint**: Defeats the entire puzzle mechanic. The irreversibility of paint is the point.
- **New templates**: 12 is sufficient. Adding more templates adds code maintenance cost without proportional diversity gain. The priority is making existing templates work better at larger grid sizes.
