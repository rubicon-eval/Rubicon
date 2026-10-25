# Tide Grid — Improvements Plan

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| Engine | Done | Supports arbitrary grid sizes, N tidal zones, M beacons, WAIT actions |
| Oracle (A*) | Done | Oxygen-pruned A*, solvability checks, Walking Dead detection |
| Renderer | Done | Single-state, trace montage, side-by-side, text-grid renderers |
| Prompts | Done | RQ1/RQ2/RQ3 + ablation prompts, dynamic rules from instance structure |
| Generator | Done | 12 templates (strait, tidal_maze, gauntlet, roundtrip, confluence, patience, switchback, tidal_bridge, deadline, pendulum, tide_pool, airlock), bait-by-construction, 10 greedy strategies (6 basic + 4 smart), oxygen recalibration |
| Deadlock discovery | Done | Greedy-trace + BFS forward-sweep + oracle-adjacent, k_horizon measurement |
| Hand-crafted instances | Done | 2 instances demonstrating bait pattern |
| Demo pipeline | Done | Hand-crafted + generated rendering, bait-vs-optimal visualization |
| D4 symmetry transforms | Done | Random rotation + flip for square grids |
| Dataset serialization | Done | `dataset.py` — JSON serialization, fingerprinting, parallel orchestrator |
| Eval build pipeline | Done | `build_eval.py` — per-RQ image + prompt generation, eval_index.jsonl |
| CLI subcommands | Done | `demo`, `generate`, `build`, `full` subcommands |
| Fingerprinting | Done | SHA-256 (16 hex) of canonical grid layout, used for deduplication |
| Grid size variation | Done | Tier-based scaling: simple (6-8), medium/complex (7-8); per-size solution ranges, deadlock minimums, oxygen slack, auto-retries |
| More templates | Done | 12 templates done (strait, tidal_maze, gauntlet, roundtrip, confluence, patience, switchback, tidal_bridge, deadline, pendulum, tide_pool, airlock) covering 12 distinct temporal reasoning skills |
| Stochastic template scaling | Done | P2: 4 independent coin-flip enhancements (E1-E4) at width>=7, applied to all 12 templates |

### Generation health (50 seeds each, current defaults)

| Template | Size | Success | Eval-ready | Solution | Deadlocks | k_horizon | Avg time |
|----------|------|---------|------------|----------|-----------|-----------|----------|
| strait | 7x6 | 50/50 (100%) | 50/50 (100%) | 13-22 (avg 17.2) | 5-61 (avg 20.6) | 202-203 | 0.18s |
| tidal_maze | 7x7 | 50/50 (100%) | 50/50 (100%) | 17-20 (avg 17.7) | 42-74 (avg 49.2) | 203-204 | 0.36s |
| gauntlet | 7x7 | 20/20 (100%) | 20/20 (100%) | 32 (avg 32.0) | 14 (avg 14.0) | 202 | 0.35s |
| gauntlet | 8x8 | 19/20 (95%) | 19/20 (95%) | 29-37 (avg 33.0) | 9-43 (avg 21.3) | 202 | 1.24s |

### Rejection breakdown (50 seeds each)

| Template | Attempts | Unsolvable | Solution range | Greedy passes | Insufficient deadlocks |
|----------|----------|------------|----------------|---------------|----------------------|
| strait | 306 (6.1/inst) | 0 | 0 | 256 | 0 |
| tidal_maze | 694 (13.9/inst) | 0 | 13 | 631 | 0 |

The dominant rejection reason is `greedy_passes` — the anti-greedy filter rejects grids where any of the 10 greedy strategies succeeds. Zero unsolvable or insufficient-deadlock rejections confirms the bait-by-construction approach produces reliably solvable, structurally rich puzzles. (Stats above predate the 10-strategy upgrade; `greedy_passes` rejection rate is expected to increase slightly with 4 additional smart strategies.)

---

## What Already Works (no changes needed)

| Component | Why it's already general |
|-----------|------------------------|
| **Engine** (`engine.py`, 256 lines) | Supports arbitrary grid dimensions, any number of `TidalZone`s with independent cycle parameters, any number of `Beacon`s. `apply_action` handles WAIT + 4 directions, auto-collects beacons on step. `is_passable_at(pos, t)` queries zone phase alignment. |
| **Oracle** (`oracle.py`, 184 lines) | A* search with oxygen pruning and admissible nearest-neighbor TSP heuristic. `is_solvable`, `is_walking_dead`, `find_fatal_step`, `count_reachable_dead_states` — all work with any grid/state configuration. |
| **Renderer** (`renderer.py`, 379 lines) | `render_state`, `render_trace` (with `highlight_fatal`), `render_side_by_side`, `render_text_grid`. Visualizes tidal zones (dry/flooded at current time), beacons (collected/uncollected), oxygen bar. Color scheme in `TIDE_COLORS`. |
| **Prompts** (`prompts.py`, 295 lines) | `build_rq1_prompt`, `build_rq2_main_prompt`, `build_rq3_prompt` + ablation variants. Dynamically generates rules text from instance structure (zone schedules, beacon positions, oxygen). |

---

## Completed Improvements

### Bait-by-construction templates — DONE

The core design innovation: tidal zone phase offsets are **calculated** so that the greedy path encounters a flooded crossing (forcing expensive WAITs) while the optimal path encounters a dry crossing (zero WAITs).

**Mechanism:**
1. Place the **bait beacon** (B1) closer to the agent by Manhattan distance, across a tidal zone with bad phase alignment
2. Place the **correct beacon** (B2) farther away, across a tidal zone with good phase alignment
3. Calculate Zone A offset so it is **flooded** at `t = manhattan(start, crossing_A)` (greedy arrival time)
4. Calculate Zone B offset so it is **dry** at `t = manhattan(start, B2) + manhattan(B2, crossing_B)` (correct-path arrival time)
5. Set oxygen = `len(oracle_solution) + randint(3, 6)` — enough for optimal, not enough for bait + WAITs

**Template 1: Strait Crossing** (7x6)
```
Row 0: .  .  .  .  .  .  .     (floor — agent region)
Row 1: .  .  .  .  .  .  .     (floor)
Row 2: #  tA  #  #  #  tB  #   (wall barrier, 2 tidal chokepoints)
Row 3: .  .  .  .  .  .  .     (floor)
Row 4: .  .  .  .  .  .  .     (floor — beacon region)
Row 5: .  .  .  .  .  .  .     (floor — exit region)
```
- Agent: top-left (near Zone A crossing)
- B1 (bait): bottom-left, below Zone A → greedy goes here first, hits flooded Zone A
- B2 (correct): top-right, accessible via floor → no tidal crossing needed initially
- Exit: bottom-right, below Zone B
- Zone A: short dry (2), long flood (5), offset forces flooding at greedy arrival
- Zone B: balanced dry/flood (3/3), phase-aligned for correct path

**Template 2: Tidal Maze** (7x7)
```
Col:  0     1     2     3     4     5     6
Row 0: .     .     .    tB     .     .     .
Row 1: .     .     .    tB     .     .     .
Row 2: .     .     .    tB     .     .     .
Row 3: tA    tA    tA    #    tA    tA    tA
Row 4: .     .     .    tB     .     .     .
Row 5: .     .     .    tB     .     .     .
Row 6: .     .     .    tB     .     .     .
```
- Two perpendicular tidal corridors divide grid into 4 quadrants
- Agent: TL quadrant → B1 (bait) in BL (nearest, crosses Zone A) → B2 (correct) in TR (crosses Zone B) → Exit in BR
- Zone A: horizontal corridor, short dry, long flood, bad timing
- Zone B: vertical corridor, phase-offset for correct-path arrival

### 10 greedy strategies (6 basic + 4 smart) — DONE

All strategies WAIT at flooded crossings instead of oscillating (UP/DOWN/UP/DOWN). This simulates realistic greedy behavior: "I'll wait here for the tide to recede."

**Basic strategies** (cheap, simple heuristics):

| Strategy | Behavior |
|----------|----------|
| `nearest_beacon` | Navigate to nearest uncollected beacon by Manhattan distance, WAIT at flooded crossings |
| `shortest_static_path` | BFS to nearest beacon by BFS distance (not manhattan), tries all 8 direction orderings for tiebreaking |
| `soonest_opening` | Head toward nearest beacon; when blocked, WAIT for soonest zone opening |
| `min_wait` | Estimate WAITs per beacon based on tidal phase, pick beacon needing fewest |
| `avoid_tidal` | Prefer non-tidal moves that strictly reduce manhattan distance; cross tidal only when forced |
| `rush_exit` | After any beacon, rush exit; skip beacons not near exit path |

**Smart strategies** (timing/oxygen-aware, model plausible LLM reasoning):

| Strategy | Behavior |
|----------|----------|
| `phase_aware_beacon` | Estimate full trip cost per beacon (manhattan + wait penalty at each zone crossing), pick lowest total cost (pos→beacon→exit). Models an LLM that reads tidal schedules |
| `oxygen_guard` | For each legal action, check if remaining oxygen after action exceeds NN-TSP heuristic lower bound. Reject budget-busting actions. Models an LLM that monitors its fuel gauge |
| `farthest_beacon_first` | Visit farthest uncollected beacon by manhattan distance first (simple `max` instead of `min`). If both nearest-first and farthest-first fail, puzzle requires genuine timing reasoning |
| `bfs_tidal_aware` | Dynamic BFS treating currently-flooded tidal tiles as impassable (time-dependent passable set). Tries all 8 direction orderings. Most expensive strategy |

Key helper functions:
- `_move_toward_manhattan()` — only returns moves that **strictly reduce** Manhattan distance (prevents oscillation)
- `_near_flooded_tidal_on_path()` — detects when agent is near a flooded crossing on the path to target (via-distance check: `agent→tile + tile→target ≤ agent→target + 2`)
- `_nearest_crossing_toward()` — finds nearest tidal tile between agent and target
- `_greedy_step()` — common dispatch: move toward target, or move toward crossing, or WAIT if near flooded crossing
- `_DIRECTION_ORDERINGS` — 8 direction orderings for BFS multi-tiebreaking (same pattern as mosaic_grid)
- `_bfs_next_step()` — now accepts optional `direction_order` parameter for tiebreaking control

**Bug fixes applied**:
- `avoid_tidal`: Fixed oscillation bug — was using raw `min(candidates, key=manhattan_distance)` which could select moves that increase distance. Now uses strict-improvement filter matching all other strategies.
- `shortest_static_path`: Fixed inconsistent beacon selection — was selecting "closest beacon" by manhattan distance but pathfinding by BFS. Now uses BFS distance for beacon selection, consistent with BFS pathfinding.

### Greedy-trace deadlock discovery — DONE

`discover_greedy_deadlocks(grid, initial_state)` traces each of the 10 greedy strategies and finds the **structural fatal step** — the action where solvability flips from True to False.

This produces deadlocks where the fatal step is meaningful: "the agent chose to approach the bait crossing instead of detouring to the correct one." Contrast with random BFS deadlocks which are often "agent wasted oxygen on random jitter."

Each `DeadlockEntry` includes:
- `strategy_name` — which greedy strategy (e.g., `"nearest_beacon"`)
- `fatal_step` — index where solvability flips
- `k_horizon` — BFS reachable states from deadlock state (all 200+ for this puzzle)
- Full action path and pre/post deadlock states

### Oracle-based oxygen recalibration — DONE

Instead of heuristic oxygen values, the validation pipeline:
1. Solves with generous placeholder oxygen
2. Computes `tight_oxygen = len(oracle_solution) + randint(3, 6)`
3. Rebuilds grid with tight oxygen
4. Re-verifies solvability

This ensures oxygen is always calibrated relative to the optimal path, making the bait trap effective — greedy paths that waste 3+ turns WAITing at the wrong crossing drain enough oxygen to make the puzzle unsolvable.

### Anti-greedy filter — DONE

`all_greedy_strategies_fail(grid, initial_state)` runs all 10 greedy strategies and returns True only if **none** solve the puzzle. This is the primary rejection filter.

The high rejection rate is expected and desirable — it means most random grids are too easy for greedy approaches, and only grids with genuine structural traps pass.

### D4 symmetry transforms — DONE

For square grids (tidal_maze at 7x7), post-generation applies random rotation (0/90/180/270) + optional horizontal flip. Transforms all positions (walls, beacons, zones, start, exit) and remaps action directions. Non-square grids (strait at 7x6) skip this.

### Triple deadlock sources — DONE

The validation pipeline combines three deadlock discovery methods:
1. **Primary**: `discover_greedy_deadlocks()` — high-quality structural deadlocks from greedy traces
2. **Supplement**: `discover_deadlocks()` — BFS forward-sweep for additional coverage
3. **Oracle-adjacent**: `discover_oracle_adjacent_deadlocks()` — single-action deviations from oracle solution path (high-quality for RQ2/RQ3)

Deduplication by deadlock state hash ensures no double-counting.

### Oracle-adjacent deadlock discovery — DONE

`discover_oracle_adjacent_deadlocks(grid, initial_state, solution)` walks the oracle solution step-by-step. At each step `i`, tries every legal action that differs from the oracle move. If any leads to a Walking Dead state, records it with `strategy_name="oracle_adjacent"`.

**Optimization**: Only tests alternatives when oxygen margin is tight (`remaining_oxygen - heuristic <= len(solution) - i + 3`). Early in the solution when oxygen is plentiful, one wrong step rarely causes deadlock.

**Unlike chain_vault** (only tests USE_KEY alternatives): Tide Grid tests ALL alternatives since any action consumes oxygen.

Produces high-quality deadlocks for RQ2/RQ3 where the divergence from the oracle path is minimal — both paths share the oracle prefix and differ only at the fatal step.

### Hard RQ2/RQ3 generation from oracle-adjacent deadlocks — DONE

- `generate_hard_rq2_pair(grid, initial_state, solution, catalog_entry)` — creates matched deadlock/solvable pairs that share the oracle prefix and differ only on the last move. Both paths are identical up to `fatal_step`, then diverge.
- `generate_hard_rq3_instance(grid, initial_state, solution, catalog_entry, target_length)` — follows the oracle for `fatal_step` actions, takes one wrong turn, then continues with random Walking Dead moves. Computes cause-symptom gap.

Follows the exact pattern from mosaic_grid and chain_vault.

### Hand-crafted instances with verified bait — DONE

Two instances in `instances.py` with manually designed zone parameters:
- **The Strait**: Zone A (bait) dry=2, flood=5, offset=0 — floods at t=2 (greedy arrival). Oxygen=22 (optimal ~20).
- **Tidal Maze**: Zone A (bait) dry=2, flood=5, offset=0 — floods at greedy arrival. Zone B offset=4. Oxygen=23.

### Bait-vs-optimal demo visualization — DONE

`demo.py` includes `_demo_bait_vs_optimal()` which renders:
1. The greedy bait trace (showing WAITs at the wrong crossing)
2. The oracle optimal trace (no/minimal WAITs)
3. Summary statistics comparing step counts and WAIT counts

---

## Template Catalog

All 12 templates are implemented, each testing a distinct temporal reasoning skill. Templates 1-2 (strait, tidal_maze) are described under Completed Improvements. Templates 3-12 are detailed below.

#### Template 3: Gauntlet — Sequential Phase Chaining — DONE

Three horizontal barrier walls at H/4, H/2, 3H/4 divide the grid into four sections. Each barrier has a single tidal crossing point that zigzags left-right-left. The agent must pass through all 3 barriers in series.

**Bait mechanism (cascading timing):**
- Beacon b2 in section 0 (free, same section as start)
- Beacon b1 in section 1 (bait, across Zone A)
- Greedy collects b2 first (nearest) → arrives at Zone A during flood phase
- Wait cascade: delay at A shifts arrival at B and C into flood phases too
- Optimal: brief wait (0-2 turns) syncs Zone A dry window → clean cascade through B and C

**Implementation details:**
- 3 zones (A, B, C), each 1 tile (single crossing per barrier), 2 beacons
- Phase offsets computed so Zone A is flooded at greedy arrival, Zones B and C are dry for optimal cascade
- Cascade verification: rejects seeds where greedy arrives dry at B or C
- Size constraints: width ≥ 6, height ≥ 7
- Solution range: 15-50 steps

**RQ value**:
- **RQ1**: Must track three independent phase schedules simultaneously
- **RQ2**: Moving forward looks reasonable — the deadlock manifests as compounding delays at all subsequent barriers
- **RQ3**: Cascading cause-symptom gap — the timing error at Zone A propagates through B and C

#### Template 4: Roundtrip — Bidirectional Timing — DONE

Single horizontal barrier at H//2 with one tidal crossing. Section 0 (above) has start, B2 (free beacon), exit. Section 1 (below) has B1 (must cross to collect). Agent must cross down to get B1 and back up to reach exit.

**Bait mechanism (phase trapping):**
- B1 is closer to start by manhattan → all greedy strategies go to B1 first
- Greedy enters at first available dry tick → round-trip distance D shifts return phase into flood window
- Agent trapped below barrier, WAITs through flood → oxygen depleted
- Optimal delays crossing (collects B2 first or WAITs) to enter at a "good phase" where both outbound and return hit dry windows

**Phase algebra:**
- D = 2 * manhattan(crossing, B1), D_mod = D % cycle
- Entry at phase p: return_phase = (p + D_mod) % cycle
- Good if return_phase < dry; bad if return_phase >= dry
- Constraint: D_mod in [flood+1, cycle-1] ensures phase 0 is bad (traps greedy) but good phases exist
- Brute-force offset search over all (dry, flood) combinations ensures both key greedy arrival times map to bad phases

**Implementation details:**
- 1 zone (A), 1 tile (single crossing), 2 beacons — simplest tidal configuration
- Searches over all dry in [2,4] x flood in [3,6] combinations for valid phase algebra (reduces rejection rate vs random sampling)
- Oracle reachability verified: at least one arrival time via direct path or B2-first path hits a good phase
- Size constraints: width >= 6, height >= 6
- Solution range: 10-35 steps
- Generation health: ~4-13 attempts per success at 6x6/7x7/8x8

**RQ value**:
- **RQ1**: Must simulate crossing → collecting → waiting → returning, tracking phase across the full round trip
- **RQ2**: Deadlock state is "stuck on the wrong side of a flooded zone" — visually obvious but the fatal step was the crossing decision, not the current position
- **RQ3**: Clean counterfactual — "if you had collected B2 first, you would have had enough oxygen for the round trip"

#### Template 5: Confluence — Schedule Intersection — DONE

Horizontal barrier with two tidal crossings (Zone A left, Zone B right) and a vertical divider splitting the bottom into left/right pockets connected by a passage at the bottom row. Agent crosses Zone A down → collects B1 in left pocket → traverses through passage → collects B2 in right pocket → crosses Zone B up → exits.

**Bait mechanism (schedule mismatch):**
- B1 closer to start → nearest-first greedy crosses Zone A first
- After traversing below (D steps), arrives at Zone B during flood window
- Farthest-first greedy crosses Zone B first → arrives at Zone A during flood
- Both orderings trapped by independent schedule mismatch
- Optimal delays entry to find time where both Zone A (at entry) and Zone B (D steps later) are dry

**Phase algebra:**
- D = below-barrier traversal distance (Zone A entry → B1 → passage → B2 → Zone B exit)
- Brute-force search over (dry_a, flood_a, dry_b, flood_b, offset_a, offset_b)
- Validates: greedy A-first traps at Zone B, greedy B-first traps at Zone A, optimal exists

**Implementation details:**
- 2 zones (A, B), each 1 tile (single crossing per barrier half), 2 beacons
- Connecting passage at row H-1 links pockets (prevents two-roundtrip degeneracy)
- Size constraints: width ≥ 7, height ≥ 6
- Solution range: 12-40 steps

**RQ value**:
- **RQ1**: Must track two independent zone phases and predict alignment over time
- **RQ2**: The deadlock is "committed to one zone's schedule without considering the other's"
- **RQ3**: Fatal step = crossing into the first zone at the wrong time, before the overlap window

#### Template 6: Patience — WAIT-Optimality — DONE

Single horizontal barrier at H//2 with one tidal crossing. Section 0 (above) has start, B2 (free beacon), exit. Section 1 (below) has B1 (must cross to collect). Structurally identical to roundtrip — the difference is entirely in zone parameters.

**Bait mechanism (WAIT-optimality):**
- Zone A has a very short dry window (dry=1-2) with long flood (4-7), cycle 5-9
- Walking distance from start to crossing is `d` steps
- At t=0, greedy moves immediately, arriving at crossing at t=d → **flooded**
- Farthest-first greedy collects B2 first, arrives at t=t_c → also **flooded**
- Both greedy orderings WAIT at crossing through flood → oxygen drain
- Optimal WAITs at start for W turns, then walks d steps → arrives at t=W+d on dry tick
- The optimal first action is WAIT (deeply counterintuitive)

**Phase algebra:**
- dry in [1, 2], flood in [4, 7], cycle = dry + flood (5-9)
- D = 2 * manhattan(crossing, B1), D_mod = D % cycle
- Roundtrip constraint: good phases exist where both entry and return hit dry
- Brute-force offset search ensures both t_g and t_c map to flood phases
- Oracle reachability: at least one wait duration via direct or B2-first path hits good phase

**Implementation details:**
- 1 zone (A), 1 tile (single crossing), 2 beacons — simplest tidal configuration (same as roundtrip)
- Size constraints: width ≥ 6, height ≥ 6
- Solution range: 10-35 steps

**RQ value**:
- **RQ1**: WAIT doesn't change position but advances time, affecting zone phases — tests understanding of WAIT's temporal effect
- **RQ2**: Fatal "choice" is moving at t=0, which looks completely correct — the subtlest possible error
- **RQ3**: Cause = first move at t=0. Symptom = oxygen depletion 15+ steps later. **Very large temporal gap**

#### Template 7: Switchback — Multi-Gate Interference — DONE

Three horizontal barrier walls at H/4, H/2, 3H/4 divide the grid into four sections. Each barrier has a single tidal crossing that zigzags right-left-right (opposite of gauntlet's left-right-left). Beacon b2 is free in section 0 (same section as start), beacon b1 sits in section 1 (across Zone A, between barriers 1 and 2), and the exit is in section 3 (below all barriers). Same cascade structure as gauntlet but with reversed zigzag creating different inter-gate distances.

**Bait mechanism (cascading timing with reversed zigzag):**
- Greedy collects b2 first (nearest, same section) → arrives at gate A late → flooded
- After waiting at A and crossing, greedy collects b1 then arrives at gate B → also flooded
- After waiting at B, greedy arrives at gate C → also flooded
- Three forced WAITs drain oxygen below threshold
- Optimal: brief initial wait (0-2 turns) syncs Zone A's dry window → clean cascade through all 3 gates

**Implementation details:**
- 3 zones (A, B, C), each 1 tile (single crossing per barrier), 2 beacons
- Gates zigzag right-left-right (opposite of gauntlet's left-right-left)
- b1 between barriers 1 and 2, b2 in section 0 (same structure as gauntlet)
- Cascade path: A → b1 (section 1) → B → C
- Phase offsets computed via direct algebra (same approach as gauntlet)
- Cascade verification: rejects seeds where greedy arrives dry at B or C
- Size constraints: width ≥ 6, height ≥ 7
- Solution range: 15-50 steps

**Key difference from gauntlet:**
- **Gauntlet**: Gates zigzag left-right-left. Same bait mechanism (b2 detour delays gate A arrival).
- **Switchback**: Gates zigzag right-left-right. The reversed zigzag creates different inter-gate manhattan distances, producing distinct interference patterns and different phase algebra solutions. Complements gauntlet for benchmark diversity.

**RQ value**:
- **RQ1**: Track 3 zone phases with different offsets through a multi-step descent
- **RQ2**: Looks like steady progress (moving through corridors, passing gates) but timing creates compounding delays
- **RQ3**: Compounding errors — was it the timing at gate 1, gate 2, or the overall descent strategy?

#### Template 8: Tidal Bridge — Duration-Aware Crossing — DONE

A 3-row-thick horizontal barrier centered in the grid. Each barrier row is solid wall except one column (the bridge column). The 3 bridge tiles form a vertical passage — the only way from section 0 to section 1. All 3 bridge tiles share one zone (Zone A).

**Bait mechanism (duration-aware crossing):**
- Zone A: dry_duration=4, flood_duration in [3,5]. Crossing requires 3 consecutive dry ticks.
- Safe entry: phases 0-1 (enough dry ticks to complete crossing).
- Dangerous entry: phase 2 — agent enters tile 1 (dry), moves to tile 2 (dry), tries tile 3 (FLOOD) — stuck on tile 2 with no legal actions (walls left/right, tiles 1&3 flooded). Terminates.
- Greedy collects B2 (free, section 0) first then heads to bridge, arriving at phase 2 due to the detour.
- Optimal WAITs for phase 0, crosses safely.

**Implementation details:**
- 1 zone (A), 3 tiles (vertical bridge passage), 2 beacons
- Phase offset computed so greedy enters at phase 2 (late dry — permanent stuck)
- Verification: greedy enters dry but gets stuck (phase+2 is flood), optimal can cross (3 consecutive dry ticks exist)
- Size constraints: width >= 5, height >= 7
- Solution range: 10-35 steps

**RQ value**:
- **RQ1**: Must simulate being *on* tidal tiles for multiple consecutive steps, tracking flood timing mid-crossing
- **RQ2**: "On a bridge that's about to flood" — impending danger vs. current safety
- **RQ3**: Fatal step = entering the bridge too late, even though the bridge was dry at entry time. **Small cause-symptom gap** — contrasts with the large gaps from gauntlet/patience

#### Template 9: Deadline — Earliest-Deadline-First Scheduling (3 beacons, 3 zones) — DONE

Single barrier row with 3 tidal crossings. Bottom section has 3 isolated vertical corridors (pockets), each containing 1 beacon. Three zones with staggered dry windows: Zone A (nearest beacon) has a very long dry window (always accessible), Zone B (middle) has a narrow window opening at EDF arrival time, Zone C (farthest) has a narrow window opening at EDF arrival time.

**Bait mechanism (EDF scheduling):**
- Greedy visits nearest beacon first (Zone A, always open) → burns time → arrives at Zone B/C during flood
- Optimal visits farthest beacon first (tightest deadline) → EDF ordering: C → B → A
- Phase offsets computed so EDF agent arrives exactly when each narrow window opens
- Greedy verification: confirms greedy arrives in flood phase at both Zone B and Zone C

**Implementation details:**
- 3 zones (A, B, C), each 1 tile (single crossing per pocket), 3 beacons — first template with 3 beacons
- Barrier row at max(2, H//3), 3 crossing columns evenly spaced
- Bottom pockets: only crossing columns are floor, everything else is wall
- Zone A: dry_a = max(30, t_edf_near + rt_near + 5), flood_a = rng(3,5) — effectively always dry
- Zone B/C: dry = rng(3,5), flood = rng(20,30) — narrow windows with long flood periods
- Phase algebra aligns narrow windows to EDF arrival times
- Size constraints: width ≥ 7, height ≥ 7
- Solution range: 15-50 steps

**RQ value**:
- **RQ1**: Must track 3 independent zone phase schedules and predict when each floods. With 3 beacons, the collection state space is richer (8 substates)
- **RQ2**: Going to B1 first looks completely safe — Zone A is dry, the path is clear, the beacon is close. The deadlock is that the agent spent its timing budget on a non-urgent task. Nothing visually signals danger
- **RQ3**: Fatal step = first move toward B1 instead of B3. Symptom = arriving at Zone B/C after flood, 8+ steps later. **Large cause-symptom gap** driven by a prioritization error rather than a spatial error

#### Template 10: Pendulum — Modular Crossing Arithmetic — DONE

```
Row 0: S  .  B1 .  .  .  .
Row 1: .  .  .  .  .  .  .
Row 2: tA tA tA tA tA tA tA    (single zone, full-width)
Row 3: .  .  .  .  .  .  .
Row 4: .  .  B2 .  .  B3 .
Row 5: .  .  .  .  .  E  .
```

**Bait mechanism**: Beacons are split across a single full-width zone — B1 above, B2+B3 below. The agent must cross Zone A **three times**: down to collect B2/B3, back up for B1, then down to exit. The key insight: the *interval between crossings* must satisfy `crossing_time mod cycle < dry_duration` for every crossing.

Zone A: dry=2, flood=5, cycle=7. Dry at t≡0,1 (mod 7).

- **Bait**: Greedy collects B1 first (above, t≈2), then heads to crossing at t≈3 — phase 3%7=3 → **FLOODED**. Must WAIT until t=7. Then collects B2, B3, needs another crossing — oxygen drained from the WAIT.
- **Correct**: Cross down at t=0 or t=1 → collect B2 (t≈3) → collect B3 (t≈5) → cross back up at t=7 (7%7=0 → DRY) → collect B1 (t≈9) → cross down at t=14 (14%7=0 → DRY) → exit. All crossing times ≡ 0 or 1 (mod 7).
- **Fatal step**: First crossing at the wrong time — the WAIT shifts ALL subsequent crossing times out of alignment.
- **Temporal skill**: Modular arithmetic — crossing times must satisfy `t mod cycle < dry_duration`. The only template where the same zone is crossed multiple times, turning the puzzle into an arithmetic constraint satisfaction problem.

**RQ value**:
- **RQ1**: Must simulate multiple crossings of the same zone, correctly computing phase at each crossing time. Arithmetic errors compound across crossings
- **RQ2**: The first crossing at the wrong time looks benign — agent waits and eventually crosses. But the WAIT shifts all subsequent crossing times, making the puzzle unsolvable
- **RQ3**: Compounding attribution — was the failure at crossing 1 (wrong time), or the route choice (should have gone below first), or a failure to WAIT before the first crossing? Ambiguous causal structure tests deep diagnostic reasoning

**Implementation details:**
- 1 zone (A), full-width row (W tiles), 3 beacons — first template with a single zone crossed 3 times
- No walls at all — the zone row itself is the only barrier (transparent when dry, blocks when flooded)
- B1 in section 0 (closest to start), B2+B3 in section 1 with spacing ≥ 2
- Brute-force search over dry in [1,3], flood in [4,8], offset and wait values
- Greedy nearest-first and farthest-first both verified flooded
- Optimal 3-crossing path pre-computed: all crossings satisfy `(t + offset) % cycle < dry`
- Size constraints: width ≥ 6, height ≥ 6
- Solution range: 12-40 steps

#### Template 11: Tide Pool — Window Allocation (3 beacons, 1 large zone) — DONE

```
Row 0: S  .  .  .  .  .  .
Row 1: .  .  .  .  .  .  .
Row 2: #  tA tA tA tA tA #
Row 3: #  tA B1 tA B2 tA #     (beacons INSIDE the tidal zone)
Row 4: #  tA tA tA B3 tA #
Row 5: #  #  #  #  #  #  #     (wall row seals bottom)
Row 6: .  .  .  E  .  .  .
```

**Bait mechanism**: Zone A is a large tidal area (3x5 block), and all 3 beacons are **inside** it. The zone has brief dry windows (e.g., dry=5, flood=8, cycle=13). During each 5-tick window, the agent can enter, collect some beacons, and must exit before the flood. But the 3 beacons are spread out — collecting all 3 in one window requires ~7 steps (impossible). The agent needs **two separate forays**, choosing which beacons to grab during each.

- **Bait**: Greedy enters during the first dry window and grabs B1+B2 (nearest pair, 5 steps — barely exits). Second window: enters for B3, but B3 is far from the entry — takes 4 steps in + 4 out = 8, only 5 ticks of dry. **Caught inside when flood hits** → must WAIT 8 turns inside → oxygen gone.
- **Correct**: First window: grab B3+B2 (farthest pair, efficient traversal arc, 5 steps). Second window: grab B1 (nearest, quick 3-step in-and-out). The model must solve a mini-TSP *within each dry window* and allocate beacons to windows optimally.
- **Fatal step**: Choosing the wrong beacons for the first window — a **batch allocation error**.
- **Temporal skill**: Window allocation / bin-packing — limited dry time must be distributed across multiple objectives. The only template where the agent spends extended time *inside* a tidal zone.

**RQ value**:
- **RQ1**: Must simulate entering, navigating within, and exiting a large tidal zone, tracking remaining dry ticks at each step. Being caught mid-zone when the flood hits is a new failure mode not tested by any crossing template
- **RQ2**: The deadlock is "entered the zone with the wrong collection plan." Agent is inside when the flood hits, can't escape, oxygen ticks away. The visual state (agent inside a dry zone) looks perfectly fine — the flood hasn't happened yet
- **RQ3**: Subtle multi-factor attribution — was the failure choosing the wrong beacons for window 1? Entering at the wrong position? Taking an inefficient route within the zone? Multiple plausible causes

#### Template 12: Airlock — Complementary Phase Navigation (2 beacons, 2 adjacent complementary zones)

```
Row 0: S  .  .  .  .  .  .
Row 1: .  .  .  .  .  B1 .
Row 2: tA tA tA tA tA tA tA    (Zone A)
Row 3: tB tB tB tB tB tB tB    (Zone B — immediately below, complementary phase)
Row 4: .  .  .  .  .  .  .
Row 5: .  B2 .  .  .  E  .
```

**Bait mechanism**: Zones A and B are adjacent rows with **complementary phases** — when A is dry, B is flooded, and vice versa. They are never passable at the same time. To cross from top to bottom, the agent must: step onto Zone A during A's dry window, then step onto Zone B exactly when the phase flips (A floods, B opens). An "airlock" — enter chamber A, wait for the phase swap, exit through chamber B.

Zone A: dry=3, flood=3, offset=0. Dry at t≡0,1,2 (mod 6).
Zone B: dry=3, flood=3, offset=3. Dry at t≡3,4,5 (mod 6). Exactly complementary.

Crossing protocol: enter A at t≡0 → walk to target column on A during t≡1,2 → step onto B at t≡3 (B just opened, A just flooded) → exit B at t≡4. Total ~5 steps if positioned correctly.

- **Bait**: Agent chases B1 (top-right) first, arrives at Zone A at column 5, at t≡2 (last dry tick). Steps onto A — but needs to traverse to a better column for the B-side exit. At t≡3, A floods — agent must be on B already. If at the wrong column, the exit from B deposits the agent far from B2/exit → extra oxygen spent repositioning → insufficient for the rest.
- **Correct**: Go directly to the optimal crossing column first → enter A at t≡0 → traverse to target column → step onto B at t≡3 → exit near B2 → collect B2 → exit. Collect B1 before crossing (it's on the top side).
- **Fatal step**: Entering Zone A at the wrong column or time — the phase flip is unforgiving. Being at the wrong place when the swap happens wastes an entire cycle (6 turns).
- **Temporal skill**: Phase transition precision — must be in exactly the right position at exactly the right timestep to catch the handoff between complementary zones. The tightest timing constraint of any template.

**RQ value**:
- **RQ1**: Must simulate being ON a tidal tile when the phase flips — one tick on dry land, next tick the agent must have moved to the adjacent zone. Requires precise step-by-step tracking of two complementary schedules
- **RQ2**: Standing on Zone A during a dry tick looks perfectly safe. The deadlock is positional: "wrong column when the phase flips, so I can't reach B in time." Invisible danger
- **RQ3**: Fatal step is entering Zone A at the wrong column. Consequence (stranded between zones, wasting a full cycle) appears 2-3 steps later. **Small cause-symptom gap** — tests precise attribution in tight-timing scenarios

#### Template diversity matrix

| Template | Beacons | Zones | Bait mechanism | Temporal skill tested | Cause-symptom gap |
|----------|---------|-------|---------------|----------------------|-------------------|
| **Strait** (done) | 2 | 2 | Wrong crossing (spatial) | Zone selection | Medium |
| **Tidal Maze** (done) | 2 | 2 | Wrong crossing (spatial) | Zone selection | Medium |
| **Gauntlet** (done) | 2 | 3 | Cascading flood timing | Sequential phase chaining | Large |
| **Roundtrip** (done) | 2 | 1 | One-way thinking | Bidirectional timing | Medium |
| **Confluence** (done) | 2 | 2 | First-available zone | Schedule intersection (LCM) | Medium |
| **Patience** (done) | 2 | 1 | Moving immediately | WAIT-optimality | Very large |
| **Switchback** (done) | 2 | 3 | Cascading flood timing (reversed zigzag) | Multi-gate interference | Large |
| **Tidal Bridge** (done) | 2 | 1 (3 tiles) | Late bridge entry | Duration-aware crossing | Small |
| **Deadline** (done) | 3 | 3 | Nearest-first priority | Urgency scheduling (EDF) | Large |
| **Pendulum** (done) | 3 | 1 (crossed 3x) | Wrong crossing time | Modular arithmetic | Medium-large |
| **Tide Pool** (done) | 3 | 1 (large area) | Wrong beacon batch | Window allocation | Medium |
| **Airlock** (done) | 2 | 2 (complementary) | Wrong entry column | Phase transition precision | Small |

All 12 templates are implemented, spanning 12 distinct temporal reasoning skills. Templates 9, 10 and 11 use 3 beacons, adding permutation complexity and richer state spaces for RQ1 simulation.

### Dataset serialization (`dataset.py`) — DONE

`dataset.py` provides full JSON serialization, fingerprinting, and parallel generation orchestration. Follows `puzzles/chain_vault/dataset.py` patterns exactly, adapted for tide_grid types.

**Serialization helpers:**
- `_pos_to_list` / `_list_to_pos` — `Pos` ↔ `[row, col]`
- `_posset_to_sorted` / `_sorted_to_posset` — `frozenset[Pos]` ↔ sorted list
- `_zone_to_dict` / `_dict_to_zone` — `TidalZone` ↔ dict (zone_id, tiles, dry_duration, flood_duration, phase_offset)
- `_beacon_to_dict` / `_dict_to_beacon` — `Beacon` ↔ dict (beacon_id, pos)
- `_grid_to_dict` / `_dict_to_grid` — `TideGridGrid` ↔ dict (width, height, walls, tidal_zones, beacons, start, exit_pos, initial_oxygen)
- `_state_to_dict` / `_dict_to_state` — `TideGridState` ↔ dict (agent_pos, time, collected_beacons)
- `_deadlock_to_dict` / `_dict_to_deadlock` — `DeadlockEntry` ↔ dict (path, fatal_step, deadlock_state, pre_deadlock_state, k_horizon, strategy_name)
- `instance_to_dict` / `dict_to_instance` — full `GeneratedInstance` round-trip with `.get()` defaults for backward compat

**Fingerprinting:** `compute_fingerprint(inst)` — SHA-256 (16 hex chars) of canonical grid: width, height, walls (linear indices), start, exit, zones (tiles + phase params), beacons (pos + id), initial_oxygen.

**Parallel orchestration:**
- `_worker(task)` — multiprocessing worker with fingerprint computation + eval-readiness check
- `generate_dataset(...)` — fixed seeds-per-config, all templates × all sizes (no tier filtering — generators self-filter by returning None for undersized grids), resume support, fingerprint deduplication
- `generate_to_target(...)` — target-driven seed-by-seed generation with stall detection + resume

**Key difference from chain_vault:** No `TEMPLATE_TIERS` / `TIER_GRID_SIZES` — all templates × all sizes are attempted; generators return None for undersized grids. Metadata uses `num_beacons`, `num_zones` instead of `num_keys`, `num_doors`.

### Eval build pipeline (`build_eval.py`) — DONE

`build_eval.py` converts raw JSONL into per-puzzle evaluation directories with prompts, ground truth, and rendered images. Follows `puzzles/chain_vault/build_eval.py` patterns exactly, adapted for tide_grid types.

**Helpers:**
- `_puzzle_dir_name(inst)` — `"{template}_{W}x{H}_{num_beacons}b_{num_zones}z_seed{seed}"`
- `_ground_truth_state(grid, state)` — `{agent_pos, time, oxygen_remaining, collected_beacons, zone_states}` where `zone_states = {zone_id: "dry"/"flooded"}`

**Per-RQ builders (same structure as chain_vault):**
- `_build_rq1_entries` — oracle prefixes at [5,10,15] + full, ablation legality check per direction, RQ2/RQ3 baselines
- `_build_rq2_entries` — up to 3 deadlocks (k_horizon≥3, fatal_step≥2), 4 items each: main_deadlock, main_solvable, ablation_deadlock, ablation_solvable
- `_build_rq3_entries` — up to 2 tier-2 deadlocks (k_horizon≥5), 3 items each: main, ablation_final, ablation_comparison
- `_build_hard_rq2_entries` / `_build_hard_rq3_entries` — oracle-adjacent variants, same structure

**RQ1 ablation legality check:** Uses `get_legal_actions(grid, state)` to check each direction (time-dependent, unlike chain_vault's static `is_passable`).

**Images rendered per puzzle:** `initial.png`, `solution_trace.png`, `rq2_state_e{i}.png`, `rq3_trace_e{i}.png`, `rq3_final_e{i}.png`, `rq3_comparison_e{i}.png`, hard variants with `_hard_` prefix.

**Orchestrator:** `build_eval_dataset(...)` — parallel build, carry-forward on skip, stratified sampling by `{template}__{size}__{beacons}b_{zones}z`, eval_index.jsonl, cleanup of unused puzzle dirs.

### CLI subcommands — DONE

Extended `__main__.py` from demo-only (34 lines) to full CLI (207 lines) with 4 subcommands:

| Command | Description |
|---------|-------------|
| `demo` | Existing — no change |
| `generate` | `--templates`, `--grid-sizes` (default "6,7,8"), `--seeds-per-config`, `--max-retries` (200), `--workers`, `--output`, `--resume`, `--seed-start` |
| `build` | `--input`, `--output`, `--workers`, `--rq1-lengths`, `--overwrite`, `--puzzle-target`, `--sample-seed`, `--require-hard` |
| `full` | Combines generate+build; single-pass or target-driven mode with `--puzzle-target`, `--max-seeds`, `--stall-seeds` |

Default grid sizes `[6,7,8]` (no 5x5 yet). Default output: `"tide_grid_dataset.jsonl"`. Default max-retries: 200.

### Grid size variation (P1) — DONE

Tier-based scaling system bringing tide_grid to parity with chain_vault's mature `TEMPLATE_TIERS` / `TIER_GRID_SIZES` / `*_BY_SIZE` architecture.

**Constants added to `generator.py`:**
- `TEMPLATE_TIERS` — maps each template to simple/medium/complex
- `TIER_GRID_SIZES` — simple: 6-8, medium: 7-8, complex: 7-8
- `TEMPLATE_SOLUTION_RANGES_BY_SIZE` — per-template, per-size `(min, max)` solution length bounds
- `TEMPLATE_MIN_DEADLOCKS_BY_SIZE` — per-template, per-size minimum deadlock count

**`_generate_validated_inner` changes:**
- Per-size solution range and deadlock minimum lookups (with flat-table fallback)
- Auto-scaling retries: `max(max_retries, 300)` at 7+, `max(max_retries, 500)` at 8+
- Size-dependent oxygen slack: `randint(2, 5)` at 6, `randint(3, 6)` at 7, `randint(3, 7)` at 8

**`dataset.py` tier filtering:**
- Both `generate_dataset` and `generate_to_target` skip size/template combos not in `TIER_GRID_SIZES`
- Beacon scaling is natural: 3-beacon templates (deadline, tide_pool) are complex tier (7+ only), 2-beacon templates span 6-8

**Tier rationale:**
| Tier | Templates | Sizes | Why |
|------|-----------|-------|-----|
| Simple | strait, roundtrip, patience, pendulum, airlock | 6, 7, 8 | 1-2 zones, straightforward barriers, work at all sizes |
| Medium | tidal_maze, confluence, switchback, tidal_bridge | 7, 8 | Multi-zone interactions need room for zone placement |
| Complex | gauntlet, deadline, tide_pool | 7, 8 | 3 zones/beacons, cascading mechanics need larger grids |

### Stochastic template-specific scaling (P2) — DONE

Per-template coin-flip (50%) for enhanced variants at larger grid sizes, following chain_vault's pattern. Four independent bait-safe enhancements that modify zone parameters without touching template bait logic:

| ID | Enhancement | Effect | Applies to |
|----|-------------|--------|------------|
| E1 | Reduced oxygen slack | Tighter `randint` ranges (e.g., +2..4 instead of +3..6 at size 7) | All 12 templates |
| E2 | Longer zone cycles | 1.5-2x scaling via `_scale_cycle_params`; bigger numbers to track | All 12 templates |
| E3 | Coprime zone cycles | `_ensure_coprime` forces gcd=1; LLM must reason about LCM | 7 multi-zone templates |
| E4 | Similar zone parameters | Copies zone A params to B/C; zones look alike but differ in offset | 7 multi-zone templates |

All gated by `width >= 7 and rng.random() < 0.5`. Size 6 grids unchanged. Coin-flips consumed inside template functions for RNG consistency, not persisted on instances.

**Implementation details:**
- Helpers: `_scale_cycle_params(dry, flood, rng)` and `_ensure_coprime(flood, dry, other_cycles)` with `from math import gcd`
- Category 1 (direct offset algebra): strait, tidal_maze, gauntlet, switchback — E2 scales raw samples, E4 copies zone A, E3 enforces coprime, offset algebra adapts automatically
- Category 2 (brute-force search): roundtrip, patience, pendulum, tide_pool — E2 extends search ranges; confluence — E4/E3 add `continue` filters inside nested search
- Category 3 (special structure): tidal_bridge — E2 scales dry_a (min 3), `target_phase = dry_a - 2`; deadline — E2 scales flood only, E4 copies C→B; airlock — E2 extends `d_values` to [3..7]

---

## Review-Driven Improvements (P4)

External review of the Tide Grid benchmark identified several concrete issues and architectural improvements. Items are prioritized by impact on dataset quality and paper readiness.

### Priority list

| # | Item | Priority | Effort | Status |
|---|------|----------|--------|--------|
| P4.1 | Fix `target_length=18` in hard/standard RQ3 | Critical | Small | DONE |
| P4.2 | Separate standard-ready from hard-ready | High | Medium | DONE |
| P4.3 | Add per-template generation caps | High | Small | DONE |
| P4.4 | Merge template families for paper presentation | Medium | Small | DONE (documented) |
| P4.5 | Document RQ2 standard vs hard structural difference | Medium | Small | DONE (documented) |
| P4.6 | Per-family uniqueness checks | Medium | Medium | DONE |
| P4.7 | Smarter Walking Dead continuation for hard RQ3 | Low | Medium | DONE |

### P4.1: Fix `target_length=18` — CRITICAL — DONE

**Bug:** `generate_hard_rq3_instance` (`generator.py:1432`) and `generate_rq3_instance` (`generator.py:4065`) both hardcode `target_length=18`. The continuation check:
```python
continuation_length = target_length - len(trace_to_deadlock)
if continuation_length < 3:
    return None
```
silently drops any instance where `fatal_step + 1 > 15`. Gauntlet, deadline, and switchback all have solution ranges starting at 15, so most oracle-adjacent deadlocks at late steps return `None`. This is a direct cause of gauntlet producing 0 hard RQ3 instances.

**Fix:** Replace `target_length: int = 18` with `target_length: int = 0` and compute dynamically:
```python
if target_length == 0:
    target_length = max(18, fatal_step + 6)
```
in both `generate_hard_rq3_instance` and `generate_rq3_instance`. The `+6` guarantees at least 6 Walking Dead continuation steps (well above the `>= 3` minimum), while preserving the existing default for short-solution templates.

**Scope:** Two lines changed in `generator.py`.

### P4.2: Separate standard-ready from hard-ready — HIGH — DONE

**Problem:** `build_eval.py:779` drops puzzles entirely if they lack hard RQ2/RQ3 variants when `require_hard=True`. This gates the *entire raw pool* on oracle-adjacent deadlock availability, which is a structural fit for point-of-no-return templates (tidal_bridge, airlock, confluence) but not for schedule-based templates (gauntlet, deadline, tide_pool).

**Fix:** Two-tier eval readiness:
1. Generate with standard filters only — keep all instances that pass standard RQ2/RQ3.
2. At build time, tag each puzzle with `has_hard: true/false`. Include all puzzles in the dataset; hard-only analyses filter by tag.
3. `require_hard` becomes a *reporting filter*, not a *generation gate*.

**Impact:** Recovers gauntlet, deadline, and tide_pool instances that currently pass standard RQ2/RQ3 but fail hard. Prevents the dataset from skewing toward templates where one wrong step immediately creates Walking Dead.

**Scope:** Changes to `build_eval.py:_build_worker` (keep puzzle, tag instead of drop) and `is_eval_ready` in `generator.py` (standard-only check during generation).

### P4.3: Add per-template generation caps — HIGH — DONE

**Problem:** `generate_to_target` (`dataset.py:401-608`) has no per-template caps. Templates with higher yield rates (tidal_bridge, airlock, confluence — which naturally produce oracle-adjacent deadlocks) dominate the accepted pool. Post-hoc sampling via `_allocate_and_sample` in `build_eval.py` partially addresses this but wastes compute generating instances that will be discarded.

**Fix:** During `generate_to_target`, track accepted count per template. Skip a template's tasks in a seed batch once it reaches `ceil(puzzle_target / num_active_templates)`. This front-loads diversity instead of relying only on post-hoc rebalancing.

**Scope:** ~15 lines in `dataset.py:generate_to_target`.

### P4.4: Merge template families for paper — MEDIUM

**Observation:** Some templates are structurally identical and differ only in parameter choices:
- **roundtrip** and **patience**: Both use 1 zone, 1 crossing, 2 beacons, identical layout. Patience has shorter dry windows to force WAIT-optimality. Same spatial structure, different temporal skill.
- **switchback** and **gauntlet**: Both use 3 zones, 3 barriers, zigzag crossings. Only differ in zigzag direction (L-R-L vs R-L-R).

Reporting these as separate families in the paper invites reviewer criticism about inflated template counts.

**Fix:** Keep all code as-is. For paper presentation, group into 8 reasoning families:

| Family | Templates included | Temporal skill |
|--------|-------------------|----------------|
| Tidal Bridge | tidal_bridge | Duration-aware crossing |
| Airlock | airlock | Phase transition precision |
| Confluence | confluence | Schedule intersection |
| Patience | patience, roundtrip | WAIT-optimality / bidirectional timing |
| Pendulum | pendulum | Modular crossing arithmetic |
| Deadline | deadline | Urgency scheduling (EDF) |
| Gauntlet | gauntlet, switchback | Sequential phase chaining |
| Tide Pool | tide_pool | Window allocation |

Demote **strait** to support/demo only (too simple, dominated by other 2-zone templates). Keep **tidal_maze** only if yield/diversity improves after P4.1/P4.2 fixes.

**Scope:** Paper-level change; no code modifications needed.

### P4.5: Document RQ2 standard vs hard structural difference — MEDIUM

**Observation:** Standard RQ2 and hard RQ2 have different internal structures:
- **Standard RQ2** (`build_eval.py:272-386`): The deadlock and solvable action sequences are *not* a shared prefix. The deadlock side replays a greedy trace; the solvable side is a fresh `oracle_solve` from the same start state. Matched by start state only.
- **Hard RQ2** (`build_eval.py:495-612`): Both sides share the oracle prefix up to `fatal_step`, then diverge on the last action. True minimal-pair design.

This is by design, not a bug. But it means standard and hard RQ2 test different things (greedy-vs-oracle divergence vs single-step deviation), and results should be reported separately.

**Fix:** Add a note in the paper methods section and in prompt documentation. No code change.

### P4.6: Per-family uniqueness checks — MEDIUM — DONE

**Problem:** Current deduplication is global fingerprint-only (`dataset.py:380`). Two instances from the same template family can be structurally very similar (same zone timings, similar geometry) but have different fingerprints due to minor wall/beacon position changes.

Templates most at risk of repetitive outputs:
- **airlock** — limited layout variation in the complementary-zone arrangement
- **pendulum** — no walls at all, so geometry is almost entirely determined by beacon positions
- **deadline** — rigid 3-pocket structure
- **patience** — single-crossing structure constrains layout

**Fix:** Add a secondary dedup pass during `build_eval.py` sampling that checks within-family structural similarity (e.g., same zone parameter tuple, same beacon count, similar solution length). Cap per-family instances during `_allocate_and_sample`.

**Scope:** ~30 lines in `build_eval.py`.

### P4.7: Smarter Walking Dead continuation for hard RQ3 — DONE

**Problem:** Oracle-adjacent hard RQ3 requires one wrong step that produces Walking Dead, followed by a continuation of legal-but-futile moves. The original inline random walk often got stuck after 1-2 steps (especially for tide_pool, where the agent gets stranded inside a flooding zone). This made many structurally valid deadlocks fail the `cause_symptom_gap >= 3` requirement.

**Root cause:** The sequential random walk had no awareness of tile types — it would walk straight back into tidal tiles that were about to flood, getting trapped immediately. BFS-reachable states (k_horizon) could be high while the sequential walk consistently failed.

**Fix implemented** (template-agnostic, in `generator.py`):
1. **`_walking_dead_continuation()` helper** — Multi-seed, floor-preferring random walk:
   - Tries 10 different RNG seeds per deadlock state
   - Prefers moves to non-tidal tiles (FLOOR, EXIT) over tidal tiles
   - Returns the longest continuation found across all seeds
   - Used by both `generate_hard_rq3_instance` and `generate_rq3_instance`
2. **Increased eligible entry limit** from `[:2]` to `[:5]` in both `is_eval_ready` and `_build_hard_rq3_entries`, giving more deadlock candidates a chance

**Yield after fix** (50 seeds per template):
| Template | std_rq3 | hard_rq3 | hard_rq2 |
|----------|---------|----------|----------|
| tide_pool | 58% | 14% | 100% |
| gauntlet | 100% | 32% | 32% |
| deadline | 100% | 100% | 100% |
| tidal_bridge | 100% | 42% | 42% |
| airlock | 100% | 100% | 100% |
| confluence | 100% | 42% | 42% |

**Scope:** ~40 lines new helper + minor edits in `generator.py`, 1-line change in `build_eval.py`.

---

## What We Don't Need

| Feature | Reason |
|---------|--------|
| **Dynamic zone schedules** | Phase offsets already provide sufficient variety; making schedules change mid-puzzle adds complexity without improving bait quality |
| **Grids > 8x8** | A* state space explodes with larger grids × more zones × oxygen; 8x8 is already at the edge of fast generation |
| **More than 3 beacons** | Each beacon doubles the state space (collected/not). 3 beacons = 8 states × positions × time × oxygen; 4+ makes oracle prohibitively slow |
| **11+ greedy strategies** | 10 strategies (6 basic + 4 smart) cover the main decision axes: distance, path, timing, avoidance, urgency, phase-aware planning, oxygen budgeting, reverse ordering, and dynamic BFS. Additional strategies would add filter cost without improving bait quality |
| **Breakable walls** | Adds a second resource axis (tools + oxygen) that complicates the timing-based bait mechanism without clear evaluation benefit |
| **Multi-agent** | Coordination reasoning is a separate capability from tidal-phase planning; would need a different puzzle design |
| **Variable oxygen costs** | Moving through flooded zones could cost extra oxygen, but this obscures the clean "timing vs distance" tradeoff that makes the bait mechanism legible |
