# Mosaic Grid — "The Pathmaker's Dilemma"

**Reasoning Dimension:** Spatial-Topological
**Core Tension:** Painting tiles activates required mechanisms AND permanently blocks future traversal.

---

## 1. Game Rules

### Overview

The agent navigates a grid, leaving a permanent paint trail behind it. Certain tiles are **Pressure Plates** — when the agent paints them (by stepping on them), they activate linked mechanisms (open doors, extend bridges). The agent must activate all required plates and reach the exit.

**The catch:** Every tile the agent visits becomes painted and impassable. The same paint that activates plates also severs paths. The agent must find a non-self-intersecting route that visits plates in the correct order and still reaches the exit.

### Formal Rules

1. **Grid:** Rectangular grid (W × H) of tiles. Each tile is one of:
   - `FLOOR` — walkable, becomes `PAINTED` after agent leaves
   - `WALL` — impassable, permanent
   - `PAINTED` — impassable, permanent (created by agent's trail)
   - `PLATE(id)` — walkable, becomes `PAINTED` after agent leaves, triggers activation of linked mechanism
   - `DOOR(id, state)` — impassable when `CLOSED`, walkable when `OPEN`. Linked to one or more plates.
   - `EXIT` — goal tile. Agent wins by stepping onto this tile.

2. **Agent Movement:**
   - Agent occupies exactly one tile at a time.
   - Legal moves: UP, DOWN, LEFT, RIGHT (4-directional).
   - Agent can move onto: `FLOOR`, `PLATE`, `OPEN DOOR`, `EXIT`.
   - Agent CANNOT move onto: `WALL`, `PAINTED`, `CLOSED DOOR`.
   - **On leaving a tile:** The tile the agent just left becomes `PAINTED` (unless it's the EXIT).
   - The tile the agent is currently standing on is NOT yet painted — painting happens on departure.

3. **Plate Activation:**
   - When the agent steps onto a `PLATE(id)` tile, the plate's linked mechanism activates immediately.
   - Activation effects: `DOOR(linked_id)` changes from `CLOSED` to `OPEN`.
   - Plates are single-activation (stepping on them once is enough; they stay active even after being painted).
   - A plate can be linked to one or more doors. A door can require one or more plates (AND logic: all linked plates must be activated for the door to open).

4. **Win Condition:** Agent steps onto the `EXIT` tile.

5. **Deadlock Condition:** The `EXIT` tile is unreachable from the agent's current position given painted tiles and door states. Formally: no non-self-intersecting path exists from agent to EXIT that visits all remaining required plates (for unopened doors blocking the path to EXIT).

---

## 2. State Representation

```python
@dataclass
class MosaicState:
    agent_pos: Tuple[int, int]          # (row, col)
    painted: FrozenSet[Tuple[int, int]] # set of painted tile coordinates
    activated_plates: FrozenSet[str]     # set of plate IDs that have been activated
    # Door states are derived: door is OPEN iff all its required plates are in activated_plates
```

**Derived properties:**
- `door_open(door_id)` → `all(plate in activated_plates for plate in door.required_plates)`
- `tile_passable(pos)` → `grid[pos] not in {WALL, PAINTED, CLOSED_DOOR}`
- `is_solved` → `agent_pos == exit_pos`

**State space size:** |positions| × 2^|paintable_tiles| × 2^|plates|

For a 5×5 grid with ~16 walkable tiles and 2 plates: 25 × 2^16 × 2^2 ≈ 6.6M states. Feasible with A*.

For an 8×8 grid with ~50 walkable tiles and 6 plates: state space is enormous in theory, but A* with an admissible heuristic explores only a small frontier. All grid sizes 5x5 through 8x8 are tractable.

**Grid sizes used:** 5x5, 6x6, 7x7, 8x8. Plates auto-scale (2→3→4→6).

---

## 3. Action Space

```python
class Action(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
```

**Legal action check:**
```
def is_legal(state, action):
    new_pos = state.agent_pos + action.value
    return (in_bounds(new_pos) and
            grid[new_pos] != WALL and
            new_pos not in state.painted and
            (grid[new_pos] != DOOR or door_open(new_pos, state)))
```

**State transition:**
```
def apply_action(state, action):
    new_pos = state.agent_pos + action.value
    new_painted = state.painted | {state.agent_pos}  # paint the tile we're leaving
    new_activated = state.activated_plates
    if grid[new_pos] is PLATE(id):
        new_activated = new_activated | {id}
    return MosaicState(new_pos, new_painted, new_activated)
```

---

## 4. Oracle Implementation

### Algorithm: A* Search

We use **A* search** (not BFS) throughout the oracle. A* with an admissible heuristic guarantees optimal solutions while exploring far fewer states than BFS. Since `is_solvable` is called millions of times during instance generation, the speedup is critical.

### Admissible Heuristic for Mosaic Grid

```
def heuristic(state):
    """
    Admissible heuristic: lower bound on steps to reach EXIT.
    h = manhattan_distance(agent, exit)
        + number_of_unactivated_required_plates

    Why admissible: each unactivated plate requires at least 1 additional step
    to reach and activate. Manhattan distance is always ≤ true path distance.
    The sum never overestimates the true cost.
    """
    h = manhattan(state.agent_pos, exit_pos)
    remaining_plates = required_plates - state.activated_plates
    h += len(remaining_plates)
    return h
```

**Tighter heuristic (optional optimization):**
```
def heuristic_tight(state):
    """
    Sum of minimum detour distances: for each unactivated plate,
    add the minimum extra distance to visit it on the way to exit.
    More expensive to compute but prunes much more aggressively.
    """
    remaining = list(required_plates - state.activated_plates)
    if not remaining:
        return manhattan(state.agent_pos, exit_pos)
    # Greedy TSP lower bound: nearest-neighbor through remaining plates then exit
    pos = state.agent_pos
    total = 0
    unvisited = set(remaining)
    while unvisited:
        nearest = min(unvisited, key=lambda p: manhattan(pos, plate_positions[p]))
        total += manhattan(pos, plate_positions[nearest])
        pos = plate_positions[nearest]
        unvisited.remove(nearest)
    total += manhattan(pos, exit_pos)
    return total
```

### is_solvable(state) → bool

A* search from current state. Returns True if EXIT is reachable.

```
def is_solvable(state):
    # Priority queue: (f_score, state)
    # f = g (steps taken) + h (heuristic)
    open_set = [(heuristic(state), 0, state)]  # (f, g, state)
    visited = {state.hashable()}

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current.agent_pos == exit_pos:
            return True
        for action in Action:
            if is_legal(current, action):
                next_state = apply_action(current, action)
                h = next_state.hashable()
                if h not in visited:
                    visited.add(h)
                    next_g = g + 1
                    next_f = next_g + heuristic(next_state)
                    heapq.heappush(open_set, (next_f, next_g, next_state))
    return False
```

### oracle_solve(state) → List[Action]

A* to find the **optimal** (shortest) solution path. Same as `is_solvable` but tracks the path.

```
def oracle_solve(state):
    open_set = [(heuristic(state), 0, state, [])]  # (f, g, state, path)
    visited = {state.hashable()}

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current.agent_pos == exit_pos:
            return path  # optimal solution
        for action in Action:
            if is_legal(current, action):
                next_state = apply_action(current, action)
                h = next_state.hashable()
                if h not in visited:
                    visited.add(h)
                    heapq.heappush(open_set, (g + 1 + heuristic(next_state),
                                               g + 1, next_state, path + [action]))
    return None  # unsolvable
```

### is_walking_dead(state) → bool

For RQ2, we specifically need **Walking Dead** states — the agent has legal moves but can't win. Physically stuck states (no legal moves) are trivially detectable and uninteresting.

```
def is_walking_dead(state):
    has_legal_moves = any(is_legal(state, a) for a in Action)
    is_unsolvable = not is_solvable(state)
    return has_legal_moves and is_unsolvable
```

**All RQ2 deadlock instances must satisfy `is_walking_dead(S_k) == True`.** Instances where the agent is physically stuck after k actions are discarded during generation.

### apply_actions(state, actions) → State

Sequential application: fold `apply_action` over the action list.

### find_fatal_step(trace) → int

Iterate through the trace. At each step, run `is_solvable()` on the resulting state. Return the first index where `is_solvable` flips from True to False.

```
def find_fatal_step(initial_state, actions):
    state = initial_state
    for i, action in enumerate(actions):
        state = apply_action(state, action)
        if not is_solvable(state):
            return i  # this action caused the deadlock
    return -1  # no deadlock in trace
```

### State Representation for Fast Hashing

Use bitmask representation for the painted set to make hashing O(1) instead of O(n):

```
# Assign each walkable tile an index 0..N-1
tile_index = {(r, c): i for i, (r, c) in enumerate(walkable_tiles)}

# Painted set as bitmask: painted_bits |= (1 << tile_index[(r, c)])
# Hashable state: (agent_pos, painted_bits, activated_bits)
# Comparison is a single integer comparison — very fast.
```

### Computational Budget

Generation time is dominated by deadlock discovery (repeated A* solvability checks). Per-instance times vary widely by grid size and template:

- **5x5:** <0.2s per instance (all templates)
- **6x6:** 0.01–1.3s per instance
- **7x7:** 0.09–25s per instance
- **8x8:** 0.35–100s per instance (slowest: island, peninsula)

Generation is a one-time cost. The parallel CLI (`--workers`) distributes across CPU cores.

---

## 5. Procedural Generation

### Instance Requirements

Every generated instance must satisfy:
1. **Solvable:** Oracle A* finds a solution from initial state to exit.
2. **Greedy-proof:** All 5 greedy strategies fail (3 manhattan-based + 2 BFS-based with 8 tiebreaking orders).
3. **Deadlock-rich:** At least 2 deadlocks with k-horizon >= 3, at least 1 with k-horizon >= 5.
4. **Doors required:** Every door lies on the mandatory start→exit path (BFS reachability check).
5. **Solution in range:** Solution length within grid-scaled bounds.

### Generation Algorithm

Generation uses 12 structural templates (see `docs/TEMPLATES.md`), each defining a wall layout with a specific planning challenge. The pipeline:

1. **Template generates grid:** Each template places walls, start, exit, plates, and doors according to its structural pattern. Parameters are randomized per seed (wall positions, plate placement, orientation flips).
2. **8-filter validation:** The candidate must pass all filters in order: grid valid, door count correct, solvable, doors required, solution in range, greedy-proof, deadlock-rich (tier-1), deadlock-rich (tier-2).
3. **Deadlock discovery:** Forward sweep from initial state. At each reachable state, check `is_solvable()`. Collect all solvability-flip transitions (Walking Dead states) and compute k-horizon per deadlock.
4. **Retry with new seed:** If any filter fails, increment seed and retry (up to `--max-retries`).

### Difficulty Scaling

Difficulty scales with four knobs that increase together:

| Grid | Plates | Doors | Solution range |
|:-----|:-------|:------|:---------------|
| 5x5  | 2      | 1     | 6–20           |
| 6x6  | 3      | 1–2   | 8–30           |
| 7x7  | 4      | 1–2   | 10–40          |
| 8x8  | 6      | 1–3   | 10–40          |

See `README.md` for the full CLI reference and dataset generation recipe.

---

## 6. Visual Rendering Spec

### Tile Sprites

| Tile Type | Visual | Color |
|:---|:---|:---|
| FLOOR | Empty square | Light gray |
| WALL | Solid filled square | Dark gray / Black |
| PAINTED | Square with paint texture/cross-hatch | Blue or teal (the agent's color) |
| PLATE (inactive) | Square with circle/button symbol | Yellow with dark border |
| PLATE (activated + painted) | Painted square with dimmed circle | Blue with faded yellow circle |
| DOOR (closed) | Square with door icon / bars | Red |
| DOOR (open) | Empty square with doorframe hint | Light green border |
| EXIT | Square with flag/star | Gold / Green |
| AGENT | Circle or arrow indicating last move direction | Bright blue |

### Layout Requirements
- Grid cells should be square, minimum 40×40 pixels for clarity.
- Plate-to-door links shown via matching labels: "P1 → D1" or matching colors.
- Current paint trail should be clearly visible as a connected path.
- A small legend/key explaining symbols should be included in the image (or in the text prompt).

---

## 7. RQ-Specific Experimental Protocols

### RQ1: Simulation Fidelity

**Setup:** Provide initial state image + sequence of n actions.
**Task:** "Predict the exact positions of the agent, which tiles are painted, and which doors are open after these n actions."
**Verification:** Compare predicted (agent_pos, painted_set, activated_plates) against Oracle ground truth.
**Metric:** Coordinate Exact Match (CEM) — percentage of state variables correctly predicted.

**What makes Mosaic Grid interesting for RQ1:**
- Every step changes the world (new painted tile). No "idle" steps.
- Plate activations cause non-local state changes (a door elsewhere opens).
- Models must track a growing set (painted tiles) + derived state (door openness from plate combinations).

### RQ2: Foresight

**Main (Natural Foresight):**
- Provide initial state image + k actions. 50% lead to deadlock, 50% keep puzzle solvable.
- Ask: "What is the next move to win the game?"
- On deadlock instances: model should refuse / indicate unsolvable.
- On solvable instances: model should provide a valid move.

**Ablation (Explicit Solvability):**
- Same initial state image + k actions.
- Ask: "Is the puzzle still solvable? YES or NO."

**What makes Mosaic Grid interesting for RQ2:**
- The board looks open — many unpainted tiles remain.
- The deadlock is topological: the agent's own trail has partitioned the grid.
- A model must perform graph connectivity analysis to detect the deadlock.
- At k=1-2, the partition might be obvious. At k=5+, the trail is complex and connectivity is hard to assess visually.

### RQ3: Causal Attribution

**Setup:** Provide initial state image + full trace (start → deadlock). Oracle identifies the fatal step T.
**Task:** "This sequence led to an unsolvable state. Identify the last step at which the puzzle was still solvable."
**Metric:** |T_model - T_oracle| (Culprit Localization Error)

**What makes Mosaic Grid interesting for RQ3:**
- The fatal step is often the one where the agent painted through a critical corridor, severing access to a required plate.
- But the agent doesn't get "stuck" until many steps later (when it tries to reach the now-inaccessible plate).
- The temporal distance between cause (painting the corridor) and symptom (failing to reach the plate) creates the diagnostic challenge.
- We expect Recency Bias: models will point to the step where the agent physically can't proceed, not the earlier step where the corridor was severed.

---

## 8. Edge Cases and Design Considerations

### The "Painting on Departure" Rule
The agent paints the tile it LEAVES, not the tile it enters. This means:
- The agent's current tile is unpainted (it's standing on it).
- If the agent backtracks immediately (step left then step right), the middle tile gets painted and blocks further traversal.
- This creates a natural tension: every exploration step is a commitment.

### Plates on Required Corridors
The most interesting deadlocks occur when a plate is located ON a corridor that's needed for future navigation. Activating the plate (required) paints the corridor (potentially fatal). The generation algorithm should prioritize placing plates at or near chokepoints.

### Door AND-Logic
If a door requires multiple plates, the agent must visit all of them before the door opens. This creates multi-checkpoint ordering constraints — the richest deadlock type for this puzzle.

### Dead-on-arrival Prevention
Some paint patterns might make the exit unreachable regardless of plate state. The oracle detects these, but generation should aim for deadlocks that are CAUSED by painting decisions, not by initial grid layout.

---

## 9. Demo

The demo renders PNG images for all 12 templates across multiple configurations (grid sizes, plate counts, door counts). For each template it generates:

- **Initial state** — the puzzle before any moves
- **Oracle solution trace** — step-by-step optimal solution
- **RQ1 demo** — initial vs. post-action state (side-by-side + trace)
- **RQ2 demo** — matched deadlock/solvable pair after k actions
- **RQ3 demo** — full trace with fatal step highlighted

Run with: `python -m puzzles.mosaic_grid demo`

Output goes to `demo_output/`. See `demo_output/README.md` for the full directory structure and file descriptions.

---

## 10. Instance Generation Pipeline

### Philosophy: Deadlock-First, Then Construct Instances

Instead of generating random action sequences and hoping they hit deadlocks (low hit rate), we **find the deadlocks first** and then construct RQ instances around them. Two-phase approach:

1. **Phase A — Grid generation + deadlock discovery:** Template generates a grid, oracle validates it, forward sweep catalogs all reachable Walking Dead states and the paths that reach them.
2. **Phase B — Instance construction:** The deadlock catalog feeds RQ2 (matched deadlock/solvable pairs) and RQ3 (full traces with cause-symptom gap). RQ1 uses the oracle solution directly.

### Deadlock Discovery

Forward sweep from S₀. At each reachable state, check `is_solvable()`. Collect all transitions where solvability flips True→False. Key optimizations:

- **Monotonicity pruning:** Painted tiles only grow → once unsolvable, always unsolvable. Children of Walking Dead states skip the oracle call entirely.
- **k-horizon:** For each deadlock, count reachable dead states (how many moves the agent can still make while doomed). Filters: k >= 3 for RQ2 (Walking Dead has room to explore), k >= 5 for RQ3 (meaningful cause-symptom gap).
- **Early stop:** Stop expanding once enough deadlocks are found.

Each deadlock entry stores: `path` (actions from S₀), `fatal_step` (solvability flip index), `deadlock_state`, `pre_deadlock_state`, `k_horizon`.

### Greedy-Proof Guarantee

Every instance must defeat all 5 greedy strategies:
- **3 manhattan-based:** nearest_plate_first, toward_exit, visit_all_then_exit
- **2 BFS-based:** shortest-path-to-nearest-plate with 8 tiebreaking orders

If any greedy strategy solves the puzzle, the instance is rejected. This ensures non-trivial planning is required.

### RQ Instance Construction

- **RQ1 (Simulation):** Uses the oracle solution. The model is given the initial state image + N actions and must predict the final state.
- **RQ2 (Foresight):** Uses the deadlock catalog. The model is given the initial state image + K actions (which may or may not reach a deadlock) and must judge solvability / find the best next move.
- **RQ3 (Causal Attribution):** Uses the deadlock catalog. The model is given the initial state image + a full trace ending in deadlock and must identify the fatal step.

See `prompts.py` for the exact prompt builders (`build_rq1_prompt`, `build_rq2_main_prompt`, `build_rq3_prompt` + ablation variants). See `README.md` for the CLI and dataset generation recipe.
