# Chain Vault — "The Locksmith's Gambit"

**Reasoning Dimension:** Logical-Resource
**Core Tension:** Every key use opens a door (progress) and consumes the key (irreversible commitment). The branching dependency tree means choosing WHICH door to open with a given key determines what future keys/paths become available.

---

## 1. Game Rules

### Overview

The agent navigates a grid containing colored keys and colored doors. Keys are single-use — consumed when used to open a door. Multiple doors may share the same color, so the agent must decide WHICH door to open with each key. Behind each door lie different keys, paths, or the exit itself. The puzzle's challenge is allocating scarce keys to the correct combination of doors.

### Formal Rules

1. **Grid:** Rectangular grid (W × H) of tiles. Each tile is one of:
   - `FLOOR` — walkable
   - `WALL` — impassable
   - `KEY(color)` — walkable, contains a key of the given color. Agent picks it up on entry.
   - `DOOR(color, lock_type)` — impassable when locked. Requires key(s) to open.
   - `EXIT` — goal tile. Agent wins by stepping onto this tile.

2. **Agent Movement:**
   - Legal moves: UP, DOWN, LEFT, RIGHT.
   - Agent can move onto: `FLOOR`, `KEY`, `OPEN DOOR`, `EXIT`.
   - Agent CANNOT move onto: `WALL`, `LOCKED DOOR`.

3. **Inventory:**
   - The agent carries a **set** of keys (multi-slot inventory, no limit).
   - Stepping onto a `KEY(color)` tile picks it up automatically. The tile becomes `FLOOR`.
   - Keys are distinguishable by color only (e.g., "Red Key"), not by instance. If you have 2 Red Keys, they're interchangeable.

4. **Door Mechanics:**
   - **Single-lock door** `DOOR(color, SINGLE)`: Requires 1 key of matching color. Opens permanently. Key is consumed.
   - **Dual-lock door** `DOOR(color1+color2, DUAL)`: Requires 1 key of each color simultaneously. Opens permanently. Both keys consumed.
   - Opening a door is a deliberate action: the agent must be adjacent to the door and choose to use a key on it. Moving into a locked door is not a legal move.
   - **Open doors** become walkable `FLOOR` tiles permanently.

5. **Key Scarcity:**
   - The total number of keys of each color is strictly limited.
   - Not all doors can be opened — there are more doors than keys allow.
   - The agent must choose which subset of doors to open.

6. **Win Condition:** Agent steps onto the `EXIT` tile.

7. **Deadlock Condition:** No remaining sequence of key uses and movements can bring the agent to the EXIT. Formally: A* search over all possible key allocations from the current state finds no path to EXIT.

---

## 2. State Representation

```python
@dataclass
class ChainVaultState:
    agent_pos: Tuple[int, int]              # (row, col)
    inventory: FrozenSet[Tuple[str, int]]    # multiset of keys: {("red", 2), ("blue", 1)}
    door_states: FrozenSet[Tuple[int, int]]  # set of door positions that are OPEN
    key_tiles: FrozenSet[Tuple[int, int]]    # set of key positions not yet picked up
```

**Alternative (simpler) inventory representation:**
```python
inventory: Dict[str, int]  # {"red": 2, "blue": 1, "green": 0}
# For hashing: tuple(sorted(inventory.items()))
```

**State space size:** |positions| × |inventory_combinations| × 2^|doors| × 2^|key_tiles|

For a 5×5 grid with 4 keys (3 colors) and 5 doors:
25 × ~20 inventory combos × 2^5 × 2^4 ≈ 256,000 states. **Comfortably feasible.**

---

## 3. Action Space

```python
class Action(Enum):
    MOVE_UP = "up"
    MOVE_DOWN = "down"
    MOVE_LEFT = "left"
    MOVE_RIGHT = "right"
    USE_KEY_UP = "use_key_up"       # use key on door above agent
    USE_KEY_DOWN = "use_key_down"
    USE_KEY_LEFT = "use_key_left"
    USE_KEY_RIGHT = "use_key_right"
```

**Movement is separate from key use.** The agent must explicitly choose to use a key on an adjacent door. This is important because:
- If two doors of the same color are adjacent, the agent must choose which one.
- The "use key" action is the irreversible commitment.

**Legal action checks:**
```
def is_legal(state, action):
    if action is MOVE_*:
        new_pos = state.agent_pos + direction
        return in_bounds(new_pos) and tile_is_passable(new_pos, state)
    if action is USE_KEY_*:
        door_pos = state.agent_pos + direction
        return (in_bounds(door_pos) and
                grid[door_pos] is DOOR and
                door_pos not in state.door_states and  # door is locked
                has_required_keys(state.inventory, grid[door_pos]))
```

**State transition:**
```
def apply_action(state, action):
    if action is MOVE_*:
        new_pos = state.agent_pos + direction
        new_inventory = state.inventory
        new_key_tiles = state.key_tiles
        # Pick up key if present
        if new_pos in state.key_tiles:
            key_color = grid[new_pos].color
            new_inventory = add_key(state.inventory, key_color)
            new_key_tiles = state.key_tiles - {new_pos}
        return ChainVaultState(new_pos, new_inventory, state.door_states, new_key_tiles)

    if action is USE_KEY_*:
        door_pos = state.agent_pos + direction
        door = grid[door_pos]
        new_inventory = consume_keys(state.inventory, door.required_keys)
        new_doors = state.door_states | {door_pos}
        return ChainVaultState(state.agent_pos, new_inventory, new_doors, state.key_tiles)
```

---

## 4. Oracle Implementation

### Algorithm: A* Search

We use **A* search** throughout the oracle. A* with an admissible heuristic guarantees optimal solutions while exploring far fewer states than BFS — critical when `is_solvable` is called millions of times during generation.

### Admissible Heuristic for Chain Vault

```
def heuristic(state):
    """
    Admissible heuristic: lower bound on actions to reach EXIT.
    h = manhattan_distance(agent, exit)
    """
    return manhattan(state.agent_pos, exit_pos)
```

Using manhattan distance alone is simpler and safe (always admissible since USE_KEY doesn't move the agent, so each action reduces manhattan distance by at most 1).

### is_solvable(state) → bool

A* search from current state. Returns True if EXIT is reachable.

```
def is_solvable(state):
    open_set = [(heuristic(state), 0, state)]
    visited = {state.hashable()}

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current.agent_pos == exit_pos:
            return True
        for action in legal_actions(current):
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

A* to find the **optimal** (shortest) solution. Same search but tracks the path.

### is_walking_dead(state) → bool

For RQ2, we need **Walking Dead** states — the agent has legal moves (can walk around, maybe even pick up keys) but no sequence of actions reaches the EXIT. Physically stuck states are trivially detectable and uninteresting.

```
def is_walking_dead(state):
    has_legal_moves = any(is_legal(state, a) for a in all_actions)
    is_unsolvable = not is_solvable(state)
    return has_legal_moves and is_unsolvable
```

**All RQ2 deadlock instances must satisfy `is_walking_dead(S_k) == True`.** This is especially important for Chain Vault, where "Walking Dead" is the common failure mode — the agent can still walk around the entire map, pick up keys, even open some doors. It just can't reach the exit because the right combination of doors can no longer be opened.

### find_fatal_step(trace) → int

**Optimization:** In Chain Vault, the fatal step is almost always a `USE_KEY` action, not a movement. Movement alone never changes solvability (it doesn't consume resources or alter door states). So we can skip `is_solvable` calls after pure movement actions and only check after `USE_KEY` actions.

```
def find_fatal_step(initial_state, actions):
    state = initial_state
    for i, action in enumerate(actions):
        state = apply_action(state, action)
        # Optimization: only check after key uses (the only irreversible actions)
        if action.type == USE_KEY:
            if not is_solvable(state):
                return i
    return -1
```

### Computational Budget

- **Per `is_solvable` call:** ~0.001-0.01s with A* on 5×5 grids (~13K state space).
- **RQ3 `find_fatal_step`:** 3-4 calls per trace × 0.01s = ~0.04s per trace. 250 traces = ~10 seconds.
- **Generation pipeline:** 10,000 candidates × 5 oracle calls = 50,000 calls × 0.005s = ~4 minutes.
- **Total oracle compute:** Under 10 minutes for full dataset generation. Chain Vault has the smallest state space of the three puzzles.

---

## 5. Non-Greedy Design Principles

### Why Non-Greediness Matters

Chain Vault tests **resource allocation reasoning** — the agent must decide how to spend scarce keys on competing doors. If a greedy agent that simply moves toward the exit (or uses keys on the nearest door) can solve the puzzle, then the puzzle fails to test strategic reasoning. The model would get the right answer without ever reasoning about key budgets, door dependencies, or future consequences.

**For the benchmark to be valid, every puzzle must be unsolvable by greedy strategies.** The correct solution must require the agent to make counter-intuitive choices — going away from the exit, saving keys for later, or opening doors that don't appear to lead toward the goal.

### The Greedy-Solvability Problem

A puzzle is **greedy-solvable** when its logical structure (which door is correct) is aligned with its spatial structure (which door is closer to the exit). Common failure modes:

1. **Correct door is closer to exit than trap door:** A "move toward exit" greedy agent encounters the correct door first.
2. **Direct path happens to be the correct path:** No detour is required, so greedy movement works.
3. **Single key-use point:** With only one USE_KEY decision where the greedy choice is correct.

### Anti-Greedy Spatial Layout: The Core Principle

**The trap door must be on the spatially direct/greedy path.** The correct door must require a spatial detour.

```
WRONG (greedy-solvable):                   RIGHT (anti-greedy):

Start ──────→ [Correct Door] ───→ Exit     Start ──────→ [Trap Door] ──→ Dead End
         ↘                                          ↘
          [Trap Door] ──→ Dead End                   [Correct Door] ──→ Exit
                                                      (requires detour)

Greedy agent → correct door → wins         Greedy agent → trap door → deadlock
```

### Non-Greedy Mechanisms

We define five mechanisms that can make puzzles non-greedy. Each mechanism defeats a specific greedy strategy. Strong puzzles combine multiple mechanisms.

#### Mechanism 1: Direct-Path Trap

The trap door lies on the shortest spatial path from start to exit. The correct door requires going sideways or backward.

**Defeated strategy:** "Move toward exit"

```
    0   1   2   3   4
  ┌───┬───┬───┬───┬───┐
0 │ S │   │   │ # │ E │  Greedy path: S → right → D_trap → ???
  ├───┼───┼───┼───┼───┤
1 │   │ K │   │D_t│   │  D_trap at (1,3) is ON the direct path to E.
  ├───┼───┼───┼───┼───┤  D_correct at (3,1) requires going SOUTH first.
2 │   │   │   │ # │   │
  ├───┼───┼───┼───┼───┤
3 │   │D_c│   │   │   │  Correct: go down, use key on D_c, find alternate
  ├───┼───┼───┼───┼───┤  path to exit from below.
4 │   │   │   │   │   │
  └───┴───┴───┴───┴───┘
```

#### Mechanism 2: Invest-Before-Progress (Detour for Future Resources)

The correct strategy requires going AWAY from the exit first to acquire a resource needed later. The agent must make an "investment" that pays off several steps later.

**Defeated strategy:** "Move toward exit," "Pick up nearest key"

```
Phase 1 (investment): Go AWAY from exit to open Door A → get Blue Key
Phase 2 (payoff):     Use Blue Key on Door B (which is on the path to exit)

A greedy agent skips Phase 1, reaches Door B, and is stuck (no Blue Key).
```

#### Mechanism 3: Nearest-Door Trap

Two same-color doors. The agent encounters the trap door first (it's closer). The correct door is farther away. Must resist spending the key on the first matching door found.

**Defeated strategy:** "Use key on nearest matching door"

```
Agent with Red Key → walks toward exit →
  encounters Red Door (trap) at distance 2 →
  encounters Red Door (correct) at distance 5 →

Greedy: use key on nearest (distance 2) → deadlock
Correct: walk past nearest, use on farther one
```

#### Mechanism 4: Save-for-Expensive (Resource Conservation)

A single-lock door tempts the agent to spend a key, but that key is also needed for a dual-lock door that actually leads to the exit. Must save keys for the more expensive requirement.

**Defeated strategy:** "Use key as soon as possible," "Use key on nearest matching door"

```
Red Key available. Two doors:
  - Red Door (single-lock, nearby, tempting) → dead end
  - Red+Blue Door (dual-lock, farther, needs BOTH keys) → exit

Greedy: spend red on single-lock immediately → can't open dual-lock → deadlock
Correct: save red key, combine with blue for dual-lock → exit
```

#### Mechanism 5: Key-Behind-Key Chain with Misdirection

Sequential dependency: Key A opens Door A → behind it is Key B → Key B opens Door B → exit. But there's also a decoy Door C (same color as A) that appears to be more direct.

**Defeated strategy:** "Open door closest to exit" combined with "Move toward exit"

```
Red Key → D1(red, leads to Blue Key) → Blue Key → D2(blue, exit behind it)
           vs.
           D3(red, closer to exit, dead end)

Greedy: opens D3 (closer to exit) → never gets Blue Key → stuck at D2
Correct: opens D1 (away from exit) → gets Blue Key → opens D2 → exit
```

### Greedy Strategies to Validate Against

Every generated puzzle must be verified against ALL of these greedy strategies:

```python
GREEDY_STRATEGIES = {
    "move_toward_exit":
        # At each step, move in the direction that minimizes
        # manhattan_distance(agent, exit). USE_KEY on first adjacent door.

    "use_key_nearest_door":
        # When holding a key, move toward and USE_KEY on the nearest
        # matching door. Otherwise move toward nearest key.

    "use_key_immediately":
        # Whenever a USE_KEY action is legal, take it immediately
        # (don't walk past a door you could open).

    "pick_up_nearest_key_first":
        # Always go for the nearest uncollected key first,
        # then head toward exit/doors.

    "move_toward_exit_smart":
        # Move toward exit, but pick up keys that are on the path.
        # USE_KEY when adjacent to any door on the path to exit.
}
```

**Validation filter:**
```python
def all_greedy_strategies_fail(grid, initial_state):
    """Every greedy strategy must deadlock (not reach exit)."""
    for strategy in GREEDY_STRATEGIES.values():
        trace, final = simulate_strategy(grid, initial_state, strategy)
        if is_solved(grid, final):
            return False  # this strategy solves it → puzzle is too easy
    return True
```

---

## 6. Procedural Generation

### Instance Requirements

1. **Solvable:** At least one key allocation sequence reaches the exit.
2. **Has deadlocks:** At least one plausible key allocation creates a state where the exit is unreachable.
3. **Non-greedy:** ALL greedy strategies must fail (deadlock or get stuck).
4. **Branching:** At least 2 doors of the same color (so the agent must CHOOSE which to open).
5. **Depth:** The correct solution requires at least 1 key-use decision.
6. **Human-tractable:** Solvable in ~30 seconds on visual inspection once the trick is seen.

### Generation Strategy: Anti-Greedy Barrier Layout

The fundamental approach: place a vertical wall barrier with two door gaps. The trap door is positioned such that it is spatially closer to the exit (on the "greedy path"), while the correct door is farther from the exit (requires a detour).

**Key spatial invariant:** `manhattan(trap_door, exit) < manhattan(correct_door, exit)`

```
def generate_anti_greedy_grid(width, height, rng):
    """
    Generate a fork-style grid where the TRAP door is on the greedy path
    and the CORRECT door requires a detour.

    Layout strategy:
    1. Barrier at a middle column
    2. Exit placed near the TRAP door's side of the barrier
    3. Correct door requires going to opposite end of barrier
    4. Trap door region walled off (dead end)
    5. Correct door region connected to exit
    """
    # Place barrier
    barrier_col = width // 2

    # Key spatial decision: exit should be near TRAP door row
    # so that greedy "move toward exit" encounters trap first

    # Trap door: row close to exit → greedy agent goes here
    # Correct door: row far from exit → requires detour

    # Example with exit at top-right:
    # Trap door at top of barrier (row 1) — close to exit
    # Correct door at bottom of barrier (row 4) — far from exit
    # Start at top-left — greedy path goes RIGHT toward exit,
    #   hits trap door first

    # Wall off the region behind trap door
    # Keep region behind correct door connected to exit
```

### Three Anti-Greedy Templates

#### Template A: The Baited Fork

The exit and trap door are in the same spatial quadrant. The correct door is in the opposite quadrant.

```
          Exit
           ↑
    ┌──────#──────┐
    │  trap │dead  │
    │  door │ end  │
    │      [#]     │
    │  correct     │
    │  door → path │
    │      │to exit│
    │ Start│       │
    └──────#───────┘

Greedy: move UP toward exit → encounter trap door → deadlock
Correct: move DOWN to correct door → path wraps around to exit
```

Grid invariants:
- `exit.row < trap_door.row < correct_door.row` (exit and trap are both "above")
- `start.row >= correct_door.row` (start is near correct door)
- Trap region is walled off (dead end behind trap door)
- Correct region connects to exit via a winding path

#### Template B: The Conservation Trap

One key of a color that appears in both a single-lock door and a dual-lock door. The single-lock is nearby and tempting; the dual-lock is the actual path to exit.

```
    Start → Red Key, Blue Key
         ↓
    [Red Door — single-lock, nearby]  → dead end
         ↓ (if you don't waste the red key)
    [Red+Blue Door — dual-lock, farther] → Exit

Greedy: use red key on single-lock (it's right there!) → can't open dual-lock
Correct: walk past single-lock, save red for dual-lock → exit
```

#### Template C: The Hidden Dependency

Red key is available. Two red doors, one closer to exit (trap), one farther (correct). Behind the correct door is a blue key needed for a blue door that guards the exit.

```
    Start → Red Key
         ↓
    Red Door (near exit, trap) — dead end
    Red Door (away from exit) → Blue Key → Blue Door → Exit

Greedy: use red on nearest/closest-to-exit door → no blue key → stuck at blue door
Correct: use red on farther door → get blue key → open blue door → exit
```

### Difficulty Parameters

| Parameter | Easy | Medium | Hard |
|:---|:---|:---|:---|
| Grid size | 5×5 | 5×6 | 6×7 |
| Colors | 1-2 | 2-3 | 3-4 |
| Total keys | 1-2 | 2-3 | 3-4 |
| Doors | 2-3 | 3-5 | 4-6 |
| Dual-lock doors | 0 | 0-1 | 1-2 |
| Solution key-uses | 1 | 2 | 2-3 |
| Anti-greedy mechanisms | 1 | 1-2 | 2-3 |

---

## 7. Visual Rendering Spec

### Tile Sprites

| Tile Type | Visual | Color |
|:---|:---|:---|
| FLOOR | Empty square | Light gray |
| WALL | Solid square | Dark gray |
| KEY(red) | Square with key icon | Red key on light background |
| KEY(blue) | Square with key icon | Blue key on light background |
| KEY(green) | Square with key icon | Green key on light background |
| DOOR(red, locked) | Square with door icon + red lock | Red |
| DOOR(red, open) | Square with open door frame | Light red / faded |
| DOOR(red+blue, dual, locked) | Square with door icon + two colored locks | Red-blue split |
| EXIT | Square with flag/star | Gold |
| AGENT | Circle/character | White with black border |

### Layout Requirements
- Each door should be clearly labeled with its color and lock type.
- Keys on the ground should be immediately recognizable as pickable items.
- Agent's current inventory should be displayed as icons in a sidebar or HUD area.
- Door labels should indicate what color key(s) they require (e.g., "R" for red, "R+B" for dual-lock).

---

## 8. RQ-Specific Experimental Protocols

### RQ1: Simulation Fidelity

**Setup:** Initial state (image or text) + sequence of n actions (movements + key uses).
**Task:** "After these actions, what is the agent's position, what keys does the agent hold, which keys are still on the ground, and which doors are open?"
**Verification:** Compare all state variables against Oracle.
**Metric:** CEM — percentage of state variables correctly predicted.

**What makes Chain Vault interesting for RQ1:**
- Multi-type state tracking: position + inventory (add/remove) + door states + floor items.
- Key pickups are automatic (stepping on a key tile), testing whether models remember implicit state changes.
- USE_KEY actions are explicit and have both a deduction (key consumed) and an effect (door opens).
- Dual-lock doors consume two resources simultaneously — a richer transition.

**n values:** {1, 3, 5, 8}

### RQ2: Foresight

**Main (Natural Foresight):**
- Provide state S₀ + k actions. 50% lead to deadlock (wrong key allocation), 50% solvable.
- Ask: "What is the next move to win the game?"
- Deadlock instances: model should recognize unsolvability.
- Solvable instances: model should provide a valid next action.

**Ablation (Explicit Solvability):**
- Same S₀ + k actions.
- Ask: "Is the puzzle still solvable? YES or NO."

**What makes Chain Vault interesting for RQ2:**
- The board looks completely normal — doors are either open or closed, keys are present or absent. Nothing is "broken."
- The deadlock is in the ABSTRACT dependency state: the remaining keys cannot satisfy the remaining doors on any path to the exit.
- Detection requires searching the key allocation tree, not just inspecting the grid.
- **Non-greediness amplifies difficulty:** Because the correct path is counter-intuitive (requires a detour), recognizing that the greedy path is a dead end requires deeper reasoning.

**k values:** {1, 3, 5, 8}

### RQ3: Causal Attribution

**Setup:** Full trace from start to deadlock state.
**Task:** "This sequence led to an unsolvable state. Identify the last step at which the puzzle was still solvable."
**Metric:** |T_model - T_oracle|

**What makes Chain Vault interesting for RQ3:**
- The fatal step is a `USE_KEY` action that allocated a key to the wrong door.
- The symptom (being stuck) occurs many moves later when the agent reaches a door it can't open.
- The cause-symptom distance is naturally large because the agent continues moving/exploring between key uses.
- We expect models to blame the LAST key use (closest in time) rather than the FIRST wrong allocation (the actual cause).
- **Anti-greedy design helps RQ3:** Because the trap door is on the "obvious" path, the fatal USE_KEY looks reasonable in isolation — it's the spatially intuitive thing to do. Identifying it as the mistake requires understanding the non-obvious correct path.

---

## 9. Edge Cases and Design Considerations

### Key Pickup is Automatic
The agent picks up keys by stepping on them. This means:
- A wrong path can accidentally pick up a key the agent didn't want (though in our design, extra keys don't hurt).
- More importantly, the agent can pick up keys in an order that tempts wrong usage.

### Open Doors are Permanent
Once opened, a door stays open forever. This simplifies the state space and prevents a class of "lock yourself out" deadlocks that would be too complex.

### Key Color Ambiguity
If the agent has 2 Red Keys and uses one on Red Door A, the remaining Red Key can still open Red Door B. Deadlocks occur when the agent runs out of keys of a needed color. This means the puzzle is about COLOR-BUDGET MANAGEMENT, not about choosing between identical keys.

### Dual-Lock Doors as Difficulty Multiplier
Dual-lock doors consume 2 keys simultaneously. This creates compound resource constraints — the agent must save BOTH a Red and Blue key for the dual-lock, which competes with using them individually on single-lock doors. This is where the most interesting deadlocks emerge.

### Preventing Trivial Solutions
The generation must ensure that:
- There's no path to the exit that requires zero key uses (bypassing all doors).
- The correct key allocation is not the only option (there must be wrong choices available).
- Not all doors need to be opened — some are distractors (useful-looking but unnecessary).

---

## 10. Demo Plan

### Paper Walkthrough Example — Anti-Greedy Fork

A minimal 5×5 example demonstrating the anti-greedy principle. The exit is at the top-right, the trap door is close to the exit, the correct door requires a detour south.

```
      0   1   2   3   4
    ┌───┬───┬───┬───┬───┐
  0 │ S │   │ # │   │ E │   Exit at (0,4)
    ├───┼───┼───┼───┼───┤
  1 │   │Kr │ # │D_t│   │   Trap door D_t at (1,3) — close to exit
    ├───┼───┼───┼───┼───┤
  2 │   │   │ # │ # │ # │   Walls seal off trap region
    ├───┼───┼───┼───┼───┤
  3 │   │   │D_c│   │   │   Correct door D_c at (3,2)
    ├───┼───┼───┼───┼───┤
  4 │   │   │ # │   │   │   Behind D_c: (3,3)→(3,4)→(2,4)→...→(0,4)=E
    └───┴───┴───┴───┴───┘

Greedy path: S → right → pick up Kr → continue right toward exit →
             encounter D_t → USE_KEY on D_t → enters (1,3) →
             (1,4) dead end (walls at (2,3),(2,4) block progress) → DEADLOCK

Correct path: S → pick up Kr → go DOWN to row 3 →
              USE_KEY on D_c → go through (3,3)→(3,4)→...→ exit
```

The trap door D_t is closer to the exit (manhattan 2 from exit) while the correct door D_c is farther (manhattan 4 from exit). Any greedy "move toward exit" agent encounters D_t first.

### Interactive Demo

1. **Terminal prototype:** ASCII grid with HUD showing inventory, door states, `is_solvable()` status.
2. **Visual demo:** Rendered grid with color-coded keys/doors and inventory HUD.
3. **Replay mode:** Step through traces, highlight fatal step.

---

## 11. Principled Instance Generation Pipeline

### Philosophy: Anti-Greedy by Construction + Deadlock-First Discovery

Chain Vault uses a three-layer generation strategy:
1. **Anti-greedy spatial layout:** Place trap door closer to exit than correct door, guaranteeing greedy strategies fail by construction.
2. **Deadlock-first discovery:** Forward sweep to catalog all Walking Dead states.
3. **Multi-strategy validation:** Verify that ALL common greedy strategies deadlock.

---

### Step 1: Anti-Greedy Grid Generation

```
def generate_anti_greedy_fork(width, height, rng):
    """
    Generate a grid where:
    - Barrier at a central column splits grid into left and right halves
    - Two doors in the barrier (same color, one key available)
    - TRAP door is closer to exit (on the greedy path)
    - CORRECT door is farther from exit (requires detour)
    - Region behind trap door is a dead end
    - Region behind correct door connects to exit
    """
    barrier_col = width // 2

    # Choose door positions:
    # Trap door close to exit row, correct door far from exit row
    # Exit in top-right → trap door in top half, correct door in bottom half

    trap_row = rng.randint(0, height // 2)          # upper half
    correct_row = rng.randint(height // 2 + 1, height - 1)  # lower half

    # Place exit in top-right area (near trap door)
    exit_pos = pick_right_side_near_row(trap_row)

    # Place start in top-left (so greedy path goes RIGHT toward exit/trap)
    start_pos = pick_left_side_pos()

    # Wall off the region behind trap door
    # The trap door region should be small (1-3 tiles, dead end)
    seal_trap_region(walls, trap_row, barrier_col)

    # Ensure correct door region connects to exit
    # Path: correct_door → bottom-right → winds up to exit
    ensure_correct_path_to_exit(walls, correct_row, barrier_col, exit_pos)

    # VERIFY anti-greedy property:
    assert manhattan(Pos(trap_row, barrier_col), exit_pos) < \
           manhattan(Pos(correct_row, barrier_col), exit_pos)
```

### Step 2: Deadlock Discovery (Forward Sweep)

After embedding, run the forward sweep to discover all reachable Walking Dead states:

```
def discover_deadlocks(initial_state, max_depth=20):
    """
    Forward sweep from S₀. At each state, check is_solvable().
    Collect all transitions where solvability flips True → False.

    OPTIMIZATION for Chain Vault: solvability can only change on USE_KEY actions
    (movement never consumes resources). So we only call is_solvable() after
    key-use actions, dramatically reducing oracle calls.
    """
    catalog = []
    queue = deque([(initial_state, [], True)])
    visited = {initial_state.hashable()}

    while queue:
        state, path, parent_solvable = queue.popleft()
        if len(path) >= max_depth:
            continue

        for action in legal_actions(state):
            next_state = apply_action(state, action)
            h = next_state.hashable()
            if h in visited:
                continue
            visited.add(h)
            new_path = path + [action]

            # Optimization: only check solvability after key uses
            if action.type == USE_KEY:
                next_solvable = is_solvable(next_state)
            elif not parent_solvable:
                next_solvable = False  # monotonicity: once dead, always dead
            else:
                next_solvable = True   # movement alone can't cause deadlock

            # Detect flip
            if parent_solvable and not next_solvable:
                if is_walking_dead(next_state):
                    catalog.append(DeadlockEntry(
                        path=new_path,
                        fatal_step=len(new_path) - 1,
                        fatal_action=action,  # always a USE_KEY
                        deadlock_state=next_state,
                        pre_deadlock_state=state
                    ))

            if next_solvable or is_walking_dead(next_state):
                queue.append((next_state, new_path, next_solvable))

    return catalog
```

**Chain Vault-specific insight:** The fatal step is ALWAYS a `USE_KEY` action. Movement, key pickup — none of these can make the puzzle unsolvable. Only consuming a key on the wrong door can.

### Step 3: Multi-Strategy Greedy Validation

```
def validate_non_greedy(grid, initial_state):
    """
    Verify that ALL common greedy strategies fail.
    A puzzle is non-greedy only if EVERY greedy approach deadlocks.
    """
    strategies = [
        simulate_move_toward_exit,     # minimize manhattan to exit
        simulate_use_key_nearest,      # use key on nearest matching door
        simulate_use_key_immediately,  # USE_KEY on first adjacent door
        simulate_pick_nearest_key,     # go to nearest key first, then doors
        simulate_move_toward_exit_smart,  # move toward exit, use keys when adjacent
    ]

    for strategy_fn in strategies:
        trace, final_state = strategy_fn(grid, initial_state)
        if is_solved(grid, final_state):
            return False  # This greedy strategy solves it → reject

    return True  # All greedy strategies fail → puzzle is genuinely non-greedy
```

**Strategy definitions:**

```python
def simulate_move_toward_exit(grid, state):
    """Greedy: always minimize manhattan(agent, exit). USE_KEY on any adjacent door."""
    # At each step:
    # 1. If USE_KEY is legal, do it (on whichever adjacent door)
    # 2. Else move in direction that minimizes manhattan to exit

def simulate_use_key_nearest(grid, state):
    """Greedy: move toward nearest matching door, use key when adjacent."""
    # At each step:
    # 1. If holding key and adjacent to matching door, USE_KEY
    # 2. If holding key, move toward nearest matching door
    # 3. Else move toward nearest key

def simulate_use_key_immediately(grid, state):
    """Greedy: USE_KEY immediately whenever possible."""
    # At each step:
    # 1. If any USE_KEY action is legal, take first one
    # 2. Else move toward nearest key/door/exit

def simulate_pick_nearest_key(grid, state):
    """Greedy: always pick up nearest key first, then use on nearest door."""
    # At each step:
    # 1. If keys remain on ground, move toward nearest
    # 2. If holding keys, USE_KEY on nearest matching door
    # 3. Else move toward exit

def simulate_move_toward_exit_smart(grid, state):
    """Greedy: move toward exit, picking up keys and opening doors along the way."""
    # A* pathfinding to exit treating locked doors as:
    #   passable if agent has the key (plan to USE_KEY and pass through)
    # Then follow that path, using keys as needed
```

### Step 4: Grid Validation Filters

```
def validate_grid(puzzle):
    state0 = puzzle.initial_state

    # Must be solvable
    if not is_solvable(state0):
        return False

    # Optimal solution in human-tractable range
    solution = oracle_solve(state0)
    if not (6 <= len(solution) <= 20):
        return False

    # ALL greedy strategies must FAIL (critical filter!)
    if not validate_non_greedy(puzzle.grid, state0):
        return False

    # Must have enough deadlocks
    catalog = discover_deadlocks(state0)
    if len(catalog) < 1:
        return False

    puzzle.solution = solution
    puzzle.deadlock_catalog = catalog
    return True
```

### Step 5: RQ Instance Construction (from Deadlock Catalog)

#### RQ1 Instances (Simulation)

```
def generate_rq1_instances(puzzle, n_values=[1, 3, 5, 8]):
    instances = []
    for n in n_values:
        # From oracle solution (correct play)
        sol_actions = puzzle.solution[:n]
        gt = apply_actions(puzzle.initial_state, sol_actions)
        instances.append(RQ1Instance(puzzle.initial_state, sol_actions, gt))

        # From deadlock paths (wrong play — includes wrong key uses)
        for entry in sample(puzzle.deadlock_catalog, 3):
            if len(entry.path) >= n:
                actions = entry.path[:n]
                gt = apply_actions(puzzle.initial_state, actions)
                instances.append(RQ1Instance(puzzle.initial_state, actions, gt))
    return instances
```

#### RQ2 Instances (Foresight) — Deadlock-First Construction

```
def generate_rq2_instances(puzzle, k_values=[1, 3, 5, 8]):
    instances = []

    for entry in puzzle.deadlock_catalog:
        path = entry.path
        T = entry.fatal_step

        for k in k_values:
            window_start = T - k + 1
            if window_start < 0:
                continue

            S_start = apply_actions(puzzle.initial_state, path[:window_start])
            deadlock_actions = path[window_start : T + 1]

            # Verify
            assert is_solvable(S_start)
            assert is_walking_dead(apply_actions(S_start, deadlock_actions))

            # Matched solvable control from same S_start
            control_solution = oracle_solve(S_start)
            if control_solution and len(control_solution) >= k:
                solvable_actions = control_solution[:k]
            else:
                solvable_actions = find_solvable_walk(S_start, k)

            if solvable_actions:
                instances.append(RQ2MatchedPair(
                    start_state=S_start,
                    k=k,
                    deadlock_actions=deadlock_actions,
                    solvable_actions=solvable_actions
                ))

    return instances
```

#### RQ3 Instances (Causal Attribution)

```
def generate_rq3_instances(puzzle, target_trace_length=18):
    instances = []

    for entry in puzzle.deadlock_catalog:
        path = entry.path
        T = entry.fatal_step
        dead_state = entry.deadlock_state

        trace_to_deadlock = path[:T + 1]
        continuation_length = target_trace_length - len(trace_to_deadlock)
        if continuation_length < 3:
            continue

        continuation = walking_dead_continuation(dead_state, continuation_length)
        full_trace = trace_to_deadlock + continuation
        cause_symptom_gap = len(full_trace) - T - 1

        if cause_symptom_gap < 3:
            continue

        instances.append(RQ3Instance(
            initial_state=puzzle.initial_state,
            trace=full_trace,
            oracle_fatal_step=T,
            trace_length=len(full_trace),
            cause_symptom_gap=cause_symptom_gap
        ))

    return filter_for_distribution(instances,
        key=lambda x: x.oracle_fatal_step / x.trace_length)
```

**Why Chain Vault produces excellent RQ3 instances:** The fatal step (wrong key use) happens early. Then the agent continues walking around, picking up keys, exploring rooms — it LOOKS like normal productive play. The puzzle appears fine. The model must look past many steps of "normal" activity to find the one key use that doomed everything. **Anti-greedy design makes RQ3 harder:** The fatal USE_KEY is the "obvious" greedy move (use key on the door that's closest to exit), so it looks even more reasonable in hindsight.

---

### Step 6: Dataset Assembly

```
TARGET_PER_RQ = {
    "RQ1": 250,        # ~63 per n value × 4 n values
    "RQ1_abl": 100,    # boundary physics (locked doors, walls)
    "RQ2_main": 250,    # ~63 per k value × 4 k values, 50/50 deadlock/solvable
    "RQ2_abl": 250,     # same instances, explicit solvability prompt
    "RQ3": 250,         # full traces, diverse cause-symptom gaps
}
```

---

## 12. Hand-Crafted Instance Designs (Anti-Greedy)

These three instances demonstrate the core anti-greedy mechanisms. Each instance is designed so that the greedy path (move toward exit) leads to the trap door, and the correct path requires a non-obvious detour.

### Instance 1: "The Baited Fork" (5×5)

**Mechanism:** Direct-Path Trap + Nearest-Door Trap

```
      0   1   2   3   4
    ┌───┬───┬───┬───┬───┐
  0 │ S │   │ # │   │ E │   Start top-left, Exit top-right
    ├───┼───┼───┼───┼───┤
  1 │   │Kr │D_t│   │   │   Trap door D_t at (1,2) — ON greedy path to exit
    ├───┼───┼───┼───┼───┤   Behind D_t: (1,3) but (2,3),(2,4) are walls → dead end
  2 │   │   │ # │ # │ # │
    ├───┼───┼───┼───┼───┤
  3 │   │   │ # │   │   │   Correct door D_c at (4,2) — requires going DOWN
    ├───┼───┼───┼───┼───┤   Behind D_c: (4,3)→(4,4)→(3,4)→(3,3)→...→(0,4)=E
  4 │   │   │D_c│   │   │
    └───┴───┴───┴───┴───┘

Trap door manhattan to exit: |(1-0)| + |(2-4)| = 3
Correct door manhattan to exit: |(4-0)| + |(2-4)| = 6  (FARTHER — anti-greedy ✓)

Greedy: Start→(1,1) pick up Kr→move right toward exit→encounter D_t→USE_KEY→dead end
Correct: Start→(1,1) pick up Kr→go DOWN to (4,1)→USE_KEY on D_c→go through→exit
```

**Properties:**
- 1 red key, 2 red doors
- Greedy path: ~4 steps to D_t (deadlock)
- Correct path: ~9 steps total
- Walking Dead after wrong choice: agent can walk freely on left side

### Instance 2: "The Conservation Trap" (5×5)

**Mechanism:** Save-for-Expensive + Direct-Path Trap

```
      0   1   2   3   4
    ┌───┬───┬───┬───┬───┐
  0 │ S │Kr │ # │ # │ # │   Start with red key right next to it
    ├───┼───┼───┼───┼───┤
  1 │   │Kb │ # │   │   │   Blue key nearby too
    ├───┼───┼───┼───┼───┤
  2 │   │   │D_r│   │   │   D_r: red single-lock at (2,2) — TRAP (closer to exit)
    ├───┼───┼───┼───┼───┤   Behind D_r: (2,3),(2,4) — dead end
  3 │   │   │ # │ # │ # │   Walls seal off D_r region from exit
    ├───┼───┼───┼───┼───┤
  4 │   │   │D_rb│  │ E │   D_rb: red+blue dual-lock at (4,2) → exit
    └───┴───┴───┴───┴───┘

Greedy: pick up keys → encounter D_r first (closer) → USE red KEY →
        now can't open D_rb (needs red+blue) → deadlock
Correct: pick up both keys → walk PAST D_r → go down to D_rb →
         USE both keys on dual-lock → (4,3)→(4,4)=E
```

**Properties:**
- 1 red key, 1 blue key, 1 red door (trap), 1 red+blue dual-lock door (correct)
- Tests resource conservation: must save red key for dual-lock
- Any "use key immediately" or "use key on nearest door" strategy fails

### Instance 3: "The Hidden Chain" (6×5)

**Mechanism:** Key-Behind-Key + Invest-Before-Progress + Direct-Path Trap

```
      0   1   2   3   4   5
    ┌───┬───┬───┬───┬───┬───┐
  0 │ S │   │Kr │ # │   │   │   Red key available in starting area
    ├───┼───┼───┼───┼───┼───┤
  1 │   │   │   │D_t│   │   │   D_t: red door at (1,3) — trap, closer to exit
    ├───┼───┼───┼───┼───┼───┤   Behind D_t: sealed dead-end
  2 │   │   │   │ # │ # │ # │   Walls seal off trap region
    ├───┼───┼───┼───┼───┼───┤
  3 │   │   │   │D_c│Kb │   │   D_c: red door at (3,3) — correct → blue key behind
    ├───┼───┼───┼───┼───┼───┤
  4 │   │   │   │ # │D_b│ E │   D_b: blue door at (4,4) → exit
    └───┴───┴───┴───┴───┴───┘

Chain: Red Key → D_c (correct red door) → Blue Key → D_b (blue door) → Exit

Greedy: move toward exit (top-right area) → encounter D_t → USE red KEY → dead end
        Now: no red key → can't open D_c → no blue key → can't open D_b → stuck
Correct: go DOWN (away from exit!) → open D_c → get Kb → open D_b → exit
```

**Properties:**
- Sequential dependency chain: red key → correct red door → blue key → blue door → exit
- Requires invest-before-progress: must go south (away from exit) first
- Two red doors: trap (closer to exit) and correct (farther from exit)
- The blue key is only accessible behind the correct red door

---

## Key Differences from Mosaic Grid Prompts

| Aspect | Mosaic Grid | Chain Vault |
|:---|:---|:---|
| Action space | 4 (UP, DOWN, LEFT, RIGHT) | 8 (4 MOVE + 4 USE_KEY) |
| State to track | position, painted tiles, plates, doors | position, inventory, ground keys, doors |
| Irreversible action | painting a tile on departure | consuming a key on a door |
| Deadlock mechanism | paint severs critical path | wrong key allocation exhausts scarce resource |
| RQ3 fatal step cause | movement that painted a critical corridor | USE_KEY that wasted a key on the wrong door |
| RQ2 detection difficulty | visual (can see paint blocking paths) | abstract (must reason about key budget vs. door requirements) |
| Non-greedy mechanism | paint-on-departure inherently non-greedy | requires explicit anti-greedy spatial layout |
