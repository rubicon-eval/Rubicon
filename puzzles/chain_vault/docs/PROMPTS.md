# Chain Vault — Model Prompt Instructions

Prompt templates sent to models during evaluation. Each prompt contains only the **game rules** and the **task question**. The puzzle state is provided separately via a rendered image.

Placeholders in `{curly_braces}` are filled per-instance at generation time.

---

## Shared: Game Rules Preamble

Prepended to every prompt.

```
You are playing a grid-based puzzle called Chain Vault.

## Rules

1. You control an agent on a rectangular grid. Each cell is one of: floor, wall, key, locked door, open door, or exit.
2. The agent can move one step at a time: MOVE_UP, MOVE_DOWN, MOVE_LEFT, or MOVE_RIGHT.
3. The agent can move onto floor, keys, open doors, and the exit. The agent CANNOT move onto walls or locked doors.
4. **Key pickup:** Stepping onto a key tile automatically picks up the key. The tile becomes floor. Keys are colored (red, blue, etc.).
5. **Using keys on doors:** The agent can USE a key on an adjacent locked door by performing USE_KEY_UP/DOWN/LEFT/RIGHT. This consumes the required key(s) and opens the door permanently. The agent stays in place.
6. **Single-lock doors** require 1 key of matching color. **Dual-lock doors** require 1 key of each of two colors simultaneously.
7. **Key scarcity:** Keys are consumed when used. There may not be enough keys to open all doors — the agent must choose which doors to open.
8. **Goal:** The agent wins by stepping onto the exit tile.

{instance_rules}
```

`{instance_rules}` is the output of `generate_rules_text(grid)`. Example:

```
Instance details:
- Red key at row 1, col 1
- Blue key at row 2, col 3
- Single-lock door at row 2, col 2 (requires 1 red key to open)
- Single-lock door at row 4, col 2 (requires 1 red key to open)
- Single-lock door at row 2, col 4 (requires 1 blue key to open)
- Exit at row 2, col 5
- Start at row 0, col 0
```

---

## RQ1: Simulation Fidelity

**Goal:** Test whether the model can accurately simulate the puzzle forward by n steps, tracking position, inventory changes, key pickups, and door openings.

### RQ1 — Main

```
{rules_preamble}

## Task

The following {n} actions are taken in order starting from the state shown in the image:
{action_list}

After all {n} actions are executed, answer the following:

1. What is the agent's final position? (row, col)
2. What keys does the agent currently hold in inventory?
3. Which keys are still on the ground (not yet picked up)?
4. Which doors are now open?

Respond in this exact format:
Agent position: (row, col)
Inventory: {color: count, ...}
Keys on ground: [list of (color, row, col)]
Open doors: [list of door locations]
```

### RQ1 — Ablation (Single-Step Boundary Physics)

Tests legality of a single action (movement or key use).

```
{rules_preamble}

## Task

Looking at the state shown in the image, the agent attempts to {attempt_description} (action: {action_name}).

What happens? Choose one:
A) The action succeeds — it is legal.
B) The action fails — it is illegal.

Respond in this format:
Answer: [A or B]
Explanation: [why the action is legal or illegal]
```

`{attempt_description}` examples:
- `"move UP"` for movement actions
- `"use a key on the cell RIGHT of the agent"` for USE_KEY actions

---

## RQ2: Foresight

**Goal:** Test whether the model can detect that a state is unsolvable (Walking Dead — the agent wasted a key on the wrong door), or find a valid next action in a solvable state.

### RQ2 — Main (Natural Foresight)

```
{rules_preamble}

## Task

Starting from the state shown in the image, the following {k} actions were taken:
{action_list}

What is the best next action to eventually reach the exit and win?

Available actions: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, USE_KEY_UP, USE_KEY_DOWN, USE_KEY_LEFT, USE_KEY_RIGHT.

If the puzzle is still solvable, provide the best next action and briefly explain your reasoning.

If the puzzle is no longer solvable from this state (no possible sequence of actions can reach the exit), respond with UNSOLVABLE and explain why.

Respond in this format:
Answer: [action name / UNSOLVABLE]
Reasoning: [your explanation]
```

### RQ2 — Ablation (Explicit Solvability)

```
{rules_preamble}

## Task

Starting from the state shown in the image, the following {k} actions were taken:
{action_list}

Is the puzzle still solvable? That is, does there exist any sequence of legal actions from the agent's current state that reaches the exit?

Consider:
- What keys the agent currently holds
- What keys are still on the ground
- Which doors are still locked and what they require
- Whether the remaining keys can open a path to the exit

Answer: [YES / NO]
Reasoning: [your explanation]
```

---

## RQ3: Causal Attribution

**Goal:** Test whether the model can identify the specific USE_KEY action that made the puzzle unsolvable by wasting a scarce key on the wrong door.

### RQ3 — Main

```
{rules_preamble}

## Task

Starting from the state shown in the image, the following {trace_length} actions were taken:
{action_list}

After this full sequence, the puzzle is **unsolvable** — no remaining sequence of actions can reach the exit.

Identify the **fatal step**: the action whose execution caused the puzzle to become unsolvable. Before this step, the puzzle was still solvable. After this step, it became impossible to win.

Think carefully:
- The fatal step is always a USE_KEY action that consumed a key on the wrong door.
- The fatal step is NOT necessarily the last step or where the agent gets physically stuck.
- The agent may continue walking around after the fatal mistake, making the true cause hard to identify.
- There may be many steps between the fatal key use and the point where the agent runs out of options.

Respond in this format:
Fatal step: [step number]
Action: [the action taken at that step]
Reasoning: [explain what this key use destroyed and why the puzzle became unsolvable after it]
```

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

---

## Prompt Construction Reference

### Per-instance variables

| Variable | Source | Description |
|:---|:---|:---|
| `{instance_rules}` | `generate_rules_text(grid)` | Key locations, door requirements, exit location |
| `{n}` / `{k}` | per-instance | Number of actions |
| `{action_list}` | per-instance | Numbered list of actions (e.g. `1. MOVE_DOWN`, `2. USE_KEY_RIGHT`) |
| `{trace_length}` | `len(trace)` | Total trace length (RQ3) |
| `{action_name}` | per-instance | Single action for RQ1 ablation (e.g. `MOVE_UP`, `USE_KEY_RIGHT`) |
| `{attempt_description}` | per-instance | Human-readable action description for RQ1 ablation |

### Expected output format per RQ

| RQ | Output parsed | Metric |
|:---|:---|:---|
| RQ1 Main | agent position, inventory, ground keys, open doors | CEM (Coordinate Exact Match) per field |
| RQ1 Ablation | A or B | Accuracy |
| RQ2 Main | action name or UNSOLVABLE | Accuracy |
| RQ2 Ablation | YES or NO | Accuracy |
| RQ3 Main | step number | \|T_model - T_oracle\| (Localization Error) |

### Python API

All prompt builders are in `puzzles.chain_vault.prompts`:

```python
from puzzles.chain_vault.prompts import (
    build_rq1_prompt,          # (grid, actions) -> str
    build_rq1_ablation_prompt, # (grid, direction, action_type) -> str
    build_rq2_main_prompt,     # (grid, actions) -> str
    build_rq2_ablation_prompt, # (grid, actions) -> str
    build_rq3_prompt,          # (grid, actions) -> str
)
```
