# Mosaic Grid — Model Prompt Instructions

Prompt templates sent to models during evaluation. Each prompt contains only the **game rules** and the **task question**. The puzzle state is provided separately via a rendered image.

Placeholders in `{curly_braces}` are filled per-instance at generation time.

---

## Shared: Game Rules Preamble

Prepended to every prompt.

```
You are playing a grid-based puzzle called Mosaic Grid.

## Rules

1. You control an agent on a rectangular grid. Each cell is one of: floor, wall, pressure plate, door, or exit.
2. The agent can move one step at a time: UP, DOWN, LEFT, or RIGHT.
3. The agent can move onto floor, pressure plates, open doors, and the exit. The agent CANNOT move onto walls, painted tiles, or closed doors.
4. **Paint on departure:** Every tile the agent leaves becomes permanently painted and impassable. The tile the agent currently stands on is NOT yet painted — painting happens when the agent moves away.
5. **Pressure plates:** Stepping onto a pressure plate activates it. Plates are linked to doors — when all plates linked to a door are activated, that door opens permanently. Plates and doors that share the same color are linked.
6. **Goal:** The agent wins by stepping onto the exit tile.

{instance_rules}
```

`{instance_rules}` is the output of `generate_rules_text(grid)`. Example:

```
- All 2 blue plates (at row 0, col 2 and row 3, col 1) must be activated to open the blue door at row 2, col 2
- The exit is at row 3, col 4
- Every tile the agent leaves becomes painted (impassable)
- The agent wins by stepping onto the exit tile
```

---

## RQ1: Simulation Fidelity

**Goal:** Test whether the model can accurately simulate the puzzle forward by n steps.

### RQ1 — Main

```
{rules_preamble}

## Task

The following {n} actions are taken in order starting from the state shown in the image:
{action_list}

After all {n} actions are executed, answer the following:

1. What is the agent's final position? (row, col)
2. Which tiles are now painted? List all painted tile coordinates.
3. Which pressure plates have been activated?
4. Which doors are now open?

Respond in this exact format:
Agent position: (row, col)
Painted tiles: [(row, col), (row, col), ...]
Activated plates: [list of plate locations]
Open doors: [list of door locations]
```

### RQ1 — Ablation (Single-Step Boundary Physics)

```
{rules_preamble}

## Task

Looking at the state shown in the image, the agent attempts to move {direction}.

What happens? Choose one:
A) The agent moves successfully — the move is legal.
B) The agent stays in place — the move is illegal.

Respond in this format:
Answer: [A or B]
Explanation: [why the move is legal or illegal]
```

---

## RQ2: Foresight

**Goal:** Test whether the model can detect that a state is unsolvable (Walking Dead), or find a valid next move in a solvable state.

### RQ2 — Main (Natural Foresight)

```
{rules_preamble}

## Task

Starting from the state shown in the image, the following {k} actions were taken:
{action_list}

What is the best next move to eventually reach the exit and win?

If the puzzle is still solvable, provide the best next action (UP, DOWN, LEFT, or RIGHT) and briefly explain your reasoning.

If the puzzle is no longer solvable from this state (no possible sequence of moves can reach the exit), respond with UNSOLVABLE and explain why.

Respond in this format:
Answer: [UP / DOWN / LEFT / RIGHT / UNSOLVABLE]
Reasoning: [your explanation]
```

### RQ2 — Ablation (Explicit Solvability)

```
{rules_preamble}

## Task

Starting from the state shown in the image, the following {k} actions were taken:
{action_list}

Is the puzzle still solvable? That is, does there exist any sequence of legal moves from the agent's current position that reaches the exit?

Consider:
- Which tiles are painted (impassable)
- Which pressure plates still need to be activated
- Whether required doors can still be reached and opened
- Whether a valid path to the exit exists

Answer: [YES / NO]
Reasoning: [your explanation]
```

---

## RQ3: Causal Attribution

**Goal:** Test whether the model can identify the specific action that made the puzzle unsolvable.

### RQ3 — Main

```
{rules_preamble}

## Task

Starting from the state shown in the image, the following {trace_length} actions were taken:
{action_list}

After this full sequence, the puzzle is **unsolvable** — no remaining sequence of moves can reach the exit.

Identify the **fatal step**: the action whose execution caused the puzzle to become unsolvable. Before this step, the puzzle was still solvable. After this step, it became impossible to win.

Think carefully:
- The fatal step is NOT necessarily the last step or where the agent gets physically stuck.
- The fatal step is the specific move that painted over a critical corridor, making a required plate or door permanently unreachable.
- There may be many steps between the fatal mistake and the point where the agent runs out of moves.

Respond in this format:
Fatal step: [step number]
Action: [the action taken at that step]
Reasoning: [explain what this move destroyed and why the puzzle became unsolvable after it]
```

---

## Prompt Construction Reference

### Per-instance variables

| Variable | Source | Description |
|:---|:---|:---|
| `{instance_rules}` | `generate_rules_text(grid)` | Plate-door linkages, exit location |
| `{n}` / `{k}` | per-instance | Number of actions |
| `{action_list}` | per-instance | Numbered list of actions (e.g. `1. DOWN`) |
| `{trace_length}` | `len(trace)` | Total trace length (RQ3) |
| `{direction}` | per-instance | Single action for RQ1 ablation |

### Expected output format per RQ

| RQ | Output parsed | Metric |
|:---|:---|:---|
| RQ1 Main | agent position, painted set, plates, doors | CEM (Coordinate Exact Match) per field |
| RQ1 Ablation | A or B | Accuracy |
| RQ2 Main | action or UNSOLVABLE | Accuracy |
| RQ2 Ablation | YES or NO | Accuracy |
| RQ3 Main | step number | \|T_model - T_oracle\| (Localization Error) |

### Python API

All prompt builders are in `puzzles.mosaic_grid.prompts`:

```python
from puzzles.mosaic_grid.prompts import (
    build_rq1_prompt,          # (grid, actions) -> str
    build_rq1_ablation_prompt, # (grid, direction) -> str
    build_rq2_main_prompt,     # (grid, actions) -> str
    build_rq2_ablation_prompt, # (grid, actions) -> str
    build_rq3_prompt,          # (grid, actions) -> str
)
```
