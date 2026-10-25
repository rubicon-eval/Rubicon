# Generation Axioms — Validation Checklist

Formal constraints every generated puzzle instance must satisfy. Use this document to validate puzzles before including them in the final dataset.

---

## Theoretical Framework

Each puzzle is a finite-horizon MDP: M = (S, A, T, R). Non-goal states partition into:

- **S_viable**: V*(s) > 0. The goal is reachable — a winning action sequence exists.
- **S_dead** (Walking Dead): V*(s) = 0 but legal actions still exist. The goal is mathematically unreachable, yet the agent can keep acting.

The benchmark measures an agent's ability to detect the exact transition s_viable → s_dead (the fatal step) and to distinguish viable from dead states at varying lookahead depths.

---

## Axiom 1: Out-of-Distribution Mechanics

**Requirement:** The puzzle mechanics must not be isomorphic to standard RL/planning environments (MiniGrid, Sokoban, Blocksworld, DoorKey) that are heavily represented in LLM training data. Models should not be able to solve the puzzle by pattern-matching against known environments.

**What this means for us:** The novelty must come from the game mechanics themselves, not from superficial renaming. Our three puzzles satisfy this through genuinely novel transition functions:

| Puzzle | Novel mechanic (not in standard envs) |
|:---|:---|
| Mosaic Grid | Paint-on-departure: every tile the agent leaves becomes permanently impassable. No standard env has self-destructing paths. |
| Chain Vault | Key consumption with dual-lock doors: a single action consumes multiple resource types simultaneously. Standard key-door envs have 1:1 key-door mappings. |
| Fuse Grid | Delayed grid mutation: arming a fuse schedules future events (walls open, bridges collapse) after a countdown. No standard env has time-delayed state changes triggered by player action. |

**Ablation opportunity:** Test whether semantic naming ("use red key on door") vs. abstract naming ("apply Δ-token to Ω-gate") affects model performance. If models score higher with semantic names, that's evidence of training data leakage. This is a prompt-level experiment, not a generation constraint.

### Compliance check

- [ ] Each puzzle has at least one transition rule that does not exist in MiniGrid, Sokoban, Blocksworld, or DoorKey
- [ ] (Optional ablation) Abstract naming prompts prepared alongside semantic naming prompts

---

## Axiom 2: Anti-Greedy by Construction

**Requirement:** Every generated instance must be provably anti-greedy. The greedy action (minimize manhattan distance to nearest target / use available resource immediately) must lead to S_dead. The optimal action must initially increase distance to target or bypass an immediately available resource.

Formally: T(s, a_greedy) ∈ S_dead. The agent must take a_optimal, which looks suboptimal by any shallow heuristic, to stay in S_viable.

**What this means for us:** This is enforced as a hard generation filter. Every instance is rejected unless all greedy strategies fail.

| Puzzle | Greedy strategies tested | How greedy leads to S_dead |
|:---|:---|:---|
| Mosaic Grid | 1 (nearest plate first) | Nearest-plate path severs the return corridor |
| Chain Vault | 4 (toward_exit, nearest_key, nearest_door, use_key_immediately) | Nearest matching door is always the trap; using key immediately wastes it |
| Fuse Grid | 2 (trigger_asap, toward_exit) | Triggering ASAP collapses bridge before crossing; moving toward exit hits the fuse first |

### Compliance check

- [x] All greedy strategies fail on every generated instance (hard filter in `generate_validated_instance`)
- [x] Greedy failure is verified by simulation, not by construction assumptions
- [ ] Store greedy traces as instance metadata (greedy action sequence, step where it enters S_dead)

---

## Axiom 3: k-Horizon Walking Dead (k >= 5)

**Requirement:** When the agent takes the fatal action entering S_dead at step t_fatal, the environment must permit at least k >= 5 subsequent legal, non-trivial actions. The deadlock must be latent, not immediate — the agent must be able to continue acting (traversing, interacting) for multiple steps before running out of options. The cause and symptom must be temporally decoupled.

**Why this matters:** A deadlock with 0 moves remaining is trivially detectable ("I can't move, something went wrong"). The interesting diagnostic question is whether models can detect a deadlock while the agent is still free to act.

**Current empirical status (measured):**

| Puzzle | Instance | Reachable S_dead states after fatal step |
|:---|:---|:---|
| Chain Vault | Baited Fork (hand-crafted) | 12 |
| Chain Vault | Conservation Trap (hand-crafted) | 17–33 |
| Chain Vault | Hidden Chain (hand-crafted) | 10 |
| Chain Vault | Fork template (generated, seeds 100–104) | 10–12 |
| Mosaic Grid | Barrier template (typical) | Varies, usually 5+ (agent has unpainted tiles to traverse) |
| Fuse Grid | Race/Two Locks (typical) | Varies widely (agent can still move in unmutated regions) |

All current puzzles satisfy k >= 5. Chain Vault is the tightest — the fork template leaves 10-12 reachable dead states, corresponding to ~8 non-trivial moves. The new templates (conservation trap, key-behind-key) with 2+ keys and 3+ doors will have larger Walking Dead state spaces since the agent has more grid and unused keys to interact with.

### Compliance check

- [x] All current instances satisfy k >= 5 (verified empirically)
- [ ] Add k >= 5 as a hard filter in deadlock discovery — reject deadlock catalog entries where the Walking Dead continuation has fewer than 5 reachable unique states
- [ ] Store k_horizon as instance metadata

---

## Axiom 4: Visual & Spatial Parity

**Requirement:** The transition from s_viable to s_dead must involve minimal visual change. The macroscopic visual state must not trivially reveal the deadlock. No spawning of massive barriers. The deadlock must be:
- **Topological**: a critical connectivity edge was severed (but the grid looks the same density)
- **Combinatorial**: an irreversible resource allocation was made (but the grid looks normal)
- **Temporal**: a delayed phase shift was triggered (but the grid hasn't changed yet)

**Why this matters:** Vision-language models might use visual shortcuts ("dead-ends look cluttered", "fewer open tiles = stuck") instead of reasoning about the game state. The deadlock should be invisible at the pixel level.

| Puzzle | Visual change at fatal step | Deadlock type | Visual parity |
|:---|:---|:---|:---|
| Mosaic Grid | One tile changes from floor to painted (same as every other step) | Topological (remaining tiles are disconnected) | Good — fatal step looks identical to any other step. Deadlock requires tracing connectivity, not visual inspection. |
| Chain Vault | One door opens (same visual as opening the correct door) | Combinatorial (resource exhausted) | Excellent — the Walking Dead state looks completely normal. Agent is in an open corridor. No visual cue at all. |
| Fuse Grid | Fuse icon changes from unarmed to armed (small visual change) | Temporal (consequences haven't materialized yet) | Excellent — grid hasn't changed at all at the moment of the fatal step. Bridge is still there, wall is still closed. |

### Compliance check

- [x] All puzzles produce deadlocks that are not trivially visible from the rendered image
- [x] Fatal step involves minimal visual mutation (1 tile change or less)
- [ ] (Optional validation) Run a visual similarity metric between pre-fatal and post-fatal rendered images to quantify pixel change

---

## Axiom 5: Instance Metadata Schema

**Requirement:** Every generated instance in the final dataset should include structured metadata enabling reproducibility, validation, and subgroup analysis.

**Target schema per instance:**

```json
{
  "instance_id": "<puzzle>_<template>_<seed>",
  "puzzle": "chain_vault | mosaic_grid | fuse_grid",
  "template": "<template_name>",
  "grid_size": [width, height],
  "seed": "<int>",

  "solution": {
    "actions": ["<action_1>", "..."],
    "length": "<int>"
  },

  "anti_greedy_proof": {
    "strategies_tested": ["<name_1>", "..."],
    "all_fail": true,
    "greedy_traces": {
      "<strategy_name>": {
        "actions": ["..."],
        "enters_s_dead": true,
        "t_fatal_greedy": "<int or null>"
      }
    }
  },

  "deadlock_catalog": [
    {
      "fatal_step": "<int>",
      "fatal_action": "<action_name>",
      "k_horizon": "<int, reachable dead states>",
      "bottleneck_mechanism": "topological | combinatorial | temporal",
      "cause_symptom_gap": "<int, steps between fatal step and physical stuck>"
    }
  ],

  "rq_instances": {
    "rq1": { "n_values": [1, 3, 5, 10] },
    "rq2": { "k_values": [1, 3, 5, 10], "pairs_generated": "<int>" },
    "rq3": { "trace_length": "<int>", "fatal_step": "<int>", "gap": "<int>" }
  }
}
```

### Compliance check

- [ ] `GeneratedInstance` dataclass includes all metadata fields above
- [ ] Greedy traces stored per instance (not just pass/fail)
- [ ] k_horizon computed and stored per deadlock entry
- [ ] Bottleneck mechanism labeled per deadlock entry
- [ ] Export to JSON implemented for dataset assembly
