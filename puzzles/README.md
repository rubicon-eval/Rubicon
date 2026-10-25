# RUBICON puzzle environments

The release contains three environments. Their current engine code is authoritative; historical per-domain design notes may describe earlier experiments.

| Environment | Actions | Irreversible constraint |
|---|---|---|
| `mosaic_grid` | UP, DOWN, LEFT, RIGHT | Paint on departure; painted cells cannot be revisited |
| `chain_vault` | UP, DOWN, LEFT, RIGHT | Moving toward an adjacent locked door with matching keys consumes the keys and opens the door **without moving the agent**; a later move enters it |
| `tide_grid` | UP, DOWN, LEFT, RIGHT, WAIT | Time advances and oxygen is consumed on every action; tidal zones alternate between dry and flooded |

All environments provide a symbolic engine, search oracle, instance generator, renderer, prompt templates and a CLI:

```bash
uv run python -m puzzles.mosaic_grid --help
uv run python -m puzzles.chain_vault --help
uv run python -m puzzles.tide_grid --help
```

Use `demo` for examples, `generate` for symbolic instances, `build` for evaluation files, or `full` for the combined pipeline. Each subcommand has `--help`.

Common utilities live in `common/`: positions/directions, A* search, and grid rendering.
The three tasks are simulation fidelity (RQ1), solvability/foresight (RQ2), and attribution of the first fatal action (RQ3).

The fixed public release is on [Hugging Face](https://huggingface.co/datasets/ajirs/rubicon). Download it with `uv run python scripts/download_dataset.py` from the repository root. See the [main README](../README.md) for evaluation and inspection commands.
