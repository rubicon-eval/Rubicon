# Chain Vault — Improvements Plan

Derived from the cross-puzzle improvements tracker in `puzzles/README.md`. Tracks what's done, what's next, and why each item matters for reviewer confidence.

---

## Status Tracker

Quick-reference for current progress on the improvement plan. See the **Master Priority Table** section for full details on each item.

Legend: ✅ Done · 🔄 In progress · ⏭ Deferred · ⬜ Todo

### P0 — Pre-generation blockers

| ID | Change | Status | Notes |
|:--|:--|:--|:--|
| P0.1 | Remove USE_KEY hint from `build_rq3_prompt` | ✅ Done | Committed 17a34c4. Ablation prompt retains hint as scaffolded variant. |
| P0.2 | Counterfactual rephrasing of RQ3 prompt | ⏭ Deferred | Will be handled in unified prompt revision pass across all puzzles. |
| P0.3 | Parallel Wings (T5): gate right-wing key behind blue single-lock | ✅ Done | 3 keys (K_b, K_r, K_b2), horizontal gate wall in right wing. 100% success all grid sizes. |
| P0.4 | Key-Behind-Key (T3): scale chain depth with grid size | ✅ Done | depth-2 at ≤6×6, stochastic depth-3 at 7×7 (~20% d3), deterministic depth-3 at 8×8+ (~54% d3). `_generate_kbk_depth3` helper. 3-key chain (K_r→K_g→K_b), 5 doors, 2 deadlocks per instance. |

### P1 — Pre-submission essentials

| ID | Change | Status | Notes |
|:--|:--|:--|:--|
| P1.1 | Replace Conservation Trap (T2) with Budget Partitioning | ✅ Done | `generate_budget_partitioning_grid`: 2 RED + 1 BLUE, 3 doors (D_A correct, D_B trap, D_C dual-lock). Vertical divider separates Zone 1_correct/Zone 1T. TEMPLATE_REGISTRY["conservation_trap"] → new fn. 100% validation success, avg 3-5 deadlocks/instance. |
| P1.2 | Key Collector (T12): Path Commitment variant at 7×7+ | ✅ Done | `_generate_kc_path_commitment`: K_r/K_b/K_g all in Zone 0, dual convergent forks (K_r near gate_r RED, K_b near gate_b BLUE), bypass (free) on far-half, D_exit (R+B+G triple-lock). 50% stochastic at w≥7. 100% pipeline success (7×7/8×8), avg 6-12 deadlocks/instance, sol 18-31 steps. |
| P1.3 | SDL (T11): stochastic sub-variant selection | ✅ Done | `_generate_sdl_same_color_fork`: 4 colors (R,B,Y,G), inner Y-fork in Zone 1 with D_trap_y + D_correct_y (both yellow single-lock). 50% stochastic at w≥7∧h≥7. 100% pipeline success (6×6/7×7/8×8, 50 seeds). 7×7: ~24% SCF instances, 8×8: 100% SCF. Sol 21-35 steps, 2-7 deadlocks. |
| P1.4 | SDL (T11): convergent-forks bait at Zone 1 | ✅ Done | Best-effort D_trap_b_z1 (blue single-lock) on outer divider adjacent to K_b2. Opens into Zone 1T_b (existing dead end). 7×7 standard: ~39% instances have Z1-bait, raises min deadlocks 2→3, avg 3.9→4.5. 6×6 too tight (0%). 100% pipeline success all sizes. |
| P1.5 | Convergent Forks (T4): sequential fork dependency variant | ✅ Done | `_generate_cf_sequential_fork`: 3 colors (R,G,B). Barrier 1: D_trap_r + D_trap_g + bypass. Barrier 2: D_correct_g (green → Area A with K_b) + D_dual (R+B). Below barrier 2: Area A | divider | exit zone. Sequential dependency G→K_b→R+B. 50% stochastic at w≥6∧h≥6. 100% pipeline success (5-8×, 50 seeds). SF instances: ~26-34% of seeds, sol 21-34, dl 3-8. |
| P1.6 | Investment Door (T7): variable investment ratio | ✅ Done | `_generate_id_1key`: 1-key investment (ratio 1:1). K_g on left side near D_trap_g on barrier (convergent bait). Investment room yields only K_b. Row ordering r_invest < r_trap_g < r_split < r_trap_r < r_bypass. 50% stochastic at h≥7. 100% pipeline success (6-8×, 50 seeds). 7×7: ~20% 1-key (dl avg 8.5), 8×8: ~46% 1-key (dl avg 8.1). |
| P1.7 | Relay (T6): seed-level color palette permutation | ✅ Done | Permute {R,B,G} palette per seed via `rng.shuffle`. Gate, zone1-key-A, zone1-key-B colors vary across instances. Key/door IDs use actual color initial. 100% pipeline success (6-8×, 50 seeds). Gate color distribution: ~⅓ each. |
| P1.8 | Scarce Resource (T9): ambiguous scarcity variant | ✅ Done | `_generate_sr_ambiguous`: 3R+1B+1G, both B and G critical via B+G dual-lock exit. Two convergent bait traps (D_trap_b, D_trap_g) on barrier 1. 50% stochastic at w≥6. 100% pipeline success (5-8×, 50 seeds). Ambig: dl avg 17-25 (vs std 10-16), significantly more deadlocks from double-bottleneck. |
| P1.9 | Cascade Trap (T10): cascade depth 2 | ✅ Done | `_generate_ct_depth2` helper; stochastic at 6×6 (50%); 3 keys (R,Y,B), 4 doors; depth-2 chain: R waste → no K_y → D_y locked → no K_b → D_b locked → stuck; ~31% depth-2 in pipeline; sol 22-29, dl=1; widened sol range to (12,32) for 6×6, lowered min_dl to 1 |
| P1.10 | Decoy Key (T13): mixed should-use / should-save stages | ✅ Done | Color-swap variant: Stage 1 flips to "USE matching" (K_y→D_correct_y, with K_y2 replacement in Zone 1C). Same spatial layout, different color assignments. Stochastic at w≥6 (50%). 44-58% mixed in pipeline. Breaks "always save matching" meta-policy. |
| P1.11 | ODL (T14): false compensation in trap zone | ✅ Done | Stochastic at w≥6 (50%): adds K_b2 (blue decoy) in Zone 1B. Agent wastes R on trap, finds extra blue — but exit needs R+B and red is gone. Passes `_all_keys_relevant` (blue used in oracle). 40-68% false-comp in pipeline. |
| P1.12 | Cross-template: seed-level color palette permutation | ✅ Done | `_permute_colors` in `_transform_instance`: collects unique colors, shuffles, remaps keys/doors/inventory. Works for all 14 templates. 2-color templates swap ~50%; 3-color templates get 6 permutations. 100% pipeline success, all solutions valid post-permutation. |

### P2 — Pre-submission enhancements

Worth doing if time allows. Each improves a specific template without being a prerequisite for submission.

| ID | Change | Status | Notes |
|:--|:--|:--|:--|
| P2.1 | Fork (T1): 3-way fork (2 traps, 1 correct) at 7×7+ | ✅ Done | Binary choice → ternary. Vertical barrier, 3 zones, convergent bait near closest trap approach. |
| P2.2 | Fork (T1): content-informed fork — both doors lead to real zones | ✅ Done | Red decoy key in trap zone at 6×6+. Model must check which zone has exit. |
| P2.3 | Budget Partitioning (T2): cross-color budget (2R+2B) at 7×7+ | ✅ Done | 3 barriers, 5 doors (red+blue traps + dual-lock exit). Both color budgets must be managed. |
| P2.4 | KBK (T3): cyclic chain (R→G→R₂→B→exit) at 7×7+ | ✅ Done | K_r2 in Zone R, D_blue→dual-lock. Red appears twice in chain. |
| P2.5 | Convergent Forks (T4): 3-way fork + triple-lock exit at 7×7+ | ✅ Done | 3 trap doors (R/B/G) + triple-lock exit. Rules + renderer updated for triple-lock. |
| P2.6 | Parallel Wings (T5): internal corridor walls forcing path commitment | ✅ Done | Horizontal strip in left wing at 6×6+, gap only at trap approach col. |
| P2.7 | Relay (T6): same-color fork in Zone 1 — two blue doors, one correct | ⏭️ Defer | Requires significant Zone 1 restructuring (internal sub-zones + doors). Relay already has good variance via P1.7 color permutation + extra traps. |
| P2.8 | Relay (T6): ordered zone collection — green behind blue door | ⏭️ Defer | Requires extra blue key + internal blue door in Zone 1. Complex layout changes for sequential collection. |
| P2.9 | Investment Door (T7): bad investment trap — second investment door yields insufficient keys | ✅ Done | Red door on barrier entering exit area (already accessible via bypass). Wastes red key. 7×7: 20%, 8×8: 28%. D_bad_invest always in deadlock catalog. |
| P2.10 | Switchback (T8): 3 keys, 4 same-color doors at 7×7+ | ✅ Done | `_generate_sb_3key`: 3 barriers, 4 doors (D_A/D_B/D_C/D_trap). 50% at 7×7+. Sol 17-31, dl 4-16. |
| P2.11 | Switchback (T8): mixed-color pair — blue door guards exit + blue trap | ✅ Done | D_B→blue, K_b+D_trap_b on barrier 1. 50% at w>=6. 6×6: 44%, 7×7: 28%, 8×8: 17%. |
| P2.12 | Scarce Resource (T9): two-color simultaneous bottleneck at 7×7+ | ✅ Done | `_generate_sr_2color`: 3 barriers, 6 doors (trap+correct for B and G). 7×7: 44%, 8×8: 49%. |
| P2.13 | Cascade Trap (T10): two-way cascade — two independent spatial traps | ✅ Done | `_generate_ct_2way`: 3 barriers, 3-zone split. D_trap_r + D_trap_g independent cascades. 7×7: 89%, 8×8: 100%. |
| P2.14 | SDL (T11): internal wall corridors in Zone 0 for non-trivial key collection | ✅ Done | Horizontal corridor wall at 7×7+ with gap, forces longer key collection path. |
| P2.15 | Key Collector (T12): hidden key behind red door at 7×7+ | ✅ Done | Pocket approach: isolate edge/corner cell in Zone 1 behind red door. K_g in pocket, K_r2 in Zone 0. 7×7: 10%, 8×8: 10%. |
| P2.16 | Decoy Key (T13): 3-stage temptation at 8×8 | ✅ Done | `_generate_dk_3stage`: 3 barriers, 4 keys (R/Y/B/G), 7 doors. Yellow saved Stage 1→3. 8×8: 34%. |
| P2.17 | Decoy Key (T13): quantity mismatch — 2 yellow (losing 1 OK) vs 1 blue (fatal) | ✅ Done | K_y2 at w>=7. Non-fatal yellow loss vs fatal blue loss. 7×7: 25%, 8×8: 12%. |
| P2.18 | ODL (T14): content-informed same-color fork — green key in trap zone | ⏭️ Skip | Green decoy key conflicts with `_all_keys_relevant` filter — green is deliberately irrelevant. Would need pipeline modification. |
| P2.19 | ODL (T14): 3-way same-color fork at 7×7+ | ✅ Done | Already implemented via existing `extra_trap_odl_b1` at w>=6. 100% of 6×6+ instances have 3+ red doors. |
| P2.20 | Cross-template: richer trap zone contents (irrelevant keys/doors) | ⏭️ Skip | Same `_all_keys_relevant` conflict as P2.18 for irrelevant keys. Locked doors in trap zones would need per-template custom logic. |
| P2.21 | Cross-template: template-aware meta-policy as validation filter #6 | ⏭️ Defer | Requires defining per-template optimal policies for 14 templates + simulation. High effort, all P1/P2 changes stable first. |

### P3 — Next version / camera-ready

Deferred without risk to the current submission. Either high effort, low urgency, or dependent on paper feedback.

| ID | Change | Status | Notes |
|:--|:--|:--|:--|
| P3.1 | Multi-decision RQ3 traces — multiple non-fatal wrong moves before fatal step | ⬜ Todo | Extended `generate_hard_rq3_instance`. High diagnostic value, complex implementation. |
| P3.2 | New template: "Conditional Allocation" — correct door depends on currently held resources | ⬜ Todo | Fills largest gap in dependency graph taxonomy. Plan as future work in paper. |
| P3.3 | Relay (T6): 4-color relay extension at 8×8 | ⬜ Todo | Architectural complexity increase. Only meaningful if 8×8 prioritized. |
| P3.4 | SDL (T11): 3 sequential dual-locks at 8×8 | ⬜ Todo | 3-candidate fatal steps makes RQ3 attribution much harder. |
| P3.5 | KBK (T3): parallel chains with shared endpoint | ⬜ Todo | Major structural redesign. |
| P3.6 | Parallel Wings (T5): 3-wing triple-lock at 7×7+ | ⬜ Todo | Natural extension of P0.3 gating. |
| P3.7 | Key Collector (T12): progressive reveal of lock requirements in image | ⬜ Todo | Renderer + prompt changes. Separate evaluation modality study. |
| P3.8 | Template-blind evaluation split — hold out 3-4 templates from few-shot | ⬜ Todo | Important for paper claims. Zero implementation cost; eval pipeline setup. |
| P3.9 | Per-template accuracy + variance reporting | ⬜ Todo | Distinguishes meta-policy learning from genuine reasoning. Paper analysis. |
| P3.10 | Stochastic trap count uniformly across all templates | ⬜ Todo | Systematize `_pick_extra_trap_col`. Low urgency — scaling already handles this. |

---

## Completed

### Core components *(always done — no changes needed)*

These components are fully general and already support multiple keys, colors, and dual-lock doors. Verified with generated instances across all implemented templates (conservation_trap uses 2 colors + dual-lock, key_behind_key uses 2 colors + 3 doors).

| Component | Why it already works |
|:---|:---|
| **Engine** (`engine.py`) | `KeyInfo.color` is a free string, `DoorInfo.required_keys` is a tuple of arbitrary length, inventory tracks per-color counts. Dual-lock doors list 2+ colors in `required_keys`. |
| **Oracle** (`oracle.py`) | A* search and `is_solvable` work over arbitrary state spaces. `find_fatal_step` checks solvability at each step regardless of key/door count. |
| **Renderer** (`renderer.py`) | `KEY_DOOR_COLORS` has entries for red, blue, green, yellow. `_draw_door_bars` alternates bar colors for dual-lock. `_door_label` produces "R+B" labels. Inventory HUD renders multiple colored keys. |
| **Prompts** (`prompts.py`) | `generate_rules_text` iterates all keys/doors, labels doors as "single-lock" or "dual-lock", lists required colors. Rule 6 already explains dual-lock mechanics. |
| **Deadlock discovery** (`generator.py`) | Only checks `is_solvable` after USE_KEY actions. Monotonicity optimization holds with any number of keys/colors. |

---

### Generator infrastructure

| Item | Details |
|:---|:---|
| **6 greedy strategies** (incl. 2 BFS) | 4 manhattan-heuristic strategies + `simulate_bfs_toward_exit` + `simulate_bfs_nearest_door`. `_bfs_to()` helper treats locked doors as true walls. `all_greedy_strategies_fail()` requires all 6 to fail. |
| **`_all_keys_relevant` filter** (filter #5) | Every key's color must appear in the oracle solution OR be the fatal key in a deadlock catalog entry. Verified: spare-key puzzles correctly rejected. |
| **Template registry + per-template config** | `TEMPLATE_REGISTRY: dict[str, Callable]`, `TEMPLATE_NAMES: list[str]`, `TEMPLATE_SOLUTION_RANGES`, `TEMPLATE_MIN_DEADLOCKS`. All defined at module level. |
| **`generate_validated_instance` `template` param** | Accepts `template: str = "fork"`, looks up grid fn + defaults from registry/config dicts. Raises `ValueError` for unknown names. |

**Validation filters — all 5 implemented:**

1. Solvable from start
2. Solution length in per-template range
3. All 6 greedy strategies fail
4. Sufficient deadlocks (per-template minimum)
5. All keys relevant (`_all_keys_relevant`)

---

### Hand-crafted instance removal

Removed all hand-crafted examples from `instances.py`, `KNOWN_SOLUTIONS` dict, and legacy `simulate_greedy` alias from `generator.py`. Updated `demo.py` to iterate over `TEMPLATE_NAMES` and generate one instance per implemented template — no hand-crafted fallback.

---

### Templates

All 14 templates implemented and stress-tested.

| # | Template | Status |
|:--|:--|:--|
| 1 | **Fork** | Done |
| 2 | **Conservation Trap** | Done |
| 3 | **Key-Behind-Key** | Done |
| 4 | **Convergent Forks** | Done |
| 5 | **Parallel Wings** | Done |
| 6 | **Relay** | Done |
| 7 | **Investment Door** | Done |
| 8 | **Switchback** | Done |
| 9 | **Scarce Resource** | Done |
| 10 | **Cascade Trap** | Done |
| 11 | **Sequential Dual-Locks** | Done |
| 12 | **Key Collector** | Done |
| 13 | **Decoy Key** | Done |
| 14 | **Order-Dependent Lock** | Done |

**Template 1: Fork** — 1 red key, 2 red doors. Right side split by horizontal divider into Zone R_trap (real walkable area, 3+ cells) and Zone R_correct (exit). Red key placed near trap door approach (convergent forks bait). Anti-greedy: convergent forks key placement + post-hoc greedy filter. Deadlocks: 1.

**Template 2: Conservation Trap** — 2 keys (red + blue), 2 doors (red trap + red+blue dual-lock). Two horizontal barriers: barrier 1 has trap door + bypass gap, barrier 2 has dual-lock as only passage. Trap column walled in middle zone to prevent BFS bypass. Exit biased toward trap side. Greedy wastes red on trap → can't open dual-lock. 20/20 pipeline runs succeed across all grid sizes (5×5–7×7).

#### Template overview

| # | Template | Keys | Colors | Dual-locks | Dependency structure | Core reasoning | Grid |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 1 | **Fork** | 1 | 1 | 0 | Binary fork | Correct door selection | 5×5–8×8 |
| 2 | **Conservation Trap** | 2 | 2 | 1 | Save both for combined | Budget conservation | 5×5–8×8 |
| 3 | **Key-Behind-Key** | 2 | 2 | 0 | 2-step chain | Sequence dependencies | 5×5–8×8 |
| 4 | **Convergent Forks** | 2 | 2 | 1 | Two areas → dual-lock exit | Both branches needed | 5×5–8×8 |
| 5 | **Parallel Wings** | 2 | 2 | 1 | Two chains → dual-lock merge | Domain alignment + combine | 5×5–8×8 |
| 6 | **Relay** | 3 | 3 | 1 | 3-step chain → dual-lock | Extended chain + combination | 6×6–8×8 |
| 7 | **Investment Door** | 3 | 3 | 1 | Spend 1 → unlock 2 → dual-lock | Investment vs. conservation | 6×6–8×8 |
| 8 | **Switchback** | 2 | 1 | 0 | Same-color used twice correctly | Reuse sequencing | 5×5–8×8 |
| 9 | **Scarce Resource** | 4 | 2 | 0 | Asymmetric abundance | Bottleneck identification | 5×5–8×8 |
| 10 | **Cascade Trap** | 2 | 2 | 0 | Spatial + resource | Spatial consequence of key use | 5×5–7×7 |
| 11 | **Sequential Dual-Locks** | 3 | 3 | 2 | Two dual-locks in sequence | Multi-stage combination planning | 6×6–8×8 |
| 12 | **Key Collector** | 3 | 3 | 1 (triple-lock) | Collect all 3, preserve for triple-lock | Sequential single-lock temptation | 6×6–8×8 |
| 13 | **Decoy Key** | 3 | 3 | 0 | Two-stage temptation: save key for hidden use | Resist visible matching door | 6×6–8×8 |
| 14 | **Order-Dependent Lock** | 3 | 2 | 1 | Same-color fork + hidden key | Temporal ordering of identical actions | 5×5–8×8 |

**Dual-lock distribution:** 0 dual-locks (templates 1, 3, 8, 9, 10, 13) → 1 dual-lock (templates 2, 4, 5, 6, 7, 14) → 2 dual-locks (templates 11, 12). Natural difficulty escalation: single-resource → single combination → multi-stage combination.

#### Acceptance criteria

| Template | Min gen success | Min deadlocks | Solution range | All greedy fail |
|:---|:---|:---|:---|:---|
| Fork | 100% | 1 | 6–20 | Yes |
| Conservation Trap | 100% | 2 | 10–22 | Yes |
| Key-Behind-Key | 100% | 1 | 12–20 | Yes |
| Convergent Forks | 100% | 3–4 | 10–22 | Yes |
| Parallel Wings | 100% | 2–5 | 15–24 | Yes |
| Relay | 100% | 2–7 | 17–26 | Yes |
| Investment Door | 100% | 5–6 | 23–24 | Yes |
| Switchback | 100% | 2–3 | 11–20 | Yes |
| Scarce Resource | 100% | ≥1 | 8–20 | Yes |
| Cascade Trap | 100% | 2 | 14–22 | Yes |
| Sequential Dual-Locks | 100% | 3–4 | 18–23 | Yes |
| Key Collector | 100% | ≥2 | 14–28 | Yes |
| Decoy Key | 100% | ≥2 | 12–28 | Yes |
| Order-Dependent Lock | 100% | 2 | 10–22 | Yes |

Stress test: 100+ seeds per template. All must produce valid RQ2 pairs (k_horizon ≥ 3) and RQ3 traces (k_horizon ≥ 5). Verified via `is_eval_ready()` gate — see P1: Hard variants section.

---

#### Generation health

**Fork (5×5):**

| Metric | Value |
|:---|:---|
| Gen success rate | 100% |
| Avg solution length | 13.1 |
| Avg deadlocks/instance | 1.0 |
| RQ3 trace success | 98% |

**Conservation Trap (5×5):**

| Metric | Value |
|:---|:---|
| Gen success rate | 100% (20/20 pipeline runs) |
| Avg solution length | 12.9 |
| Avg deadlocks/instance | 2.0 |
| Solution range | 10–16 steps |

**Key-Behind-Key (5×5):**

| Metric | Value |
|:---|:---|
| Gen success rate | 100% (100/100 pipeline runs) |
| Avg solution length | 15.3 |
| Avg deadlocks/instance | 1.0 |
| Solution range | 12–20 steps |

**Convergent Forks (5×5):**

| Metric | Value |
|:---|:---|
| Gen success rate | 100% (25/25 pipeline runs) |
| Avg solution length | 13–18 |
| Avg deadlocks/instance | 3–4 |
| Solution range | 10–22 steps |

**Switchback:** Raw yield 63.0% (5×5) / 64.0% (6×6). Pipeline 100% (5×5) / 100% (6×6). Solution lengths 11–20, deadlocks 2–3 (avg 2.8).

**Relay:** Raw yield 100% (6×6) / 100% (7×7). Pipeline 100% (6×6) / 100% (7×7). Solution lengths 17–26, deadlocks 2–7 (avg 4.1).

**Investment Door:** Raw yield 35.6% (6×6) / 62.6% (7×7). Pipeline 100% (6×6) / 100% (7×7). Solution lengths 23–24, deadlocks 5–6 (avg 5.5).

**Scarce Resource:** Raw yield 100% (5×5) / 100% (6×6). Pipeline 100% (5×5) / 100% (6×6). Solution lengths 8–20, avg 15.4 (5×5) / 18.0 (6×6). Deadlocks avg 10.5 (5×5) / 11.8 (6×6).

**Cascade Trap:** Raw yield 58.0% (5×5) / 60.6% (6×6). Pipeline 100% (5×5) / 100% (6×6). Solution lengths 14–18 (5×5) / 14–22 (6×6). Deadlocks: 2 consistently.

**Sequential Dual-Locks:** Raw yield 12.0% (6×6) / 28.4% (7×7). Pipeline 100% (6×6, 25 seeds) / 100% (7×7, 25 seeds). Solution lengths 18 (6×6) / 21 (7×7). Deadlocks: min 2, avg 3.7–4.0.

**Key Collector:** Raw yield ~50-70% (6×6) / ~60-80% (7×7). Pipeline 100% (25 seeds). Solution lengths 14-28. Deadlocks: 2-6 (avg ~4).

**Order-Dependent Lock:**

| Metric | 5×5 | 6×6 |
|:---|:---|:---|
| Gen success rate | 100% (100/100) | 100% (100/100) |
| Solution lengths | 17–18 | 14–22 |
| Avg solution length | 17.5 | 19.1 |
| Deadlocks | 2 | 2–3 (avg 2.0) |
| All greedy fail | Yes | Yes |

---

### Template details

#### Template 2: Conservation Trap

**Mechanic:** 2 keys (red + blue), 1 red single-lock trap door on barrier 1, 1 red+blue dual-lock on barrier 2. Bypass gap on barrier 1 avoids the trap. Greedy wastes red key on trap → can't open dual-lock.

**Two-barrier design:**
- Barrier 1 (upper): walls + red trap door gap + bypass gap (edge column)
- Barrier 2 (lower): walls + red+blue dual-lock as only gap
- Trap column walled in middle zone (prevents BFS from reaching trap from below)
- Dual-lock placed on bypass side of trap column (middle-zone path doesn't cross trap)
- Exit below barrier 2, biased toward trap side

**Anti-greedy property:** Verified by 6 greedy strategy filters (no manhattan-distance check). Trap is encountered because greedy agents heading toward exit pass through the trap column in the upper zone. The bypass is on the far edge.

**Connectivity constraints:**
- Upper zone: start + both keys + bypass_above + trap_above (all connected)
- Middle zone: bypass_below → dl_above connected (trap side isolated)
- Exit zone: dl_below → exit connected

**Deadlocks:** 2 (waste red alone on trap; waste red while holding blue on trap).

**Grid:** 5×5–8×8. 5×5 minimum (b1=1, b2=3 fixed), larger grids add more variety and extra traps.

---

#### Template 3: Key-Behind-Key

**Mechanic:** Sequential dependency chain. Red key (start side) -> correct red door -> blue key (behind correct door) -> blue door -> exit. Red trap door leads to real walkable zone (zone divider pattern). Wasting red key on trap makes blue key permanently unreachable.

**Zone divider design:**
- Vertical barrier at column b ∈ [1, width-4]
- Trap red door on barrier at edge row
- Correct red door on barrier far from trap
- Right side split by horizontal divider into Zone R_trap (3+ cells) and Zone R_correct
- Exit pocket at far corner of Zone R_correct (opposite trap_row)
- Blue door gates the exit pocket (standard pocket wall pattern)

**Anti-greedy invariant:** Convergent forks key placement (red key near trap door approach on left side) + post-hoc greedy filter (all 6 strategies must fail).

**Connectivity constraints:**
- Start -> red key -> both red door approach sides (left of barrier)
- D_correct -> blue key + blue door approach (Zone R_correct)
- Blue door -> exit pocket (isolated by pocket wall)
- D_trap -> Zone R_trap (real walkable area, disconnected from Zone R_correct)

**Deadlocks:** 1 (waste red on trap → blue key permanently unreachable behind correct door). Min deadlocks set to 1 because the template naturally produces exactly one deadlock scenario.

**Grid:** 5×5–8×8. 100% success rate across all tested sizes.

---

#### Template 4: Convergent Forks

**Mechanic:** Two spatially separated areas, each guarded by a single-lock door (red guards area A, blue guards area B). The exit is behind a red+blue dual-lock door. Agent has 2 red keys + 2 blue keys. One red opens area A, one blue opens area B, then both remaining keys open the dual-lock exit. Trap doors consume extra keys so the dual-lock budget runs out.

**Anti-greedy invariant:** Trap doors are adjacent to keys (greedy uses key immediately on trap rather than routing to the area doors first).

**Layout:**
- Two-wing grid: left wing (area A, red door) and right wing (area B, blue door)
- Central corridor connecting both wings
- Dual-lock exit at end of central corridor
- Single-lock red trap and single-lock blue trap placed on approach paths

**Deadlocks:** 3+ (waste red on trap → can't open dual-lock; waste blue → same; waste both).

**Grid:** 5×5–8×8.

---

#### Template 5: Parallel Wings

**Mechanic:** Two completely independent resource chains, one per side of the grid. Left wing requires red key → red door → progress. Right wing requires blue key → blue door → progress. Main exit behind red+blue dual-lock. Deadlock: use red key in right wing (or blue in left) — cross-contaminate resources across domains.

**Key distinction from Conservation Trap:** No single bypass corridor; both wings must be completed. The challenge is spatial alignment — keeping resources matched to their correct domain.

**Layout:**
- Bilateral symmetric grid, vertical divider
- Left side: red key + red door + left-area objective
- Right side: blue key + blue door + right-area objective
- Bottom corridor with dual-lock exit
- Cross-contamination traps: red door stub on right side, blue door stub on left side

**Deadlocks:** 2–3 (cross-contaminate R, cross-contaminate B, both).

**Grid:** 5×5–8×8.

---

#### Template 6: Relay

**Mechanic:** 3-color sequential gating with bypass + exit pocket. Red key -> red gate (correct) -> zone 1 with blue+green keys -> bypass gap -> zone 2 -> blue+green dual-lock exit pocket -> exit. Red trap door leads to Zone 1T (real walkable area split by vertical divider). Blue and green traps on barrier 2 lead to zone 2 (genuinely tempting — real passage) but waste keys needed for the dual-lock.

**Key design:** The dual-lock is moved OFF barrier 2 into an exit pocket at a bottom corner (parallel wings pattern). Barrier 2 has a bypass gap on the far-side edge (conservation trap pattern). All barrier 2 openings (traps + bypass) lead to zone 2 — traps are tempting because they provide real passage, but waste a dual-lock key.

**Anti-greedy invariant:** Barrier 1: red trap on exit-side half (closer to exit); correct door on far side. Barrier 2: keys directly above matching traps -> USE_KEY fires immediately on pickup adjacency.

**Layout:**
- Zone 0 (rows 0..b1-1): Start + red key, biased toward exit side
- Barrier 1 (row b1): Red gate door + red trap door
- Zone 1 (rows b1+1..b2-1): Split by vertical divider:
  - Zone 1T (trap side): Real walkable area (2+ cells, no keys, no barrier 2 access)
  - Zone 1C (correct side): Blue key + green key (above matching traps)
- Barrier 2 (row b2): Blue trap + green trap + bypass gap (far-side edge)
- Zone 2 (rows b2+1..h-1): Open area + dual-lock exit pocket at corner

**Exit pocket:** exit_side=0 → exit (h-1,0), DL (h-1,1), pocket wall (h-2,0). exit_side=w-1 → exit (h-1,w-1), DL (h-1,w-2), pocket wall (h-2,w-1). Exit isolated: only reachable through dual-lock.

**Correct path:** Start → red key → red gate → zone 1 → blue+green keys → bypass → zone 2 → dual-lock → exit.

**Deadlocks:** 3+ (waste red on trap -> Zone 1T (real area but no keys, no barrier 2 access); waste blue on trap -> enters zone 2 but can't open dual-lock; waste green on trap -> enters zone 2 but can't open dual-lock).

**Grid:** 6×6–8×8 (5×5 impossible — too small for 3-stage layout).

---

#### Template 7: Investment Door

**Mechanic:** Agent picks up 1 red key. A red "investment door" on a vertical barrier leads to an investment room containing both a blue key and a green key. Exit requires blue+green dual-lock. The correct play requires *spending* your only key as an investment to gain two — then returning to cross a bypass gap.

**Anti-greedy invariant:** Red trap door closer to exit than investment door (manhattan), guaranteed by row ordering (r_invest < r_trap, exit at bottom). Keys directly above matching split traps (convergent forks pattern) cause greedy agents to USE_KEY immediately on pickup.

**Layout:**
- Left side (cols 0..b-1): Start + Red key
- Vertical barrier (col b): Investment door (top) + Red trap (bottom) + Bypass gap (near bottom)
- Right side: Investment room (rows 0..r_split-1) with blue+green keys above split traps
- Split wall (row r_split): Blue trap door + Green trap door
- Exit area (rows r_split+1..h-1): Open area + dual-lock exit pocket at corner

**Deadlocks:** 3+ (red on red trap → exit area but no blue/green; blue on blue split trap → exit area with only green → can't open DL; green on green split trap → exit area with only blue → can't open DL; both on split traps → no keys for DL).

**Grid:** 6×6–8×8 (5×5 impossible — too small for vertical barrier + split).

---

#### Template 8: Switchback

**Mechanic:** 2 red keys (both collectable), 3 red doors. Agent must use one red key on door A (first correct door), navigate to a different grid region, then use the second red key on door B (second correct door). Third red door is a trap — placed spatially between A and B on the greedy route.

**Key distinction:** Same color used twice *correctly*. The reasoning challenge is identifying which same-color doors are the correct ones.

**Anti-greedy invariant (triple bait):**
- Key 1 directly above trap door (convergent forks pattern) — agent picks up key and is immediately adjacent to trap → strong USE_KEY temptation.
- Zone 1T (trap side) is larger than Zone 1A — divider biased toward correct side so trap area looks like the "main" path.
- Trap on exit-side half (closer to exit), exit below trap side. `manhattan(trap, exit) < manhattan(correct, exit)` — hard filter.

**Layout:**
- Two horizontal barriers dividing grid into 3 zones
- Zone 0 (top): Start + Key 1 (above trap) + Key 2, biased toward exit side
- Barrier 1: Door A (far-side) + Trap (exit-side), both lead to real areas
- Zone 1 (middle): Split by internal divider wall into Zone 1A (correct side, connects to Door B) and Zone 1T (trap side, larger but disconnected from Door B)
- Barrier 2: Door B (on correct side of divider)
- Zone 2 (bottom): Exit, biased toward exit side

**Deadlocks:** ≥2 (agent picks up K1 above trap → opens trap → Zone 1T → back to Zone 0 → insufficient keys for A+B).

**Grid:** 5×5–8×8.

---

#### Template 9: Scarce Resource

**Mechanic:** 3 red keys + 1 blue key, 2 red doors + 1 blue correct door + 1 blue trap door. Red is abundant (1 spare key), blue is the bottleneck (1 key, 2 blue doors — only 1 leads forward). The model must identify blue as the limiting constraint rather than focusing on the more numerous red keys.

**Key distinction:** Asymmetric scarcity — tests whether model can identify the *limiting constraint* (blue) rather than treating all resources as equally precious. Red keys are a distraction.

**Anti-greedy invariant:**
- D_trap_b on exit-side half of barrier 1 (closer to exit column).
- D_r1 on far-side half (farther from exit). `manhattan(D_trap_b, exit) < manhattan(D_r1, exit)` — hard filter.
- K_b directly above D_trap_b (convergent forks bait pattern) — agent picks up blue key and is immediately adjacent to trap.
- Both barrier 1 doors lead to the SAME Zone 1 — trap provides genuinely useful passage, making it harder to dismiss.

**Layout:**
- Zone 0 (top): Start + K_r1 + K_r2 + K_r3 + K_b (blue key above trap)
- Barrier 1: D_r1 (far-side) + D_trap_b (exit-side, blue) + walls — both lead to same Zone 1
- Zone 1 (middle): Open passage, no divider
- Red Gate (row h-2): D_r2 (red, only opening) + walls — pocket isolation for free
- Exit Row (bottom): D_correct_b (blue) + Exit at corner (standard exit pocket)

**Correct path:** Start → collect all keys → D_r1 (red) → Zone 1 → D_r2 (red) → Exit Row → D_correct_b (blue) → Exit. Uses 2 red + 1 blue = 3 of 4 keys. 1 red key spare.

**Deadlocks:** ≥1 (agent opens D_trap_b with blue key → Zone 1 → D_r2 (red) → Exit Row → D_correct_b needs blue → 0 blue keys → stuck).

**Grid:** 5×5–8×8.

---

#### Template 10: Cascade Trap

**Mechanic:** 2 keys (red + blue), 3 doors. Using the red key on the red trap door is legal and appears to make progress — but the trap door's location means that once opened, the path through it walls the agent into a subregion that doesn't contain the blue key. Blue key becomes spatially unreachable.

**Key distinction:** The deadlock has a *spatial* component on top of resource depletion. The opened door changes the topology — the routing consequence is the actual cause.

**Anti-greedy invariant:** Red trap door is on the greedy path toward the blue key. Greedy opens it thinking it's making progress, then finds itself cut off from blue.

**Connectivity constraints:**
- After opening red trap: agent can no longer reach blue key (BFS confirms)
- After opening red correct: agent can reach blue key and blue door and exit

**Deadlocks:** 2+ (use red on trap → spatially isolated from blue; use blue on blue trap if one exists).

**Grid:** 5×5–7×7 (8×8: greedy "toward_exit" strategy solves it — grid too open for cascade's barriers to block greedy navigation).

---

#### Template 11: Sequential Dual-Locks

**Mechanic:** 3 colors (red, blue, green). Two dual-lock doors in sequence: first is red+blue, second is blue+green. Only 1 blue key. The solution: open the red+blue dual-lock first, which reveals a second blue key behind it; then use the new blue key + green key on the second dual-lock. Deadlock: use blue key on anything before the first dual-lock.

**Key distinction:** Model must reason about a key being needed at multiple future points, and that the solution itself *provides* the resource for the second use.

**Anti-greedy invariant:** Blue single-lock trap door placed closer to exit than the first dual-lock. Greedy wastes blue on trap.

**Layout:**
- Three-zone layout (start → zone 1 → zone 2 → exit)
- First dual-lock (R+B) gates zone 1 → zone 2
- Zone 1 contains second blue key
- Second dual-lock (B+G) gates zone 2 → exit
- Blue trap in start zone, closer to exit

**Deadlocks:** 2+ (waste blue early on trap; waste blue between zones on other stubs).

**Grid:** 6×6–8×8 (5×5 impossible — too small for 2 dual-locks in sequence).

---

#### Template 12: Key Collector

**Mechanic:** 3 keys (R, B, G), one triple-lock exit (R+B+G), and 2 single-lock trap doors (blue, green). The exit demands ALL three keys, so wasting any single colored key on a trap is immediately fatal — true scarcity. Each barrier offers a cheaper single-lock temptation.

**Key distinction:** Must collect and PRESERVE all 3 keys for a single triple-lock. Each barrier's single-lock trap is independently fatal.

**Anti-greedy invariant:** Convergent forks pattern — K_b placed above trap_b on barrier 1, K_g placed above trap_g on barrier 2. Greedy agent collects key and immediately wastes it on adjacent cheap single-lock.

**Layout:**
- Zone 0 (rows 0..b1-1): Start + K_r + K_b (near trap_b via convergent forks)
- Barrier 1 (row b1): D_trap_b (blue, trap-side) + bypass (free, far-side)
- Zone 1 (rows b1+1..b2-1): K_g (near trap_g via convergent forks)
- Barrier 2 (row b2): D_trap_g (green, trap-side) + D_exit (R+B+G, far-side)
- Zone 2 split by divider: Zone 2T (trap side, real area) + Zone 2C (exit)

**False hope door (D_tease):** A green door sits on the Zone 2 divider wall. An agent in Zone 2T (having already wasted green on trap_g) sees this door and may think it's an alternate path to the exit side — but the green key is gone, so D_tease is provably unopenable. Pure psychological cruelty that adds no valid path but increases perceived complexity.

**Deadlocks:** 3+ (waste blue on trap_b -> can't open R+B+G; waste green on trap_g -> can't open R+B+G; waste both -> stuck).

**Grid:** 6×6–8×8 (5×5 impossible — too small for 3-key collect layout).

---

#### Template 13: Decoy Key

**Mechanic:** 3 keys (red, yellow, blue), 5 doors. Two-stage temptation design: at each barrier, a visible matching-color trap door tempts the agent to waste a key that's needed later for a hidden same-color door deeper in the puzzle. The agent must resist the immediate temptation and save the key for a non-obvious future use.

**Key distinction from existing templates:**
- Fork/KBK: same-color doors, choose right one. Decoy: different-color doors on barrier.
- Key Collector: save all keys for triple-lock. Decoy: save key for hidden same-color door.
- Cascade Trap: spatial consequence. Decoy: resource depletion + hidden dependency.

**Two-stage temptation design:**
- Stage 1 (Barrier 1): Yellow key + visible yellow trap door (convergent forks bait). Red door is the correct path.
- Stage 2 (Barrier 2): Blue key + visible blue trap door (convergent forks bait). Yellow door (D_gate_y) is the correct path.
- At each stage, the greedy action (use key on adjacent matching door) is fatal.

**Layout:**
- Zone 0 (rows 0..b1-1): Start + K_r (red) + K_y (yellow, near D_trap_y)
- Barrier 1 (row b1): D_trap_y (yellow, exit-side) + D_correct_r (red, far-side)
- Zone 1 split by vertical divider:
  - Zone 1T (exit-side): Real walkable area (>=2 cells), dead end
  - Zone 1C (far-side): K_b (blue, near D_trap_b)
- Barrier 2 (row b2): D_trap_b (blue) + D_gate_y (yellow) [both in Zone 1C cols]
- Zone 2 (rows b2+1..h-1): Open area + exit pocket (D_exit_b blue + exit)

**Correct path:** Zone 0 → collect K_r + K_y → D_correct_r (red) → Zone 1C → collect K_b → D_gate_y (yellow) → Zone 2 → D_exit_b (blue) → exit.

**Deadlocks:** 2+ (USE_KEY(yellow) on D_trap_y → Zone 1T dead end → yellow gone for D_gate_y; USE_KEY(blue) on D_trap_b → Zone 2 → blue gone for D_exit_b).

**Grid:** 6×6–8×8 (5×5 impossible — too small for 2-stage temptation layout).

---

#### Template 14: Order-Dependent Lock

**Mechanic:** 3 keys (2 red + 1 blue), 3 doors. Two same-color red doors on barrier 1 — the agent can't use color to distinguish them. Must reason about what's behind each door: Door_A (correct, far-side) leads to Zone 1A containing a hidden red key K_r2, while Door_B (trap, exit-side) leads to Zone 1B (dead end). Opening Door_B first wastes the only accessible red key, making Door_A and the dual-lock exit permanently unreachable.

**Key distinction from Switchback (T8):** Switchback tests "use 2 same-color keys on the right 2 of 3 doors" — resource counting. Order-Dependent Lock tests "use 1 key on door A before door B, because opening B first blocks A" — the constraint is topological (what lies behind each door), not resource-based.

**Anti-greedy invariant (triple bait):**
- K_r directly above Door_B (convergent forks pattern) — USE_KEY temptation
- Door_B on exit-side half (closer to exit column)
- Zone 1B >= Zone 1A in size (trap area looks like "main" path)
- `manhattan(Door_B, exit) < manhattan(Door_A, exit)` — hard filter

**Layout:**
- Zone 0 (rows 0..b1-1): Start + K_r (near Door_B via convergent forks) + K_b
- Barrier 1 (row b1): Door_A (red, far-side) + Door_B (red, exit-side, trap)
- Zone 1 split by vertical divider:
  - Zone 1A (far-side): K_r2 (hidden red key) + connects to D_exit approach
  - Zone 1B (exit-side): Real walkable area (>=2 cells), disconnected from D_exit
- Barrier 2 (row b2): D_exit (red+blue dual-lock, Zone 1A side)
- Zone 2 (rows b2+1..h-1): Exit

**Correct path:** Zone 0 → collect K_r + K_b → Door_A (red) → Zone 1A → collect K_r2 → D_exit (red+blue) → Zone 2 → exit. Uses 2 red + 1 blue = 3 keys.

**Deadlocks:** 2 (open Door_B with {1 red, 0 blue} → Zone 1B → stuck; open Door_B with {1 red, 1 blue} → Zone 1B → stuck — can't open Door_A or D_exit).

**Grid:** 5×5–8×8.

---

### Trap door quality

**Problem:** Three templates (Fork T1, Key-Behind-Key T3, Relay T6) had trap doors leading to obvious dead-end pockets (0-2 cells). No rational agent would open these doors — they were "cosmetic" traps that didn't actually test reasoning.

**The pocket wall anti-pattern:**
- A single wall cell placed behind the trap door entry, creating a 0-1 cell dead end
- Agent sees the pocket immediately and knows the door is pointless
- Doesn't test forward-looking reasoning — only tests "can you see a wall?"

**The zone divider fix pattern (proven in switchback, cascade_trap, key_collector):**
- Replace pocket walls with zone dividers (horizontal or vertical walls spanning the zone)
- Each trap leads to a **real walkable area** (2-8 cells) disconnected from the correct path
- Agent must reason about whether the area leads to progress (it doesn't — no keys, no exits)
- Anti-greedy shifted from hard manhattan checks to convergent forks key placement + post-hoc greedy filter

**Templates fixed:**

| Template | Old trap | New trap | Divider type |
|:---|:---|:---|:---|
| Fork (T1) | 1-cell pocket (trap row walls + blocker) | Zone R_trap (3+ cells) | Horizontal, right side |
| Key-Behind-Key (T3) | 1-cell pocket + exit on trap row | Zone R_trap (3+ cells), exit relocated to far corner | Horizontal, right side |
| Relay (T6) | 0-cell pocket (single wall below barrier 1) | Zone 1T (2+ cells) | Vertical, Zone 1 |

**Verified OK (not fixed):**
- Investment Door (T7): D_trap_r leads to the full exit area (real multi-cell region). Agent enters but lacks blue+green keys for dual-lock. Genuinely convincing trap.

---

### Anti-greedy mechanism evolution

Documented here for consistency across future template work.

**Old pattern (hard manhattan check):** `manhattan(trap_door, exit) < manhattan(correct_door, exit)`. Applied as an explicit filter during grid generation. Brittle — breaks when exit location changes (e.g., zone divider fixes moved exits to far corners).

**New pattern (convergent forks + post-hoc greedy filter):**
1. **Convergent forks key placement:** Key placed adjacent to trap door approach cell (Chebyshev distance ≤ 1). Greedy agent picks up key and is immediately adjacent to trap -> USE_KEY fires.
2. **Cost asymmetry:** Single-lock traps are "cheap" (1 key) vs dual/triple-lock exits that are "expensive" (2-3 keys). Greedy prefers cheap immediate actions.
3. **Post-hoc greedy filter:** All 6 greedy strategies (4 manhattan-heuristic + 2 BFS-based) must fail on the generated instance. Applied as validation filter #3.

**Templates using new pattern:** Fork (T1), Key-Behind-Key (T3), Convergent Forks (T4), Relay (T6), Switchback (T8), Cascade Trap (T10), Key Collector (T12), Decoy Key (T13), Order-Dependent Lock (T14).

**Templates using cost asymmetry:** Conservation Trap (T2), Relay (T6), Investment Door (T7), Key Collector (T12).

---

### P1: Orientation randomization *(completed)*

**Problem:** Templates have implicit directionality — Fork always has a vertical barrier with start on the left. Without randomization, models could learn positional biases (e.g., "exit is always bottom-right").

**Solution:** Apply a random element from D4 (8 symmetries of a square: 4 rotations × 2 flip states) to the fully-constructed, fully-validated instance. Composition order: rotate first, then flip.

**Implementation:** Six private helpers in `generator.py`:
- `_transform_pos`: rotation (0°/90°/180°/270°) then optional horizontal flip
- `_transform_direction`: direction mapping via precomputed lookup tables
- `_transform_action`: same action type with transformed direction
- `_transform_grid`: walls, keys, doors, start, exit_pos
- `_transform_state`: agent_pos only (inventory/doors/keys unchanged)
- `_transform_instance`: picks random D4 element, transforms all `GeneratedInstance` fields

**Design choice — transform after validation:** Template generates in canonical orientation, all 5 validation filters run on that, then the transform is applied just before returning. This avoids any risk of breaking validation filters. The model sees the transformed grid.

**Verification:**
- 225/225 generated instances (25 seeds × 10 templates at 5×5) pass solution replay — 100% pipeline success
- Oracle re-solve succeeds on all transformed grids
- Start/exit positions span all 4 quadrants across 100 fork instances (34/14/24/28 distribution)
- Identity transform (rotation=0, flip=False) returns the exact same object

---

### P1: Grid size variation *(completed)*

**Auto-scaling trap complexity** at larger grids. No tier gating — templates generate at any size they physically support. `generate_validated_instance` returns `None` only when a template genuinely cannot produce a valid puzzle at a given size (structural constraint). 50 working (template, size) pairs stress-tested at 100 seeds each — 100% pipeline success.

**Actual grid support (from 100-seed stress test):**

| Template | 5×5 | 6×6 | 7×7 | 8×8 | Constraint |
|:---|:---|:---|:---|:---|:---|
| Fork | 100% | 100% | 100% | 100% | — |
| Switchback | 100% | 100% | 100% | 100% | — |
| Scarce Resource | 100% | 100% | 100% | 100% | — |
| Conservation Trap | 100% | 100% | 100% | 100% | — |
| Key-Behind-Key | 100% | 100% | 100% | 100% | — |
| Convergent Forks | 100% | 100% | 100% | 100% | — |
| Parallel Wings | 100% | 100% | 100% | 100% | — |
| Cascade Trap | 100% | 100% | 100% | — | 8×8: greedy solves it (grid too open) |
| Order-Dependent Lock | 100% | 100% | 100% | 100% | — |
| Relay | — | 100% | 100% | 100% | 5×5: too small for 3-stage layout |
| Investment Door | — | 100% | 100% | 100% | 5×5: too small for vertical barrier + split |
| Sequential Dual-Locks | — | 100% | 100% | 100% | 5×5: too small for 2 dual-locks |
| Key Collector | — | 100% | 100% | 100% | 5×5: too small for 3-key collect |
| Decoy Key | — | 100% | 100% | 100% | 5×5: too small for 2-stage temptation |

**Config dicts:** `TEMPLATE_SOLUTION_RANGES_BY_SIZE`, `TEMPLATE_MIN_DEADLOCKS_BY_SIZE` provide per-size solution ranges and deadlock minimums. Original `TEMPLATE_SOLUTION_RANGES` and `TEMPLATE_MIN_DEADLOCKS` preserved as fallbacks for sizes without explicit entries. `TEMPLATE_TIERS` and `TIER_GRID_SIZES` retained as metadata (not used for gating).

**Solution range capping:** `TEMPLATE_SOLUTION_RANGES_BY_SIZE` enforces both lower and upper bounds on solution length. Lower bound rejects trivially short solutions that wouldn't test reasoning. Upper bound rejects excessively long solutions where the puzzle becomes tedious pathfinding rather than key-dependency reasoning — a 40-step solution on a 5×5 grid means unnecessary detours, not harder logic. Ranges widen at larger grid sizes (+2 lower, +4 upper per size step) to accommodate naturally longer paths.

**Auto-scaling retries:** `generate_validated_instance` auto-scales max retries at larger grids (w>=7: 300, w>=8: 500) to compensate for lower raw yield.

**Auto-scaling traps:** Each template adds extra trap doors at larger grid sizes via `_pick_extra_trap_col` helper (best-effort — silently falls back to base layout if no valid column found).

| Grid | Simple templates | Medium templates | Complex templates |
|:---|:---|:---|:---|
| 5×5 | Base key/door counts | Base counts | N/A or base counts |
| 6×6 | +1 trap door on barrier 1 | +1 trap on barrier 1 | Base counts |
| 7×7 | +1 trap (same as 6×6) | +1 more trap (barrier 1 or 2) | +1 trap on barrier 1 |
| 8×8 | +1 trap (same as 6×6) | +1 more trap (barrier 1 or 2) | +1 trap on barrier 2 |

**8×8 4-key variants deferred to P2** (Stochastic template-specific scaling): Convergent Forks quad-lock, Conservation Trap triple-lock, Parallel Wings 4-key — these require major structural generator changes (new key colors, upgraded lock types).

---

### P1: Hard variants (oracle-adjacent deadlocks) *(completed)*

**Problem:** Current `discover_deadlocks` finds deadlocks via exhaustive forward BFS from the initial state — "easy" deadlocks the agent may stumble into from any state. **Oracle-adjacent deadlocks** are harder: the agent follows the optimal solution perfectly, then makes one wrong move. These are maximally deceptive because the path looks correct up to the final divergence.

**Implementation:** Ported from mosaic_grid's `discover_oracle_adjacent_deadlocks` with one key optimization: only USE_KEY actions can cause deadlocks in chain_vault (movement never flips solvability), so only non-oracle USE_KEY actions are tested as branch points.

**New components in `generator.py`:**

| Component | Purpose |
|:---|:---|
| `DeadlockEntry.k_horizon` | New field (default 0): count of reachable states from deadlock via BFS, capped at 200. Invariant under D4 transform (scalar count). |
| `count_reachable_dead_states()` | BFS from Walking Dead state counting unique reachable states. Measures continuation depth for RQ3. |
| `discover_oracle_adjacent_deadlocks()` | Walks oracle solution step-by-step; at each step `i`, tries every legal USE_KEY action that differs from oracle move. Records Walking Dead results with `k_horizon` pre-computed. |
| `generate_hard_rq2_pair()` | Both paths share oracle prefix `solution[:i]`, diverge at step `i` (oracle action vs. bad action). Returns `(initial_state, deadlock_actions, solvable_actions)`. |
| `generate_hard_rq3_instance()` | Oracle prefix + wrong turn + Walking Dead continuation to `target_length`. Uses `random.Random(42)` for reproducibility. Requires `cause_symptom_gap >= 3`. |
| `is_eval_ready()` | Four-gate check: standard RQ2 (k_horizon≥3, fatal_step≥2), standard RQ3 (k_horizon≥5), hard RQ2, hard RQ3. Oracle-adjacent catalog computed on-the-fly (not stored on instance). |

**k_horizon computation in `generate_validated_instance`:** Computed lazily after `discover_deadlocks` returns, with tier counting (tier1: k_horizon≥3, tier2: k_horizon≥5).

**Design decision — no `hard_deadlock_catalog` field on `GeneratedInstance`:** Matches mosaic_grid's pattern. Oracle-adjacent catalog is computed on demand in `is_eval_ready` and hard RQ generators, not stored. Keeps the instance dataclass lean.

**Verification results:**

| Test | Result |
|:---|:---|
| k_horizon populated | 25/25 fork instances: all entries have k_horizon > 0 |
| Oracle-adjacent discovery | 10/14 templates produce OA deadlocks (4 templates need 6×6+ grids) |
| Hard RQ2 round-trip | 15/15 OA entries → valid pairs (dead path → Walking Dead, solvable path → solvable) |
| Hard RQ3 traces | 15/15 OA entries → valid traces with cause_symptom_gap ≥ 3 |
| is_eval_ready (std) | 80–100% pass rate across tested templates |
| is_eval_ready (hard) | 48–100% pass rate (convergent_forks: 100%, switchback: 48%) |
| Regression | 100% pipeline success on all templates that generate at 5×5 |
| Performance | k_horizon BFS + OA discovery both < 1ms per instance |

---

### P1: Key and door placement distance constraints *(completed)*

**Problem:** Template generators placed non-bait keys by shuffling a candidate list and taking the first N entries (`rng.shuffle` + index slicing) or `rng.choice`. This risked clustering multiple keys on adjacent cells, reducing navigation challenge. Additionally, 6 templates had identical inline `manhattan_distance(trap, exit) >= manhattan_distance(correct, exit)` checks that should be consolidated.

**New helpers in `generator.py`:**

| Component | Purpose |
|:---|:---|
| `_select_key_positions(candidates, n, rng, min_spacing, already_placed, reference)` | Three-phase greedy selection mirroring mosaic_grid's `_select_plate_positions`. Phase 1: spacing + distinct distances from `reference` (exit_pos). Phase 2: spacing only. Phase 3: relaxed `min_spacing=1`. Supports `already_placed` (frozenset) so bait keys placed via `_convergent_key_pos` are respected. |
| `_trap_closer_to_exit(trap, correct, exit_pos)` | Shared predicate replacing 6 identical inline manhattan checks. Returns `True` when trap door is strictly closer to exit than correct door. |

**Templates updated with `_select_key_positions` (7 total):**

| Template | Keys in same zone | Previous code | New code |
|:---|:---|:---|:---|
| scarce_resource | 4 (K_b bait + K_r1, K_r2, K_r3) | `rng.shuffle; [0],[1],[2]` | `_select_key_positions(candidates, 3, rng, already_placed={kb_pos}, reference=exit_pos)` |
| conservation_trap | 3 (start + K_r + K_b) | `rng.shuffle; [0],[1],[2]` | Place start first, then `_select_key_positions(candidates, 2, rng, reference=exit_pos)` |
| switchback | 2 (K1 bait + K2) | `rng.choice(candidates)` | `_select_key_positions(candidates, 1, rng, already_placed={key1_pos}, reference=exit_pos)` |
| sequential_dual_locks | 3 (K_b1 bait + K_r + K_g) | Biased sort + `rng.choice` | Biased top-third pool → `_select_key_positions(pool, 2, ...)`, fallback to full candidates |
| key_collector | 2 (K_b bait + K_r) | `rng.choice(candidates)` | `_select_key_positions(candidates, 1, rng, already_placed={kb_pos}, reference=exit_pos)` |
| cascade_trap | 2 (K_r bait + K_b) | Biased sort + `rng.choice` | Biased top-third pool → `_select_key_positions(pool, 1, ...)`, fallback to full candidates |
| order_dependent_lock | 2 (K_r bait + K_b) | Biased sort + `rng.choice` | Biased top-third pool → `_select_key_positions(pool, 1, ...)`, fallback to full candidates |

**Templates updated with `_trap_closer_to_exit` (6 total):** switchback, scarce_resource, cascade_trap, sequential_dual_locks, decoy_key, order_dependent_lock.

**Not changed:** `_convergent_key_pos` logic (bait key placement), template structural layout (barriers, zones, exit pockets), `generate_validated_instance` filters, templates with keys in separate zones or single key per zone (fork, key_behind_key, convergent_forks, parallel_wings, relay, investment_door, decoy_key).

**Verification results:**

| Test | Result |
|:---|:---|
| Regression (5×5) | 9/9 templates that generate at 5×5: 25/25 seeds each (100%) |
| Regression (6×6) | sequential_dual_locks + key_collector: 25/25 seeds each (100%) |
| Key spacing (scarce_resource 6×6) | 52% of instances have min non-bait pairwise manhattan ≥ 2 (up from ~0% with pure shuffle) |
| `_trap_closer_to_exit` equivalence | Helper produces identical accept/reject decisions as inline code on all test cases |
| Yield impact | Three-phase fallback prevents degradation — zero generation failures from spacing constraints |

---

### Full dataset pipeline (P2)

| Component | Details |
|:---|:---|
| **`generator.py` metadata** | `GeneratedInstance` extended with `template_name`, `grid_width`, `grid_height`, `num_keys`, `num_doors`, `solution_length`, `tier1_deadlocks`, `tier2_deadlocks`, `max_k_horizon`, `generation_attempts`, `fingerprint`. `_transform_instance` preserves all metadata. |
| **`GenerationResult` + stats** | `_generate_validated_inner` returns `(instance, rejection_counts, total_attempts)` with named counters: `unsolvable`, `solution_range`, `greedy_passes`, `insufficient_deadlocks`, `irrelevant_key`. `generate_validated_instance_with_stats` wraps in `GenerationResult` with timing. |
| **`prompts.py` RQ3 ablation** | `build_rq3_ablation_prompt(grid, actions, show_initial)` — removes simulation burden, shows final state (or initial+final side-by-side). |
| **`dataset.py`** | Full serialization (`instance_to_dict`/`dict_to_instance`), SHA-256 fingerprinting, `_worker` for multiprocessing, `generate_dataset` (parallel batch), `generate_to_target` (seed-by-seed with stall detection). Task key: `{template}_{w}x{h}_s{seed}` (no door axis). Grid sizes filtered per template by `TIER_GRID_SIZES[TEMPLATE_TIERS[template]]`. |
| **`build_eval.py`** | All 7 RQ entry builders mirroring mosaic_grid: `_build_rq1_entries`, `_build_rq2_entries`, `_build_rq3_entries`, `_build_hard_rq2_entries`, `_build_hard_rq3_entries`. Per-puzzle directory: `{template}_{w}x{h}_{k}k_{d}d_seed{s}`. Worker renders all images (initial, solution trace, RQ2/RQ3 states, hard variants). `build_eval_dataset` orchestrator with stratified sampling by `(template, grid_size, num_keys, num_doors)`. |
| **`__main__.py`** | CLI subcommands: `demo`, `generate`, `build`, `full`, `verify`, `filter`. No `--doors` argument (templates have fixed door structure). Grid sizes default `5,6,7,8` filtered per template by tier. |

---

### Stochastic template-specific scaling (P2)

50% coin-flip (`rng.random() < 0.5`) per generation attempt activates an "enhanced" variant at specified grid size thresholds. Adds structural complexity (extra traps, extra keys) rather than just extra walls.

| Template | Activates at | Standard → Enhanced | Observed rate |
|:---|:---|:---|:---|
| **Parallel Wings** | w≥6 | 1 extra trap → 2 extra traps on divider | ~0% at 6×6 (divider too short); structural at 7×7+ (deterministic) |
| **Relay** | w≥7 | b1 trap only → b1+b2 traps | ~44% at 7×7 |
| **Investment Door** | w≥7 | barrier trap only → barrier+split traps | ~32% at 7×7 |
| **Sequential Dual-Locks** | w≥7 | b2 trap only → b1+b2 traps | ~21% at 7×7 |
| **Key Collector** | w≥6 | 3 keys + triple-lock exit → 4 keys (+ yellow) + quad-lock exit | ~41% at 6×6 |

Solution ranges widened at activation sizes (+4 upper bound). Full regression: 25/25 across all 50 template×size combinations.

---

## Remaining Work

---

### P3: Difficulty tiers

| Tier | Templates | Grid | Keys | Colors | Dual-locks | Solution range |
|:---|:---|:---|:---|:---|:---|:---|
| Simple | Fork, Switchback, Scarce Resource | 5×5–8×8 | 1–4 | 1–2 | 0 | 6–32 |
| Medium | Conservation Trap, KBK, Convergent Forks, Parallel Wings, Cascade Trap, Order-Dependent Lock | 5×5–8×8 (Cascade Trap: 5×5–7×7) | 2–4 | 2 | 0–1 | 10–40 |
| Complex | Relay, Investment Door, Sequential DL, Key Collector, Decoy Key | 6×6–8×8 | 3–4 | 2–3 | 0–2 | 12–38 |

---

### P3: Gap-stratified RQ3 sampling

**Plan:** Bucket deadlock entries by cause-symptom gap:
- Small (2–3 steps): key wasted just before a required door
- Medium (4–6 steps): key wasted, agent walks some before hitting a locked door
- Large (7+ steps): key wasted early; downstream consequence appears much later

Templates that naturally produce large gaps: Key-Behind-Key, Relay, Sequential Dual-Locks.

---

## Review Findings: Template Weaknesses & Proposed Improvements

Documented from reviewer feedback and design review (March 2026). These are improvements to pursue before finalizing the benchmark dataset.

---

### Template weakness audit

#### Tier 1: Genuinely weak (need fix or replacement)

**Template 12: Key Collector**

*Reviewer finding:* "Doesn't require much planning. Its greedy heuristic to collect all keys and then use all of them at the final door, as you already know that the final door requires all keys. Does not involve any intermediate door openings."

This critique is correct. The current design is:
- Collect 3 keys (R, B, G)
- Avoid 2 single-lock trap doors along the way
- Use all 3 at the triple-lock exit

The optimal strategy is also the most natural strategy: don't touch any single-lock door, then use all keys at the exit. A model that simply reads the rules ("triple-lock requires all 3") and applies "save everything" reasoning succeeds without any genuine planning. The single-lock temptation doors are weak traps because a model that understands the final objective will trivially avoid them.

*Decision:* Keep as-is for now as a lower-bound diagnostic ("can the model even understand the final objective and resist visible temptation?"), but note this is the easiest template in the set. It should not be weighted equally with harder templates in the benchmark paper.

**Template 13: Decoy Key**

*Structural finding:* Decoy Key is Fork (Template 1) with extra keys. The core reasoning task — "I see a matching-color door, but should I use this key here or save it?" — is identical to Fork's core reasoning. Fork has 1 key and 2 same-color doors. Decoy Key has 3 keys and a 2-stage trap where the visible door matches the key in hand. The meta-structure is the same binary "resist the visible match" decision.

The claimed distinction from the DESIGN doc — "Fork/KBK: same-color doors, choose right one. Decoy: different-color doors on barrier" — holds structurally but not cognitively: the reasoning test is still "should I use key X on door Y that I'm standing next to?"

*Decision:* Decoy Key stays but its distinctiveness claim needs to be weakened in the paper. It tests the same decision as Fork but at a higher complexity level (3 keys, 2 stages).

---

#### Tier 2: Structurally similar across templates (layout redundancy)

Templates 2 (Conservation Trap), 4 (Convergent Forks), 5 (Parallel Wings), 6 (Relay), 7 (Investment Door) all share the same visual DNA: one or two horizontal barriers, a bypass gap on a grid edge, trap doors on the exit-side half, and an exit pocket gated by a dual-lock. While the *logical* structure differs (different dependency graphs), the *visual* structure is repetitive. A model that learns the layout convention of "bypass gap at edge column = safe path" would generalize incorrectly across these templates.

D4 orientation randomization partially mitigates this (the bypass gap rotates to any edge), but within a single orientation, the pattern is recognizable.

*Decision:* Acceptable for now — the D4 transform prevents systematic exploitation. Worth noting as a limitation in the paper.

---

### Proposed improvements

#### P1: Key Collector → "Key Collector with Path Commitment" (recommended)

**Problem:** No intermediate USE_KEY decisions. The correct strategy is identical to the naive strategy.

**Fix:** Add two structurally distinct paths to the triple-lock exit. The shorter path requires spending keys on intermediate single-lock gates; the longer path bypasses those gates. Arriving at the triple-lock with fewer than 3 keys (because you used some on the intermediate gates) means you cannot win.

```
Conceptual layout:
  Path A (short, direct):  Start → [RED gate] → [BLUE gate] → triple-lock exit
  Path B (long, bypass):   Start → winding corridor → triple-lock exit

Keys available: RED, BLUE, GREEN (all needed for triple-lock)

Greedy: takes Path A (shorter), uses RED on gate → uses BLUE on gate
        → arrives at triple-lock with only GREEN → stuck (Walking Dead).

Correct: takes Path B (longer, no intermediate gates),
         arrives with RED+BLUE+GREEN → opens triple-lock → wins.
```

**What this tests:** Path commitment under budget constraint. The model must reason that spending keys on intermediate gates — even though it appears to make progress — depletes the budget required for the final lock. This is a genuine planning decision absent from the current Key Collector.

**RQ impact:**
- *RQ2:* The Walking Dead state after using RED + BLUE on intermediate gates looks completely normal — the agent is in a corridor, moving freely, approaching the exit. Nothing visually signals the deadlock. High-quality RQ2 case.
- *RQ3:* The fatal step is the first gate use (USE_KEY on Path A's first gate). The symptom appears much later at the triple-lock. Creates a genuine cause-symptom gap.

**Generator approach:** Extend the current two-barrier layout. Barrier 1 has two openings: a short gate requiring RED (trap), and a free bypass (correct path). Barrier 2 has two openings: a short gate requiring BLUE (trap, only reachable if you opened Barrier 1's gate), and a free bypass (correct path). Triple-lock exit at the bottom. Keys RED, BLUE, GREEN all placed in Zone 0. The correct path threads both bypass gaps; greedy threads both gates.

---

#### P2: RQ3 prompt — remove the USE_KEY hint (high impact, low effort)

**Problem:** The RQ3 main prompt (`build_rq3_prompt` in `prompts.py`) currently contains:

> "Think carefully: The fatal step is **always a USE_KEY action** that consumed a key on the wrong door."

This hint collapses the attribution search space from all N trace steps to only the (typically 2–5) USE_KEY actions in the trace. In a 20-step trace with 3 USE_KEY events, the model only needs to select among 3 candidates — this is not a diagnostic of causal attribution, it is near-trivial elimination.

**Why this matters for the paper:** The core RQ3 claim is that models exhibit *recency bias* — attributing failure to recent actions rather than the true root cause. If the model knows the answer is always a USE_KEY, it will simply select the LAST USE_KEY action. This produces an apparent "recency bias" signal that is in fact an artifact of the hint, not a genuine finding about causal reasoning.

**Fix:** Remove the two-sentence hint block from the prompt:
```
# REMOVE this block from build_rq3_prompt:
"- The fatal step is always a USE_KEY action that consumed a key on "
"the wrong door.\n"
```

After removal, the model must reason about the full trace (all MOVE and USE_KEY actions) to identify the fatal step. This makes recency bias genuinely measurable: if the model attributes failure to the last action (which is often a MOVE, not a USE_KEY), that is a real signal.

**Ablation option:** Keep a version of the prompt WITH the hint as a scaffolded ablation variant. The gap between the hinted and un-hinted performance measures how much the model's attribution relies on this structural shortcut vs. genuine causal reasoning.

**Implementation:** Modify `build_rq3_prompt` in `prompts.py`. The "Think carefully" bullet list should be replaced with a more neutral framing:

```
"Think carefully:\n"
"- The agent may continue taking actions after the fatal mistake, "
"so the error may not be obvious from the final position.\n"
"- There may be many steps between the fatal action and the point "
"where the agent runs out of options.\n"
"- Consider what each action does to the agent's resources and "
"position, and how that constrains future options.\n"
```

---

#### P3: New template — "Budget Partitioning" (fills a reasoning gap)

**Reasoning gap:** No current template tests the skill of *counting allocation errors* — where the model has enough keys in total but assigns them incorrectly across multiple requirements. Scarce Resource (T9) tests identifying the bottleneck color. Budget Partitioning tests whether the model can correctly allocate a fixed budget across competing mandatory uses.

**Mechanic:** 2 RED keys are available. Three RED doors exist: Door A (mandatory — only path to blue key), Door B (trap — leads to dead end), Door C (mandatory — dual-lock RED+BLUE, final exit). Total RED needed for correct path: 1 (Door A) + 1 (Door C component) = exactly 2. Using either red key on Door B exhausts the budget: agent can open one mandatory door but not both.

```
Layout:
  Zone 0:  Start + 2x RED key + 1x BLUE key
  Door A (RED, far-side): → Zone 1 with BLUE key pickup
  Door B (RED, exit-side trap): → Zone 1T (dead end, no keys)
  Zone 1 → Door C (RED+BLUE dual-lock) → exit

Correct path: Door A → Zone 1 → BLUE key → Door C (RED+BLUE) → exit
  Uses: RED (Door A) + RED+BLUE (Door C) = 2 RED + 1 BLUE ✓

Trap: Door B → Zone 1T (dead end) → back to Zone 0 → now only 1 RED left
  → can open Door A OR Door C's RED component, but not both → stuck
```

**Key distinction from existing templates:**
- *Fork (T1):* 1 key, binary choice, no budget arithmetic.
- *Scarce Resource (T9):* Identify the bottleneck color (blue). Budget Partitioning: correctly allocate a non-bottleneck resource (both colors available, red is not scarce in total — just over-committed).
- *Conservation Trap (T2):* Don't waste red; you need it for the dual-lock. Budget Partitioning: you MUST spend both reds, but on the right doors.

**What this tests:** The model must perform explicit resource accounting: "I have 2 red keys, I need 1 red for Door A and 1 red for Door C's dual-lock component — that's exactly my budget. I cannot spend any red on Door B, even though Door B is closer."

---

### Priority ranking

| Priority | Change | Effort | Impact |
|:--|:--|:--|:--|
| **P0** | Remove USE_KEY hint from `build_rq3_prompt` | Low (2-line edit in `prompts.py`) | High — makes RQ3 a genuine diagnostic rather than elimination |
| **P1** | Key Collector → Path Commitment variant | Medium (new generator function) | High — turns trivial template into real planning test |
| **P2** | New "Budget Partitioning" template | Medium (new generator function) | Medium — fills counting/allocation gap |
| **P3** | Note Decoy Key / Key Collector as easy baselines in paper | Low (paper writing) | Medium — sets correct expectations for reviewers |

---

### Second review batch (March 2026)

Additional weaknesses identified from further review of generated instances.

---

#### Template 11: Sequential Dual-Locks — low seed variance

*Reviewer finding:* "Becomes too similar across seeds. Collect all keys that you can, use it on a multi-key door, there you will get another key, and then use that key along with what you have on the multi-door to open."

The current layout always follows the same logical arc:
1. Collect R + B + G keys in Zone 0
2. Open first dual-lock (R+B) → enter Zone 1 which contains a second B key
3. Open second dual-lock (B+G) → exit

Every instance of this template is structurally identical. The only variation is spatial (where keys and doors appear on the grid), not logical (what decisions need to be made). Once a model understands the structure, it transfers perfectly across all seeds.

**Root cause:** The two dual-locks are adjacent in the dependency graph with nothing between them. There are no intermediate USE_KEY decisions separating the two commitments. A model that learns "collect everything, use both for first dual-lock, collect new key, use both for second" has a complete decision policy that works for all seeds.

**Fix — add investment doors between the dual-locks:**

The two dual-locks must be separated by at least one intermediate door that requires a USE_KEY decision. Two placement options:

*Option A: Investment door before the first dual-lock*
```
Zone 0: Start + R key (only)
Investment door (R): → Zone I with B + G keys
[RED trap door in Zone 0 → dead end]
Zone I → First dual-lock (R+B) → Zone 1 (has second B key)
Zone 1 → Second dual-lock (B+G) → exit

Correct path: R on investment door → collect B+G → R+B dual-lock → collect B → B+G dual-lock → exit
Trap: R on red trap → now no key for investment door → stuck in Zone 0
```

This separates planning into two phases: first decide whether to invest the R key correctly, then navigate the two dual-locks.

*Option B: Investment door between the two dual-locks*
```
Zone 0: Start + R + B + G keys
First dual-lock (R+B) → Zone 1
Zone 1: Investment door (B or G) → Zone I with hidden second key
Zone I → connects to approach for second dual-lock
Second dual-lock (B+G) → exit

Correct path: R+B dual-lock → B on investment door → collect new key → B+G dual-lock → exit
Trap: B on trap in Zone 1 → lose B needed for investment door OR second dual-lock
```

Both options add a genuine USE_KEY branch point that creates divergent seed outcomes. Which specific doors appear, where the traps are, and how Zone I is structured will vary across seeds in ways that matter for the decision.

**Generator change:** In `generate_sequential_dual_locks_grid`, insert an intermediate investment door zone between the two barrier zones. The new door's required_keys and the trap configuration should vary based on the rng. Minimum viable change: on 7×7+ grids, always add Option A or B; on 6×6, use a simplified version.

---

#### Template 5: Parallel Wings — open-world problem

*Reviewer finding:* "Very simple, because there is an open path to all the keys and the exit. It is also known that the exit requires both keys, hence, greedy-heuristic (collect all keys and open door finally) can be followed very easily."

The current design places both keys in their respective wings with no locks on either path. An agent can walk from start → left wing → blue key → bypass → right wing → red key → exit approach → USE_KEY dual-lock → done. There is no point in the correct path where the agent must make a USE_KEY decision before having both keys. The only way to fail is to use a key on a wing trap door before collecting the other wing's key.

**Root cause:** All keys are accessible without any USE_KEY commitment. The "collect all, then open exit" greedy policy is always valid. The trap doors only fire if the agent is pathologically impatient (uses key immediately on adjacent trap door).

**Fix — gate at least one key behind a door, forcing commitment order:**

Introduce a single-lock door inside one wing that blocks access to that wing's key. The lock color comes from the other wing. This forces the agent to cross to the unlocked wing first, collect its key, return, and use that key to unlock the other wing's key room. The trap doors on the divider then become genuinely costly — using the shortcut trap consumes the key the agent will need to unlock the other wing's key room.

```
Revised layout:
  Left wing:  blue key (free access)
  Divider:    blue trap door + red trap door + bypass gap at bottom
  Right wing: [RED gate → red key room (1-2 cells)] + red key inside

Correct path: blue key (left) → bypass → right wing → USE blue on red gate
              → collect red key → return via bypass → USE red+blue on exit pocket dual-lock

Wait — but blue was used on the red gate, so only red remains for the dual-lock.
  This means the exit must require only RED, not red+blue dual-lock.
  OR: there must be a second blue key behind the red gate.

Better version (two keys behind red gate):
  Left wing: blue key (free)
  Right wing: red gate (needs blue) → room with [red key + second blue key]
  Exit: red+blue dual-lock

Correct path: collect blue → unlock red gate with blue → collect red + second blue
              → exit with red + (second) blue
Trap: use blue on divider blue-trap → enter right wing via trap → reach right wing
      but can't open red gate (blue spent) → stuck
      OR: try to go through bypass without using blue, reach red gate,
          fail because blue key needed
```

This creates a genuine planning decision: the agent must use the blue key to unlock the red gate (investment), which grants both the red key and a replacement blue key. The bypass is still free but leads to a dead end at the red gate.

**Additional path blocking (if the above is too complex):** A simpler intermediate fix is adding internal walls within each wing to create corridor structures. Currently wings are open rooms — adding 1-2 internal walls forces the agent along a specific path, meaning it encounters the trap door before it reaches the key. This alone makes the convergent-forks bait more effective (agent can't "walk around" the trap). This does not fix the "collect all, open exit" problem but makes the traps harder to avoid.

**Recommended approach:** Implement the key-gating fix (Option A above) on 6×6+ grids. Keep the current open layout only for 5×5 (too small for the additional zone).

---

#### Template 3: Key-Behind-Key — too short on larger grids

*Reviewer finding:* "Should involve more keys. In its current shape it's much simpler. At least put more keys on larger grids."

Current structure across all grid sizes: 1 red key → correct red door → 1 blue key → blue door → exit. This is a 2-step dependency chain regardless of grid size. On a 7×7 or 8×8 grid, this is just the same simple chain with more empty space.

**Fix — scale chain length with grid size:**

On larger grids, extend the chain by adding an intermediate key/door pair:

| Grid | Chain depth | Structure |
|:--|:--|:--|
| 5×5 | 2 | R → correct_R_door → B key → B_door → exit (unchanged) |
| 6×6 | 3 | R → correct_R_door → G key → correct_G_door → B key → B_door → exit |
| 7×7–8×8 | 3–4 | R → correct_R_door → G key → correct_G_door → B key → B_door → exit (or add one more hop) |

At 6×6+, add a second intermediate zone: the correct red door leads to a green key (not directly to blue). A green door then leads to the blue key. Blue door leads to the exit.

**Trap scaling:** Add trap doors at each new intermediate zone. The red trap door already exists (wastes red). Add a green trap door at the second barrier. Wasting either key breaks a different link in the chain.

**Generator change:** In `generate_key_behind_key_grid`, the current single-zone right side (Zone R_correct has the blue key and the blue door) becomes two zones on 6×6+: Zone R_1 (behind correct red door, has green key + green trap door + correct_green_door), Zone R_2 (behind correct green door, has blue key + blue door + exit).

**Impact on RQ3:** Longer chains increase the cause-symptom gap. Wasting the red key at step 5 in a 25-step trace on a 7×7 grid produces a much more diagnostic RQ3 instance than the current 15-step trace.

---

#### Conservation Trap (T2) vs Convergent Forks (T4) — redundancy

*Reviewer finding:* "Convergent Forks is very similar to Conservation Trap. Keep Convergent Forks because you have to avoid more bait."

This is correct. Both templates have the same dependency structure:
- 2 keys, 2 colors
- First barrier: trap door(s) + bypass gap
- Second barrier: dual-lock (requires both colors)
- Exit pocket behind dual-lock

The only difference is that Conservation Trap has 1 trap on the first barrier, while Convergent Forks has 2 traps (one per color). Convergent Forks is strictly harder (more bait to avoid) and subsumes Conservation Trap's reasoning challenge. From a benchmark diversity perspective, they are measuring the same skill.

**Decision: deprecate Conservation Trap, replace with a genuinely distinct template.**

Conservation Trap's slot (Template 2) should be replaced with a template that tests something not covered by the remaining 13. Candidates:

*Option A: "Forced Detour" template*
The correct path to the exit is longer than an incorrect path. There is no trap door per se — the issue is that the shorter route leads through a zone with no keys, and the agent must recognize it must take the longer detour to collect the key it needs. Tests whether the model can commit to a suboptimal-looking step count in service of resource acquisition.

*Option B: "Budget Partitioning" template* (already proposed above under P3 improvements)
2 RED keys available, 3 RED doors: Door A (mandatory, leads to blue key), Door B (trap, dead end), Door C (dual-lock RED+BLUE, exit). Exactly 2 reds needed for the correct path. Using one on Door B makes it impossible to open both Door A and Door C. Tests explicit resource counting under tight allocation constraints.

*Option C: Promote the existing "Budget Partitioning" idea to this slot and remove Conservation Trap from the template set*
Keeps the total template count at 14 and eliminates the most redundant template.

**Recommended decision:** Replace Conservation Trap (T2) with Budget Partitioning. This preserves 14 templates while eliminating the redundancy and adding a genuinely new reasoning test.

---

#### Template 10: Cascade Trap — subset of Order-Dependent Lock

*Reviewer finding:* "Cascade Trap becomes a subset of Order-Dependent Lock."

This observation is partially correct. Both templates test the consequence of making the wrong USE_KEY decision on barrier 1 (the decision isolates the agent from a necessary key or path). The key difference is mechanism:
- Cascade Trap: using the wrong door changes spatial accessibility (topological change)
- Order-Dependent Lock: using the wrong door wastes a key needed for the next door in sequence

The reasoning tasks are different enough to justify keeping both. However, Cascade Trap's unique contribution — that opening a door can spatially isolate the agent from future keys — is not well-emphasized in the current generator. Cascade Trap instances that look visually like "agent is cut off from blue key room" are more diagnostic than instances where the spatial consequence is subtle.

**Decision:** Keep Cascade Trap as-is. Its spatial-consequence mechanism is meaningfully distinct from Order-Dependent Lock even if the surface narrative is similar.

---

### Updated priority ranking

| Priority | Change | Effort | Impact |
|:--|:--|:--|:--|
| **P0** | Remove USE_KEY hint from `build_rq3_prompt` | Low | High — makes RQ3 genuinely diagnostic |
| **P1** | Sequential Dual-Locks: add investment door between the two dual-locks | Medium | High — eliminates seed homogeneity |
| **P1** | Parallel Wings: gate one wing's key behind a door requiring the other wing's key | Medium | High — eliminates the open-world "collect all, open exit" policy |
| **P1** | Key-Behind-Key: scale chain depth with grid size (3-hop chain at 6×6+) | Medium | High — leverages larger grids for harder reasoning |
| **P2** | Replace Conservation Trap (T2) with Budget Partitioning template | Medium | Medium — removes redundancy, adds counting/allocation test |
| **P2** | Key Collector → Path Commitment variant | Medium | Medium — turns trivial template into real planning test |
| **P3** | New "Budget Partitioning" template (if Conservation Trap slot is not repurposed) | Medium | Medium |
| **P4** | Note Decoy Key / Key Collector as easy baselines in paper | Low | Low |

---

### Consolidated brainstorm: four weakness types

All identified weaknesses can be grouped into four root-cause categories. This taxonomy is useful for prioritization and for framing the template improvement story in the paper.

| Weakness type | Affected templates | Root cause | Fix strategy |
|:---|:---|:---|:---|
| **Arc homogeneity** — every seed follows the same logical arc, only spatial layout changes | T11 SDL, T5 Parallel Wings | Dependency graph has only one topological path; no USE_KEY branch point varies across seeds | Add at least one intermediate USE_KEY decision whose outcome varies structurally (investment door or same-color fork) |
| **Open-world greedy** — "collect all keys, open final door" policy is always valid | T5 Parallel Wings, T12 Key Collector | No door guards a key; every key is freely reachable without committing any prior key | Gate at least one key behind a door whose color comes from the other wing/zone |
| **Template redundancy** — two templates test the same cognitive skill | T2 Conservation Trap ≈ T4 Convergent Forks | Same dependency graph (bypass + dual-lock), only bait count differs | Replace redundant template with genuinely distinct reasoning test (Budget Partitioning) |
| **Depth not scaling** — larger grids add space but not reasoning steps | T3 Key-Behind-Key at 6×6–8×8 | Chain depth fixed at 2 regardless of grid size | Scale chain depth with grid size: depth 2 at 5×5, depth 3 at 6×6, depth 3–4 at 7×7–8×8 |

---

### T11 Sequential Dual-Locks — same-color fork alternative design

The investment-door designs (Options A and B above) are one fix. An alternative that more directly targets the seed-homogeneity problem is a **same-color fork in Zone 1**, modeled after what Template 14 (Order-Dependent Lock) does on barrier 1.

**Design:**

```
Zone 0:  Start + R + B + Y keys
First dual-lock (R+B) → Zone 1

Zone 1:  Two Y doors (same-color fork):
  D_correct_Y (far-side) → Zone 1C: {Y₂ key, G key}
  D_trap_Y (exit-side)   → Zone 1T: dead end (real walkable area, no keys)

Zone 1C → Second dual-lock (G+Y₂) → exit

Correct path:  Collect R+B+Y → open R+B DL → Zone 1 → Y on D_correct_Y →
               collect Y₂ + G → open G+Y₂ DL → exit.
               (Y consumed, but Y₂ recovered)

Fatal trap:    Y on D_trap_Y → Zone 1T (walkable, no keys) →
               back to Zone 1 → no Y for D_correct_Y → can't collect Y₂ + G →
               can't open second DL → Walking Dead.
```

**Seed variance:**  Across seeds, D_correct_Y and D_trap_Y swap positions on the Zone 1 barrier (column randomization). Which side is "correct" changes per seed. Color randomization: the base palette {R, B, Y, G} can be permuted (e.g., {R, B, G, Y} → first DL is R+G, zone 1 fork is B, second DL is Y+B₂) to produce structurally identical but visually distinct instances.

**Key distinction from Investment Door (T7):** T7's investment door is always on the initial barrier; the "investment" happens before the agent has any keys for the dual-lock. Here the fork happens *between* two dual-locks — the agent has already committed one USE_KEY action and must reason about which of two identical-color doors leads to the keys needed for the second dual-lock. The temporal structure is different even though both involve spending a key to gain keys.

**Comparison with investment-door options (A and B above):** Options A/B add an investment door before or between the dual-locks. The same-color fork creates an additional dilemma in Zone 1 that more directly mirrors T14's mechanism. Both approaches address the homogeneity problem; the same-color fork produces more distinctive visual layouts across seeds.

**Recommended approach:** Implement the same-color fork on 7×7+ grids (enough space for Zone 1 fork area), use investment-door Option A on 6×6 (less spatial budget). This gives three distinct sub-variants across grid sizes.

---

### Pre-emptive flags: T6 Relay and T7 Investment Door

These templates have not yet been flagged as weak by reviewers, but their structure shares properties with the templates that were flagged. Proactively noting them here.

**T6 Relay — potential arc homogeneity at scale:**
The logical arc is always: R key → R gate → collect B+G → bypass → B+G dual-lock → exit. On any given seed, every instance follows this arc. The only structural variation is which keys go above which traps (greedy bait placement). If a model learns "collect B and G, bypass both barrier-2 traps, open the dual-lock pocket", it transfers perfectly across all Relay seeds.

*If flagged:* Add a same-color fork inside Zone 1 (between barrier 1 and barrier 2). Zone 1 has two B doors: D_correct_B → B₂ key + G key (needed for dual-lock); D_trap_B → Zone 1T (real area, no keys). The agent must reason about which B door leads to the keys for the dual-lock. This is the same pattern as the T11 same-color fork design above.

**T7 Investment Door — near-deterministic structure:**
The arc is always: spend R on investment door → collect B+G → bypass → B+G dual-lock → exit. With the convergent forks bait (red trap closer to exit than investment door), the only interesting decision is whether to invest. Once a model learns the "invest to gain 2" pattern, it applies perfectly. The two split traps (blue and green) are secondary distractors.

*If flagged:* The simplest fix is adding a second investment room: after collecting B+G from the first investment room, a second door (e.g., B "investment") leads to a room with a second G key needed for the dual-lock. This creates a two-step investment chain where both spends must be correctly ordered. Alternatively, add a same-color fork on the split wall.

*Current decision:* Both T6 and T7 are functioning correctly and have not been flagged. Do not change them for the current benchmark round — monitor reviewer feedback after initial submission.

---

### Cross-cutting concerns

**1. Visual layout pattern leakage (bypass-gap convention)**

Templates 2, 4, 5, 6, 7 all use a bypass gap at an edge column as the "safe path" alongside one or more trap doors in the interior of a barrier. D4 orientation randomization rotates which edge the bypass is on, but within a single orientation, the pattern is consistent. A model that learns "the door closest to a grid edge is the safe bypass, doors in the interior are traps" would have above-chance performance across all five templates — not from understanding the key-dependency logic, but from reading the visual layout convention.

*Mitigation already in place:* D4 randomization ensures the bypass gap is at a random edge per instance, so no positional bias (top vs. bottom vs. left vs. right edge) is learnable.
*Residual risk:* Within a given orientation, "nearest to edge = safe" is still a consistent cue. This is worth a sentence in the paper as a limitation. Structural fix would require making some "correct" paths go through interior barriers and some "trap" paths go through edge columns — this would break the current bypass-gap generation pattern substantially.

**2. RQ3 hint removal — implementation note**

The removal of `"The fatal step is always a USE_KEY action"` from `build_rq3_prompt` (Priority P0 above) is a 2-line edit. After removal, the `build_rq3_ablation_prompt` (in `prompts.py`) should be updated to make the ablation explicit: add a comment in the ablation prompt that "hints" about USE_KEY are present, so the ablation variant is clearly the "scaffolded" version and the main prompt is the unscaffolded version. The ablation serves two purposes: (a) upper-bound estimate of RQ3 performance when the search space is reduced, (b) measures how much of the model's performance depends on the structural hint vs. genuine causal attribution.

**3. Difficulty calibration after T2 replacement**

Replacing Conservation Trap (T2) with Budget Partitioning changes the difficulty distribution. Current difficulty tiers:
- Simple: Fork, Switchback, Scarce Resource (3 templates, no dual-locks)
- Medium: Conservation Trap, KBK, Convergent Forks, Parallel Wings, Cascade Trap, ODL (6 templates, 0–1 dual-locks)
- Complex: Relay, Investment Door, SDL, Key Collector, Decoy Key (5 templates, 1–2 dual-locks)

Budget Partitioning should slot into Medium (2 RED keys, 1 BLUE key, 1 dual-lock exit). After the T2 replacement, Medium still has 6 templates (KBK, Convergent Forks, Parallel Wings, Cascade Trap, ODL, Budget Partitioning). The difficulty balance is preserved.

---

## Extended Brainstorm: Structural Approaches to Mitigating All Weaknesses

The piecemeal template fixes above address individual issues. This section steps back and asks: what are the deeper structural causes of these weaknesses, and are there approaches that address them more holistically?

---

### The real problem: templates are finite policies

The reviewer critiques point to something deeper than individual template weaknesses. The core issue is:

> **Every template has a fixed dependency graph topology. A model that learns the topology can solve all seeds of that template without doing any per-instance reasoning.**

The greedy filter catches naive greedy heuristics, but it does not catch **template-level policies** — strategies of the form "if I recognize this as an SDL instance, I know the correct sequence is: collect all → open dual-lock 1 → collect new key → open dual-lock 2." A model that sees enough SDL instances during training learns this meta-policy. Instances that fool a 6-strategy greedy baseline will not fool a meta-learned policy.

This means the real evaluation threat is a **14-way template classifier followed by a per-template optimal policy.** If such a policy exists and is learnable, the benchmark measures template recognition, not causal reasoning.

---

### Thread 1: Stochastic topology variation within templates

**The idea:** Within each template call, randomly select among 2–3 structurally distinct sub-variants using the seed's rng. The model can't apply a per-template policy if it doesn't know which sub-variant it's seeing — it must actually read the instance.

**For SDL specifically:**
- Sub-variant A: investment door before first dual-lock (Options A from above)
- Sub-variant B: same-color fork in Zone 1 between the two dual-locks
- Sub-variant C: original structure (collect all, DL1, collect key, DL2) — kept to prevent the model from assuming a complex variant always exists

The model must observe the specific layout to determine which variant it's facing, forcing per-instance reasoning.

**Generalizes across templates:**

| Template | Sub-variant A | Sub-variant B | Selection |
|:---|:---|:---|:---|
| SDL (T11) | Investment door before DL1 | Same-color fork in Zone 1 | `rng.choice(["A","B"])` at 7×7+; only A at 6×6 |
| Parallel Wings (T5) | Key room gated (investment) | Open key room, internal corridor walls force path | `rng.random() < 0.5` at 6×6+ |
| KBK (T3) | Depth-2 chain | Depth-3 chain | Grid-size-driven (5×5 → 2, 6×6+ → 3) |
| Fork (T1) | One same-color trap | Two traps (different colors, only one fatal) | `rng.random() < 0.3` for enhanced variant |

**Why this is more powerful than D4 randomization:** D4 randomizes spatial layout within a fixed topology. Sub-variant selection randomizes the topology itself. These are orthogonal dimensions of diversity — both should be applied.

**Implementation cost:** Low to medium. Each template's generator needs an if/elif branch per sub-variant. The existing rng-based generation infrastructure handles this naturally. The oracle and greedy filter run on the generated instance regardless of which sub-variant was selected.

---

### Thread 2: The greedy filter is the wrong filter

**Current filter:** reject instances where any of 6 strategy-agnostic greedy policies succeed.

**Missing filter:** reject instances where a **template-aware greedy policy** succeeds. For each template, define the most obvious template-level meta-policy and run it as an additional validation filter.

**Example meta-policies:**

| Template | Meta-policy |
|:---|:---|
| SDL | Collect all keys → open first dual-lock → collect key inside → open second dual-lock → exit |
| Parallel Wings | Collect blue key (left wing) → collect red key (right wing) → open dual-lock exit |
| Key Collector | Collect all keys without using any → open triple-lock exit |
| KBK | Use red key on first door encountered → collect key behind it → use that key on next door |
| Relay | Use red key → collect both keys in zone 1 → bypass both traps → open dual-lock |

Any instance solvable by its template's meta-policy should be **rejected** by the validator, forcing the generator to produce instances where even the template-specific strategy fails.

**What "template meta-policy fails" means in practice:** It means the instance has a sub-variant that deviates from the default arc — e.g., the investment door is on the wrong side for the meta-policy's traversal order, or the same-color fork makes the default choice wrong. This naturally pressures the generator toward the more complex sub-variants from Thread 1.

**Implementation:** Define a `TEMPLATE_META_POLICIES` dict mapping template name → a `Callable[[ChainVaultGrid, ChainVaultState], list[ChainVaultAction]]`. Add this as validation filter #6. If the meta-policy successfully reaches the exit, reject the instance.

---

### Thread 3: Multi-decision traces for RQ3 (the "one fatal step" problem)

**Current design:** Every RQ3 trace has exactly one fatal step — the first point of no return, identified by `find_fatal_step`. The model's task is to find that one step among N actions.

**Problem:** A model with recency bias picks the most recent USE_KEY action. Because the fatal step is always a USE_KEY, and the most recent USE_KEY is sometimes the fatal one, recency bias produces above-chance accuracy on RQ3 without any genuine causal reasoning.

**Fix:** Generate traces where the agent makes **several non-fatal suboptimal moves before the fatal one**. Specifically:
- Oracle prefix up to step i
- At step i: the agent makes a wrong-but-non-fatal USE_KEY move (e.g., using a surplus key on a valid but non-critical door — the puzzle remains solvable)
- At step j > i: the agent makes the actually-fatal USE_KEY move
- Continuation after step j until the trace reaches target length

The `find_fatal_step` oracle still correctly identifies step j as fatal (step i left the puzzle solvable; step j did not). But the model now faces a trace with **multiple wrong-looking USE_KEY actions**, only one of which is the fatal one. Recency bias would guess step j (the last wrong USE_KEY), which is actually correct — but on instances where j is early in the trace, recency bias would fail.

**Generator change:** Extend `generate_hard_rq3_instance` to support a "multi-decision" mode. After the oracle prefix at step i, instead of immediately making the fatal wrong move, first make 1–2 valid-but-suboptimal USE_KEY moves (non-fatal divergences), then make the fatal move.

**Requires:** Instances with more than one USE_KEY action after the prefix point. KBK (3-hop at 6×6+), SDL, Relay, and Investment Door are the natural candidates — they have enough key-use events.

---

### Thread 4: Counterfactual rephrasing of RQ3

**Current RQ3 prompt:** *"Identify the fatal step: the action whose execution caused the puzzle to become unsolvable."*

**Counterfactual rephrasing:** *"If you could undo exactly one action in this trace, which action would you undo to restore solvability from that point forward?"*

These are logically equivalent (undoing the fatal step restores solvability; undoing any other step doesn't). But the counterfactual framing changes the reasoning process:
- The model must consider each action's reversible effect on game state, not just scan for "wrong-looking" moves
- Undoing a MOVE action never helps (movement doesn't consume resources), so the model must narrow to USE_KEY actions by reasoning, not by hint
- The phrase "restore solvability" forces the model to reason about what solvability requires, connecting to the RQ2 skill

**Additional benefit:** The counterfactual framing naturally extends to multi-decision traces (Thread 3). Even if there are multiple wrong-looking actions, only undoing the true fatal step restores solvability — the model must verify each candidate by reasoning about the post-undo state.

**Implementation:** Trivial prompt change in `build_rq3_prompt`. No generator changes. Keep the original phrasing in `build_rq3_ablation_prompt` as the "direct" ablation variant, making the ablation a comparison between direct attribution vs. counterfactual reasoning.

---

### Thread 5: Missing dependency graph types

The 14 templates claim diverse dependency structures, but several map to the same graph. Actual coverage:

| Graph type | Templates | Coverage |
|:---|:---|:---|
| Binary fork (choose right door, same color) | T1, T13 (partially) | Covered |
| Save-for-later (don't spend X, need for Y+Z) | T2, T4 | Redundant — T2 ≈ T4 |
| Sequential chain (X → Y → Z) | T3, T6, T10 (spatial variant) | Covered |
| Investment (spend X to gain X+Y) | T7, T11 | Covered |
| Collect-all (need all N for final lock) | T12 | Covered |
| Bottleneck identification (X scarce, Y not) | T9 | Covered |
| Temporal ordering (A before B, same color) | T8, T14 | Covered |
| **Conditional allocation** (correct door depends on what other resources you have) | — | **Missing** |
| **Cycle breaking** (agent must traverse a cyclic path, temptation is to short-circuit) | — | **Missing** |

**Conditional allocation — what it tests:**
The correct door to use a key on isn't fixed — it depends on what other keys the agent currently holds. E.g., if the agent has both red and blue, it should use red on Door A; if it only has red, it should NOT use red on Door A (save it for Door B). This tests whether the model can reason about resource allocation as a function of global state, not just local proximity.

**Cycle-breaking — what it tests:**
The correct path is a cycle: start → collect key K1 → go left → unlock door D1 → go right → collect key K2 → go right → unlock door D2 → go left → proceed. The greedy temptation is to take a shorter path that skips K1 — but D1 is then locked, forcing backtrack. Tests whether the model can plan a path that temporarily moves away from the goal to acquire a necessary resource.

**Priority:** Conditional allocation is the higher-priority gap — it's a genuinely new reasoning test that requires holding multiple constraints in mind simultaneously. Cycle breaking is partially covered by KBK (the agent has to go "backward" to collect the key behind the correct door), but a dedicated template would make it more explicit.

---

### Thread 6: Evaluation-level fixes (what we measure)

Some problems can't be fixed by generator changes. They need to be addressed in how results are reported and evaluated.

**Per-template vs. cross-template accuracy:**
Report accuracy separately for each template AND the aggregate. If a model achieves 80% on SDL but the SDL meta-policy exists and is learnable, that 80% reflects policy recognition, not reasoning. High aggregate + low per-template variance = meta-policy learning. High aggregate + high per-template variance = genuine reasoning.

**Template-blind evaluation split:**
Hold out 3–4 templates from any few-shot examples in the evaluation prompt. Test whether models that see few-shot examples from T1–T10 correctly solve T11–T14. If performance drops significantly on held-out templates, the model is relying on template pattern matching. If performance is stable, the rules are genuinely being followed.

**Ablation on rules prompt:**
Test performance with game rules stripped (image only, no rules text). This isolates whether models are following stated rules or pattern-matching on visual features. Expected result: performance degrades significantly on RQ2/RQ3 but less so on RQ1 (which depends more on spatial reading).

**Difficulty calibration reporting:**
Rather than reporting a single aggregate accuracy, report accuracy as a function of: (a) k_horizon (cause-symptom gap), (b) trace length, (c) number of USE_KEY actions in trace. This shows where model performance degrades and lets the paper make a more specific claim about what the benchmark is diagnosing.

---

### Synthesis and revised priority table

| Priority | Change | Thread | Effort | Impact | When |
|:---|:---|:---|:---|:---|:---|
| **P0** | Remove USE_KEY hint from RQ3 prompt | — | Minimal | High — makes RQ3 genuinely diagnostic | Now |
| **P0** | Counterfactual rephrasing of RQ3 prompt | 4 | Minimal | High — eliminates hint-free shortcut reasoning | Now |
| **P1** | Stochastic sub-variant selection: SDL (fork + investment) | 1 | Medium | High — eliminates arc homogeneity at source | Before dataset gen |
| **P1** | Stochastic sub-variant selection: Parallel Wings | 1 | Medium | High — eliminates open-world greedy | Before dataset gen |
| **P1** | KBK depth scaling (already decided) | — | Medium | High | Before dataset gen |
| **P1** | Budget Partitioning replaces Conservation Trap (already decided) | — | Medium | Medium–High | Before dataset gen |
| **P2** | Template-aware meta-policy as filter #6 | 2 | Medium | High — directly targets policy-learning threat | Before dataset gen |
| **P2** | Multi-decision RQ3 traces | 3 | Medium | Medium — harder causal attribution | Before dataset gen |
| **P3** | Conditional allocation new template | 5 | High | High — genuinely new reasoning graph | Next round |
| **P3** | Per-template accuracy + template-blind eval split | 6 | Low | Medium — correct evaluation framing | Paper |
| **P3** | Difficulty calibration by k_horizon / trace length | 6 | Low | Medium — richer diagnostic claim | Paper |
| **P4** | Pre-emptive flags for T6 Relay + T7 Investment Door | — | None | Low — monitor only | After submission |

---

## Per-Template Improvement Brainstorm

For each template: current mechanic, specific weaknesses, and concrete improvement options. Options are labeled **Quick** (generator tweak, ≤1 day), **Medium** (structural generator change, 2–4 days), or **Major** (new sub-variant or template redesign).

---

### T1: Fork

**Mechanic:** 1 red key, 2 red doors. Choose the correct door; wrong door leads to a walkable dead-end zone.

**Weaknesses:**
- Binary choice with 1 key — the simplest possible template. Once a model learns "avoid the door closer to the exit", it transfers perfectly.
- The correct vs. trap distinction is always spatial (manhattan distance + convergent-forks bait). No content reasoning required: the model never needs to think about what's *behind* each door.

**Improvement options:**

*Option A — 3-way fork at larger grids.* **Quick.** At 6×6+, add a third red door (a second trap). The agent has 1 key and 3 red doors; 2 are traps, 1 is correct. Increases elimination difficulty: "closer to exit" heuristic still fails (both traps are placed strategically), but the model must now rule out 2 options rather than 1. Pairs naturally with D4 rotation (which door is "closest to exit" changes per orientation).

*Option B — prerequisite chain before the fork.* **Medium.** Add a blue door blocking access to the red key. The agent must first find and use the blue key to unlock the corridor to the red key, then face the fork. Now Fork is a 2-step puzzle: blue-key retrieval (linear, no decision) + red-door selection (the fork). This does not add reasoning complexity to the fork decision itself but makes the trace longer and more diagnostic for RQ1/RQ3.

*Option C — content-informed fork.* **Medium.** Both red doors lead to genuine zones, but only one zone contains the exit. The other zone contains a visible green key on a locked shelf (behind a green door inside the zone) — it looks like it might help, but there's no green door on the exit path. The model must reason: "does this key lead to anything useful, or is this a content dead-end?" Tests forward-looking reasoning about zone contents, not just spatial proximity.

*Option D — fork with a spare key at larger grids.* **Quick.** At 7×7+, add a second red key. Now the agent can open the trap door AND the correct door — the puzzle becomes about whether the agent can identify the correct door, not whether it has enough keys. Deadlock scenario: use first key on trap (Zone 1T) → walk around → use second key on trap again (there's a second trap) → zero keys left, correct door still locked. Tests key accounting across multiple choices.

**Recommended:** Option A (3-way fork) at 6×6+ is the highest-leverage, lowest-effort improvement. Implement as a stochastic variant (50% chance at 6×6, deterministic at 7×7+).

---

### T2: Conservation Trap → Budget Partitioning (replacement)

**Current mechanic:** 2 keys (R+B), red single-lock trap + R+B dual-lock exit. Bypass gap on barrier 1. Greedy wastes red on trap, can't open dual-lock.

**Weakness:** Redundant with T4 (Convergent Forks) — same dependency graph, fewer traps. Replacement already decided.

**Budget Partitioning design:** (Detailed above in the Review Findings section.) 2 RED keys, 3 RED doors: Door A (mandatory → blue key zone), Door B (trap → dead end), Door C (dual-lock R+B, exit). Exactly 2 reds needed; using one on Door B leaves agent one short for A+C.

**Additional improvement options for Budget Partitioning:**

*Option A — asymmetric budget.* **Medium.** 3 RED keys, 4 RED doors: 2 mandatory uses (Door A and Door C), 1 trap (Door B), 1 spare (Door D — optional, leads to a key that opens a shortcut but isn't needed). The model must identify that 1 spare exists and 1 is a trap — spending the trap key AND the spare leaves exactly 2 for the mandatory uses, but spending 2 traps is fatal. Tests counting under asymmetric allocation with one slack unit.

*Option B — cross-color budget.* **Medium.** 2 RED + 2 BLUE keys. Three mandatory uses: Door A (R → blue key zone), Door B (B → shortcut), Door C (R+B dual-lock, exit). One red trap (Door D → dead end). Agent must budget BOTH colors: if it spends either red on D, or either blue incorrectly, the dual-lock becomes unreachable. Tests multi-color budget partitioning simultaneously.

*Option C — "almost enough" framing.* **Quick.** Keep the 2-RED / 3-door design but make Door B's zone contain a decoy blue key (wrong color for the dual-lock — e.g., green, not the blue needed for the dual-lock). The agent wastes a red on Door B, finds what looks like a useful key, but it's the wrong color. The Walking Dead state is discovered only when the agent reaches Door C with insufficient reds. Adds a false-recovery temptation: the agent might think the green key compensates.

**Recommended:** Start with the base 2-RED design (existing plan). Add Option B (cross-color) at 7×7+ as a stochastic variant.

---

### T3: Key-Behind-Key

**Mechanic:** Sequential dependency chain. R key → correct R door → B key → B door → exit. Red trap door (wrong zone, no blue key). Chain depth 2.

**Weaknesses:**
- Chain depth fixed at 2 regardless of grid size. 7×7 grid has the same 2-step reasoning as 5×5 — just more empty space.
- The "correct door" can always be identified spatially (correct door is farther from exit). Content reasoning (what's behind each door) is not required.

**Improvement options:**

*Option A — scale chain depth with grid size.* **Medium.** (Already decided.) 5×5 → depth 2, 6×6 → depth 3 (R→G→B→exit), 7×7+ → depth 3–4. Each hop adds an intermediate key/door pair and a new trap. Wasting any key in the chain breaks a different link.

*Option B — parallel chains with a merge point.* **Major.** Two separate chains (left: R→G→blue_zone, right: Y→P→blue_zone) that both lead to the same blue key pickup area. The exit requires a blue door. Agent must complete at least one chain to collect the blue key, but trap doors on both chains consume the chain's key, making that chain's blue key access permanently unavailable. If both chains are broken, blue key is unreachable. Tests reasoning about parallel resource paths with a shared endpoint.

*Option C — chain with a "loan" mechanic.* **Medium.** R → correct R door → B key → B door → R₂ key (a second red inside) → second R door → exit. The chain cycles back to the first key's color. The trap: a red door at the start that looks like the correct chain entrance but leads to a zone with only a dead-end and a false B key (wrong color). Tests whether the model can distinguish the start of the chain from a decoy entrance.

*Option D — hidden chain length.* **Medium.** The agent doesn't know how deep the chain goes from the initial state. It sees only R key and R doors. Behind the correct R door: a zone with a B key AND another R door. Behind that R door: another zone with a G key. Only then is the exit reachable. The chain depth is not signaled visually — the agent discovers it one step at a time. Tests whether models can reason under uncertainty about chain length.

**Recommended:** Option A (depth scaling) is already planned. Add Option C at 7×7+ as a stochastic variant — the cyclic chain is a meaningfully different reasoning structure.

---

### T4: Convergent Forks

**Mechanic:** Two areas (left: red-guarded, right: blue-guarded), dual-lock exit. 2R + 2B keys: use 1R on area A, 1B on area B, remaining R+B on dual-lock exit. Trap doors adjacent to keys waste the matching color.

**Weaknesses:**
- The reasoning test ("visit both areas, save one of each color for the dual-lock") is learned after a single instance. No per-instance variation in the dependency graph.
- The two areas are interchangeable (visiting left before right vs. right before left are equally valid).

**Improvement options:**

*Option A — asymmetric area yields.* **Medium.** Area A (behind red door) contains a green key in addition to being traversable. Area B (behind blue door) contains nothing extra. Exit requires R+B+G triple-lock. Now the agent must visit Area A first (to collect green), and must budget: 1R for Area A door, 1B for Area B door, then R+B+G for exit — but only 2R+2B+1G are available (1R spare, 0G spare). Introduces an ordering constraint (Area A must be visited to collect G) and tighter budget.

*Option B — sequential fork dependency.* **Medium.** Area A contains not only a zone to traverse but also the blue key needed for Area B's door. The blue key is behind the red door. So the agent must open Area A first (using red) to get the blue key to open Area B. This converts the "unordered parallel visit" into a "forced sequential visit" — a meaningfully different reasoning task. Trap: blue trap door at start (convergent-forks bait) wastes blue before the agent enters Area A, making Area B permanently inaccessible.

*Option C — 3-way fork at 7×7+.* **Medium.** Add a third area (green-guarded). Exit requires R+B+G triple-lock. 2R+2B+2G keys available; use 1 of each on the three area doors, remaining 1R+1B+1G on triple-lock. Three trap doors (one per color, adjacent to corresponding keys). Tests whether the model can reason about three simultaneous budget constraints.

*Option D — false area contents.* **Quick.** One of the two areas contains a decoy key (a key whose color doesn't match any remaining door requirement). The agent that opens this area might think the decoy key gives more options, but it's irrelevant to the exit. Tests whether models can identify which keys are budget-relevant — but conflicts with the `_all_keys_relevant` filter (would need to relax this filter for this template).

**Recommended:** Option B (sequential fork dependency) at 6×6+ as a stochastic variant. It converts the unordered parallel structure to an ordered sequential one — a more diagnostic reasoning test. Option C (3-way) at 7×7+.

---

### T5: Parallel Wings

**Mechanic:** Left wing (blue key + blue door), right wing (red key + red door), dual-lock exit. All keys freely reachable. Trap doors on divider waste cross-wing keys.

**Weaknesses:**
- Open-world: no door guards any key. "Collect all, open exit" always valid.
- Trap doors only fire if agent is pathologically impatient (uses key immediately on adjacent trap).

**Improvement options:**

*Option A — gate one wing's key.* **Medium.** (Already decided.) Right wing key room gated by a blue single-lock. Agent must collect blue key (left wing) → bypass → unlock right wing key room → collect red key + second blue key → exit with R+B dual-lock. The bypass is still free but leads to a dead end at the red gate without blue.

*Option B — internal corridor structure.* **Quick.** Add 1–2 internal walls within each wing, converting open rooms into narrow corridors. The agent must now navigate through the wing in a specific path that passes directly adjacent to the trap door before reaching the key. This makes the convergent-forks bait more effective (agent can't walk around the trap) without changing the dependency graph. Does not fix the "collect all" policy but makes individual instances harder to navigate.

*Option C — shared resource between wings.* **Major.** Add a single green key in the center corridor (accessible without opening any door). A green door inside the left wing leads to a shortcut to the exit approach — but is completely optional. If the agent wastes the green key on a green trap door, it loses the shortcut, must take a longer path, AND the key count doesn't change (green isn't needed for the dual-lock). Tests whether the model can correctly identify which keys are "budget-critical" vs. "convenience-only" — and avoid wasting convenience keys on traps.

*Option D — three wings + triple-lock.* **Major.** At 7×7+, add a third wing (green key + green door area). Exit requires R+B+G triple-lock. All three wings must be completed. Three trap doors (one per color on the divider). Extends the parallel structure to three dimensions, testing three-color budget conservation simultaneously.

**Recommended:** Option A (gate one wing's key) is already the decided fix. Combine with Option B (internal corridor structure) for stronger trap placement — both are complementary.

---

### T6: Relay

**Mechanic:** R key → R gate → Zone 1 with B+G keys → bypass → B+G dual-lock exit. Trap doors on both barriers waste the keys needed for the dual-lock.

**Weaknesses:**
- Arc homogeneity: same logical sequence every seed (invest red → collect both → bypass both traps → exit). Pre-emptively flagged.
- Zone 1 is open: the agent can freely choose between B and G trap doors in any order. No ordering constraint within Zone 1.

**Improvement options:**

*Option A — same-color fork in Zone 1.* **Medium.** Add two blue doors in Zone 1 (a same-color fork, like ODL does on barrier 1). D_correct_B → yields B₂ key (replacement) + G key for the dual-lock. D_trap_B → dead end. Agent must reason about which B door leads to the keys needed for the exit. Seed variance: which side is correct randomized per seed. Now Zone 1 is not just "collect and bypass" but "choose the right B door, collect G, bypass green trap, open dual-lock."

*Option B — ordered zone collection.* **Medium.** The green key in Zone 1 is placed behind the blue door (B door leads to G key zone). So the agent must: open the B door (correct) → collect G key → then bypass the G trap on barrier 2 using the bypass gap. If the agent wastes blue on the B trap, both B and the path to G are lost simultaneously. The dependency chain inside Zone 1 becomes B→G (sequential), not B||G (parallel). Tests whether the model understands the within-zone ordering constraint.

*Option C — relay extension at 8×8.* **Major.** Add a third barrier (barrier 3) between Zone 2 and the exit. Barrier 3 has a purple single-lock (correct) + purple trap. Zone 2 contains the purple key. The relay chain becomes: R→B+G→P→exit. Each barrier tests a different decision. Tests 4-color relay reasoning.

*Option D — color randomization per seed family.* **Quick.** The base palette {R, B, G} is permuted per seed (e.g., some seeds use {R, G, B}, some use {B, R, G} as the relay order). The logical structure is identical but the specific key used at each step differs. A model that memorizes "use red first, then blue, then green" fails on permuted-color seeds. Forces genuine rule-following rather than color-order memorization.

**Recommended:** Option D (color randomization) is a near-zero-effort improvement that directly addresses the policy-memorization threat. Option A (same-color fork in Zone 1) at 7×7+ adds genuine per-instance decision variance.

---

### T7: Investment Door

**Mechanic:** Spend R on investment door → room yields B+G keys → B+G dual-lock exit. Red trap door closer to exit. Two split traps (B and G) inside right side.

**Weaknesses:**
- Arc homogeneity: "invest the red key, collect both, open the dual-lock" is the clear template policy, learnable after 1–2 instances. Pre-emptively flagged.
- The investment ratio is fixed: spend 1, gain 2. A model that learns "spend to gain 2" always knows the correct action without reading the specific layout.

**Improvement options:**

*Option A — variable investment ratio.* **Quick (as stochastic variant).** Some seeds: invest R → room yields B+G (ratio 1:2, existing design). Other seeds: invest R → room yields only B (ratio 1:1 — the agent must ALSO find G elsewhere). Other seeds: room yields B+G+P (ratio 1:3 — extra purple key, but exit still needs only B+G, so P is irrelevant). The model can't rely on "investment always gives both keys" — it must actually check what's behind the investment door. Implement via `rng.choice(["2-key", "1-key", "3-key"])`.

*Option B — two-stage investment.* **Medium.** First investment (R → B): spend red on investment door 1, yields only blue. Second investment (B → G): inside the investment room, a blue door leads to a second investment room yielding green. Exit needs B+G dual-lock, but agent used B on the second investment — so it needs B₂. This means: first investment room also yields B₂ (alongside B for the inner door). Tests whether the model can reason about a two-step investment chain and correctly sequence the spends.

*Option C — bad investment trap.* **Medium.** Add a second investment-looking door (same layout, different column on the barrier). D_invest_correct yields B+G as usual. D_invest_trap yields only B (insufficient for the B+G dual-lock). Both doors look structurally identical — same door type, same zone. The agent must reason about which investment yields sufficient keys. The anti-greedy placement: D_invest_trap is closer to start, D_invest_correct is farther. Tests whether the model checks investment yields rather than taking the first investment opportunity.

*Option D — non-investment bypass.* **Quick.** Add a free path to the exit zone that doesn't require the investment door. However, the free path ends at the dual-lock with only the initial R key — insufficient. The agent might think "I can reach the exit without investing." Forces the model to reason that merely reaching the exit zone is not enough; it needs the specific keys the dual-lock requires.

**Recommended:** Option A (variable investment ratio) as a stochastic variant — low effort, directly breaks the fixed-ratio assumption that makes the template policy trivially learnable.

---

### T8: Switchback

**Mechanic:** 2 red keys, 3 red doors. Use 2 correct doors in sequence. Third door is trap, placed spatially between correct doors A and B on the greedy route.

**Weaknesses:**
- Once a model learns "choose 2 of 3 same-color doors", it applies a "pick the 2 that are correctly spatially positioned" heuristic that works across seeds.
- The ordering is always A then B (forced by spatial layout: you must pass Zone 1A to reach Zone B's approach). Ordering constraint isn't a decision — it's structural.

**Improvement options:**

*Option A — 3 keys, 4 same-color doors at 7×7+.* **Medium.** Agent has 3 red keys, 4 red doors. Must use exactly 3 of the 4 in the correct sequence. One door is still a trap (wastes a key without contributing to progress, but doesn't block the correct path — however, using the trap key and one of the correct keys leaves insufficient keys for the third mandatory door). Tests counting under same-color redundancy.

*Option B — same-color with an ordering constraint that isn't spatially forced.* **Medium.** Zone 1A and Zone 1B are reachable in either order (no linear dependency). But the contents of Zone 1A affect what must be done in Zone 1B: Zone 1A has the second key AND a green door (D_ga) leading to a shortcut past Zone 1B's approach. If the agent visits Zone 1B first, they can't use D_ga and must take a longer path. Not a Walking Dead deadlock — but tests whether the model prefers the optimal ordering vs. a valid-but-suboptimal ordering.

*Option C — same-color with a temporal key constraint.* **Medium.** Zone 1 contains a third red key (K_r3) alongside the existing passage. The agent now has 3 red keys total for 3 red doors. The trap door is still wrong (using K_r3 on the trap before collecting K_r3... wait, K_r3 is inside Zone 1, accessible only after opening Door A). So: collect K_r1 and K_r2 in Zone 0 → open Door A (K_r1) → Zone 1 → collect K_r3 → open Door B (K_r2 OR K_r3) → proceed. Trap: if agent uses both K_r1 and K_r2 on the trap and Door A → enters Zone 1 → uses K_r3 on Door B → OK (still solvable). So instead, place trap where using K_r3 inside Zone 1 on an internal trap door makes Door B unreachable with 0 remaining keys. Tests multi-zone key accounting.

*Option D — mixed-color pair.* **Quick.** Add a single blue key and a blue door late in the puzzle (after Zone 1B, guarding the exit approach). Now the agent has 2R+1B: must use 2R correctly on the same-color fork, AND preserve 1B for the exit blue door. Adding a blue single-lock trap adjacent to the blue key (convergent forks bait) means both the same-color decision AND the blue conservation are tested simultaneously. A model that fails either loses.

**Recommended:** Option D (mixed-color pair) is the simplest addition that adds a genuinely new constraint on top of the existing same-color sequencing. Option A (4-door counting) at 7×7+ as a natural size-scaling extension.

---

### T9: Scarce Resource

**Mechanic:** 3R+1B keys, 2R doors (correct path) + 1B correct door + 1B trap door. Red is abundant (1 spare), blue is the bottleneck (1 key, 2 blue doors).

**Weaknesses:**
- Once the model identifies blue as the bottleneck (1 key, 2 doors), the decision is obvious: don't use blue on the trap. The bottleneck color is usually visually obvious (fewest keys vs. most doors for that color).
- Red abundance is not a distractor — the model correctly ignores red scarcity and focuses on blue. There's no reasoning required about red.

**Improvement options:**

*Option A — ambiguous scarcity.* **Medium.** Add a green color: 3R+1B+1G keys, with red being abundant, blue and green each having exactly 1 key and 1 door requiring each. One of {blue, green} is the "real" bottleneck (its door is mandatory), the other is a "pseudo-bottleneck" (its door leads to a dead end — wastes the key but doesn't prevent the exit). The agent must determine WHICH single-key color is actually critical. Requires reasoning about what each door leads to, not just counting keys vs. doors.

*Option B — dynamic scarcity.* **Medium.** The "scarce" resource isn't determined at the start — it becomes scarce based on the agent's choices. 2R+2B keys at start, 2R doors + 2B doors on two barriers. One R door and one B door are correct (open useful paths); one of each color is a trap. Whichever color the agent wastes first becomes the bottleneck for the second barrier. Tests whether the model can reason about how early decisions affect later resource availability. This is more of a "two-barrier scarce resource" than a "fixed-bottleneck" design.

*Option C — false abundance.* **Quick.** 3R+2B keys, 2R correct doors + 1B correct door + 1B trap. Red appears abundant (3 keys, 2 doors = 1 spare). But one of the "red" keys is actually inaccessible without first using blue on a blue door — so effective red count is 2 until blue is committed correctly. If the agent uses blue on the trap first, the third red becomes accessible (path opens) but the blue correct door is now locked with no key left. Tests whether the model correctly accounts for resource accessibility, not just raw counts.

*Option D — two-color simultaneous bottleneck.* **Medium.** 1R+1B+3G keys. Red: 1 key, 1 mandatory door (correct path). Blue: 1 key, 1 mandatory door (correct path). Green: 3 keys, 3 green doors (1 correct, 2 traps). Green is abundant but has many traps; red and blue are both scarce with no traps. The bottleneck identification task now has two answers: "red and blue are both critical, green has slack." Tests multi-color bottleneck identification.

**Recommended:** Option A (ambiguous scarcity — which single-key color is critical?) is the most diagnostic improvement. It converts the task from "identify the color with fewest keys" to "identify which door the scarce color MUST be used on."

---

### T10: Cascade Trap

**Mechanic:** Opening the red trap door changes spatial topology — blue key becomes unreachable. Spatial consequence on top of resource depletion.

**Weaknesses:**
- The spatial isolation after opening the trap is often visually apparent: the agent can see it's now walled off from the blue key room. "Does this door change my spatial access?" is the reasoning test, but it's visible from the new position.
- Cascade is depth-1: one wrong action causes immediate Walking Dead. No multi-step cascade.

**Improvement options:**

*Option A — cascade depth 2.* **Medium.** Opening red trap door D1 (wrong) → closes off yellow key access. Without yellow key → cannot open yellow door D2. Without opening D2 → cannot reach blue key. Without blue key → cannot open exit. The cascade now propagates 2 steps: D1 wrong → yellow gone → D2 locked → blue inaccessible → exit locked. Each step is individually recoverable-looking but the chain breaks the puzzle. Creates a longer cause-symptom gap for RQ3.

*Option B — invisible cascade.* **Medium.** The spatial isolation caused by opening the trap door is not immediately visible from the resulting position — the agent must navigate 3–5 cells before discovering it can't reach the blue key room. The Walking Dead state is reached quickly but the symptom (blocked path to blue key) appears only after additional movement. This directly increases the RQ3 difficulty without changing the fundamental mechanic.

*Option C — two-way cascade.* **Medium.** Two trap doors (D_trap_r and D_trap_b), each of which independently causes spatial isolation of a different resource. Opening D_trap_r cuts off blue key. Opening D_trap_b cuts off green key. Exit needs B+G dual-lock. If the agent opens either trap, one required key becomes unreachable → exit is inaccessible. Opening both traps is "doubly fatal" (not fatal in any new way, but the model must account for both risks). Tests multi-threat spatial reasoning.

*Option D — reversible-looking cascade.* **Quick.** After opening the trap door and becoming spatially isolated from the blue key, there's a visible door in the new zone that looks like it might lead back to the blue key area (same corridor direction). But this door requires a color the agent doesn't have. The false recovery option makes the Walking Dead state less obvious: the agent might think "I can still get back if I find a purple key" — but no purple key exists. Increases the deceptiveness of the Walking Dead state.

**Recommended:** Option A (cascade depth 2) provides the most diagnostic value for RQ3 — it extends the cause-symptom gap and makes causal attribution genuinely harder. Implement on 6×6+ grids.

---

### T11: Sequential Dual-Locks

**Mechanic:** R+B first dual-lock → Zone 1 yields second B key → B+G second dual-lock → exit.

**Weaknesses:** Arc homogeneity (same logical arc every seed). Already extensively analyzed and two fix approaches documented (investment door options + same-color fork). Combined recommendation: stochastic sub-variant selection.

**Additional improvement options not yet documented:**

*Option A — key-generating trap in Zone 1.* **Medium.** Zone 1 contains the B₂ key (as now) AND a blue single-lock trap door adjacent to B₂ (convergent-forks bait). Agent picks up B₂ and is immediately adjacent to the trap. Using B₂ on the trap → enters Zone 1T → returns → no B₂ for second dual-lock. This adds a third decision point (first dual-lock, Zone 1 trap, second dual-lock) and makes the convergent-forks bait relevant at Zone 1's level, not just at the initial approach.

*Option B — non-trivial Zone 0 collection.* **Quick.** Keys in Zone 0 (R, B, G) are placed in separate subrooms separated by internal walls rather than open floor. Agent must navigate into each subroom to collect each key. Doesn't change the dependency graph but increases RQ1 complexity (the agent's path during key collection is non-trivial) and makes greedy strategies less effective (BFS-to-exit heuristic leads through walls, not through subroom corridors).

*Option C — 3 dual-locks at 8×8.* **Major.** Three sequential dual-locks: R+B → (yields B₂+G) → B+G → (yields G₂+P) → G+P → exit. Each dual-lock unlock yields a fresh set of keys for the next stage. Requires 4 colors and 8×8 grid. Tests whether models can maintain a 3-stage combination plan. RQ3 traces from this template would have 3 candidate fatal steps (one per dual-lock), making causal attribution significantly harder.

**Recommended:** Option A (key-generating trap in Zone 1) combined with the existing investment-door / same-color-fork sub-variants. Option B (subroom collection) is a quick addition that increases trace complexity for RQ1.

---

### T12: Key Collector

**Mechanic:** 3 keys (R+B+G), triple-lock exit, 2 single-lock trap doors. "Collect all, avoid traps" is the obvious correct strategy.

**Weaknesses:** No intermediate USE_KEY decisions. Correct strategy is identical to naive strategy: save all keys for the triple-lock. Already extensively analyzed.

**Improvement options:**

*Option A — Path Commitment variant.* **Medium.** (Already decided.) Short path through two single-lock gates (uses R and B), long bypass path (free). Arriving at triple-lock via short path leaves only G — insufficient. Tests planning under path commitment. Implement as primary variant at 7×7+.

*Option B — One key hidden, must be unlocked.* **Medium.** The green key is not in the open — it's inside a room gated by a red door. Agent must: collect R (open) + collect B (open) → use R to unlock G key room → collect G → then have R+B+G but used 1R → only B+G remain. Exit requires R+B+G, so the agent must find a second red key. A second R is placed in Zone 1 (behind the first R door). Correct path: collect R₁+B → use R₁ on G room door → collect G → enter Zone 1 → collect R₂ → exit with R₂+B+G. Tests whether the model can plan a "use key to gain key" step even when the final goal needs all keys.

*Option C — Progressive reveal of lock requirements.* **Quick (prompt change, not generator change).** In the current design, the triple-lock's requirements (R+B+G) are stated in the rules text at the start. Change: the exit door label shows only "???" in the rendered image; the model must infer from context what's needed. This tests whether models can infer lock requirements from available evidence (number of keys present, past door patterns) rather than reading them directly. Primarily a RQ2/RQ3 prompt-level change.

*Option D — False triple-lock.* **Medium.** Add a door that LOOKS like a triple-lock (R+B+G) but actually only requires R+B (the G label is a decoy label in the image). Opening it with only R+B succeeds; the agent doesn't need G for this door. But there's a second door requiring only G for the exit. The correct plan: open the "fake triple-lock" with R+B → collect exit key inside → use G on exit door → win. Wastes less than expected; tests whether models verify actual lock requirements rather than assuming visual complexity maps to required keys.

**Recommended:** Option A (Path Commitment) already planned. Option B (hidden key requiring unlock) at 7×7+ as a stochastic variant — it adds a genuine "use key to gain key" sub-decision even when the final lock requires all keys.

---

### T13: Decoy Key

**Mechanic:** 3 keys (R+Y+B), 2-stage temptation. At each stage, a visible matching-color trap lures the key that's needed for a hidden same-color door further in the puzzle.

**Weaknesses:**
- Cognitive structure is similar to Fork: "resist the visible match, save the key for later." Just 2 stages and 3 keys instead of 1 stage and 1 key.
- The correct doors at each stage are a different color from the trap — a model that learns "use the NON-matching key at each barrier" succeeds on all seeds.

**Improvement options:**

*Option A — 3-stage temptation at 8×8.* **Medium.** Three barriers, each with a matching-color trap and a different-color correct door. Stage 3 is the reverse of Stage 1: the key saved from Stage 1 is needed at Stage 3. A model that learns "skip your matching key at each stage" still fails if it doesn't track which key is needed where. Tests resistance to temptation over 3 steps.

*Option B — "should use" vs. "should save" mixed stages.* **Major.** At Stage 1: using the yellow key on the yellow door IS correct (the door leads to a replacement yellow key inside + the correct red door). At Stage 2: using the blue key on the blue trap is wrong (no replacement). The model cannot apply a fixed "always save the matching-color key" rule — it must reason about replacement availability per stage. This directly breaks the meta-policy that makes Decoy Key predictable.

*Option C — Non-obvious color correspondence.* **Medium.** At Stage 2, the "hidden same-color door" is the exit door itself (B key → B exit door), not an intermediate door. The agent must recognize that the blue key collected in Stage 1 must be preserved all the way to the exit — not just to the next stage. Increases the planning horizon for the key-saving decision.

*Option D — Decoy quantity mismatch.* **Quick.** At Stage 1, agent has 2 yellow keys, 1 matching yellow trap. Using 1 yellow on the trap leaves 1 yellow for the gate — so using yellow on the trap is NOT fatal at Stage 1 (still has 1 left). But at Stage 2, only 1 blue key and 1 blue trap: using blue on trap IS fatal. Tests whether models correctly track quantities (losing 1 of 2 is fine; losing 1 of 1 is fatal) rather than treating "use key on matching door" as uniformly bad.

**Recommended:** Option B (mixed "should use / should save" stages) is the most impactful redesign — it breaks the "always avoid matching doors" meta-policy. Option D (quantity mismatch) is a quick variant that tests counting alongside decoy resistance.

---

### T14: Order-Dependent Lock

**Mechanic:** 2R+1B keys, barrier with 2 red doors (same color). Correct door leads to K_r2 (needed for R+B dual-lock exit). Trap door leads to dead end with no keys.

**Weaknesses:**
- Reviewer flagged this as fine. Structurally distinct from Switchback.
- The distinction between the two same-color doors is always spatial: one is farther from exit (correct), one is closer (trap, convergent-forks bait). A model that learns "for same-color forks, the farther door is correct" transfers across all seeds after D4 randomization is applied.

**Improvement options:**

*Option A — content-informed same-color fork.* **Medium.** Both same-color doors lead to genuine zones with content. D_correct leads to Zone 1A with K_r2. D_trap leads to Zone 1B with a green key (K_g). The exit, however, requires R+B dual-lock (not R+B+G). So K_g is irrelevant — but the agent must reason that entering Zone 1B gives an extra key that can't help, making D_trap a walking dead (can open D_correct from Zone 1B... wait, no — once in Zone 1B, both red keys are already spent).

Revised: agent enters Zone 1B (trap, uses K_r1) → collects K_g → cannot open D_correct (no red left) → cannot open exit (needs R for dual-lock) → stuck. The green key in Zone 1B is a false hope: it looks like compensation but isn't. Forces the model to verify that K_g has no role in the exit requirements.

*Option B — 3-way same-color fork at 7×7+.* **Medium.** Three red doors (2 traps, 1 correct). Agent has 2R+1B keys: enough to try 2 wrong red doors but not 3. If both reds are wasted on wrong doors, exit is inaccessible. If one red is wasted and one is correct, there's still no second red for the dual-lock. Correct path: use FIRST red only on the correct door → enter Zone 1A → collect K_r2 → exit with K_r2+K_b. Tests counting: the model must recognize it cannot afford any wrong-door attempts.

*Option C — same-color fork with ordering constraint.* **Medium.** Two red doors accessible from Zone 0, both leading to useful zones: D_A leads to Zone 1A with K_g (green key), D_B leads to Zone 1B with K_r2. Exit requires R+B+G triple-lock. Correct path: open D_A first (with K_r1) → collect K_g → open D_B (with K_r2... wait, K_r2 is IN Zone 1B, not held by agent).

Revised structure: K_r1 at start, Zone 0 → D_A (red, far-side) → Zone 1A (has K_r2 + K_g) → D_B (red, Zone 1A side) → Zone 1B (has K_b) → exit (R+B+G). D_trap_r (red, exit-side) → dead end. Correct path: K_r1 → D_A → collect K_r2+K_g → K_r2 → D_B → collect K_b → exit with K_r2(already used)+K_g+K_b...

Actually this is getting complex. The key insight is: add an ordering dependency between the two same-color doors so the "correct" one must be opened BEFORE the other becomes accessible. Like a sequential same-color chain rather than a parallel fork.

*Option D — same-color fork with decoy contents at larger grids.* **Quick (stochastic variant).** Zone 1B (trap side) now contains a blue key (K_b₂) in addition to being a walkable dead-end. The agent might think K_b₂ compensates for entering the wrong zone. But the exit needs R+B dual-lock: the agent used their only red (K_r1) opening D_trap, so there's no red for the dual-lock regardless of having 2 blue keys. The extra blue key in Zone 1B is a false compensation. Tests whether models correctly account for which specific resources are depleted, not just total key count.

**Recommended:** Option D (false compensation in trap zone) as a quick stochastic variant — low effort, adds a false-recovery temptation that increases RQ2 difficulty (Walking Dead with extra keys looks more solvable). Option B (3-way fork) at 7×7+.

---

### Cross-template: Quick wins available for all templates

Beyond individual template improvements, three changes apply uniformly across all templates:

1. **Color randomization (seed-level palette permutation).** Currently, most templates hardcode which colors play which roles (e.g., KBK is always R→B, Relay is always R→{B,G}). Permuting the color assignments per seed (via a stable rng mapping at instance creation time) prevents color-role memorization ("in Relay, the first key used is always red"). D4 transform already handles spatial randomization; color permutation handles semantic randomization. Implementation: add a `_permute_colors(rng, instance)` pass in `_transform_instance` that maps the base palette to a random permutation.

2. **Richer trap zone contents.** Most trap zones are empty (just walkable cells). Adding contextually irrelevant content (a key the agent can't use, a locked door the agent can't open, a decorative dead end) makes the trap zone look more like a genuine path. This increases the difficulty of RQ2 ("is this a Walking Dead state?") because the agent is in a zone with apparent resources, not an obviously empty dead end.

3. **Stochastic trap count.** For templates that currently have a fixed number of traps, add a 50% coin-flip at generation time that doubles the trap count (adding a second trap on the same barrier). The model can't assume "there is exactly one trap per barrier." This is already partially implemented via the `_pick_extra_trap_col` helper but should be made more systematic.

---

## Master Priority Table

Consolidates all improvement items — from earlier review rounds, the extended brainstorm, and the per-template brainstorm — into a single ranked list. Supersedes earlier scattered priority rankings.

**Priority definitions:**
- **P0** — Must complete before any dataset generation. Fixes fundamental validity flaws; running the benchmark without these produces misleading results.
- **P1** — Should complete before first submission. Meaningfully improves benchmark quality and reviewer confidence.
- **P2** — Good-to-have. Adds depth or robustness; do if time allows before submission.
- **P3** — Next version / camera-ready. Deferred without risk to the current submission.

---

### P0 — Pre-generation blockers

These are non-negotiable. They affect every instance generated.

| # | Change | Template | Type | Rationale |
|:--|:--|:--|:--|:--|
| P0.1 | Remove USE_KEY hint from `build_rq3_prompt` | All | Prompt edit | Hint reduces RQ3 to 2–5-choice elimination. The benchmark's core causal attribution claim is invalid while this hint exists. |
| P0.2 | Counterfactual rephrasing of RQ3 prompt ("which action would you undo?") | All | Prompt edit | Forces resource-state reasoning rather than scan-for-wrong-move. Orthogonal to P0.1; both should be applied together. |
| P0.3 | Parallel Wings (T5): gate right-wing key room behind blue single-lock door | T5 | Generator | Open-world "collect all, open exit" is always valid. Greedy filter passes only because agents don't immediately use the key — the policy itself is trivial once known. |
| P0.4 | Key-Behind-Key (T3): scale chain depth with grid size (depth 3 at 6×6+) | T3 | Generator | 7×7 and 8×8 instances currently have identical reasoning complexity to 5×5. Larger grids must be harder, not just spatially larger. |

---

### P1 — Pre-submission essentials

High-impact changes that meaningfully change what the benchmark measures.

| # | Change | Template | Type | Rationale |
|:--|:--|:--|:--|:--|
| P1.1 | Replace Conservation Trap (T2) with Budget Partitioning | T2 | New template | T2 ≈ T4 (same dependency graph). Slot must test something genuinely distinct: explicit resource counting under tight allocation. |
| P1.2 | Key Collector (T12): Path Commitment variant as primary variant at 7×7+ | T12 | Generator | Current template has no intermediate USE_KEY decisions. "Collect all, avoid traps" is identical to the correct policy. Path Commitment forces genuine path-planning under budget constraint. |
| P1.3 | SDL (T11): stochastic sub-variant selection — investment door (Option A) at 6×6, same-color fork at 7×7+ | T11 | Generator | Every SDL seed currently follows the same logical arc. Sub-variant selection randomizes the dependency graph topology, forcing per-instance reasoning. |
| P1.4 | SDL (T11): add convergent-forks bait at Zone 1 (key-generating trap adjacent to B₂ pickup) | T11 | Generator | Even with sub-variants, Zone 1 currently has no USE_KEY temptation. Adding a trap adjacent to the B₂ key creates a mid-trace Walking Dead risk and improves RQ2/RQ3 instance quality. |
| P1.5 | Convergent Forks (T4): sequential fork dependency variant — Area A contains the key needed for Area B's door | T4 | Generator | Converts the unordered parallel "visit both" structure into an ordered sequential chain. Different dependency graph; different reasoning test. Implement as 50% stochastic variant at 6×6+. |
| P1.6 | Investment Door (T7): variable investment ratio as stochastic variant — seeds randomly yield {B,G}, {B} only, or {B,G,P} | T7 | Generator | Fixed 1:2 investment ratio is the learnable policy. Variable ratio forces the model to check what's actually behind the investment door rather than assuming "invest always gives 2 keys." |
| P1.7 | Relay (T6): seed-level color palette permutation | T6 | Generator | "Use red first, then collect blue and green" is a memorizable policy. Permuting the base palette {R, B, G} per seed (via rng at instance creation) breaks color-role memorization with near-zero generator complexity. |
| P1.8 | Scarce Resource (T9): ambiguous scarcity variant — add green color, one single-key color is critical, the other is a pseudo-bottleneck leading to a dead end | T9 | Generator | Currently the bottleneck color is visually obvious (fewest keys vs. most doors). Ambiguous scarcity forces the model to reason about what each scarce door *leads to*, not just count keys. |
| P1.9 | Cascade Trap (T10): cascade depth 2 — wrong door D1 cuts off yellow key → yellow door D2 locked → blue key inaccessible → exit locked | T10 | Generator | Current cascade is depth-1: one wrong action → immediate Walking Dead. Depth-2 extends the cause-symptom gap, making RQ3 attribution genuinely harder. |
| P1.10 | Decoy Key (T13): mixed "should-use / should-save" stages — at one stage using the matching key IS correct (replacement inside); at another it's wrong | T13 | Generator | Current "always skip the matching door" meta-policy transfers perfectly. Mixed stages break this policy; model must reason per-stage about whether a replacement exists. |
| P1.11 | ODL (T14): false compensation in trap zone — Zone 1B (trap side) contains a blue key that looks like it compensates but can't open the dual-lock (no red remaining) | T14 | Generator (quick) | Makes the Walking Dead state after entering Zone 1B less visually obvious. Model must reason: "I have 2 blue keys but 0 red — can I still open R+B dual-lock?" Better RQ2 instances. |
| P1.12 | Cross-template: seed-level color palette permutation for all templates that hardcode color roles | All | Generator | Prevents color-role memorization (e.g., "in KBK, the first key used is always red"). Single `_permute_colors(rng, instance)` pass in `_transform_instance`. Complements D4 spatial randomization. |

---

### P2 — Pre-submission enhancements

Worth doing if time allows. Each improves a specific template without being a prerequisite for submission.

| # | Change | Template | Type | Rationale |
|:--|:--|:--|:--|:--|
| P2.1 | Fork (T1): 3-way fork (2 traps, 1 correct) at 6×6+ as stochastic variant | T1 | Generator | Binary choice is solved by "pick the non-trap." 3-way requires eliminating 2 options. Pairs well with D4 rotation (which door is "closest to exit" changes). |
| P2.2 | Fork (T1): content-informed fork — both doors lead to real zones, only one zone contains the exit | T1 | Generator | Converts spatial heuristic to content reasoning. Model must check what's behind each door rather than relying on position. |
| P2.3 | Budget Partitioning (T2): cross-color budget variant at 7×7+ — 2R+2B keys, mandatory uses in both colors, one red trap | T2 | Generator | Extends the allocation task to two colors simultaneously. Single-color BP is the base; this is the harder version for larger grids. |
| P2.4 | KBK (T3): cyclic chain variant at 7×7+ — chain returns to first key's color (R→G→R₂→B→exit) | T3 | Generator | Adds a non-linear dependency structure. Agent must track that collecting R₂ (behind the G door) is needed even though it already used an R key earlier. |
| P2.5 | Convergent Forks (T4): 3-way fork (3 areas, triple-lock exit) at 7×7+ | T4 | Generator | Scales the parallel structure to 3 colors. Triple-lock exit requires all 3 area visits to be budgeted correctly. |
| P2.6 | Parallel Wings (T5): internal corridor walls within each wing to force path commitment past trap doors | T5 | Generator (quick) | Complements P0.3 (key gating). Even with gating, open wings mean agents can approach traps from multiple directions. Corridors make the convergent-forks bait unavoidable. |
| P2.7 | Relay (T6): same-color fork in Zone 1 — two blue doors, D_correct yields B₂+G for dual-lock, D_trap is dead end | T6 | Generator | Adds a USE_KEY decision inside Zone 1 that varies per seed (which side is correct changes). Significantly improves seed variance beyond color permutation alone (P1.7). |
| P2.8 | Relay (T6): ordered zone collection — green key placed behind blue door in Zone 1, forcing B→G sequence | T6 | Generator | Converts Zone 1 from parallel (collect B and G in any order) to sequential (must open B door to access G). Creates an ordering constraint within the relay chain. |
| P2.9 | Investment Door (T7): bad investment trap — second investment-looking door on barrier that yields insufficient keys for the dual-lock | T7 | Generator | Adds a second candidate investment door that looks correct but fails. Model must verify yields rather than selecting the first investment opportunity. |
| P2.10 | Switchback (T8): 3 keys, 4 same-color doors at 7×7+ | T8 | Generator | Scales the same-color counting task. Must correctly choose 3 of 4 correct doors in sequence. Natural size escalation. |
| P2.11 | Switchback (T8): mixed-color pair addition — blue door guards exit approach; blue trap adjacent to blue key | T8 | Generator (quick) | Adds a second resource conservation constraint on top of the same-color sequencing. Model must simultaneously manage same-color decisions AND preserve blue for exit. |
| P2.12 | Scarce Resource (T9): two-color simultaneous bottleneck at 7×7+ — both blue and green are scarce (1 key each, 2 doors each) | T9 | Generator | Scales the bottleneck task to two colors. Both must be correctly allocated; wasting either is fatal. Tests multi-color bottleneck identification. |
| P2.13 | Cascade Trap (T10): two-way cascade — two trap doors, each independently isolates a different required key | T10 | Generator | Adds a second independent spatial trap. Model must reason about two simultaneous cascade risks, not just one. |
| P2.14 | SDL (T11): non-trivial Zone 0 key collection via internal wall corridors | T11 | Generator (quick) | Keys in separate subrooms increase RQ1 trace complexity and reduce BFS-to-exit greedy effectiveness even before the dual-lock decisions. |
| P2.15 | Key Collector (T12): hidden key variant at 7×7+ — green key locked behind a red door; second red available in Zone 1 | T12 | Generator | Adds a "use key to gain key" sub-decision even when the final goal needs all keys. Tests whether models can plan an instrumental unlock within a collect-all context. |
| P2.16 | Decoy Key (T13): 3-stage temptation at 8×8 — third barrier with matching-color trap; key saved from Stage 1 needed at Stage 3 | T13 | Generator | Extends resistance-to-temptation over 3 steps. Stage 3 requires remembering a Stage 1 key, increasing the planning horizon. |
| P2.17 | Decoy Key (T13): quantity mismatch variant — Stage 1 agent has 2 yellow keys (losing 1 is OK), Stage 2 has 1 blue (losing it is fatal) | T13 | Generator (quick) | Tests quantity-aware decoy reasoning: "skip all matching doors" is wrong when 2 of the same color exist and only 1 is needed later. |
| P2.18 | ODL (T14): content-informed same-color fork — trap zone (Zone 1B) contains a green key that looks compensatory but cannot open the R+B dual-lock (no red remaining) | T14 | Generator | Upgrades from P1.11. Zone 1B contains a key, not just empty space. Model must verify that the green key has no role in the exit requirements. |
| P2.19 | ODL (T14): 3-way same-color fork at 7×7+ — 2 trap doors, 1 correct; 2 red keys total, cannot afford to try 2 wrong doors | T14 | Generator | Scales the same-color fork. With only 2 reds and 3 doors, any wrong attempt depletes the budget for the correct door + dual-lock. |
| P2.20 | Cross-template: richer trap zone contents — add contextually irrelevant key or locked door inside trap zones | All | Generator | Makes Walking Dead states less obviously "stuck" — the agent is in a zone with apparent resources. Improves RQ2 diagnostic quality. |
| P2.21 | Cross-template: template-aware meta-policy as validation filter #6 | All | Generator | Rejects instances solvable by a template-specific optimal policy (not just template-agnostic greedy). Directly targets the policy-learning threat identified in the extended brainstorm. |

---

### P3 — Next version

Deferred without risk to the current submission. Either high effort, low urgency, or dependent on paper feedback.

| # | Change | Template | Type | Rationale |
|:--|:--|:--|:--|:--|
| P3.1 | Multi-decision RQ3 traces — traces with multiple non-fatal wrong moves before the true fatal step | All | Generator | Requires extended `generate_hard_rq3_instance`. High diagnostic value but complex to implement correctly. |
| P3.2 | Add "conditional allocation" template — correct door depends on currently held resources, not fixed | New | New template | Fills the largest gap in the dependency graph taxonomy. Genuinely new reasoning test. Plan in paper as future work. |
| P3.3 | Relay (T6): 4-color relay extension at 8×8 — adds a 4th barrier with purple key/door | T6 | Generator | Architectural complexity increase. Only meaningful if 8×8 grids are prioritized in the dataset. |
| P3.4 | SDL (T11): 3 sequential dual-locks at 8×8 — each unlock yields the resource for the next | T11 | Generator | Significant structural change. 3-candidate fatal steps makes RQ3 attribution much harder. Worth building after the 2-DL variant is validated. |
| P3.5 | KBK (T3): parallel chains with shared endpoint | T3 | Generator | Parallel resource paths with a single merge point. Major structural redesign; hold until depth scaling (P0.4) is proven out. |
| P3.6 | Parallel Wings (T5): 3-wing triple-lock at 7×7+ | T5 | Generator | Natural extension of the gating fix (P0.3). Only pursue after P0.3 is implemented and validated. |
| P3.7 | Key Collector (T12): progressive reveal of lock requirements in rendered image | T12 | Renderer + Prompt | Changes what's shown in the image. Requires renderer changes. Better tackled as a separate evaluation modality study. |
| P3.8 | Template-blind evaluation split — hold out 3–4 templates from few-shot examples | All | Evaluation | Important for paper claims. Zero implementation cost; requires evaluation pipeline setup. |
| P3.9 | Per-template accuracy + variance reporting | All | Evaluation | Distinguishes meta-policy learning from genuine reasoning. Paper-level analysis item. |
| P3.10 | Stochastic trap count uniformly across all templates | All | Generator | Already partially done via `_pick_extra_trap_col`. Systematize and make deterministic per-seed. Low urgency since scaling already handles this at larger grids. |

---

### Dependency order for implementation

Some P1 items must be sequenced correctly:

```
P0.3 (Parallel Wings gating)
  └─ must be done before P2.6 (corridor walls) — walls augment the gating, not replace it

P0.4 (KBK depth scaling)
  └─ before P2.4 (cyclic chain) — depth-3 chain must work before adding cyclic variant

P1.1 (Budget Partitioning template)
  └─ before P2.3 (cross-color BP variant) — base design must stabilize first

P1.3 (SDL sub-variant selection)
  └─ before P1.4 (Zone 1 trap) — sub-variants define Zone 1 structure; trap must match

P1.12 (color permutation cross-template)
  └─ after all P0/P1 generator changes — permutation is applied as a final transform pass,
     adding it early risks interacting with hardcoded color checks in templates

P2.21 (template-aware meta-policy filter)
  └─ after all P1 generator changes — the meta-policy must reflect the final template design,
     not the pre-improvement version
```

---

## What We Don't Need

- **Key crafting/combining**: Changes the puzzle from resource allocation to crafting logic — different reasoning axis entirely.
- **Breakable doors**: Defeats the scarcity mechanic. Key consumption is the point.
- **Doors requiring multiple keys of the same color**: Equivalent to reducing supply. Already captured by having fewer keys than doors.
- **Grids > 8×8**: With 2–4 colors and 3–8 doors, 8×8 state space is the practical ceiling. Larger grids add navigation padding without adding reasoning complexity.
- **More than 6 greedy strategies**: The 4 heuristic strategies plus 2 BFS-based ones is the right ceiling. Beyond 6, additional strategies have diminishing returns.
- **4+ key colors**: 3 colors is the upper bound. 4 colors would require larger grids and prompt complexity without proportional reasoning benefit.
- **More than 14 templates**: The 14 implemented templates cover all core dependency graph structures (fork, chain, parallel, conservation, cascade, investment, collection, decoy, temporal ordering). Additional templates would provide diminishing returns on reasoning diversity.
