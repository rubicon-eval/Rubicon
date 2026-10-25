# Mosaic Grid — Template Catalog

12 structural templates for procedural puzzle generation. Each template defines a wall layout that creates a planning challenge using the core mechanic: **irreversible paint forces the agent to plan tile-visits in a specific order**.

All templates pass the same validation pipeline:
1. Oracle-solvable (A* finds a solution)
2. Solution length within grid-scaled range (5x5: 6–20, 6x6: 8–30, 7x7+: 10–40)
3. All 5 greedy strategies fail (3 manhattan + 2 BFS with 8 tiebreaking orders)
4. At least 2 deadlocks with k-horizon >= 3 (for RQ2)
5. At least 1 deadlock with k-horizon >= 5 (for RQ3)

---

## Template Summary

| # | Name | Wall shape | Door mechanism | Plate placement |
|---|---|---|---|---|
| 1 | Barrier | Full vertical column | Door = gap in wall | Start side |
| 2 | Bottleneck | Partial vertical (40–60%) | Exit pocket | Both sides |
| 3 | Horizontal | Full horizontal row + free gap | Exit pocket | Start side |
| 4 | Spiral | 2 horizontal walls, opposing gaps | Exit pocket | Start side + middle band |
| 5 | Cross | Perpendicular segments at center | Exit pocket | Spread across grid |
| 6 | Peninsula | Dead-end corridor from edge | Exit pocket | Main area only |
| 7 | L-Shape | L-wall extending to 2 edges | Door = gap in L-wall | Inside L enclosure |
| 8 | Island | Central 2x2/2x3 block | Exit pocket | Ring + outer |
| 9 | Corridor | 3–5 random short segments | Exit pocket | Across branches |
| 10 | Moat | U-shape from one edge | Exit pocket | Main area |
| 11 | Pinch | 2 clusters from opposite corners | Exit pocket | Both sides |
| 12 | Zigzag | Alternating stubs on interior rows | Exit pocket | Along path |

---

## Door Mechanisms

Templates use one of three mechanisms to ensure the door is structurally required:

| Mechanism | How it works | Templates |
|---|---|---|
| **Door = wall gap** | The wall fully partitions the grid; the only crossing point IS the door. | Barrier, L-Shape |
| **Exit pocket** | All exit neighbours are walled off except one, which becomes the door. The exit is unreachable without opening the door. | Bottleneck, Horizontal, Spiral, Cross, Peninsula, Island, Corridor, Moat, Pinch, Zigzag |

---

## Template Grouping and Distinctiveness Analysis

At 5x5, the 12 templates produce grids with 4–10 walls on a 25-tile grid. After applying exit pocket uniformly, 8 of 12 templates share the same downstream structure (exit pocket → plate placement → extra walls). The **only differentiator is the initial wall geometry**, and on 5x5 some wall patterns converge visually.

### Group A: Linear wall dividing grid into 2 halves

| Template | Wall | Notes |
|---|---|---|
| **Barrier** | Full vertical column, door = gap | Cleanest design. The wall is the puzzle. |
| **Horizontal** | Full horizontal row, free gap + exit pocket | Same concept as barrier, rotated 90 degrees. |
| **Bottleneck** | Partial vertical column (40–60%), free gap + exit pocket | Shorter barrier with exit pocket. |

**Distinctiveness at 5x5:** Low between these three. Barrier is the archetype. Horizontal adds a free gap (non-door crossing point) which is structurally interesting. Bottleneck is the weakest — a shorter wall with exit pocket.

**At larger grids (7x7+):** A full-column barrier on 7x7 is a stronger structural feature than on 5x5. Bottleneck's partial wall becomes more meaningful when the partial coverage leaves a substantial gap section. These may differentiate better.

### Group B: Two parallel walls (serpentine)

| Template | Wall | Notes |
|---|---|---|
| **Spiral** | Two horizontal walls, opposing 2-tile gaps + exit pocket | Creates 3 zones (top / middle / bottom) with serpentine routing. |

**Distinctiveness at 5x5:** Moderate. The 3-zone structure is genuinely different from single-wall templates. With `height >= 5` and 2 walls at least 2 rows apart, the middle band is just 1 row — barely a zone.

**At larger grids:** On 7x7+ the middle band has 2–3 rows, making the serpentine path longer and more paint-prone. Spiral should improve at scale.

### Group C: Central obstacle

| Template | Wall | Notes |
|---|---|---|
| **Cross** | Perpendicular segments at center, 1 gap per arm + exit pocket | 4 semi-separated quadrants connected through narrow arm gaps. |
| **Island** | Central 2x2/2x3 block + exit pocket | Forces ring routing around the block. |

**Distinctiveness at 5x5:** Moderate between them. Cross creates linear barriers through the center (quadrant division). Island creates a solid block (ring routing). Different topologies.

**At larger grids:** Both improve. Cross arms get longer, quadrants get bigger. Island block can scale to 3x3 or 3x4, making ring routing more meaningful.

### Group D: Wall extending from edges, creating enclosure

| Template | Wall | Notes |
|---|---|---|
| **L-Shape** | L-wall to 2 edges, door = gap in wall | Seals off a corner. Door is in the structural wall. |
| **Moat** | U-wall from one edge + exit pocket | U-shape creates routing obstacle. |
| **Peninsula** | Dead-end corridor from edge + exit pocket | Corridor walls add constraints, but no plate at tip. |

**Distinctiveness at 5x5:** L-Shape is strong (unique corner-sealing topology, door in the wall). Moat is constrained (10 walls on 5x5 = 40% walled). Peninsula is the weakest — after removing the tip plate (entering traps you), the dead-end is purely decorative wall structure with no game element.

**At larger grids:** Moat's U-shape becomes more proportional. Peninsula could host a plate at the tip if the corridor is wide enough for backtracking. L-Shape scales naturally.

### Group E: Random/semi-random wall clusters

| Template | Wall | Notes |
|---|---|---|
| **Corridor** | 3–5 random short segments (len 2–3) | Exit pocket door. |
| **Pinch** | 2 clusters from opposite corners (2–4 tiles each) | Exit pocket door. |

**Distinctiveness at 5x5:** Low between them. Both place stochastic walls and use exit-pocket door placement. Pinch constrains clusters to opposite corners; Corridor doesn't. The outputs overlap.

**At larger grids:** More room for wall segments means more branches (Corridor) and wider pinch passages (Pinch). May differentiate better.

### Group F: Repeating edge pattern

| Template | Wall | Notes |
|---|---|---|
| **Zigzag** | Alternating 1–2 tile stubs on interior rows + exit pocket | Comb-teeth pattern. |

**Distinctiveness at 5x5:** Low. Stubs of 1–2 tiles produce only ~4 walls. Barely distinguishable from random walls.

**At larger grids:** On 7x7+ with 5 interior rows and stubs of 2–3 tiles, the comb-teeth pattern becomes a real serpentine corridor. Should improve significantly.

### Summary: Distinctiveness tiers at 5x5

| Tier | Templates | Reasoning |
|---|---|---|
| **Clearly distinct** | Barrier, L-Shape, Spiral, Cross, Island | Each has a unique wall topology and/or door mechanism that is visible even at 5x5. |
| **Moderately distinct** | Horizontal, Pinch, Moat | Horizontal adds a free gap vs Barrier. Pinch uses chokepoints. Moat has a U-shape. Differences are present but subtle at 5x5. |
| **Low distinctiveness at 5x5** | Bottleneck, Peninsula, Corridor, Zigzag | These converge toward "a few walls + exit pocket/chokepoint." May differentiate at larger grid sizes. |

**All 12 templates are retained** because:
1. They all pass the full validation pipeline (greedy-proof, deadlock-rich).
2. The distinctiveness gap narrows at larger grid sizes (7x7+), where wall patterns have more room to express their intended geometry.
3. For dataset generation, having more templates increases instance diversity even if some overlap at 5x5.

---

## Detailed Template Descriptions

### 1. Barrier

**Wall pattern:** Full vertical wall spanning the entire column, placed in the middle third of the grid. A single row is left open as the door gap.

**Door mechanism:** Door = gap in wall. The wall fully partitions the grid; the door is the only crossing point. No exit pocket needed.

**Plates:** All on the start side. Orientation randomized (left/right coin flip).

**Anti-greedy:** Collecting plates on the start side paints tiles that may block the return path to the door.

---

### 2. Bottleneck

**Wall pattern:** Partial vertical wall (40–60% of column height) with a free floor gap.

**Door mechanism:** Exit pocket. Exit surrounded by walls, one neighbour = door.

**Plates:** Both sides of the barrier, forcing multi-crossing through the free gap.

**Anti-greedy:** Multi-crossing through the gap paints approach tiles, eventually blocking passage.

---

### 3. Horizontal

**Wall pattern:** Full horizontal wall with a single free gap (not a door). Exit must not be adjacent to the gap.

**Door mechanism:** Exit pocket on the far side.

**Plates:** All on the start side. Orientation randomized (top/bottom coin flip).

**Anti-greedy:** Collecting plates on the start side paints paths needed to reach the gap, then the exit pocket door on the far side.

---

### 4. Spiral

**Wall pattern:** Two horizontal walls (at least 2 rows apart) with opposing 2-tile-wide gaps — one gap near the right, one near the left. Creates a serpentine path through 3 zones.

**Door mechanism:** Exit pocket.

**Plates:** Start side + middle band.

**Anti-greedy:** Serpentine traversal paints gap tiles needed for the return trip through the opposing gap.

---

### 5. Cross

**Wall pattern:** Horizontal segment along `mid_row` and vertical segment along `mid_col`, both skipping edges. Center tile is open (intersection). 1 gap opened per arm for connectivity. Creates 4 semi-separated quadrants.

**Door mechanism:** Exit pocket. Start/exit in diagonal-opposite quadrants.

**Plates:** Spread across grid.

**Anti-greedy:** Routing through arm gaps paints shared transit tiles, limiting future inter-quadrant movement.

---

### 6. Peninsula

**Wall pattern:** 2–3 tile dead-end corridor extending from a random edge (top/bottom/left/right), with walls flanking both sides.

**Door mechanism:** Exit pocket. Start, exit, and all plates are in the main area (not the peninsula).

**Plates:** Main area only. Entering the dead-end to reach a plate paints the escape route, making the puzzle unsolvable — so no plate is placed at the tip.

**Anti-greedy:** The peninsula walls constrain routing in the main area. Combined with exit pocket and paint mechanics.

**Note:** The peninsula is purely structural — no game element sits there. This makes it the weakest template at 5x5. At larger grids, a wider peninsula could host a plate with enough room for backtracking.

---

### 7. L-Shape

**Wall pattern:** L-shaped wall (horizontal + vertical arm) extending from a corner to two grid edges. Seals off a corner region. One gap in the wall = door.

**Door mechanism:** Door = gap in L-wall. The L-wall partitions the grid into inside (corner) and outside. The only crossing point is the door.

**Plates:** Inside the L enclosure (start side), with minimum spacing.

**Anti-greedy:** Collecting plates inside the enclosure paints tiles needed to reach the door. The asymmetric L-shape makes routing non-obvious.

---

### 8. Island

**Wall pattern:** Central 2x2 or 2x3 solid wall block. Creates a ring of floor tiles around the block.

**Door mechanism:** Exit pocket. Start/exit on opposite sides of the ring (randomized offset).

**Plates:** Ring and outer floor.

**Anti-greedy:** Choosing the wrong direction around the ring paints tiles blocking the reverse direction.

---

### 9. Corridor

**Wall pattern:** 3–5 short wall segments (length 2–3, random orientation), placed iteratively with connectivity validation. Creates branching paths with junction points.

**Door mechanism:** Exit pocket.

**Plates:** Across branches.

**Anti-greedy:** Taking the wrong branch paints a junction tile, cutting off access to other branches.

---

### 10. Moat

**Wall pattern:** U-shaped wall (two arms + base) opening toward one edge. Dimensions: 2–3 tiles deep, clamped to grid size.

**Door mechanism:** Exit pocket. Start far from moat, exit near it.

**Plates:** Main area outside moat.

**Anti-greedy:** U-wall constrains routing; exit pocket forces door usage.

---

### 11. Pinch

**Wall pattern:** Two wall clusters from opposite corners (top-left area: 2–4 tiles, bottom-right area: 2–4 tiles), placed with connectivity validation. Narrows traversable space to a 1–2 tile passage.

**Door mechanism:** Exit pocket.

**Plates:** Both sides of the pinch.

**Anti-greedy:** Each crossing through the pinch paints passage tiles. With plates on both sides, multiple crossings are needed.

---

### 12. Zigzag

**Wall pattern:** Alternating wall stubs on interior rows only (skip edge rows). Odd rows: stub from left edge. Even rows: stub from right edge. Stub length: 1 to `width // 2` tiles.

**Door mechanism:** Exit pocket. Orientation randomized (top/bottom coin flip).

**Plates:** Along the path.

**Anti-greedy:** Alternating direction changes paint corridor tiles at turns.

---

## Intra-Template Diversity

Every template includes three layers of randomization:

### 1. Randomized structural parameters

| Template | Randomized parameters |
|---|---|
| Barrier | Barrier column (`[width//3, width*2//3]`), door gap row, orientation flip |
| Bottleneck | Barrier column, wall coverage (40–60%), gap row, orientation flip |
| Horizontal | Wall row (`[2, height-3]`), gap column, orientation flip |
| Spiral | Row1/row2 positions (>= 2 apart), gap1/gap2 columns (opposing sides), orientation flip |
| Cross | Center position (`[2, dim-3]`), which arm gaps to open, quadrant pair for start/exit |
| Peninsula | Edge direction (4 options), corridor column/row, corridor length (2–3) |
| L-Shape | Corner position, arm directions (h_dir, v_dir), gap position in L-wall |
| Island | Block height (2–3), block width (2–3), ring offset for start/exit |
| Corridor | Segment count (3–5), per-segment length (2–3), orientation (h/v), position |
| Moat | U open direction (4 options), U depth (2–3), U half-span (1–2) |
| Pinch | Cluster sizes (2–4 each), cluster tile selection within quadrant |
| Zigzag | Stub base length (1 to `width//2`), per-row length variation, orientation flip |

### 2. Randomized game-element positions

- **Start/Exit:** Random within their constrained regions
- **Plates:** `_select_plate_positions()` enforces min spacing (manhattan >= 2) and tries for distinct distances from start
- **Door:** Template-specific (wall gap or exit pocket)

### 3. Extra random walls

After primary wall placement, `_add_extra_walls()` adds random walls while preserving connectivity. This changes corridor shapes, available detours, and deadlock structure across seeds.
