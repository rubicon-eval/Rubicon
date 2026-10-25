"""Tide Grid hand-crafted instances demonstrating bait-by-construction."""

from __future__ import annotations

from puzzles.common.types import Pos
from .engine import (
    TideGridGrid,
    TideGridState,
    TidalZone,
    Beacon,
    make_initial_state,
)


def example_the_strait() -> tuple[TideGridGrid, TideGridState]:
    """Template 1: Strait Crossing with structural bait (7x6).

    Two floor regions (top and bottom) separated by a wall barrier at row 2
    with two tidal crossing points (chokepoints).

    Layout (7 cols x 6 rows):
        Row 0: FLOOR  FLOOR  FLOOR  FLOOR  FLOOR  FLOOR  FLOOR
        Row 1: FLOOR  FLOOR  FLOOR  FLOOR  FLOOR  FLOOR  FLOOR
        Row 2: WALL   TIDAL_A WALL   WALL   WALL   TIDAL_B WALL
        Row 3: FLOOR  FLOOR  FLOOR  FLOOR  FLOOR  FLOOR  FLOOR
        Row 4: FLOOR  FLOOR  FLOOR  FLOOR  FLOOR  FLOOR  FLOOR
        Row 5: FLOOR  FLOOR  FLOOR  FLOOR  FLOOR  EXIT   FLOOR

    Bait mechanism:
    - Agent starts at (0,0) — top-left, near Zone A crossing.
    - Beacon b1 at (4,0) — bottom-left, below Zone A (bait beacon).
    - Beacon b2 at (1,5) — top-right, reachable via floor.
    - Exit at (5,5) — bottom-right, below Zone B.

    Zone A (left crossing at (2,1)): dry=2, flood=5, offset=0
        Greedy arrival ~t=2: phase=(2+0)%7=2 >= 2 → FLOODED! Agent must
        WAIT 5 turns for Zone A to reopen.  This is the trap.

    Zone B (right crossing at (2,5)): dry=3, flood=3, offset=2
        Correct-path arrival ~t=7 (via b2 first): phase=(7+2)%6=3 >= 3 → FLOODED?
        Actually t=8: phase=(8+2)%6=4 >= 3 → FLOODED.  t=10: (10+2)%6=0 < 3 → DRY.
        Correct path: go to b2 at t=5, WAIT or loop until zone B opens.

    The bait: greedy b1-first path needs ~5 WAITs at Zone A, then a long
    backtrack.  Optimal avoids Zone A entirely, using Zone B with proper timing.
    Oxygen = 22 (solution 20 + 2 slack).
    """
    width, height = 7, 6

    # Wall barrier at row 2 with two tidal gaps
    walls: set[Pos] = set()
    for c in range(width):
        if c not in (1, 5):  # gaps for tidal crossings
            walls.add(Pos(2, c))

    # Zone A (bait): short dry, long flood, offset=0
    # Dry at t=0,1.  Flood at t=2..6.  Dry t=7,8.  Flood t=9..13.  ...
    zone_a = TidalZone(
        zone_id="A",
        tiles=frozenset({Pos(2, 1)}),
        dry_duration=2,
        flood_duration=5,
        phase_offset=0,
    )
    # Zone B (correct): offset so dry window aligns with correct path
    # Dry at t=0..2.  Flood t=3..5.  Dry t=6..8.  Flood t=9..11.  ...
    zone_b = TidalZone(
        zone_id="B",
        tiles=frozenset({Pos(2, 5)}),
        dry_duration=3,
        flood_duration=3,
        phase_offset=0,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=Pos(4, 0)),
        "b2": Beacon(beacon_id="b2", pos=Pos(1, 5)),
    }

    grid = TideGridGrid(
        width=width,
        height=height,
        walls=frozenset(walls),
        tidal_zones={"A": zone_a, "B": zone_b},
        beacons=beacons,
        start=Pos(0, 0),
        exit_pos=Pos(5, 5),
        initial_oxygen=22,
    )
    return grid, make_initial_state(grid)


def example_tidal_maze() -> tuple[TideGridGrid, TideGridState]:
    """Template 2: Tidal Maze with structural bait (7x7).

    Two perpendicular tidal corridors divide the grid into quadrants.
    Zone A: horizontal corridor at row 3 (cols 0-2, 4-6).
    Zone B: vertical corridor at col 3 (rows 0-2, 4-6).
    Center cell (3,3) is a wall.

    Layout (7x7):
        Col:  0     1     2     3     4     5     6
    Row 0: FLOOR FLOOR FLOOR TID_B FLOOR FLOOR FLOOR
    Row 1: FLOOR FLOOR FLOOR TID_B FLOOR FLOOR FLOOR
    Row 2: FLOOR FLOOR FLOOR TID_B FLOOR FLOOR FLOOR
    Row 3: TID_A TID_A TID_A WALL  TID_A TID_A TID_A
    Row 4: FLOOR FLOOR FLOOR TID_B FLOOR FLOOR FLOOR
    Row 5: FLOOR FLOOR FLOOR TID_B FLOOR FLOOR FLOOR
    Row 6: FLOOR FLOOR FLOOR TID_B FLOOR FLOOR FLOOR

    Bait mechanism:
    - Agent at (0,0) — TL quadrant.
    - Beacon b1 at (4,1) — BL quadrant (bait — nearest, md=5).
      Requires crossing Zone A at row 3. Greedy arrives at Zone A ~t=3.
      Zone A is FLOODED at t=3: phase=(3+0)%7=3 >= 2 → FLOOD.
      Agent must WAIT 4 turns → drains oxygen.
    - Beacon b2 at (2,5) — TR quadrant (correct, md=7).
      Requires crossing Zone B at col 3.
    - Exit at (6,6) — BR quadrant.

    Zone A (horizontal, row 3): dry=2, flood=5, offset=0
        Dry t=0,1. Flood t=2..6. Dry t=7,8. Flood t=9..13. ...
        Greedy arrival at t=3 → FLOODED. Must WAIT until t=7.

    Zone B (vertical, col 3): dry=3, flood=3, offset=4
        Phase at t: (t+4)%6.  Dry when phase < 3.
        t=2: (2+4)%6=0 < 3 → DRY. Agent heading right can cross at t=2.
        t=5: (5+4)%6=3 >= 3 → FLOOD.
    """
    width, height = 7, 7

    walls = frozenset({Pos(3, 3)})

    # Zone A (bait): short dry, long flood
    zone_a_tiles = frozenset(
        {Pos(3, c) for c in range(7) if c != 3}
    )
    zone_a = TidalZone(
        zone_id="A",
        tiles=zone_a_tiles,
        dry_duration=2,
        flood_duration=5,
        phase_offset=0,
    )

    # Zone B (correct): offset so dry at correct-path arrival
    zone_b_tiles = frozenset(
        {Pos(r, 3) for r in range(7) if r != 3}
    )
    zone_b = TidalZone(
        zone_id="B",
        tiles=zone_b_tiles,
        dry_duration=3,
        flood_duration=3,
        phase_offset=4,
    )

    beacons = {
        "b1": Beacon(beacon_id="b1", pos=Pos(4, 1)),
        "b2": Beacon(beacon_id="b2", pos=Pos(2, 5)),
    }

    grid = TideGridGrid(
        width=width,
        height=height,
        walls=walls,
        tidal_zones={"A": zone_a, "B": zone_b},
        beacons=beacons,
        start=Pos(0, 0),
        exit_pos=Pos(6, 6),
        initial_oxygen=23,
    )
    return grid, make_initial_state(grid)
