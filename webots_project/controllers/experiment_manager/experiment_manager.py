"""
experiment_manager.py
ARAIP - Project Part 2
Author: Arul MithunKumar (408AIAUM)

Supervisor controller I use for testing. It does NOT help the e-puck search,
it only sets up the experiment:

  1. Puts the red, blue and green blocks at random positions (no overlaps,
     not too close to the walls, not in front of the robot at the start) and
     random rotations, so every run is a different layout like the brief asks.
  2. Watches the e-puck's customData field. The e-puck writes "DONE <time>"
     there when it has found all three blocks.
  3. Writes one line per run to results/runs.csv and, in batch mode, quits
     the simulation so the next run can start.

Environment variables (all optional):
    ARAIP_SEED      random seed for the block layout (default: random)
    ARAIP_BATCH     "1" = quit Webots when the run ends (for batch testing)
    ARAIP_LABEL     text saved in the CSV (e.g. "v1_random" or "v2_planned")
    ARAIP_TIMEOUT   give up after this many simulated seconds (default 180)
"""

import math
import os
import random

from controller import Supervisor

BLOCK_DEFS = {"red": "RED_BLOCK", "blue": "BLUE_BLOCK", "green": "GREEN_BLOCK"}
BLOCK_HALF_DIAG = 0.212          # half the diagonal of a 0.3 m square
ARENA_HALF = 1.0
WALL_CLEARANCE = 0.08            # gap between a block corner and the wall
MIN_BLOCK_GAP = 0.15             # gap between two blocks
ROBOT_CLEAR_RADIUS = 0.35        # keep blocks away from the robot's start spot

seed_env = os.environ.get("ARAIP_SEED")
seed = int(seed_env) if seed_env else random.randrange(1_000_000)
batch = os.environ.get("ARAIP_BATCH") == "1"
label = os.environ.get("ARAIP_LABEL", "interactive")
timeout = float(os.environ.get("ARAIP_TIMEOUT", "180"))

sup = Supervisor()
timestep = int(sup.getBasicTimeStep())
rng = random.Random(seed)


def random_layout():
    """Rejection sampling: keep trying until the three blocks fit."""
    limit = ARENA_HALF - BLOCK_HALF_DIAG - WALL_CLEARANCE
    while True:
        spots = []
        for _ in BLOCK_DEFS:
            x, y = rng.uniform(-limit, limit), rng.uniform(-limit, limit)
            if math.hypot(x, y) < ROBOT_CLEAR_RADIUS + BLOCK_HALF_DIAG:
                break
            if any(math.hypot(x - sx, y - sy) < 2 * BLOCK_HALF_DIAG + MIN_BLOCK_GAP
                   for sx, sy, _ in spots):
                break
            spots.append((x, y, rng.uniform(0, math.pi)))
        if len(spots) == len(BLOCK_DEFS):
            return spots


layout = random_layout()
for (colour, def_name), (x, y, rot) in zip(BLOCK_DEFS.items(), layout):
    node = sup.getFromDef(def_name)
    if node is None:
        print(f"[manager] WARNING: could not find {def_name} in the world")
        continue
    node.getField("translation").setSFVec3f([x, y, 0.05])
    node.getField("rotation").setSFRotation([0, 0, 1, rot])
    node.resetPhysics()

layout_text = "; ".join(f"{c} ({x:+.2f},{y:+.2f})"
                        for c, (x, y, _) in zip(BLOCK_DEFS, layout))
print(f"[manager] seed={seed} label={label} layout: {layout_text}")

epuck = sup.getFromDef("EPUCK")
data_field = epuck.getField("customData") if epuck else None
if data_field:
    data_field.setSFString("")

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, "runs.csv")

outcome, finish_time, extra = "timeout", timeout, ["", "", "", ""]
while sup.step(timestep) != -1:
    t = sup.getTime()
    text = data_field.getSFString() if data_field else ""
    if text.startswith("DONE"):
        parts = text.split()
        outcome, finish_time = "success", float(parts[1])
        extra = (parts[2:] + ["", "", "", ""])[:4]   # frames, full scans, ms/frame, unsticks
        break
    if t >= timeout:
        break

print(f"[manager] run finished: {outcome} at {finish_time:.1f} s")
new_file = not os.path.exists(csv_path)
with open(csv_path, "a", encoding="utf-8") as f:
    if new_file:
        f.write("label,seed,outcome,time_s,frames,full_scans,vision_ms,unsticks,red_xy,blue_xy,green_xy\n")
    xy = [f"{x:+.2f} {y:+.2f}" for x, y, _ in layout]
    f.write(f"{label},{seed},{outcome},{finish_time:.1f},{','.join(extra)},"
                f"{xy[0]},{xy[1]},{xy[2]}\n")

if batch:
    sup.simulationQuit(0)
