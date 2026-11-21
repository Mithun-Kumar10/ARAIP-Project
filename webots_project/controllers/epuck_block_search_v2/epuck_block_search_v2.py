"""
epuck_block_search_v2.py
ARAIP - Assignments 3, 4 and 5 (Project Part 2)
Author: Arul MithunKumar (408AIAUM)

This is the upgraded version of my Part 1 controller. The e-puck still has to
find the red, blue and green blocks, say "I see <colour>" the first time it
meets each one and print a summary, but I changed how it does it:

  * HSV colour detection instead of plain RGB rules (more robust to shading).
  * A camera PRE-FILTER: every step I only look at a sparse grid of pixels.
    The full image is only processed when the sparse grid spots a colour.
    This is the same idea I propose for running the CNN on the robot, only
    run the expensive model when something interesting is in view.
  * A planned search instead of a pure random walk: the robot does a 360
    degree SCAN, and if it sees nothing it drives to the nearest cell of a
    coverage grid it hasn't visited yet (wheel odometry tells it where it is),
    then scans again.
  * A stuck detector that reverses the robot if the wheels are turning but
    the robot isn't actually moving.
  * Every time a block is found the camera frame is saved to captures/ so the
    images can be fed into the CNN in Google Colab.
  * OpenCV vision (Session 9): if cv2 is installed the frame is turned into a
    NumPy array, converted to HSV with cvtColor() and each colour is found
    with an inRange() mask, like in the lecture. This replaces my per-pixel
    Python loop with a few vectorised calls. If cv2 isn't installed (e.g. on
    a lab PC) the controller falls back to the pure Python version, so it
    always runs. Set ARAIP_VISION=python to force the old version.
  * For each block found I also save the OpenCV filtered image (only the
    block's colour left, everything else blacked out) and print the mean
    R, G, B of the filtered pixels, the same experiment as in Session 9.

State machine:
    SCAN -> APPROACH -> BACK_OFF -> SCAN ...
    SCAN (nothing seen) -> GOTO_CELL -> SCAN ...
    any moving state -> AVOID (IR) / UNSTICK (stuck detector)
    all three found -> DONE
"""

import math
import os
import random
import time

from controller import Robot

try:                                  # OpenCV is optional (pip install opencv-python numpy)
    import cv2 as cv
    import numpy as np
    HAVE_OPENCV = True
except ImportError:
    HAVE_OPENCV = False

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
MAX_SPEED = 6.28                 # e-puck wheel limit (rad/s)
CRUISE_SPEED = 0.6 * MAX_SPEED
SCAN_SPEED = 0.25 * MAX_SPEED    # slow spin so the camera doesn't blur past a block

WHEEL_RADIUS = 0.0205            # m (e-puck datasheet / Webots model)
AXLE_LENGTH = 0.052              # m, distance between the two wheels

OBSTACLE_THRESHOLD = 80.0        # IR reading that means something is close

# HSV colour ranges. Hue is in degrees (0-360), S and V are 0-1.
# Red wraps around 0 degrees, so it has two ranges.
HUE_RANGES = {
    "red":   [(0, 20), (340, 360)],
    "green": [(90, 150)],
    "blue":  [(200, 260)],
}
MIN_SATURATION = 0.55            # the blocks are very saturated, grey/white isn't
MIN_VALUE = 0.20                 # ignore the black floor and dark shadows

SEEN_FRACTION = 0.01             # >1% of the image  -> colour is "visible"
FOUND_FRACTION = 0.15            # >=15% of the image -> block is "found"
PREFILTER_STRIDE = 4             # pre-filter samples every 4th pixel in x and y
PREFILTER_MIN_HITS = 1           # sparse hits needed before a full scan runs
# set ARAIP_PREFILTER=0 to switch the pre-filter off (used for my comparison tests)
USE_PREFILTER = os.environ.get("ARAIP_PREFILTER", "1") != "0"
# "opencv" (default when cv2 is installed) or "python" (my own per-pixel loop)
VISION_BACKEND = os.environ.get("ARAIP_VISION", "opencv" if HAVE_OPENCV else "python")
if VISION_BACKEND == "opencv" and not HAVE_OPENCV:
    VISION_BACKEND = "python"


def safe_get_image(camera):
    """camera.getImage(), but returns None instead of crashing when the camera
    has no frame yet (newer Webots Python API raises ValueError on a NULL image,
    e.g. on the very first step)."""
    try:
        return camera.getImage()
    except ValueError:
        return None


# Coverage grid used for the planned search (arena is 2 m x 2 m)
ARENA_HALF = 1.0
GRID_MARGIN = 0.2                # keep waypoints this far from the walls
GRID_CELLS = 4                   # 4 x 4 grid of waypoints inside the margin
WAYPOINT_TOLERANCE = 0.08        # m, close enough to the waypoint

STUCK_WINDOW_S = 2.5             # check progress over this many seconds
STUCK_MIN_DIST = 0.02            # less than 2 cm in that window = stuck

COLOURS = ["red", "blue", "green"]
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def wrap_angle(a):
    """Keep an angle between -pi and pi."""
    return math.atan2(math.sin(a), math.cos(a))


def rgb_to_hsv(r, g, b):
    """Fast integer-friendly RGB (0-255) -> HSV (deg, 0-1, 0-1).
    I wrote my own instead of colorsys because it runs ~2000 times per step."""
    mx = max(r, g, b)
    mn = min(r, g, b)
    v = mx / 255.0
    if mx == 0:
        return 0.0, 0.0, v
    delta = mx - mn
    s = delta / mx
    if delta == 0:
        return 0.0, s, v
    if mx == r:
        h = 60.0 * (((g - b) / delta) % 6)
    elif mx == g:
        h = 60.0 * (((b - r) / delta) + 2)
    else:
        h = 60.0 * (((r - g) / delta) + 4)
    return h, s, v


def classify_pixel(r, g, b):
    """Return 'red', 'green', 'blue' or None for one pixel."""
    h, s, v = rgb_to_hsv(r, g, b)
    if s < MIN_SATURATION or v < MIN_VALUE:
        return None
    for colour, ranges in HUE_RANGES.items():
        for lo, hi in ranges:
            if lo <= h < hi:
                return colour
    return None


# ----------------------------------------------------------------------------
# Vision: pre-filter + full HSV blob detection
# ----------------------------------------------------------------------------
class Vision:
    def __init__(self, camera):
        self.camera = camera
        self.w = camera.getWidth()
        self.h = camera.getHeight()
        self.total = self.w * self.h
        # stats so I can measure how much the pre-filter saves
        self.frames = 0
        self.full_scans = 0
        self.time_total = 0.0

    def _empty_result(self):
        return {c: (0.0, self.w / 2) for c in COLOURS}

    def prefilter(self, image):
        """Look at a sparse grid of pixels only. Returns True if any
        coloured pixel was spotted (so a full scan is worth doing)."""
        hits = 0
        w4 = self.w * 4
        for y in range(0, self.h, PREFILTER_STRIDE):
            row = y * w4
            for x in range(0, self.w, PREFILTER_STRIDE):
                i = row + x * 4
                if classify_pixel(image[i + 2], image[i + 1], image[i]):
                    hits += 1
                    if hits >= PREFILTER_MIN_HITS:
                        return True
        return False

    def full_scan(self, image):
        """Classify every pixel. Returns {colour: (fraction, mean_x)}."""
        counts = {c: 0 for c in COLOURS}
        x_sum = {c: 0 for c in COLOURS}
        w4 = self.w * 4
        for y in range(self.h):
            row = y * w4
            for x in range(self.w):
                i = row + x * 4
                colour = classify_pixel(image[i + 2], image[i + 1], image[i])  # BGRA
                if colour:
                    counts[colour] += 1
                    x_sum[colour] += x
        result = {}
        for c in COLOURS:
            frac = counts[c] / self.total
            mean_x = x_sum[c] / counts[c] if counts[c] else self.w / 2
            result[c] = (frac, mean_x)
        return result

    def to_frame(self, image):
        """The pure Python version works straight on the raw BGRA bytes."""
        return image

    def process(self, use_prefilter=True):
        """Grab a frame and return the colour blobs in it."""
        t0 = time.perf_counter()
        image = safe_get_image(self.camera)
        self.frames += 1
        if image is None:
            result = self._empty_result()
        else:
            frame = self.to_frame(image)
            if use_prefilter and not self.prefilter(frame):
                result = self._empty_result()   # nothing coloured -> skip full scan
            else:
                self.full_scans += 1
                result = self.full_scan(frame)
        self.time_total += time.perf_counter() - t0
        return result

    def stats_text(self):
        if self.frames == 0:
            return "no frames processed"
        skipped = 100.0 * (1 - self.full_scans / self.frames)
        avg_ms = 1000.0 * self.time_total / self.frames
        return (f"{self.frames} frames, {self.full_scans} full scans "
                f"({skipped:.1f}% skipped by pre-filter), "
                f"avg vision time {avg_ms:.2f} ms/frame ({VISION_BACKEND})")


def opencv_hsv_bounds():
    """Turn my HUE_RANGES (0-360 deg, S/V 0-1) into OpenCV inRange bounds.
    OpenCV stores hue as 0-179 (degrees / 2) and S, V as 0-255."""
    s_lo, v_lo = int(MIN_SATURATION * 255), int(MIN_VALUE * 255)
    bounds = {}
    for colour, ranges in HUE_RANGES.items():
        bounds[colour] = [((lo // 2, s_lo, v_lo), (min(179, hi // 2 - 1), 255, 255))
                          for lo, hi in ranges]
    return bounds
    # red   -> H 0-9 and 170-179, green -> H 45-74, blue -> H 100-129


class OpenCVVision(Vision):
    """Same interface as Vision, but uses OpenCV like in Session 9:
    BGRA bytes -> NumPy -> cvtColor(BGR2HSV) -> inRange() mask per colour."""

    def __init__(self, camera):
        super().__init__(camera)
        self.bounds = {c: [(np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in rs]
                       for c, rs in opencv_hsv_bounds().items()}

    def to_frame(self, image):
        bgra = np.frombuffer(image, np.uint8).reshape(self.h, self.w, 4)
        return cv.cvtColor(bgra, cv.COLOR_BGRA2BGR)

    def colour_mask(self, hsv, colour):
        """Binary mask (255 = pixel is this colour). Red needs two ranges."""
        mask = None
        for lo, hi in self.bounds[colour]:
            m = cv.inRange(hsv, lo, hi)
            mask = m if mask is None else cv.bitwise_or(mask, m)
        return mask

    def prefilter(self, bgr):
        """Every 4th pixel only (a 13 x 10 image for the 52 x 39 camera)."""
        small = np.ascontiguousarray(bgr[::PREFILTER_STRIDE, ::PREFILTER_STRIDE])
        hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
        hits = sum(cv.countNonZero(self.colour_mask(hsv, c)) for c in COLOURS)
        return hits >= PREFILTER_MIN_HITS

    def full_scan(self, bgr):
        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        result = {}
        for c in COLOURS:
            mask = self.colour_mask(hsv, c)
            m = cv.moments(mask, binaryImage=True)
            count = m["m00"]                       # number of pixels in the mask
            mean_x = m["m10"] / count if count else self.w / 2
            result[c] = (count / self.total, mean_x)
        return result

    def save_filtered(self, colour, folder):
        """Session 9 experiment: keep only the block's colour (bitwise AND with
        the mask), save it, and return the mean R, G, B of the kept pixels."""
        image = safe_get_image(self.camera)
        if image is None:
            return None
        bgr = self.to_frame(image)
        mask = self.colour_mask(cv.cvtColor(bgr, cv.COLOR_BGR2HSV), colour)
        filtered = cv.bitwise_and(bgr, bgr, mask=mask)
        cv.imwrite(os.path.join(folder, f"{colour}_block_filtered.png"), filtered)
        if cv.countNonZero(mask) == 0:
            return (0.0, 0.0, 0.0)
        b, g, r, _ = cv.mean(bgr, mask=mask)
        return (r, g, b)


# ----------------------------------------------------------------------------
# Odometry from the wheel encoders
# ----------------------------------------------------------------------------
class Odometry:
    def __init__(self, left_sensor, right_sensor, x=0.0, y=0.0, theta=0.0):
        self.ls = left_sensor
        self.rs = right_sensor
        self.x, self.y, self.theta = x, y, theta
        self.prev_l = None
        self.prev_r = None

    def update(self):
        l, r = self.ls.getValue(), self.rs.getValue()
        if math.isnan(l) or math.isnan(r):
            return
        if self.prev_l is None:
            self.prev_l, self.prev_r = l, r
            return
        dl = (l - self.prev_l) * WHEEL_RADIUS
        dr = (r - self.prev_r) * WHEEL_RADIUS
        self.prev_l, self.prev_r = l, r
        d = (dl + dr) / 2.0
        dth = (dr - dl) / AXLE_LENGTH
        self.x += d * math.cos(self.theta + dth / 2.0)
        self.y += d * math.sin(self.theta + dth / 2.0)
        self.theta = wrap_angle(self.theta + dth)


# ----------------------------------------------------------------------------
# Coverage grid for the planned search
# ----------------------------------------------------------------------------
class CoverageGrid:
    def __init__(self):
        span = 2 * (ARENA_HALF - GRID_MARGIN)
        step = span / (GRID_CELLS - 1)
        start = -(ARENA_HALF - GRID_MARGIN)
        self.cells = [(start + i * step, start + j * step)
                      for i in range(GRID_CELLS) for j in range(GRID_CELLS)]
        self.visited = set()
        self.radius = step / 2.0

    def mark_near(self, x, y):
        for idx, (cx, cy) in enumerate(self.cells):
            if math.hypot(cx - x, cy - y) < self.radius:
                self.visited.add(idx)

    def skip(self, idx):
        """Give up on a waypoint (e.g. a block is sitting on it)."""
        self.visited.add(idx)

    def next_target(self, x, y, theta):
        """Nearest unvisited waypoint, with a small penalty for big turns
        so the robot prefers cells roughly in front of it."""
        best, best_cost = None, float("inf")
        for idx, (cx, cy) in enumerate(self.cells):
            if idx in self.visited:
                continue
            dist = math.hypot(cx - x, cy - y)
            turn = abs(wrap_angle(math.atan2(cy - y, cx - x) - theta))
            cost = dist + 0.15 * turn
            if cost < best_cost:
                best, best_cost = idx, cost
        if best is None:                 # whole grid done -> start a new lap
            self.visited.clear()
            return self.next_target(x, y, theta)
        return best


# ----------------------------------------------------------------------------
# Robot setup
# ----------------------------------------------------------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for m in (left_motor, right_motor):
    m.setPosition(float("inf"))          # velocity control
    m.setVelocity(0.0)

left_enc = robot.getDevice("left wheel sensor")
right_enc = robot.getDevice("right wheel sensor")
left_enc.enable(timestep)
right_enc.enable(timestep)

ps = []
for i in range(8):
    sensor = robot.getDevice(f"ps{i}")
    sensor.enable(timestep)
    ps.append(sensor)

camera = robot.getDevice("camera")
camera.enable(timestep)

vision = OpenCVVision(camera) if VISION_BACKEND == "opencv" else Vision(camera)
odom = Odometry(left_enc, right_enc)
grid = CoverageGrid()

os.makedirs(CAPTURE_DIR, exist_ok=True)

found = {c: False for c in COLOURS}
found_order = []


def set_speed(left, right):
    left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left)))
    right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right)))


def still_to_find_message():
    left = [c for c in COLOURS if not found[c]]
    if not left:
        return "I've found all three blocks now!"
    if len(left) == 1:
        return f"I still need to find the {left[0]} block."
    return f"I still need to find the {' and '.join(left)} blocks."


def print_summary(t):
    seen = [c for c in COLOURS if found[c]]
    not_seen = [c for c in COLOURS if not found[c]]
    print("---------- Summary ----------")
    for c in COLOURS:
        print(f"  {c.capitalize():<6}: {'FOUND' if found[c] else 'not found yet'}")
    print(f"  Encountered so far ({len(seen)}/3): {', '.join(seen) if seen else 'none'}")
    print(f"  Order found: {' -> '.join(found_order)}")
    if not_seen:
        print(f"  Still searching for: {', '.join(not_seen)}")
    print(f"  Robot position (odometry): x={odom.x:+.2f} m, y={odom.y:+.2f} m")
    print("-----------------------------")


def save_snapshot(colour):
    """Save what the camera sees so it can be classified by the CNN later."""
    path = os.path.join(CAPTURE_DIR, f"{colour}_block_found.png")
    try:
        camera.saveImage(path, 100)
        print(f"  (camera frame saved to captures/{os.path.basename(path)})")
        if isinstance(vision, OpenCVVision):
            rgb = vision.save_filtered(colour, CAPTURE_DIR)
            if rgb:
                print(f"  (OpenCV filter: mean R={rgb[0]:.0f} G={rgb[1]:.0f} B={rgb[2]:.0f}, "
                      f"saved captures/{colour}_block_filtered.png)")
    except Exception as exc:              # don't crash the search over a file error
        print(f"  (could not save camera frame: {exc})")


def detect_new_blocks(blobs, t):
    """Check for first-time encounters. Returns True if a block was just found."""
    new_find = False
    for c in COLOURS:
        frac, _ = blobs[c]
        if not found[c] and frac >= FOUND_FRACTION:
            found[c] = True
            found_order.append(c)
            print(f"\n[t = {t:6.1f} s] I see {c}")
            print(still_to_find_message())
            print_summary(t)
            save_snapshot(c)
            new_find = True
    return new_find


def best_unfound_target(blobs):
    """The unfound colour that takes up the most of the image (if visible)."""
    target, best = None, 0.0
    for c in COLOURS:
        frac, _ = blobs[c]
        if not found[c] and frac > SEEN_FRACTION and frac > best:
            target, best = c, frac
    return target


def steer_towards_x(mean_x):
    """Proportional steering towards a blob centre (same idea as Part 1)."""
    error = (mean_x - vision.w / 2) / (vision.w / 2)     # -1 left .. +1 right
    set_speed(CRUISE_SPEED * (1 + 0.8 * error), CRUISE_SPEED * (1 - 0.8 * error))


def drive_to_point(tx, ty):
    """Turn on the spot if the heading error is big, otherwise drive and steer."""
    heading = math.atan2(ty - odom.y, tx - odom.x)
    err = wrap_angle(heading - odom.theta)
    if abs(err) > 0.5:
        turn = SCAN_SPEED * 1.6 * (1 if err > 0 else -1)
        set_speed(-turn, turn)
    else:
        k = 2.0 * err
        set_speed(CRUISE_SPEED - k, CRUISE_SPEED + k)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
state = "SCAN"
state_timer = 0
turn_dir = 1
scan_turned = 0.0                    # how far we've spun in the current scan
scan_prev_theta = 0.0
target_cell = None
stuck_ref = (0.0, 0.0, 0.0)          # (time, x, y) at the start of the window
unstick_count = 0

print(f"E-puck block search v2 started (HSV + pre-filter + planned search, vision: {VISION_BACKEND}).")
print("Looking for the red, blue and green blocks...")

while robot.step(timestep) != -1:
    t = robot.getTime()
    odom.update()
    grid.mark_near(odom.x, odom.y)
    ir = [s.getValue() for s in ps]

    # The pre-filter is skipped while approaching, because then I already
    # know a block is in view and need the exact blob size every step.
    blobs = vision.process(use_prefilter=USE_PREFILTER and state != "APPROACH")

    # 1) first-time encounters
    if detect_new_blocks(blobs, t):
        state = "BACK_OFF"
        state_timer = 25
        turn_dir = random.choice([-1, 1])

    if all(found.values()):
        set_speed(0, 0)
        print(f"\nAll three blocks found in {t:.1f} s of simulated time. Search complete!")
        print(f"Unstick manoeuvres used: {unstick_count}")
        print("Vision stats: " + vision.stats_text())
        # lets the test supervisor log the run (time + vision stats)
        robot.setCustomData(f"DONE {t:.1f} {vision.frames} {vision.full_scans} "
                            f"{1000.0 * vision.time_total / max(1, vision.frames):.3f} {unstick_count}")
        break

    front_left = max(ir[7], ir[6])
    front_right = max(ir[0], ir[1])
    obstacle = max(ir[0], ir[1], ir[6], ir[7]) > OBSTACLE_THRESHOLD

    # 2) stuck detector (only for states that are meant to move forwards)
    if state in ("GOTO_CELL", "APPROACH"):
        t0, x0, y0 = stuck_ref
        if t - t0 > STUCK_WINDOW_S:
            if math.hypot(odom.x - x0, odom.y - y0) < STUCK_MIN_DIST:
                print(f"[t = {t:6.1f} s] Looks like I'm stuck, reversing.")
                unstick_count += 1
                state = "UNSTICK"
                state_timer = 30
                turn_dir = random.choice([-1, 1])
                if target_cell is not None:
                    grid.skip(target_cell)
                    target_cell = None
            stuck_ref = (t, odom.x, odom.y)
    else:
        stuck_ref = (t, odom.x, odom.y)

    # 3) timed manoeuvres
    if state in ("BACK_OFF", "UNSTICK"):
        if state_timer > 15:
            set_speed(-CRUISE_SPEED, -CRUISE_SPEED)                       # reverse
        else:
            set_speed(-turn_dir * CRUISE_SPEED, turn_dir * CRUISE_SPEED)  # spin
        state_timer -= 1
        if state_timer <= 0:
            state = "SCAN"
            scan_turned, scan_prev_theta = 0.0, odom.theta
        continue

    if state == "AVOID":
        set_speed(-turn_dir * CRUISE_SPEED, turn_dir * CRUISE_SPEED)
        state_timer -= 1
        if state_timer <= 0:
            state = "GOTO_CELL" if target_cell is not None else "SCAN"
            scan_turned, scan_prev_theta = 0.0, odom.theta
        continue

    # 4) an unfound block is in view -> go for it
    target = best_unfound_target(blobs)
    if target is not None:
        if state != "APPROACH":
            print(f"[t = {t:6.1f} s] Spotted something {target}, heading over.")
        state = "APPROACH"
        steer_towards_x(blobs[target][1])
        continue
    if state == "APPROACH":           # lost sight of it -> look around again
        state = "SCAN"
        scan_turned, scan_prev_theta = 0.0, odom.theta

    # 5) obstacle avoidance (walls and blocks that are already found)
    if obstacle:
        state = "AVOID"
        turn_dir = 1 if front_right > front_left else -1
        state_timer = random.randint(8, 20)
        if target_cell is not None:
            cx, cy = grid.cells[target_cell]
            # if the obstacle is right where the waypoint is, give up on it
            if math.hypot(cx - odom.x, cy - odom.y) < 0.25:
                grid.skip(target_cell)
                target_cell = None
        continue

    # 6) SCAN: spin slowly on the spot for one full turn
    if state == "SCAN":
        set_speed(-SCAN_SPEED, SCAN_SPEED)
        scan_turned += abs(wrap_angle(odom.theta - scan_prev_theta))
        scan_prev_theta = odom.theta
        if scan_turned >= 2 * math.pi:
            target_cell = grid.next_target(odom.x, odom.y, odom.theta)
            cx, cy = grid.cells[target_cell]
            print(f"[t = {t:6.1f} s] Nothing new in view, moving to waypoint "
                  f"({cx:+.2f}, {cy:+.2f}).")
            state = "GOTO_CELL"
        continue

    # 7) GOTO_CELL: drive to the chosen waypoint, then scan there
    if state == "GOTO_CELL":
        if target_cell is None:
            state = "SCAN"
            scan_turned, scan_prev_theta = 0.0, odom.theta
            continue
        cx, cy = grid.cells[target_cell]
        if math.hypot(cx - odom.x, cy - odom.y) < WAYPOINT_TOLERANCE:
            grid.skip(target_cell)
            target_cell = None
            state = "SCAN"
            scan_turned, scan_prev_theta = 0.0, odom.theta
            continue
        drive_to_point(cx, cy)
        continue
