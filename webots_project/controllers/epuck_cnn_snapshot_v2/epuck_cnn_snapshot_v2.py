"""
epuck_cnn_snapshot_v2.py
ARAIP - Project Part 2 (CNN camera test, version 2)
Author: Arul MithunKumar (408AIAUM)

Used with cnn_object_test_v2.wbt. The world has a plain background and three
realistic models of the same classes as my own photos (bird, cat, automobile),
placed around the e-puck at different headings. For each object the robot:
  1. turns on the spot to face it (heading from the wheel encoders),
  2. takes a photo, creeps forward 15 cm, takes another, and so on (3 photos),
  3. reverses back to the middle and moves on to the next object.
The frames are saved to captures_v2/ and classified by my CNN in Colab.
"""

import math
import os

from controller import Robot

MAX_SPEED = 6.28
TURN_SPEED = 0.25 * MAX_SPEED
DRIVE_SPEED = 0.3 * MAX_SPEED
WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.052
SHOT_STEP_M = 0.15            # distance between photos
SHOTS = 3
HEADING_TOLERANCE = 0.02      # rad

# (label, heading in rad) - must match the object positions in the world file
TARGETS = [("bird", 0.0), ("cat", 2.0944), ("automobile", 4.1888)]

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.join(HERE, "captures_v2")
os.makedirs(CAPTURE_DIR, exist_ok=True)

robot = Robot()
ts = int(robot.getBasicTimeStep())
lm, rm = robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")
for m in (lm, rm):
    m.setPosition(float("inf"))
    m.setVelocity(0.0)
le, re_ = robot.getDevice("left wheel sensor"), robot.getDevice("right wheel sensor")
le.enable(ts)
re_.enable(ts)
camera = robot.getDevice("camera")
camera.enable(ts)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def heading():
    return WHEEL_RADIUS * (re_.getValue() - le.getValue()) / AXLE_LENGTH


def travelled():
    return WHEEL_RADIUS * (le.getValue() + re_.getValue()) / 2.0


def step():
    if robot.step(ts) == -1:
        raise SystemExit


def stop(n=10):
    lm.setVelocity(0)
    rm.setVelocity(0)
    for _ in range(n):         # let the robot settle so the photo isn't blurred
        step()


def turn_to(target):
    while True:
        err = wrap(target - heading())
        if abs(err) < HEADING_TOLERANCE:
            break
        v = max(0.15 * MAX_SPEED, min(TURN_SPEED, abs(err) * 4)) * (1 if err > 0 else -1)
        lm.setVelocity(-v)
        rm.setVelocity(v)
        step()
    stop()


def object_centre_x():
    """Mean x of pixels that don't look like the plain background (beige floor,
    light walls/sky). Returns None if the object isn't in view."""
    import numpy as np
    w, h = camera.getWidth(), camera.getHeight()
    img = np.frombuffer(camera.getImage(), np.uint8).reshape(h, w, 4)[:, :, :3].astype(int)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    mx, mn = img.max(axis=2), img.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    # background is bright and nearly grey; the objects are darker or more colourful
    obj = (mx < 150) | (sat > 0.30)
    cols = np.where(obj.sum(axis=0) > 3)[0]
    if len(cols) < 5:
        return None
    return float(cols.mean()), w


def centre_on_object(max_steps=200):
    """Odometry turns drift a little, so finish the turn by centring the object
    in the image (a very simple region-of-interest step)."""
    for _ in range(max_steps):
        res = object_centre_x()
        if res is None:
            break
        cx, w = res
        err = (cx - w / 2) / (w / 2)          # -1 left .. +1 right
        if abs(err) < 0.06:
            break
        v = 0.08 * MAX_SPEED * (1 if err < 0 else -1)
        lm.setVelocity(-v)
        rm.setVelocity(v)
        step()
    stop()


def drive(dist):
    start = travelled()
    sign = 1 if dist > 0 else -1
    while abs(travelled() - start) < abs(dist):
        lm.setVelocity(sign * DRIVE_SPEED)
        rm.setVelocity(sign * DRIVE_SPEED)
        step()
    stop()


step()
print("CNN camera test v2 started.")
for label, hdg in TARGETS:
    turn_to(hdg)
    for k in range(1, SHOTS + 1):
        centre_on_object()
        path = os.path.join(CAPTURE_DIR, f"{label}_v2_shot{k}.png")
        camera.saveImage(path, 100)
        print(f"Saved {os.path.basename(path)}")
        if k < SHOTS:
            drive(SHOT_STEP_M)
    drive(-SHOT_STEP_M * (SHOTS - 1))     # back to the middle
print("All snapshots taken.")
stop()
