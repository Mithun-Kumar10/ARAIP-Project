"""
epuck_block_search.py
ARAIP - Assignment 2 (Project Part 1)

Controller for the e-puck. The robot wanders round the arena, uses its
8 IR sensors to avoid the walls, and uses its camera to look for the red,
blue and green blocks. The first time it finds each block it says
"I see red", "I see blue" or "I see green" (as the brief asks), then tells
you which blocks it still needs to find and prints a summary of everything
found so far.
"""

import random
from controller import Robot

# ---------------- settings ----------------
MAX_SPEED = 6.28              # e-puck wheel limit (rad/s)
CRUISE_SPEED = 0.6 * MAX_SPEED
OBSTACLE_THRESHOLD = 80.0     # IR reading that means something is close

MIN_PIXEL_VALUE = 60          # colour channel has to be at least this bright
DOMINANCE_RATIO = 1.6         # and 1.6x bigger than the other two channels
SEEN_FRACTION = 0.01          # more than 1% of the image -> colour is "visible"
FOUND_FRACTION = 0.15         # 15% or more of the image -> block is "found"

COLOURS = ["red", "blue", "green"]

# ---------------- devices ----------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for m in (left_motor, right_motor):
    m.setPosition(float("inf"))   # velocity control
    m.setVelocity(0.0)

ps = []
for i in range(8):
    sensor = robot.getDevice(f"ps{i}")
    sensor.enable(timestep)
    ps.append(sensor)

camera = robot.getDevice("camera")
camera.enable(timestep)
cam_w = camera.getWidth()
cam_h = camera.getHeight()
total_pixels = cam_w * cam_h

# ---------------- what has been found ----------------
found = {c: False for c in COLOURS}
found_order = []


def classify_image():
    """Count the red, green and blue pixels in the camera image.
    Returns {colour: (fraction_of_image, average_x_position)}"""
    image = camera.getImage()        # raw BGRA bytes
    counts = {c: 0 for c in COLOURS}
    x_sum = {c: 0 for c in COLOURS}
    for y in range(cam_h):
        row = y * cam_w * 4
        for x in range(cam_w):
            i = row + x * 4
            b, g, r = image[i], image[i + 1], image[i + 2]
            if r > MIN_PIXEL_VALUE and r > DOMINANCE_RATIO * max(g, b):
                colour = "red"
            elif g > MIN_PIXEL_VALUE and g > DOMINANCE_RATIO * max(r, b):
                colour = "green"
            elif b > MIN_PIXEL_VALUE and b > DOMINANCE_RATIO * max(r, g):
                colour = "blue"
            else:
                continue
            counts[colour] += 1
            x_sum[colour] += x
    result = {}
    for c in COLOURS:
        frac = counts[c] / total_pixels
        mean_x = x_sum[c] / counts[c] if counts[c] else cam_w / 2
        result[c] = (frac, mean_x)
    return result


def still_to_find_message():
    """Makes the 'I still need to find...' line."""
    left = [c for c in COLOURS if not found[c]]
    if not left:
        return "I've found all three blocks now!"
    if len(left) == 1:
        return f"I still need to find the {left[0]} block."
    return f"I still need to find the {' and '.join(left)} blocks."


def print_summary():
    """Summary of which blocks have been encountered so far."""
    seen = [c for c in COLOURS if found[c]]
    not_seen = [c for c in COLOURS if not found[c]]
    print("---------- Summary ----------")
    for c in COLOURS:
        status = "FOUND" if found[c] else "not found yet"
        print(f"  {c.capitalize():<6}: {status}")
    print(f"  Encountered so far ({len(seen)}/3): "
          f"{', '.join(seen) if seen else 'none'}")
    print(f"  Order found: {' -> '.join(found_order)}")
    if not_seen:
        print(f"  Still searching for: {', '.join(not_seen)}")
    print("-----------------------------")


def set_speed(left, right):
    left_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left)))
    right_motor.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right)))


# ---------------- behaviour state ----------------
state = "WANDER"
state_timer = 0                      # steps left in a timed move
turn_dir = 1                         # +1 = turn left, -1 = turn right
wander_timer = random.randint(60, 150)
wander_bias = 0.0

print("E-puck block search started. Looking for the red, blue and green blocks...")

while robot.step(timestep) != -1:
    t = robot.getTime()
    ir = [s.getValue() for s in ps]
    vision = classify_image()

    # 1) has a new block just been found? (first encounter only)
    for c in COLOURS:
        frac, _ = vision[c]
        if not found[c] and frac >= FOUND_FRACTION:
            found[c] = True
            found_order.append(c)
            print(f"\n[t = {t:6.1f} s] I see {c}")
            print(still_to_find_message())
            print_summary()
            # back off and turn so it doesn't sit on the same block
            state = "BACK_OFF"
            state_timer = 25
            turn_dir = random.choice([-1, 1])

    if all(found.values()):
        set_speed(0, 0)
        print(f"\nAll three blocks found in {t:.1f} s of simulated time. Search complete!")
        robot.setCustomData(f"DONE {t:.1f}")   # Part 2: lets the test supervisor log the run
        break

    front_left = max(ir[7], ir[6])
    front_right = max(ir[0], ir[1])
    obstacle = max(ir[0], ir[1], ir[6], ir[7]) > OBSTACLE_THRESHOLD

    # 2) timed moves
    if state == "BACK_OFF":
        if state_timer > 15:
            set_speed(-CRUISE_SPEED, -CRUISE_SPEED)                       # reverse
        else:
            set_speed(-turn_dir * CRUISE_SPEED, turn_dir * CRUISE_SPEED)  # spin
        state_timer -= 1
        if state_timer <= 0:
            state = "WANDER"
        continue

    if state == "AVOID":
        set_speed(-turn_dir * CRUISE_SPEED, turn_dir * CRUISE_SPEED)
        state_timer -= 1
        if state_timer <= 0:
            state = "WANDER"
        continue

    # 3) if a block we HAVEN'T found yet is in view, drive towards it
    target = None
    best = 0.0
    for c in COLOURS:
        frac, mean_x = vision[c]
        if not found[c] and frac > SEEN_FRACTION and frac > best:
            target, best = c, frac
    if target is not None:
        state = "APPROACH"
        _, mean_x = vision[target]
        error = (mean_x - cam_w / 2) / (cam_w / 2)    # -1 (left) .. +1 (right)
        set_speed(CRUISE_SPEED * (1 + 0.8 * error),
                  CRUISE_SPEED * (1 - 0.8 * error))
        continue

    # 4) obstacle avoidance (walls, or blocks that are already found)
    if obstacle:
        state = "AVOID"
        turn_dir = 1 if front_right > front_left else -1   # turn away
        state_timer = random.randint(8, 25)
        continue

    # 5) random wander: forward with a slight random curve, and every so
    #    often a random turn so the robot covers more of the arena
    state = "WANDER"
    wander_timer -= 1
    if wander_timer <= 0:
        wander_timer = random.randint(60, 150)
        wander_bias = random.uniform(-0.3, 0.3)
        if random.random() < 0.4:
            state = "AVOID"
            turn_dir = random.choice([-1, 1])
            state_timer = random.randint(10, 40)
            continue
    set_speed(CRUISE_SPEED * (1 - wander_bias), CRUISE_SPEED * (1 + wander_bias))
