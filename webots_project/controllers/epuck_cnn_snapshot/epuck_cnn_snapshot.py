"""
epuck_cnn_snapshot.py
ARAIP - Project Part 2 (CNN camera test)
Author: Arul MithunKumar (408AIAUM)

Simple controller for cnn_object_test.wbt. The e-puck faces each object in
turn (dog in front, cat behind), creeps towards it and saves camera frames
at a few distances. The PNGs in captures/ are then uploaded to Google Colab
and classified by my CIFAR-10 CNN (cnn/ARAIP_CIFAR10_CNN.ipynb).

If TensorFlow happens to be installed for the Python that Webots uses, and
the trained model file cnn/cifar10_cnn.tflite exists, the robot also runs the
CNN on-board and prints the prediction. Otherwise it just saves the images,
the search robot never depends on it.
"""

import math
import os

from controller import Robot

MAX_SPEED = 6.28
CREEP_SPEED = 0.3 * MAX_SPEED
TURN_SPEED = 0.3 * MAX_SPEED
WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.052
STOP_IR = 80.0                       # stop before touching the object
SHOT_EVERY_M = 0.15                  # take a picture every 15 cm travelled
MAX_SHOTS = 3

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.join(HERE, "captures")
MODEL_PATH = os.path.join(HERE, "..", "..", "..", "cnn", "cifar10_cnn.tflite")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

robot = Robot()
timestep = int(robot.getBasicTimeStep())
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for m in (left_motor, right_motor):
    m.setPosition(float("inf"))
    m.setVelocity(0.0)
left_enc = robot.getDevice("left wheel sensor")
right_enc = robot.getDevice("right wheel sensor")
left_enc.enable(timestep)
right_enc.enable(timestep)
ps = [robot.getDevice(f"ps{i}") for i in range(8)]
for s in ps:
    s.enable(timestep)
camera = robot.getDevice("camera")
camera.enable(timestep)
os.makedirs(CAPTURE_DIR, exist_ok=True)


def load_onboard_model():
    """Optional on-robot inference with TensorFlow Lite."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import numpy as np            # noqa: F401
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=MODEL_PATH)
        interp.allocate_tensors()
        print("On-board CNN loaded (TFLite).")
        return interp
    except Exception as exc:
        print(f"On-board CNN not available ({exc.__class__.__name__}), "
              "images will be classified in Colab instead.")
        return None


def classify_onboard(interp):
    import numpy as np
    w, h = camera.getWidth(), camera.getHeight()
    img = np.frombuffer(camera.getImage(), np.uint8).reshape(h, w, 4)[:, :, 2::-1]
    # nearest-neighbour resize to 32x32 without needing OpenCV
    ys = (np.arange(32) * h / 32).astype(int)
    xs = (np.arange(32) * w / 32).astype(int)
    small = img[ys][:, xs].astype(np.float32) / 255.0
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], small[None, ...])
    interp.invoke()
    probs = interp.get_tensor(out["index"])[0]
    k = int(np.argmax(probs))
    return CLASSES[k], float(probs[k])


def wheel_distance():
    return WHEEL_RADIUS * (left_enc.getValue() + right_enc.getValue()) / 2.0


def wheel_heading():
    return WHEEL_RADIUS * (right_enc.getValue() - left_enc.getValue()) / AXLE_LENGTH


model = load_onboard_model()
targets = ["dog", "cat"]             # dog is in front (+x), cat is behind (-x)
target_i = 0
phase = "APPROACH"
shots = 0
last_shot_d = None
turn_start = None

robot.step(timestep)                 # let the encoders give their first value
print("CNN camera test started.")

while robot.step(timestep) != -1:
    name = targets[target_i]
    ir_front = max(ps[0].getValue(), ps[7].getValue())
    d = wheel_distance()

    if phase == "APPROACH":
        if last_shot_d is None or d - last_shot_d >= SHOT_EVERY_M or ir_front > STOP_IR:
            shots += 1
            last_shot_d = d
            path = os.path.join(CAPTURE_DIR, f"{name}_shot{shots}.png")
            camera.saveImage(path, 100)
            msg = f"Saved {os.path.basename(path)}"
            if model is not None:
                label, conf = classify_onboard(model)
                msg += f" -> CNN says {label} ({conf:.0%})"
            print(msg)
        if shots >= MAX_SHOTS or ir_front > STOP_IR:
            phase = "TURN"
            turn_start = wheel_heading()
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)
            continue
        left_motor.setVelocity(CREEP_SPEED)
        right_motor.setVelocity(CREEP_SPEED)

    elif phase == "TURN":
        target_i += 1
        if target_i >= len(targets):
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)
            print("All snapshots taken. Upload the captures folder to Colab.")
            break
        # spin 180 degrees to face the next object
        if abs(wheel_heading() - turn_start) < math.pi:
            target_i -= 1            # keep turning, same step logic next time
            left_motor.setVelocity(-TURN_SPEED)
            right_motor.setVelocity(TURN_SPEED)
            continue
        phase = "APPROACH"
        shots = 0
        last_shot_d = None
        # drive back past the centre before shooting the second object
        print("Turned round, heading for the next object.")
