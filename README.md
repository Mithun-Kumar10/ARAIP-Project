# ARAIP Project: Seek, Detect, Classify

**Applied Robotics and AI Project (5FTC2068)**, University of Hertfordshire, BEng (Hons) Robotics and AI
Arul MithunKumar · 408AIAUM · Lecturer: Ms Sarina Tajudin

This repo has all my code for the ARAIP module:

| Part | What it is | Folder |
|---|---|---|
| Assignment 1 | Smart bin lid simulation (Git/Trello practice) | `src/`, `demo/` |
| Project Part 1 (Assignment 2) | E-puck searches a Webots arena for red, blue and green blocks | `webots_project/controllers/epuck_block_search/` |
| Project Part 2 (Assignments 3-5) | Upgraded search controller, CIFAR-10 CNN in Google Colab, CNN test on the e-puck camera, optimisation study | `webots_project/`, `cnn/` |

Trello board: https://trello.com/b/rmcSxhfL/araip-project

---

## 1. Project Part 2 in short

1. **Better block search (Webots, Python).** `epuck_block_search_v2` replaces my Part 1 random walk with a *scan → approach → back off → scan* loop. When nothing is in view it drives to the nearest unvisited waypoint on a 4×4 coverage grid (position from wheel odometry). Colour detection is now in HSV, and a **camera pre-filter** only runs the full-image scan when a sparse sample of pixels spots a colour.
2. **CNN for object classification (Google Colab).** `cnn/ARAIP_CIFAR10_CNN.ipynb` trains a VGG-style CNN on CIFAR-10, shows every layer, the accuracy, predictions on three of my own photos and on frames from the e-puck camera.
3. **Optimisation for the robot.** The same notebook compares TFLite float32, dynamic-range int8, full int8 and pruned models (accuracy, size, CPU latency), and the controller's pre-filter shows how to only run the CNN when it's needed.

### Batch test results (10 random block layouts each, same seeds)

| Controller | Success | Mean time to find all 3 | Worst run |
|---|---|---|---|
| Part 1 (`epuck_block_search`, random wander, RGB) | 10/10 | 21.7 s | 52.1 s |
| Part 2 without pre-filter | 10/10 | 6.8 s | 12.3 s |
| **Part 2 (`epuck_block_search_v2`)** | **10/10** | **6.7 s** | **10.9 s** |

The pre-filter skipped the full scan on 37% of frames and cut average vision time from 1.07 ms to 0.78 ms per frame. Raw data: `webots_project/results/runs.csv`.

---

## 2. Folder structure

```
ARAIP-Project/
├── webots_project/
│   ├── worlds/
│   │   ├── block_search_part2.wbt      # 2x2 m arena, 3 blocks, e-puck, experiment manager
│   │   └── cnn_object_test.wbt         # dog + cat (CIFAR-10 classes) for the CNN camera test
│   ├── controllers/
│   │   ├── epuck_block_search/         # Part 1 controller (kept as the baseline)
│   │   ├── epuck_block_search_v2/      # Part 2 controller (HSV, pre-filter, planned search)
│   │   ├── experiment_manager/         # supervisor: random layouts + logging (testing only)
│   │   └── epuck_cnn_snapshot/         # drives up to the dog/cat and saves camera frames
│   ├── results/runs.csv                # batch test results
│   └── run_experiments.ps1             # runs the 30 batch tests headless
├── cnn/
│   ├── ARAIP_CIFAR10_CNN.ipynb         # Google Colab notebook
│   ├── my_images/                      # my own 3 photos (file name = true label)
│   └── webots_captures/                # frames saved by the e-puck camera
├── src/, demo/, hello.txt              # Assignment 1
└── README.md
```

---

## 3. Setup

### Webots
1. Install **Webots R2023b** (https://cyberbotics.com) and Python 3 (I used 3.14). In Webots: *Tools → Preferences → Python command* = `python`.
2. Open `webots_project/worlds/block_search_part2.wbt`. The PROTOs are downloaded from GitHub the first time, so you need internet once.
3. Press play. The console shows the robot's messages (`I see red`, what's still missing and a summary). Frames are saved in `controllers/epuck_block_search_v2/captures/`.
4. To reload for a new random layout use **Ctrl+Shift+R** (reload world). I found Reset sometimes froze the console in Part 1.

Optional environment variables (read by the supervisor/controller):

| Variable | Meaning |
|---|---|
| `ARAIP_SEED` | fixed seed for the block layout (otherwise random every run) |
| `ARAIP_PREFILTER=0` | turn the camera pre-filter off (for comparison) |
| `ARAIP_BATCH=1` | quit Webots when the run ends (used by `run_experiments.ps1`) |

Batch tests: `powershell -ExecutionPolicy Bypass -File webots_project/run_experiments.ps1 -Runs 10`

### World / block setup
- Arena: `RectangleArena` 2 m × 2 m, **black floor** (`baseColor 0 0 0`, roughness 1). The chequered floor's brown tiles were detected as red in Part 1.
- Blocks: three `SolidBox` nodes (`DEF RED_BLOCK`, `BLUE_BLOCK`, `GREEN_BLOCK`), **size 0.3 × 0.3 × 0.1 m**, pure colours, roughness 1 (no specular highlights).
- **Random positions:** the `experiment_manager` supervisor moves the blocks at the start of every run (random x, y and yaw). It keeps them inside the walls, at least 0.15 m apart, and out of a 0.35 m circle around the robot's start point. It doesn't help the robot search, it only sets up the layout and logs the result.

### Google Colab
1. Open `cnn/ARAIP_CIFAR10_CNN.ipynb` in Colab (File → Open notebook → GitHub → `Mithun-Kumar10/ARAIP-Project`).
2. Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all (about 15 minutes).
3. The notebook clones this repo to read `cnn/my_images`, `cnn/webots_captures` and `results/runs.csv`, and downloads a zip of all the figures plus `cifar10_cnn.tflite` at the end.

---

## 4. Hardware notes (e-puck in Webots)

| Device | Webots name | How I use it |
|---|---|---|
| 2 × DC wheel motors | `left wheel motor`, `right wheel motor` | velocity control, max 6.28 rad/s, cruise at 60% |
| 2 × wheel encoders | `left wheel sensor`, `right wheel sensor` | odometry (wheel radius 0.0205 m, axle 0.052 m) |
| 8 × IR proximity | `ps0`–`ps7` | obstacle avoidance; `ps0/ps1` front-right, `ps6/ps7` front-left, threshold 80 |
| Colour camera | `camera` | 52 × 39 px, 0.84 rad FOV in the search world; 160 × 120 in the CNN test world |

The real e-puck has a dsPIC30F6014A microcontroller (64 MHz, 8 KB RAM), so a CNN can't run on the robot itself. My proposal is to run it on a companion computer (e.g. a Raspberry Pi extension board or a laptop over Wi-Fi/Bluetooth) using the int8 TFLite model, and only when the pre-filter says something is in view.

---

## 5. How the v2 controller works

```
           ┌──────── nothing seen after 360° ────────┐
           ▼                                          │
  ┌─────► SCAN ──── unfound colour >1% ───► APPROACH  │
  │        ▲                                   │      │
  │        │ reached waypoint                  │ ≥15% of image
  │    GOTO_CELL ◄─────────────────────────────┼──────┘
  │        │                                   ▼
  │        │ IR > 80            "I see <colour>" + summary + save frame
  │        ▼                                   │
  └──── AVOID / UNSTICK ◄── stuck detector     ▼
                                           BACK_OFF ──► SCAN
```

- `Vision.prefilter()` samples every 4th pixel in x and y (~1/16 of the image). The full HSV scan only runs if it finds a coloured pixel, or while approaching.
- `classify_pixel()` uses HSV: saturation ≥ 0.55 and value ≥ 0.20 removes the black floor, grey walls and the sky. Hue ranges: red 0–20° and 340–360°, green 90–150°, blue 200–260°.
- `CoverageGrid.next_target()` picks the nearest unvisited waypoint, with a small penalty for big turns.
- The stuck detector reverses the robot if it moved less than 2 cm in 2.5 s while it was meant to be driving.

---

## 6. Git workflow

Same as Assignment 1: `main` stays working, every piece of work goes on its own branch (`feature/...`, `docs/...`, `fix/...`) and is merged through a pull request. Each Trello card names its branch.
