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
2. **CNN for object classification (Google Colab).** `cnn/ARAIP_CIFAR10_CNN.ipynb` first trains the Session 8 class model (`cnn_v3`) as a baseline, then my improved VGG-style CNN on CIFAR-10. It shows every layer, the accuracy, predictions on three of my own photos and on frames from the e-puck camera.
3. **Optimisation for the robot.** The notebook has a section that compares TFLite float32, dynamic-range int8, full int8 and pruned models (accuracy, size, CPU latency), and the controller's pre-filter shows how to only run the CNN when it's needed. In the 18 Nov run this section crashed (TensorFlow `InvalidArgumentError`, caused by timing the Keras model on the CPU while its weights were on the GPU). It is fixed and was re-run on 21 Nov (see below).

### Batch test results (10 random block layouts each, same seeds)

| Controller | Success | Mean time to find all 3 | Worst run | Vision time per frame |
|---|---|---|---|---|
| Part 1 (`epuck_block_search`, random wander, RGB) | 10/10 | 31.7 s | 52.8 s | not measured |
| Part 2, pure Python HSV, no pre-filter | 10/10 | 6.7 s | 10.9 s | 1.03 ms |
| Part 2, pure Python HSV + pre-filter | 10/10 | 7.1 s | 10.0 s | 0.79 ms |
| **Part 2, OpenCV HSV + pre-filter (`epuck_block_search_v2`)** | **10/10** | **6.9 s** | **12.3 s** | **0.15 ms** |

The planned search makes the robot about 4.5× faster than my Part 1 random wander (its random turns are unseeded, so its times vary a lot between batches). The pre-filter skipped the full scan on about 30% of frames, and moving to OpenCV (Session 9) cut vision time per frame by about 7× compared with my Python loop. Raw data: `webots_project/results/runs.csv`.

### CNN results (Colab T4 GPU run, 18 Nov 2026)

| Test | Result |
|---|---|
| My CNN, CIFAR-10 test set (10,000 images) | **85.90%** (top-3 97.47%), 306,154 parameters, 60 epochs, 8.8 min |
| Session 8 class model (`cnn_v3`), same run and test set | 67.3%, 122,570 parameters (train/val gap 12.5 points) |
| Most common mistakes | dog → cat 119, cat → dog 101, cat → frog 77 |
| My own 3 photos | 1/3: car ✓ (100%), bird → ship (39%), cat → airplane (87%) |
| E-puck camera, first world (sky/sea background) | 0/6, still 0/6 after cropping to the animal |
| E-puck camera, plain-background world (Sketchfab bird, cat, car) | **7/9** at 160×120, 8/9 at the real 52×39 resolution |

The first robot test failed because the CNN was partly classifying the sky and sea background (CIFAR-10 airplanes and ships have the same backgrounds), so I built a second world with a plain background and models that look like my photos. Figures and the raw numbers are in `cnn/results_2026-11-18/`.

### Final run (Colab T4 GPU, 21 Nov 2026), used in my final report

| Test | Result |
|---|---|
| My CNN / Session 8 `cnn_v3` | 84.69% (top-3 97.45%) / 67.9% |
| My own 3 photos | 2/3 (bird right at 54%, cat → airplane 79%) |
| E-puck camera: world 1 / world 2 | 0/6 / **9/9** at 160×120 (8/9 at 52×39) |

| Model | Accuracy | Size | gzip | CPU ms / image |
|---|---|---|---|---|
| Keras float32 | 84.69% | 3698 KB | 3282 KB | 25.62 |
| TFLite float32 | 84.69% | 1198 KB | 1107 KB | 1.09 |
| TFLite dynamic-range int8 | 84.71% | 316 KB | 283 KB | 1.20 |
| TFLite full int8 | 84.74% | 321 KB | 282 KB | 1.26 |
| Pruned 50% + dynamic int8 | 82.64% | 316 KB | 203 KB | 1.19 |
| Pruned 75% + dynamic int8 | 73.46% | 316 KB | 126 KB | 1.19 |

Quantisation made the model ~4× smaller with no accuracy loss but wasn't faster on Colab's x86 CPU; moving from a Keras call to the TFLite interpreter gave the big speed-up. Figures and numbers: `cnn/results_2026-11-21/`.

---

## 2. Folder structure

```
ARAIP-Project/
├── webots_project/
│   ├── worlds/
│   │   ├── block_search_part2.wbt      # 2x2 m arena, 3 blocks, e-puck, experiment manager
│   │   ├── cnn_object_test.wbt         # dog + cat (CIFAR-10 classes), sky/sea background
│   │   └── cnn_object_test_v2.wbt      # plain background, Sketchfab bird, cat and car
│   ├── models/                         # converted Sketchfab models (CC BY 4.0, see CREDITS.md)
│   ├── controllers/
│   │   ├── epuck_block_search/         # Part 1 controller (kept as the baseline)
│   │   ├── epuck_block_search_v2/      # Part 2 controller (HSV, pre-filter, planned search)
│   │   ├── experiment_manager/         # supervisor: random layouts + logging (testing only)
│   │   ├── epuck_cnn_snapshot/         # drives up to the dog/cat and saves camera frames
│   │   └── epuck_cnn_snapshot_v2/      # turns to each model, centres it in view, takes 3 shots
│   ├── results/runs.csv                # batch test results
│   └── run_experiments.ps1             # runs the 30 batch tests headless
├── cnn/
│   ├── ARAIP_CIFAR10_CNN.ipynb         # Google Colab notebook
│   ├── my_images/                      # my own 3 photos (file name = true label)
│   ├── webots_captures/                # e-puck frames from the first world
│   ├── webots_captures_v2/             # e-puck frames from the plain-background world
│   └── results_2026-11-18/             # figures + numbers from the T4 run
├── src/, demo/, hello.txt              # Assignment 1
└── README.md
```

---

## 3. Setup

### Webots
1. Install **Webots R2023b** (https://cyberbotics.com) and Python 3 (I used 3.14). In Webots: *Tools → Preferences → Python command* = `python`.
   Then install OpenCV, as in Session 9: `pip install opencv-python numpy`. This is optional: without it the controller falls back to my pure Python colour detection.
2. Open `webots_project/worlds/block_search_part2.wbt`. The PROTOs are downloaded from GitHub the first time, so you need internet once.
3. Press play. The console shows the robot's messages (`I see red`, what's still missing and a summary). Frames are saved in `controllers/epuck_block_search_v2/captures/`.
4. To reload for a new random layout use **Ctrl+Shift+R** (reload world). I found Reset sometimes froze the console in Part 1.

Optional environment variables (read by the supervisor/controller):

| Variable | Meaning |
|---|---|
| `ARAIP_SEED` | fixed seed for the block layout (otherwise random every run) |
| `ARAIP_PREFILTER=0` | turn the camera pre-filter off (for comparison) |
| `ARAIP_VISION=python` | use my pure Python HSV loop instead of OpenCV (for comparison) |
| `ARAIP_BATCH=1` | quit Webots when the run ends (used by `run_experiments.ps1`) |

Batch tests: `powershell -ExecutionPolicy Bypass -File webots_project/run_experiments.ps1 -Runs 10`

### World / block setup
- Arena: `RectangleArena` 2 m × 2 m, **black floor** (`baseColor 0 0 0`, roughness 1). The chequered floor's brown tiles were detected as red in Part 1.
- Blocks: three `SolidBox` nodes (`DEF RED_BLOCK`, `BLUE_BLOCK`, `GREEN_BLOCK`), **size 0.3 × 0.3 × 0.1 m**, pure colours, roughness 1 (no specular highlights).
- **Random positions:** the `experiment_manager` supervisor moves the blocks at the start of every run (random x, y and yaw). It keeps them inside the walls, at least 0.15 m apart, and out of a 0.35 m circle around the robot's start point. It doesn't help the robot search, it only sets up the layout and logs the result.

### Google Colab
1. Open `cnn/ARAIP_CIFAR10_CNN.ipynb` in Colab (File → Open notebook → GitHub → `Mithun-Kumar10/ARAIP-Project`).
2. Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all (about 15 minutes).
3. The notebook clones this repo to read `cnn/my_images`, `cnn/webots_captures`, `cnn/webots_captures_v2` and `webots_project/results/runs.csv`, and downloads a zip of all the figures plus `cifar10_cnn.tflite` at the end. Keep the tab active while it runs, Colab disconnects idle runtimes.

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
- **OpenCV vision (Session 9 method):** `OpenCVVision` turns the camera bytes into a NumPy array, converts it with `cv.cvtColor(..., COLOR_BGR2HSV)` and makes one `cv.inRange()` mask per colour. `cv.moments()` gives the blob size and centre. When a block is found it also saves the filtered image (`cv.bitwise_and` with the mask) and prints the mean R, G, B of the block, like the Session 9 experiments.
- `classify_pixel()` (pure Python fallback) uses HSV: saturation ≥ 0.55 and value ≥ 0.20 removes the black floor, grey walls and the sky. Hue ranges: red 0–20° and 340–360°, green 90–150°, blue 200–260°. OpenCV stores hue as 0–179, so these become red 0–9 and 170–179, green 45–74, blue 100–129 (S ≥ 140, V ≥ 51).
- `CoverageGrid.next_target()` picks the nearest unvisited waypoint, with a small penalty for big turns.
- The stuck detector reverses the robot if it moved less than 2 cm in 2.5 s while it was meant to be driving.

---

## 6. Git workflow

Same as Assignment 1: `main` stays working, every piece of work goes on its own branch (`feature/...`, `docs/...`, `fix/...`) and is merged through a pull request. Each Trello card names its branch.
