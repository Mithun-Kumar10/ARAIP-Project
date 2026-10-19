# Smart Bin Lid (Simulation)

A small Python project for the **Applied Robotics and AI Project** module (University of Hertfordshire, BEng Robotics and AI). It simulates a touch-free smart bin: an ultrasonic distance sensor detects a hand, the lid opens, and it closes again after a short delay.

The sensor is simulated in software for now, so the logic can be built and tested without hardware. The sensor function can later be swapped for a real HC-SR04 reading.

## Features

- Simulated ultrasonic sensor that returns a distance in cm
- Lid opens when a hand is closer than a set threshold (default 20 cm)
- Lid closes automatically after a delay (default 3 seconds)
- Counts how many times the lid was opened

## Project Structure

```
ARAIP-Project/
├── src/
│   └── smart_bin.py    # main program
├── hello.txt           # practice file from the initial Git exercise
└── README.md
```

## Requirements

- Python 3.x (developed on Python 3.14)
- No external libraries needed

## How to Run

1. Clone the repository:
```
   git clone https://github.com/Mithun-Kumar10/ARAIP-Project.git
   cd ARAIP-Project
```
2. Run the program:
```
   python src/smart_bin.py
```
3. You will see simulated distance readings, with "Lid OPEN" and "Lid CLOSED" messages as the lid reacts, followed by a count of how many times it opened.

## Configuration

Two values at the top of `src/smart_bin.py` can be changed:

| Setting | Default | Meaning |
|---|---|---|
| `OPEN_THRESHOLD_CM` | 20 | Hand closer than this opens the lid |
| `AUTO_CLOSE_DELAY_S` | 3 | Seconds before the lid closes again |

## Git Workflow

Changes follow the same cycle each time:

1. **Edit** a file in VS Code and save it.
2. **`git add <file>`** moves the change to the staging area (choosing what goes into the next snapshot).
3. **`git commit -m "message"`** saves the staged changes as a snapshot in the **local** repository.
4. **`git push`** uploads the local commits to the **remote** repository on GitHub, where they are backed up and visible to others.

In short: *add* selects, *commit* saves locally, *push* shares to GitHub.

### Other commands used

| Command | Purpose |
|---|---|
| `git init` | Create a new local repository |
| `git clone <url>` | Copy a remote repository to the computer |
| `git status` | Show what has changed |
| `git log --oneline` | Show commit history |
| `git checkout -b <name>` | Create and switch to a new branch |
| `git pull` | Download and merge changes from GitHub |

## Branching Strategy

- **`main`** holds the stable, working version of the project.
- Every new piece of work gets its own branch, named by type:
  - `feature/...` for new functionality (e.g. `feature/sensor-reading`, `feature/lid-logic`)
  - `docs/...` for documentation changes
  - `fix/...` for bug fixes
- Each branch is merged into `main` through a **Pull Request** on GitHub, then synced locally with `git pull`.

This keeps `main` working at all times and gives a clear history of what each change did.

## Project Management

Tasks are tracked on a Trello board, with one card per feature and each card linked to its branch and pull request.

Trello board: (https://trello.com/invite/b/6ad093fbe7015888a876be78/ATTI4e514846bd68888f418b87e82ec719c387016FF6/araip-project)

Lists: Project Specific Works, To Do, In Progress, Completed

## Future Work

- Replace the simulated sensor with a real HC-SR04 ultrasonic sensor
- Drive a real servo to open and close the lid
- Add an indicator LED for a full bin
- Log lid-open events to a file

Name: Arul MithunKumar
Student ID: 408AIAUM
University of Hertfordshire, Applied Robotics and AI Project
