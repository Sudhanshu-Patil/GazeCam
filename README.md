# Gaze Correction Camera

**Make natural eye contact on every video call.**

Gaze Correction Camera is a macOS virtual camera extension that automatically adjusts your eye direction in real-time — so you always look directly at the people you're talking to, even when you're reading notes or looking at your own video tile.

<img src="https://github.com/user-attachments/assets/66e2355a-20d7-4ac5-b711-cb1b2ff653d7" style="width: 160px; display: block;">

## Windows: GazeCam

`gazecam_win.py` runs the same gaze model on Windows and publishes the result as **OBS Virtual Camera**, which Google Meet, Zoom and Teams list as a normal webcam. It is built for reading notes while appearing to look at the camera, and adds four things to the original model:

- **Gaze following.** It measures where your eyes are on every frame, with no smoothing delay on eye jumps, and cancels the movement so they hold still while you scan a line. The original applies one fixed angle.
- **Real blinks.** Correction drops to zero the moment an eye closes, so your own blink passes through.
- **Fade-out.** Correction fades when your head turns more than about 15 degrees or you look well away from the lens, so you never get eyes pinned to the camera on a turned head.
- **Eyelid-shaped blend.** The corrected eye is pasted through a feathered mask the shape of your eye opening, so glasses frames and skin are untouched and no patch edge shows.

### Setup (once)

1. Install [OBS Studio](https://obsproject.com/). It provides the virtual camera driver; it does not need to be running.
2. Create the environment:
   ```powershell
   uv venv --python 3.12 .venv
   uv pip install --python .venv -r requirements-win.txt
   ```
3. Download `weights.zip` from the [v0.1.1 release](https://github.com/WangWilly/gaze-correction-cam/releases/tag/v0.1.1) and extract it here, so that `weights/warping_model/flx/12/L` exists.
4. Download [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) into `models/`.

### Run

```powershell
.venv\Scripts\python gazecam_win.py
```

Then in Meet: **Settings -> Video -> Camera -> OBS Virtual Camera**. Close OBS itself while GazeCam runs, and close any other app using the webcam.

### Calibrate (once)

Sit as you would in a meeting, look straight into the lens with your eyes open, click the preview window, and press `c`. Until you do, the video passes through unchanged.

Calibration records what "looking at the lens" looks like for you, and also measures how strongly the model moves your particular eyes, which sets the correction strength. It is saved in `user_settings.db`. Recalibrate if you change glasses, lighting or seating distance.

| Key | Action |
| --- | ------ |
| `c` | Calibrate: you are looking into the lens right now |
| `g` | Correction on / off |
| `]` / `[` | More / less gaze following |
| Arrow keys | Nudge the corrected gaze by 1 degree |
| `q` | Quit |

### Getting a natural result

- **Put your notes in a narrow window at the top centre of the screen, directly under the webcam.** The model can lift the gaze cleanly by only a few degrees vertically and about 8 degrees to each side. Keep the text column under about 8 cm wide; a wider column pushes the correction to its limit at both ends of every line, where it starts to draw halos around the iris.
- **Fine-tune against a recording if needed.** Calibration sets the strength for you, so start there. If a recording through the Windows Camera app ("OBS Virtual Camera" selected) still shows your eyes drifting along the lines, press `]`. If they move against your reading direction, press `[`.
- **Glasses:** frames stay straight, but screen glare on the lenses sits over the iris and moves with it. A dimmer screen, a dark notes window and light from the front all help.

### Check

```powershell
.venv\Scripts\python gazecam_win.py --selftest
```

On a Ryzen 5 5600H, CPU only: roughly 25-28 ms of processing per 1280x720 frame, close to 30 fps.

## Demo

<a href="https://www.youtube.com/watch?v=tOobANsNzOQ" target="_blank">
  <img src="https://img.youtube.com/vi/tOobANsNzOQ/0.jpg" style="width: 320px; display: block;">
</a>

## Download

**[⬇️ Download the macOS App](https://drive.google.com/file/d/1E47OZ66YPab1QuTbxN97hL2u3GYwyUbz/view?usp=drive_link)**

Install the `.app`, grant camera access, and you're ready to go.

## Requirements

- macOS 14 (Sonoma) or later
- A connected webcam or built-in camera
- Camera access permission granted to the app

## Getting Started

1. **Download and open** the app from the link above.
2. **Grant camera access** when prompted by macOS.
3. **Select your camera** from the dropdown in the top-right corner of the app window.
4. **Enable gaze correction** using the toggle in the settings panel.
5. **Set this app as your camera** in Zoom, Teams, FaceTime, or any video app — look for **"Gaze Correction Camera"** in the camera device list.

## Controls

| Key | Action |
| --- | ------ |
| `g` | Toggle gaze correction on / off |
| `c` | Toggle calibration panel |
| `q` | Quit |

## Calibration

Press `c` to open the calibration panel and fine-tune the correction for your setup:

| Control | Action |
| ------- | ------ |
| `↑` `↓` `←` `→` | Adjust camera position up/down/left/right |
| `+` / `-` | Adjust camera distance (closer/further) |
| `[` / `]` | Adjust focal length |
| `r` | Reset all values to default |

> **Tip:** Start with the defaults. Only calibrate if the gaze correction looks off for your specific desk setup.

---

## Advanced: Build from Source / Python CLI

If you want to run from source or use the Python CLI directly:

### Prerequisites

- [Python 3.12+](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/)
- [CMake](https://cmake.org/download/)
- [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/)

### Install

```bash
brew install pkg-config cmake
poetry install
```

### Download model weights

Download from [GitHub Releases](https://github.com/WangWilly/gaze-correction-cam/releases) and place in the correct directories:

- `lm_feat/shape_predictor_68_face_landmarks.dat`
- `weights/warping_model/flx/12/L/` and `.../R/` — checkpoint + weight files
- *(Optional)* `models/face_landmarker.task` — for MediaPipe backend

### Run

```bash
# Default (dlib backend)
poetry run python bin_single_window.py

# MediaPipe backend
poetry run python bin_single_window.py --backend mediapipe

# Specific camera
poetry run python bin_single_window.py --camera 1
```

---

## How It Works

1. Captures your webcam feed in real-time
2. Detects your face and eye positions using computer vision
3. Calculates where your eyes need to point to look at the camera
4. Applies a learned neural network warp to redirect your gaze
5. Outputs the corrected video as a virtual camera device

---

*Looking for architecture details or module documentation? See [docs/architecture.md](docs/architecture.md).*
