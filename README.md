# GazeCam

**Real-time eye-contact correction for Windows, as a virtual camera.**

GazeCam reads your webcam, turns your eyes toward the lens, and publishes the result as **OBS Virtual Camera**, which Google Meet, Zoom and Teams list as a normal webcam. It is built for reading notes during a call while appearing to look at the camera. Everything runs locally on the CPU; no GPU is needed.

> **Based on [WangWilly/gaze-correction-cam](https://github.com/WangWilly/gaze-correction-cam)**, a macOS gaze-correction camera, which in turn builds on the model from Hsu et al., *Look at Me! Correcting Eye Gaze in Live Video Communication* (2019). This repository is a fork of that project's code with a Windows application added in `gazecam_win.py`. For the macOS app, its demo and its download, go to the original repository.

## What GazeCam adds

The original applies one fixed correction angle and pastes a rectangle over each eye. GazeCam keeps its eye-warping model and adds:

- **Gaze following.** It measures where your eyes are on every frame and cancels the movement, so they hold still while you scan a line.
- **Real blinks.** Correction drops to zero the moment an eye closes, so your own blink passes through.
- **Fade-out.** Correction fades when your head turns more than about 15 degrees or you look well away from the lens, so you never get eyes pinned to the camera on a turned head.
- **Eyelid-shaped blend.** The corrected eye is pasted through a feathered mask the shape of your eye opening, so glasses frames and skin are untouched and no patch edge shows.
- **Noise-aware smoothing.** It measures how noisy its own eye reading is and smooths in proportion, so tracking noise never reaches the corrected eye as flicker.
- **Per-user calibration.** One key press records what "looking at the lens" looks like for you and measures how strongly the model moves your particular eyes.

## Requirements

- Windows 10 or 11, and a webcam
- [OBS Studio](https://obsproject.com/) installed. It provides the virtual camera driver; it does not need to be running.
- [uv](https://docs.astral.sh/uv/), or any Python 3.12 environment

## Setup (once)

1. Create the environment:
   ```powershell
   uv venv --python 3.12 .venv
   uv pip install --python .venv -r requirements-win.txt
   ```
2. Download `weights.zip` from the original project's [v0.1.1 release](https://github.com/WangWilly/gaze-correction-cam/releases/tag/v0.1.1) and extract it here, so that `weights/warping_model/flx/12/L` exists.
3. Download [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) into `models/`.

## Run

```powershell
.venv\Scripts\python gazecam_win.py
```

Then in Meet: **Settings -> Video -> Camera -> OBS Virtual Camera**.

Close OBS itself while GazeCam runs, and close any other app using the webcam. Only one app can hold the webcam at a time.

## Calibrate (once)

Sit as you would in a meeting, look straight into the lens with your eyes open, click the preview window, and press `c`. Hold still for a moment. Until you do this, the video passes through unchanged.

Calibration is saved in `user_settings.db`. Recalibrate if you change glasses, lighting or seating distance.

| Key | Action |
| --- | ------ |
| `c` | Calibrate: you are looking into the lens right now |
| `g` | Correction on / off |
| `]` / `[` | More / less gaze following |
| Arrow keys | Nudge the corrected gaze by 1 degree |
| `q` | Quit |

The preview's status line shows the processing time per frame in milliseconds and a `noise` figure in degrees.

## Getting a natural result

- **Put your notes in a narrow window at the top centre of the screen, directly under the webcam.** The model can lift the gaze cleanly by only a few degrees vertically and about 8 degrees to each side. Keep the text column under about 8 cm wide.
- **Fill the frame with your face.** The bigger your eyes are in the picture, the more precisely they can be tracked. Sit closer or raise the laptop.
- **Watch the `noise` figure.** Below about 1.0, GazeCam follows reading jumps instantly. Near 3 it can only smooth, and more of your reading movement shows.
- **Glasses:** frames stay straight, but screen glare on the lenses sits over the iris and adds noise. A dimmer screen, a dark notes window and light from the front all help.
- **Fine-tune if needed.** If a recording still shows your eyes drifting along the lines, press `]`. If they move against your reading direction, press `[`.

## How it works

1. MediaPipe finds the face landmarks on each frame.
2. Iris position, head pose and where the face sits in the frame are combined into how far the gaze is from the lens.
3. Each eye is cropped and passed through the warping model with the angle that cancels that offset.
4. The warped eye is blended back through an eyelid-shaped mask and the frame is sent to the virtual camera.

The model is restored from the original TensorFlow checkpoints and re-hosted on TFLite at startup, which roughly halves inference time.

## Check

```powershell
.venv\Scripts\python gazecam_win.py --selftest
```

On a Ryzen 5 5600H, CPU only: roughly 23-28 ms of processing per 1280x720 frame, close to 30 fps.

## Limits

- It adds about one frame (roughly 30 ms) of video delay. Audio is untouched.
- The model dates from 2019 and works at 64x48 pixels per eye. Large corrections draw halos around the iris, so GazeCam caps them.
- Notes on a second monitor, or anywhere far from the lens, are out of its range.
- If another app takes the webcam while GazeCam is running, the picture goes black; restart GazeCam.

## Repository layout

`gazecam_win.py` and `requirements-win.txt` are the Windows application. Everything else (`tf_models/`, `model_managers/`, `displayers/`, `utils/`, `docs/`, `bin_*.py`, `pyproject.toml`) is the original project's code, unchanged; GazeCam imports its model and eye-crop code. See [docs/architecture.md](docs/architecture.md) for the original's design notes.

## Credits and licence

- Original project: [WangWilly/gaze-correction-cam](https://github.com/WangWilly/gaze-correction-cam)
- Model and weights: Chih-Fan Hsu et al., [Look at Me! Correcting Eye Gaze in Live Video Communication](https://doi.org/10.1145/3311784), ACM TOMM 2019 ([chihfanhsu/gaze_correction](https://github.com/chihfanhsu/gaze_correction))
- Face landmarks: [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

Distributed under the BSD 3-Clause licence in [LICENSE](LICENSE), which carries the original copyright notice.
