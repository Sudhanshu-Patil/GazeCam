"""
GazeCam for Windows: webcam -> gaze-corrected frames -> "OBS Virtual Camera".

Run:    .venv\\Scripts\\python gazecam_win.py     then pick "OBS Virtual Camera" in Meet
Check:  .venv\\Scripts\\python gazecam_win.py --selftest

Keys (click the preview window first):
  c      calibrate: look straight into the lens, then press
  g      correction on/off
  arrows nudge the corrected gaze up/down/left/right by 1 degree
  [ ]    lock strength: 0 = no gaze following, 1 = eyes fully pinned to the lens
  q      quit
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

SIDES = ("L", "R")  # model names; "R" is the eye at image-left (the subject's right eye)
# MediaPipe face-mesh indices. a/b: eye corners at image-left/right, six: the model's anchor points.
EYES = {
    "R": dict(a=33, b=133, iris=468, up=159, lo=145, six=[33, 160, 158, 133, 153, 144],
              contour=[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]),
    "L": dict(a=362, b=263, iris=473, up=386, lo=374, six=[362, 385, 387, 263, 373, 380],
              contour=[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]),
}
HEAD = dict(left=234, right=454, top=10, chin=152)  # face edges at image-left/right, forehead, chin
USED = sorted({i for e in EYES.values() for i in [e["a"], e["b"], e["iris"], e["up"], e["lo"], *e["six"], *e["contour"]]}
              | set(HEAD.values()))

K_EYE = 145.0  # degrees of eyeball rotation per unit of iris offset (offset / eye width): ~11.5 mm radius, ~29 mm eye
# How far the model really turns an eye, in degrees [down, right] per degree of its [vertical, horizontal] input.
# It under-delivers, differently for every face, and its vertical input is positive-up, so `c` measures this on
# the user's own eyes. These are the fallback and the believable range.
RESPONSE = np.array([-0.145, 0.41])
RESPONSE_MIN, RESPONSE_MAX = np.array([-0.5, 0.2]), np.array([-0.08, 1.0])
PROBE = 12.0       # model degrees each way used to measure that response
SNAP_DEG = 1.2     # a gaze change larger than this is a real eye jump and is followed at once; smaller ones are smoothed
SMOOTH = 0.4       # share of a small gaze change taken per frame
# shortcut: focal length guessed as 0.9 x frame width (~58 deg horizontal FOV), measure the webcam if the
# corrected eyes drift when you slide sideways in frame
FOCAL = 0.9
MAX_ANGLE = np.array([15.0, 20.0])  # model input limits [vertical, horizontal]; past these it draws halos round the iris
RISE_PER_S = 5.0   # correction fades back in over ~200 ms; it drops at once
MASK_GROW = 1.2    # eyelid mask scale, so the lid edge blends instead of cutting through the lashes
SETTING = "gazecam_win_v2"  # v2: calibration also stores the measured model response
ARROWS = {0x260000: (-1, 0), 0x280000: (1, 0), 0x250000: (0, -1), 0x270000: (0, 1)}  # up, down, left, right


def smoothstep(lo, hi, x):
    t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return t * t * (3 - 2 * t)


class OneEuro:
    """One-Euro filter over a numpy array: steady when still, fast when moving."""

    def __init__(self, min_cutoff=1.5, beta=0.1, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    def __call__(self, x, t):
        if self.x is None or t <= self.t:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = t - self.t
        alpha = lambda cutoff: 1 / (1 + 1 / (2 * math.pi * cutoff * dt))
        self.dx = self.dx + alpha(self.d_cutoff) * ((x - self.x) / dt - self.dx)
        self.x = self.x + alpha(self.min_cutoff + self.beta * np.abs(self.dx)) * (x - self.x)
        self.t = t
        return self.x


def eye_feature(P, e):
    """Iris offset from the eye-corner midpoint, in eye widths, [down, right] along the eye's own axes."""
    a, b = P[e["a"], :2], P[e["b"], :2]
    width = np.linalg.norm(b - a)
    u = (b - a) / width
    d = P[e["iris"], :2] - (a + b) / 2
    return np.array([d @ np.array([-u[1], u[0]]), d @ u]) / width


def openness(P, e):
    return np.linalg.norm(P[e["up"], :2] - P[e["lo"], :2]) / np.linalg.norm(P[e["b"], :2] - P[e["a"], :2])


def head_pose(P):
    """Degrees [pitch down, yaw toward image-right]. Landmark z grows away from the camera."""
    left, right, top, chin = (P[HEAD[k]] for k in ("left", "right", "top", "chin"))
    yaw = math.atan2(right[2] - left[2], right[0] - left[0])
    pitch = math.atan2(chin[2] - top[2], chin[1] - top[1])
    return np.degrees([pitch, yaw])


def eye_gaze(P):
    """Where the eyes point within the head, degrees [down, right], up to a constant that calibration removes."""
    return K_EYE * np.mean([eye_feature(P, EYES[s]) for s in SIDES], axis=0)


def follow(prev, gaze, noise):
    """Gaze smoothing scaled to how noisy the reading is (degrees per axis). A change well clear of the noise is a
    real eye jump and is taken whole; anything else is averaged, harder the noisier it is, so noise never
    reaches the corrected eye as flicker."""
    if prev is None:
        return gaze
    step = gaze - prev
    jump = np.abs(step) > np.maximum(SNAP_DEG, 3 * noise)
    return np.where(jump, gaze, prev + np.clip(SMOOTH / (1 + noise), 0.1, SMOOTH) * step)


def measure(P, raw, w, h):
    """Per-frame readings. raw: this frame's landmarks in px, so eye jumps and blinks register with no lag.
    P: the same landmarks smoothed, for the slow things (head pose, where the face sits)."""
    eyes_px = (P[468, :2] + P[473, :2]) / 2
    return {
        "gaze": eye_gaze(raw),
        "head": head_pose(P),
        "ray": np.degrees(np.arctan2([eyes_px[1] - h / 2, eyes_px[0] - w / 2], FOCAL * w)),
        "ear": np.array([openness(raw, EYES[s]) for s in SIDES]),
    }


def deviation_deg(m, ref):
    """How far the gaze is from the lens, degrees [down, right]: eye-in-head + head turn + where the face sits in frame."""
    return (m["gaze"] - ref["gaze"]) + (m["head"] - ref["head"]) + (m["ray"] - ref["ray"])


def model_angle(dev, lock, trim, response):
    """Model input that moves the iris back onto the lens. dev and trim are degrees [down, right]."""
    return np.clip((trim - lock * dev) / response, -MAX_ANGLE, MAX_ANGLE)


def gate(open_ratio, d_head, dev):
    """0..1 share of correction for one eye: none while blinking, head turned away, or looking well off the lens."""
    return (
        smoothstep(0.45, 0.75, open_ratio)
        * (1 - smoothstep(12, 20, abs(d_head[0])))
        * (1 - smoothstep(15, 25, abs(d_head[1])))
        * (1 - smoothstep(15, 22, float(np.max(np.abs(dev)))))
    )


def blend(frame, eye, warped, contour, opacity):
    """Paste the warped eye back through a feathered eyelid-shaped mask, so glasses frames and skin stay untouched."""
    top, left = eye.top_left
    hh, ww = eye.original_size
    centre = contour.mean(axis=0)
    pts = (centre + (contour - centre) * MASK_GROW - (left, top)).round().astype(np.int32)
    mask = np.zeros((hh, ww), np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1.0)
    k = max(3, ww // 8) | 1
    mask = (cv2.GaussianBlur(mask, (k, k), 0) * opacity)[..., None]
    roi = frame[top:top + hh, left:left + ww]
    roi[:] = np.clip(roi * (1 - mask) + warped * 255 * mask, 0, 255).astype(np.uint8)


def to_tflite(model):
    """Re-host the two loaded TF eye models on TFLite: same output, half the time on a quarter of the CPU threads."""
    import tensorflow as tf

    runners = {}
    for side, sess, inputs, output in (
        ("L", model.l_sess, [model.le_img, model.le_fp, model.le_ang], model.le_pred),
        ("R", model.r_sess, [model.re_img, model.re_fp, model.re_ang], model.re_pred),
    ):
        with sess.graph.as_default():
            converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, inputs, [output])
            # The warp's index maths uses an integer MatMul that TFLite lacks; that one op falls back to TF.
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            interpreter = tf.lite.Interpreter(model_content=converter.convert(), num_threads=3)
        interpreter.allocate_tensors()
        slot = {int(d["shape"][-1]): d["index"] for d in interpreter.get_input_details()}  # 3 image, 12 anchors, 2 angle
        out = interpreter.get_output_details()[0]["index"]

        def run(eye, angle, interpreter=interpreter, slot=slot, out=out):
            """Warped eye, float 0..1, at the eye box's original size."""
            interpreter.set_tensor(slot[3], eye.image[None].astype(np.float32))
            interpreter.set_tensor(slot[12], eye.anchor_map[None].astype(np.float32))
            interpreter.set_tensor(slot[2], np.array([angle], np.float32))
            interpreter.invoke()
            h, w = eye.original_size
            return cv2.resize(interpreter.get_tensor(out).reshape(eye.image.shape), (w, h))

        runners[side] = run
    return runners


def open_devices(camera):
    """Open the webcam and the OBS virtual camera. Returns (capture, virtual camera, first frame)."""
    import pyvirtualcam

    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # never queue stale frames
    ok, frame = cap.read()
    for _ in range(15):  # on Windows a webcam held by another app still "reads", but every pixel is zero
        if not ok or frame.any():
            break
        ok, frame = cap.read()
    if not ok or not frame.any():
        sys.exit(f"Camera {camera} gives no picture. Close any app that is using it (OBS, Meet, Camera), "
                 "or open its privacy shutter, then start GazeCam again.")
    h, w = frame.shape[:2]
    try:
        cam = pyvirtualcam.Camera(w, h, 30, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs")
    except RuntimeError as err:
        sys.exit(f"OBS Virtual Camera is not available ({err}). Close OBS, or stop its virtual camera.")
    print(f"Sending {w}x{h} to '{cam.device}'. Pick it as the camera in Meet.")
    return cap, cam, frame


class GazeCam:
    def __init__(self):
        import mediapipe as mp
        from displayers.face_predictor import EyeExtractionConfig, EyeLandmarks, MediaPipeFacePredictor
        from model_managers.gaze_corrector_v1 import GazeCorrector

        self.mp, self.EyeLandmarks = mp, EyeLandmarks
        self.predictor = MediaPipeFacePredictor()  # kept for its eye-crop code only
        self.predictor.landmarker.close()
        # The repo's VIDEO-mode tracker smooths landmarks over ~5 frames, so every reading jump showed on camera
        # before the correction caught up. Detecting each frame on its own has no lag and costs ~2 ms more.
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="./models/face_landmarker.task")))
        self.corrector = GazeCorrector()  # restores the checkpoints and owns the settings database
        self.infer = to_tflite(self.corrector.model)
        self.eye_cfg = EyeExtractionConfig()
        self.pool = ThreadPoolExecutor(2)
        self.filter = OneEuro()
        saved = self.corrector.db.get_setting(SETTING) or {}
        self.ref = {k: np.array(v) for k, v in saved["ref"].items()} if saved.get("ref") else None
        self.lock = saved.get("lock", 1.0)
        self.trim = np.array(saved.get("trim", [0.0, 0.0]))
        self.enabled = True
        self.calibrate = 0  # frames of calibration still to collect; `c` sets it
        self.samples = []
        self.weight = dict.fromkeys(SIDES, 0.0)
        self.gaze = None
        self.recent, self.noise = [], np.zeros(2)
        self.t_prev = None

    def landmarks(self, frame):
        """This frame's landmarks in px (z on the same scale), unsmoothed, or None when there is no face."""
        h, w = frame.shape[:2]
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = self.landmarker.detect(image).face_landmarks
        if not faces:
            return None
        raw = np.zeros((478, 3))
        for j in USED:  # converting all 478 landmarks costs ~2 ms a frame; only these are read
            p = faces[0][j]
            raw[j] = p.x * w, p.y * h, p.z * w
        return raw

    def eye(self, frame, P, side):
        """The model's input for one eye, or None when its box runs off the frame."""
        e = EYES[side]
        points = self.EyeLandmarks(points=[tuple(P[j, :2]) for j in e["six"]], center=tuple(P[e["iris"], :2]))
        eye = self.predictor._extract_single_eye(frame, points, side, self.eye_cfg)
        return None if eye is None or min(eye.top_left) < 0 else eye

    def probe(self, frame, P):
        """Measure how far the model really turns these eyes: warp them PROBE degrees each way and re-read the gaze."""
        eyes = {s: self.eye(frame, P, s) for s in SIDES}
        response = RESPONSE.copy()
        if None in eyes.values():
            return response

        def gaze_after(angle):
            out = frame.copy()
            for s, eye in eyes.items():
                blend(out, eye, self.infer[s](eye, angle), P[EYES[s]["contour"], :2], 1.0)
            raw = self.landmarks(out)
            return None if raw is None else eye_gaze(raw)

        for axis in (0, 1):
            angle = np.zeros(2)
            angle[axis] = PROBE
            plus, minus = gaze_after(angle), gaze_after(-angle)
            if plus is not None and minus is not None:
                response[axis] = (plus - minus)[axis] / (2 * PROBE)
        return np.clip(response, RESPONSE_MIN, RESPONSE_MAX)

    def process(self, frame, t):
        """Correct the eyes in `frame` (modified in place) and return it. t: seconds."""
        h, w = frame.shape[:2]
        raw = self.landmarks(frame)
        dt, self.t_prev = (t - self.t_prev if self.t_prev is not None else 0.0), t
        if raw is None:
            self.filter.x = self.gaze = None
            self.recent = []
            self.weight = dict.fromkeys(SIDES, 0.0)
            return frame
        P = self.filter(raw, t)
        m = measure(P, raw, w, h)
        # Noise of the raw gaze reading, from its frame-to-frame wobble (second difference; slow drift cancels out).
        self.recent = (self.recent + [m["gaze"]])[-3:]
        if len(self.recent) == 3:
            wobble = np.abs(self.recent[2] - 2 * self.recent[1] + self.recent[0]) / 1.95  # -> sigma for Gaussian noise
            self.noise += 0.05 * (wobble - self.noise)
        m["gaze"] = self.gaze = follow(self.gaze, m["gaze"], self.noise)
        if self.calibrate:  # averaged over several frames: one frame's reading is too noisy to trust
            self.samples.append(dict(m, response=self.probe(frame, P)))
            self.calibrate -= 1
            if not self.calibrate:
                self.ref = {k: np.median([s[k] for s in self.samples], axis=0) for k in self.samples[0]}
                self.samples = []
                self.save()
        if self.ref is None:
            return frame
        dev = deviation_deg(m, self.ref)
        angle = model_angle(dev, self.lock, self.trim, self.ref["response"])
        jobs = []
        for i, side in enumerate(SIDES):
            target = self.enabled * gate(m["ear"][i] / self.ref["ear"][i], m["head"] - self.ref["head"], dev)
            wt = self.weight[side] = min(target, self.weight[side] + RISE_PER_S * dt)
            eye = self.eye(frame, P, side) if wt >= 0.01 else None
            if eye is not None:
                # Fading scales the angle, not just opacity: a half-faded eye is a half-turned eye, never two irises.
                jobs.append((eye, side, angle * wt, P[EYES[side]["contour"], :2], min(1.0, 4 * wt)))
        # Both eyes at once: inference is the largest share of the frame time and it releases the GIL.
        warped = self.pool.map(lambda j: self.infer[j[1]](j[0], j[2]), jobs)
        for (eye, _, _, contour, opacity), img in zip(jobs, warped):
            blend(frame, eye, img, contour, opacity)
        return frame

    def save(self):
        self.corrector.db.save_setting(SETTING, {
            "ref": {k: v.tolist() for k, v in self.ref.items()} if self.ref else None,
            "lock": self.lock,
            "trim": self.trim.tolist(),
        })

    def handle_key(self, key):
        ch = chr(key).lower() if 0 < key < 256 else ""
        if ch == "g":
            self.enabled = not self.enabled
        elif ch == "c":
            self.calibrate, self.samples = 8, []  # process() collects the next 8 frames that have a face
        elif ch == "[" or ch == "]":
            self.lock = round(min(1.5, max(0.0, self.lock + (0.05 if ch == "]" else -0.05))), 2)
            self.save()
        elif key in ARROWS:
            self.trim = self.trim + ARROWS[key]
            self.save()

    def preview(self, out, ms):
        view = cv2.resize(out, (640, out.shape[0] * 640 // out.shape[1]))
        if not self.enabled:
            state = "OFF"
        elif self.ref is None:
            state = "LOOK INTO THE LENS, PRESS C"
        else:
            state = f"ON lock {self.lock:.2f} trim {self.trim[0]:+.0f},{self.trim[1]:+.0f} noise {self.noise.max():.1f}"
        cv2.putText(view, f"{state}  {ms:.0f} ms", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("GazeCam", view)
        return cv2.waitKeyEx(1)

    def run(self, camera):
        cap, cam, frame = open_devices(camera)
        ok = True
        ms = 0.0
        with cam:
            while ok:
                t0 = time.perf_counter()
                out = self.process(frame, t0)
                cam.send(out)
                ms += 0.1 * ((time.perf_counter() - t0) * 1000 - ms)
                key = self.preview(out, ms)
                if key == ord("q"):
                    break
                self.handle_key(key)
                ok, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()
        self.corrector.close()


def selftest():
    still = np.zeros(2)
    assert gate(1.0, still, still) == 1.0                     # eyes open, facing the lens
    assert gate(0.3, still, still) == 0.0                     # blink: the real eyelid passes through
    assert gate(1.0, np.array([0.0, 30.0]), still) == 0.0     # head turned away
    assert gate(1.0, still, np.array([0.0, 25.0])) == 0.0     # looking well off the lens
    assert 0 < gate(1.0, np.array([0.0, 20.0]), still) < 1    # fades, no hard switch

    assert np.allclose(model_angle(np.array([1.0, -3.0]), 1.0, still, RESPONSE), [-1.0, 3.0] / RESPONSE)  # cancels it
    assert np.all(np.abs(model_angle(np.array([90.0, -90.0]), 1.0, still, RESPONSE)) <= MAX_ANGLE)

    at = np.array([2.0, 2.0])
    assert np.allclose(follow(at, at + [0.0, 5.0], still), at + [0.0, 5.0])  # a clear eye jump is followed whole
    assert np.allclose(follow(at, at + [0.0, 0.5], still), at + [0.0, 0.2])  # jitter is averaged down
    assert np.allclose(follow(at, at + [0.0, 5.0], still + 3), at + [0.0, 0.5])  # noisy reading: 5 deg is not a jump

    P = np.zeros((478, 3))
    P[33, :2], P[133, :2], P[468, :2] = (100, 100), (130, 100), (118, 103)
    assert np.allclose(eye_feature(P, EYES["R"]), [0.1, 0.1])  # iris 3 px down and right of centre, eye 30 px wide
    P[33, :2], P[133, :2], P[468, :2] = (100, 100), (100, 130), (97, 115)  # same eye, head rolled 90 degrees
    assert np.allclose(eye_feature(P, EYES["R"]), [0.1, 0.0])
    P[234], P[454], P[10], P[152] = (0, 50, 0), (100, 50, 20), (50, 0, 0), (50, 100, 20)
    assert np.all(head_pose(P) > 11)                           # right side and chin further away: yaw right, pitch down

    f = OneEuro()
    x = f(np.array([0.0]), 0.0)
    for i in range(1, 60):
        x = f(np.array([10.0]), i / 30)
    assert abs(x[0] - 10) < 0.1                                # settles on a step within 2 s
    print("selftest ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaze-corrected virtual camera for Windows")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--selftest", action="store_true", help="check the gaze logic and exit")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        # TensorFlow and MediaPipe print pages of harmless start-up chatter; keep real errors only.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import logging
        import warnings
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        os.chdir(os.path.dirname(os.path.abspath(__file__)))  # weights and settings are found relative to here
        GazeCam().run(args.camera)
