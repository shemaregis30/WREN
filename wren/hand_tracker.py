"""
WREN — Hand Tracking Module (Layer 02-A)
Uses the MediaPipe Tasks API (mediapipe >= 0.10).

FIRST-TIME SETUP
----------------
The new API requires a model file. On first run this module will
download it automatically into the wren/ package folder.
If you are offline, download it manually and place it at:

    wren/hand_landmarker.task

Download URL:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

import cv2
import numpy as np
import urllib.request
from pathlib import Path
from dataclasses import dataclass

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


# ── Model path ────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
MODEL_PATH = _HERE / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model():
    """Download the model file if it isn't already present."""
    if MODEL_PATH.exists():
        return
    print(f"[WREN] Downloading hand landmarker model to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[WREN] Model downloaded successfully.")
    except Exception as exc:
        raise RuntimeError(
            f"[WREN] Could not download model: {exc}\n"
            f"Please download it manually from:\n  {MODEL_URL}\n"
            f"and place it at: {MODEL_PATH}"
        ) from exc


# ── Data class ────────────────────────────────────────────────────────

@dataclass
class HandData:
    """Processed data for one detected hand."""
    landmarks: list        # list of NormalizedLandmark (x, y, z)
    finger_states: list    # [thumb, index, middle, ring, pinky] True=extended
    index_tip: tuple       # (x, y, z) normalised
    wrist: tuple           # (x, y, z) normalised


# ── Tracker ───────────────────────────────────────────────────────────

class HandTracker:
    """
    Detects hands and extracts 21 landmarks per frame.

    Usage
    -----
        tracker = HandTracker()
        while True:
            ret, frame = cap.read()
            hands = tracker.process(frame)   # list[HandData]
        tracker.close()
    """

    _CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ):
        _ensure_model()

        base_opts = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._frame_ts_ms = 0

    # ── Public API ────────────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray) -> list:
        """Run hand detection. Returns list[HandData], may be empty."""
        self._frame_ts_ms += 33   # must be monotonically increasing for VIDEO mode

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        if not result.hand_landmarks:
            return []

        return [self._build_hand_data(lm_list) for lm_list in result.hand_landmarks]

    def draw_landmarks(self, bgr_frame: np.ndarray, hands: list) -> np.ndarray:
        """
        Overlay hand skeleton using pure OpenCV (no mp.solutions dependency).
        Modifies frame in-place and returns it.
        """
        h, w = bgr_frame.shape[:2]
        for hand in hands:
            lms = hand.landmarks
            for a, b in self._CONNECTIONS:
                x1, y1 = int(lms[a].x * w), int(lms[a].y * h)
                x2, y2 = int(lms[b].x * w), int(lms[b].y * h)
                cv2.line(bgr_frame, (x1, y1), (x2, y2), (0, 220, 160), 1, cv2.LINE_AA)
            for lm in lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(bgr_frame, (cx, cy), 4, (255, 255, 255), -1)
                cv2.circle(bgr_frame, (cx, cy), 4, (0, 180, 120), 1)
        return bgr_frame

    def close(self):
        self._landmarker.close()

    # ── Internal ──────────────────────────────────────────────────────

    def _build_hand_data(self, lms) -> HandData:
        return HandData(
            landmarks=lms,
            finger_states=self._get_finger_states(lms),
            index_tip=(lms[8].x, lms[8].y, lms[8].z),
            wrist=(lms[0].x, lms[0].y, lms[0].z),
        )

    def _get_finger_states(self, lms) -> list:
        """
        Returns [thumb, index, middle, ring, pinky] — True = extended.
        Thumb uses horizontal distance from wrist; others use tip-vs-PIP y.
        """
        states = []

        # Thumb: tip (4) further from wrist than IP joint (3)
        wrist_x = lms[0].x
        states.append(abs(lms[4].x - wrist_x) > abs(lms[3].x - wrist_x))

        # Index → pinky: tip y < PIP y (y grows downward in normalised coords)
        for tip_id, pip_id in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            states.append(lms[tip_id].y < lms[pip_id].y)

        return states
