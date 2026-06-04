"""
WREN — Main Entry Point

Gestures
--------
  Index finger only             → move  (relative, accelerated)
  Index + middle, move up/down  → scroll
  Quick pinch & release         → select
  Pinch & hold 0.3s, then move  → drag
  Stay still 5 s (index only)   → dwell-select

Keyboard
--------
  Q  — quit
  D  — toggle debug overlay
  +  — increase sensitivity
  -  — decrease sensitivity
"""

import cv2
import sys
import time
import math

from wren.hand_tracker import HandTracker
from wren.gesture_classifier import DebouncedClassifier, Gesture
from wren.cursor_mapper import CursorMapper


CAMERA_INDEX   = 0
FRAME_W        = 640
FRAME_H        = 480

DWELL_SECONDS  = 3.0    # seconds of stillness required to trigger dwell-select
DWELL_RADIUS   = 0.028  # normalised-coord radius (~1.2 % of frame width); adjust to taste

_COLOR = {
    Gesture.MOVE:        (0,   255, 180),
    Gesture.SELECT:      (0,   200, 255),
    Gesture.DRAG:        (255, 180,   0),
    Gesture.SCROLL_UP:   (160, 255, 100),
    Gesture.SCROLL_DOWN: (100, 160, 255),
    Gesture.NONE:        (100, 100, 100),
}
_LABEL = {
    Gesture.MOVE:        "MOVE",
    Gesture.SELECT:      "SELECT",
    Gesture.DRAG:        "DRAG",
    Gesture.SCROLL_UP:   "SCROLL UP",
    Gesture.SCROLL_DOWN: "SCROLL DOWN",
    Gesture.NONE:        "",
}


# ---------------------------------------------------------------------------
# Dwell tracker
# ---------------------------------------------------------------------------

class DwellTracker:
    """Fires a select action when the index tip stays within DWELL_RADIUS
    of its anchor position for DWELL_SECONDS, but only while the current
    gesture is MOVE (i.e. index finger only — no pinch, no scroll)."""

    def __init__(self, radius: float = DWELL_RADIUS, seconds: float = DWELL_SECONDS):
        self._radius   = radius
        self._seconds  = seconds
        self._anchor_x = None
        self._anchor_y = None
        self._start    = None
        self.progress  = 0.0   # 0.0 – 1.0, for the HUD arc

    def update(self, nx: float, ny: float, gesture: Gesture) -> bool:
        """Call every frame with the normalised index-tip position.
        Returns True exactly once when dwell threshold is reached."""
        if gesture is not Gesture.MOVE:
            self._reset()
            return False

        if self._anchor_x is None:
            self._anchor_x = nx
            self._anchor_y = ny
            self._start    = time.time()
            self.progress  = 0.0
            return False

        dist = math.hypot(nx - self._anchor_x, ny - self._anchor_y)
        if dist > self._radius:
            # Finger moved out of the dwell zone — restart from new position
            self._anchor_x = nx
            self._anchor_y = ny
            self._start    = time.time()
            self.progress  = 0.0
            return False

        elapsed       = time.time() - self._start
        self.progress = min(elapsed / self._seconds, 1.0)

        if elapsed >= self._seconds:
            self._reset()   # reset so it doesn't keep firing every frame
            return True     # ← trigger select

        return False

    def _reset(self):
        self._anchor_x = None
        self._anchor_y = None
        self._start    = None
        self.progress  = 0.0


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def draw_dwell_arc(frame, nx: float, ny: float, progress: float):
    """Draw a circular progress arc around the cursor while dwelling."""
    if progress <= 0.0:
        return
    h, w  = frame.shape[:2]
    cx    = int(nx * w)
    cy    = int(ny * h)
    r     = 22
    start_angle = -90                         # 12-o'clock
    end_angle   = start_angle + int(360 * progress)
    color = (0, 220, 255)
    cv2.ellipse(frame, (cx, cy), (r, r), 0, start_angle, end_angle, color, 2, cv2.LINE_AA)
    # Small centre dot
    cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)


def draw_hud(frame, gesture, fps, debug, sensitivity, dwell: DwellTracker, dwell_nx, dwell_ny):
    h, w = frame.shape[:2]
    label = _LABEL.get(gesture, "")
    color = _COLOR.get(gesture, (100, 100, 100))

    if label:
        cv2.putText(frame, label, (16, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"{fps:.0f} fps", (w - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"sens {sensitivity:.0f}", (w - 110, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q=quit  D=debug  +/-=sensitivity", (16, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 100, 100), 1, cv2.LINE_AA)

    # Dwell progress arc (only when there is meaningful progress)
    if dwell_nx is not None and dwell.progress > 0.01:
        draw_dwell_arc(frame, dwell_nx, dwell_ny, dwell.progress)

    if debug:
        lines = [
            "index only           MOVE (accel, relative)",
            "index + middle       SCROLL (move hand up/down)",
            "quick pinch          SELECT",
            "pinch + hold 0.3s    DRAG",
            f"still {DWELL_SECONDS:.0f}s (index only)  DWELL-SELECT",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (16, 78 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[WREN] ERROR: Could not open webcam.")
        sys.exit(1)

    tracker    = HandTracker(max_hands=1)
    classifier = DebouncedClassifier()
    mapper     = CursorMapper(
        base_sensitivity = 900.0,
        accel_threshold  = 0.012,
        accel_exponent   = 1.6,
        smooth_window    = 2,
        deadzone         = 0.003,
        scroll_speed     = 3,
    )
    dwell = DwellTracker(radius=DWELL_RADIUS, seconds=DWELL_SECONDS)

    debug          = False
    prev_time      = time.time()
    active_gesture = Gesture.NONE
    was_dragging   = False
    sensitivity    = mapper._base
    dwell_pos      = (None, None)   # last known normalised index position for arc drawing

    print("[WREN] Running.  Q=quit  D=debug  +/-=sensitivity")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        hands = tracker.process(frame)

        if hands:
            hand           = hands[0]
            result         = classifier.classify(hand)
            active_gesture = result.gesture
            nx, ny, _      = hand.index_tip
            dwell_pos      = (nx, ny)

            # --- Dwell check (evaluated before the gesture match so that a
            #     dwell-triggered select is processed in the same frame) ----
            if dwell.update(nx, ny, active_gesture):
                # Dwell threshold reached — fire a select
                mapper.select()
                active_gesture = Gesture.SELECT   # update HUD label for this frame

            else:
                match active_gesture:
                    case Gesture.MOVE:
                        if was_dragging:
                            mapper.end_drag()
                            was_dragging = False
                        mapper.move(nx, ny)

                    case Gesture.SELECT:
                        if was_dragging:
                            mapper.end_drag()
                            was_dragging = False
                        mapper.select()

                    case Gesture.DRAG:
                        if not was_dragging:
                            mapper.begin_drag(nx, ny)
                            was_dragging = True
                        else:
                            mapper.continue_drag(nx, ny)

                    case Gesture.SCROLL_UP:
                        if was_dragging:
                            mapper.end_drag()
                            was_dragging = False
                        mapper.scroll(1)

                    case Gesture.SCROLL_DOWN:
                        if was_dragging:
                            mapper.end_drag()
                            was_dragging = False
                        mapper.scroll(-1)

                    case Gesture.NONE:
                        pass

        else:
            active_gesture = Gesture.NONE
            dwell_pos      = (None, None)
            dwell.update(0, 0, Gesture.NONE)   # resets the dwell tracker
            mapper.reset()
            was_dragging = False

        tracker.draw_landmarks(frame, hands if hands else [])
        draw_hud(frame, active_gesture, fps, debug, sensitivity,
                 dwell, *dwell_pos)
        cv2.imshow("WREN", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            debug = not debug
        elif key in (ord("+"), ord("=")):
            mapper._base = min(mapper._base + 100, 2500)
            sensitivity  = mapper._base
            print(f"[WREN] Sensitivity: {sensitivity:.0f}")
        elif key == ord("-"):
            mapper._base = max(mapper._base - 100, 200)
            sensitivity  = mapper._base
            print(f"[WREN] Sensitivity: {sensitivity:.0f}")

    if was_dragging:
        mapper.end_drag()
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[WREN] Stopped.")


if __name__ == "__main__":
    main()