"""
WREN — Main Entry Point

Gestures
--------
  Index finger only             → move  (relative, accelerated)
  Index + middle, move up/down  → scroll
  Quick pinch & release         → select
  Pinch & hold 0.3s, then move  → drag

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

from wren.hand_tracker import HandTracker
from wren.gesture_classifier import DebouncedClassifier, Gesture
from wren.cursor_mapper import CursorMapper


CAMERA_INDEX = 0
FRAME_W      = 640
FRAME_H      = 480

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


def draw_hud(frame, gesture, fps, debug, sensitivity):
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

    if debug:
        lines = [
            "index only           MOVE (accel, relative)",
            "index + middle       SCROLL (move hand up/down)",
            "quick pinch          SELECT",
            "pinch + hold 0.3s    DRAG",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (16, 78 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)


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

    debug          = False
    prev_time      = time.time()
    active_gesture = Gesture.NONE
    was_dragging   = False
    sensitivity    = mapper._base

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
            mapper.reset()
            was_dragging = False

        tracker.draw_landmarks(frame, hands if hands else [])
        draw_hud(frame, active_gesture, fps, debug, sensitivity)
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
