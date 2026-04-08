"""
WREN — Cursor Mapping Module (Layer 03-C)

Relative movement with pointer acceleration.

The cursor moves by how much the hand moved, scaled by a non-linear
acceleration curve:

  effective_speed = raw_speed * (1 + (raw_speed / threshold) ^ exponent)

  - Slow hand (raw_speed < threshold) → multiplier ≈ 1  (precise, 1:1)
  - Fast hand (raw_speed >> threshold) → multiplier grows quickly

This lets you cross the screen in a short fast flick, while still
landing precisely when you move slowly — exactly how a real mouse works.
"""

import numpy as np
from collections import deque
import pyautogui

pyautogui.PAUSE    = 0.0
pyautogui.FAILSAFE = False


class CursorMapper:
    """
    Parameters
    ----------
    base_sensitivity : float
        Baseline pixels-per-normalised-unit. ~800 at 1080p is a good start.
    accel_threshold : float
        Hand speed (normalised units/frame) below which acceleration is
        minimal. Above this it ramps up. ~0.012 works well at 30 fps.
    accel_exponent : float
        Controls how aggressive the ramp is. 1.5 = moderate, 2.0 = steep.
    smooth_window : int
        Rolling average over delta history. 2–3 keeps it responsive.
    deadzone : float
        Hand movement below this (normalised) is treated as zero.
    scroll_speed : int
        Lines scrolled per scroll-gesture frame.
    """

    SCREEN_W, SCREEN_H = pyautogui.size()

    def __init__(
        self,
        base_sensitivity: float = 900.0,
        accel_threshold:  float = 0.012,
        accel_exponent:   float = 1.6,
        smooth_window:    int   = 2,
        deadzone:         float = 0.003,
        scroll_speed:     int   = 3,
    ):
        self._base       = base_sensitivity
        self._accel_thr  = accel_threshold
        self._accel_exp  = accel_exponent
        self._deadzone   = deadzone
        self._scroll_spd = scroll_speed

        self._buf: deque      = deque(maxlen=smooth_window)
        self._prev: tuple     = None
        self._dragging: bool  = False

    # ── Public actions ────────────────────────────────────────────────

    def move(self, nx: float, ny: float):
        dx, dy = self._delta(nx, ny)
        if dx == 0 and dy == 0:
            return
        cx, cy = pyautogui.position()
        pyautogui.moveTo(
            int(np.clip(cx + dx, 0, self.SCREEN_W - 1)),
            int(np.clip(cy + dy, 0, self.SCREEN_H - 1)),
        )

    def select(self):
        pyautogui.click(button="left")

    def begin_drag(self, nx: float, ny: float):
        if not self._dragging:
            pyautogui.mouseDown(button="left")
            self._dragging = True

    def continue_drag(self, nx: float, ny: float):
        dx, dy = self._delta(nx, ny)
        if dx == 0 and dy == 0:
            return
        cx, cy = pyautogui.position()
        pyautogui.moveTo(
            int(np.clip(cx + dx, 0, self.SCREEN_W - 1)),
            int(np.clip(cy + dy, 0, self.SCREEN_H - 1)),
        )

    def end_drag(self):
        if self._dragging:
            pyautogui.mouseUp(button="left")
            self._dragging = False

    def scroll(self, direction: int):
        pyautogui.scroll(self._scroll_spd * direction)

    def reset(self):
        self._prev = None
        self._buf.clear()
        if self._dragging:
            self.end_drag()

    # ── Internal ──────────────────────────────────────────────────────

    def _delta(self, nx: float, ny: float):
        if self._prev is None:
            self._prev = (nx, ny)
            return 0, 0

        raw_dx = nx - self._prev[0]
        raw_dy = ny - self._prev[1]
        self._prev = (nx, ny)

        speed = (raw_dx ** 2 + raw_dy ** 2) ** 0.5

        if speed < self._deadzone:
            self._buf.append((0.0, 0.0))
            return 0, 0

        # ── Acceleration multiplier ───────────────────────────────────
        # At speed == threshold, multiplier = 2.0 (doubles).
        # Grows as (speed/threshold)^exponent above that.
        ratio      = speed / self._accel_thr
        multiplier = 1.0 + (ratio ** self._accel_exp)
        scale      = self._base * multiplier

        # Flip x — webcam is mirrored
        self._buf.append((-raw_dx * scale, raw_dy * scale))

        avg_dx = float(np.mean([d[0] for d in self._buf]))
        avg_dy = float(np.mean([d[1] for d in self._buf]))
        return int(avg_dx), int(avg_dy)
