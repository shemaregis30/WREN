"""
WREN — Gesture Recognition Engine (Layer 03-A)

Uses hand-plane normalised coordinates so classification is
rotation-invariant. Finger extension is measured as tip-to-wrist
distance ratio (not tip-Y vs PIP-Y), which holds at any palm angle.

Gesture map
-----------
  Index only                  → MOVE
  Index + middle, moving      → SCROLL_UP / SCROLL_DOWN
  Quick pinch & release       → SELECT
  Pinch & hold (>0.32 s)      → DRAG
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np

from .hand_tracker import HandData
from .hand_normaliser import NormalisedHand, normalise_hand


class Gesture(Enum):
    NONE        = auto()
    MOVE        = auto()
    SELECT      = auto()
    DRAG        = auto()
    SCROLL_UP   = auto()
    SCROLL_DOWN = auto()


@dataclass
class GestureResult:
    gesture: Gesture
    confidence: float
    hand_data: Optional[HandData]
    norm_hand: Optional[NormalisedHand] = None
    timestamp: float = field(default_factory=time.time)


# ── Thresholds ────────────────────────────────────────────────────────

# A finger is "extended" when its tip-to-wrist distance (normalised by
# hand size) exceeds this ratio relative to its fully-extended baseline.
# Empirically ~0.65 separates curled from extended across palm angles.
EXTENDED_RATIO   = 0.65

# Pinch: index tip to thumb tip distance, normalised by hand size
PINCH_RATIO      = 0.20   # below this = pinching

PINCH_HOLD_S     = 0.30   # seconds before pinch becomes DRAG
SCROLL_DY_THRESH = 0.008  # min movement in hand-plane to register scroll


# ── Finger state via distance ratios ──────────────────────────────────

# Landmark indices: tip, then PIP (used for reference length)
_FINGERS = [
    (4,  2,  0),   # thumb:  tip=4,  IP=2,   MCP=1  (compare tip vs IP vs wrist)
    (8,  6,  5),   # index:  tip=8,  PIP=6,  MCP=5
    (12, 10, 9),   # middle: tip=12, PIP=10, MCP=9
    (16, 14, 13),  # ring:   tip=16, PIP=14, MCP=13
    (20, 18, 17),  # pinky:  tip=20, PIP=18, MCP=17
]


def _finger_states(norm: NormalisedHand) -> list[bool]:
    """
    Returns [thumb, index, middle, ring, pinky] True=extended.

    For each finger: tip-to-wrist distance vs MCP-to-wrist distance.
    Extended fingers have tip >> MCP from wrist; curled ones don't.
    This is computed in hand-plane coords so palm tilt doesn't matter.
    """
    coords = norm.coords   # (21, 2) in hand-plane
    hs     = norm.hand_size

    states = []
    for tip_id, pip_id, mcp_id in _FINGERS:
        tip_dist = np.linalg.norm(coords[tip_id])    # distance from wrist (origin)
        mcp_dist = np.linalg.norm(coords[mcp_id])
        # Normalise by hand size so small/large hands work the same
        tip_norm = tip_dist / hs
        mcp_norm = mcp_dist / hs
        # Extended = tip is substantially further from wrist than its MCP
        states.append(tip_norm > mcp_norm * (1.0 + EXTENDED_RATIO * 0.6))

    return states


def _pinch_dist_normalised(norm: NormalisedHand) -> float:
    """Index tip (8) to thumb tip (4), normalised by hand size."""
    c = norm.coords
    d = np.linalg.norm(c[8] - c[4])
    return d / norm.hand_size


# ── Classifier ────────────────────────────────────────────────────────

class GestureClassifier:

    def __init__(self):
        self._pinch_start:        Optional[float] = None
        self._drag_active:        bool            = False
        self._prev_scroll_y:      Optional[float] = None   # hand-plane v coord

    def classify(self, hand: HandData) -> GestureResult:
        norm = normalise_hand(hand.landmarks)
        fs   = _finger_states(norm)
        g, c = self._rules(fs, norm)
        return GestureResult(gesture=g, confidence=c, hand_data=hand, norm_hand=norm)

    def _rules(self, fs: list[bool], norm: NormalisedHand):
        thumb, index, middle, ring, pinky = fs
        now = time.time()

        # ── Two-finger scroll ─────────────────────────────────────────
        if index and middle and not ring and not pinky:
            self._pinch_start = None
            self._drag_active = False

            # Use hand-plane v-coordinate of midpoint between two fingertips
            mid_v = float((norm.coords[8, 1] + norm.coords[12, 1]) / 2.0)

            if self._prev_scroll_y is not None:
                dv = mid_v - self._prev_scroll_y
                self._prev_scroll_y = mid_v
                if dv < -SCROLL_DY_THRESH:
                    return Gesture.SCROLL_UP, 0.95
                if dv > SCROLL_DY_THRESH:
                    return Gesture.SCROLL_DOWN, 0.95
            else:
                self._prev_scroll_y = mid_v

            return Gesture.NONE, 0.0

        self._prev_scroll_y = None

        # ── Index only → MOVE ─────────────────────────────────────────
        if index and not middle and not ring and not pinky:
            self._pinch_start = None
            self._drag_active = False
            return Gesture.MOVE, 1.0

        # ── Pinch ─────────────────────────────────────────────────────
        pd       = _pinch_dist_normalised(norm)
        pinching = pd < PINCH_RATIO and not middle and not ring and not pinky

        if pinching:
            if self._pinch_start is None:
                self._pinch_start = now
            held = now - self._pinch_start
            if held >= PINCH_HOLD_S:
                self._drag_active = True
            if self._drag_active:
                return Gesture.DRAG, 1.0
            # Still in tap window — emit NONE until released
            return Gesture.NONE, 0.0
        else:
            if self._pinch_start is not None and not self._drag_active:
                # Quick release → SELECT
                self._pinch_start = None
                return Gesture.SELECT, 1.0
            self._pinch_start = None
            self._drag_active = False

        return Gesture.NONE, 0.0


# ── Debounce / one-shot wrapper ───────────────────────────────────────

_INSTANT = {Gesture.MOVE, Gesture.SCROLL_UP, Gesture.SCROLL_DOWN, Gesture.DRAG}


class DebouncedClassifier:
    """
    Instant gestures pass through with zero delay.
    SELECT fires once per trigger with a 500 ms re-arm lockout.
    """

    def __init__(self):
        self._clf            = GestureClassifier()
        self._select_lockout = 0.0

    def classify(self, hand: HandData) -> GestureResult:
        result = self._clf.classify(hand)

        if result.gesture in _INSTANT:
            return result

        if result.gesture == Gesture.SELECT:
            now = time.time()
            if now < self._select_lockout:
                return GestureResult(Gesture.NONE, 0.0, hand)
            self._select_lockout = now + 0.5
            return result

        return result
