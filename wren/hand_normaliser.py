"""
WREN — Hand Normaliser
Projects raw 3D MediaPipe landmarks onto the hand's own orientation plane,
so all downstream classifiers see the hand as if it were always flat-on to
the camera regardless of wrist rotation or palm tilt.

How it works
------------
Three landmarks that are stable regardless of finger state define the palm plane:
  - Wrist        (landmark 0)
  - Index MCP    (landmark 5)
  - Pinky MCP    (landmark 17)

We compute two edge vectors from those three points, cross-product them to get
the plane normal, then build an orthonormal basis (u, v, n).  Every landmark
is projected onto (u, v) giving 2D coordinates in hand-space.

Hand-space coords are:
  u  — roughly points from wrist toward middle finger (palm "up" axis)
  v  — roughly points from pinky to index side (palm "left/right" axis)

These are invariant to camera-space rotation, so tip-vs-base comparisons
work even when the palm faces sideways or tilts 45°.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class NormalisedHand:
    """
    Landmarks reprojected into hand-plane 2D coordinates.
    coords[i] = (u, v) for landmark i, in metres-ish (MediaPipe z is in
    the same unit as x/y when the hand fills a normalised frame).
    raw_lms is kept for pinch distance calculations in camera space.
    """
    coords: np.ndarray      # shape (21, 2)  — hand-plane (u, v)
    raw_lms: list           # original NormalizedLandmark list
    hand_size: float        # wrist-to-middle-MCP distance, used for normalisation
    normal: np.ndarray      # unit normal of the palm plane (for debug)


def _lm_to_vec(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def normalise_hand(lms) -> NormalisedHand:
    """
    Build a NormalisedHand from a MediaPipe landmark list.
    Falls back gracefully if the plane is degenerate (very flat z data).
    """
    wrist     = _lm_to_vec(lms[0])
    idx_mcp   = _lm_to_vec(lms[5])
    pinky_mcp = _lm_to_vec(lms[17])
    mid_mcp   = _lm_to_vec(lms[9])

    # Two edge vectors of the palm plane
    edge1 = idx_mcp   - wrist
    edge2 = pinky_mcp - wrist

    # Normal via cross product
    normal = np.cross(edge1, edge2)
    n_len  = np.linalg.norm(normal)

    if n_len < 1e-6:
        # Degenerate — palm is edge-on or z data is absent.
        # Fall back: use image-plane axes directly.
        u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        normal = normal / n_len
        # u = direction from wrist to middle MCP, projected off the normal
        u_raw = mid_mcp - wrist
        u_raw = u_raw - np.dot(u_raw, normal) * normal
        u_len = np.linalg.norm(u_raw)
        u = u_raw / u_len if u_len > 1e-6 else edge1 / (np.linalg.norm(edge1) + 1e-9)
        v = np.cross(normal, u)

    # Hand size: wrist → middle MCP distance (scale reference)
    hand_size = float(np.linalg.norm(mid_mcp - wrist)) + 1e-6

    # Project all 21 landmarks onto (u, v)
    coords = np.zeros((21, 2), dtype=np.float32)
    for i in range(21):
        p = _lm_to_vec(lms[i]) - wrist   # translate to wrist origin
        coords[i, 0] = float(np.dot(p, u))
        coords[i, 1] = float(np.dot(p, v))

    return NormalisedHand(
        coords=coords,
        raw_lms=lms,
        hand_size=hand_size,
        normal=normal,
    )
