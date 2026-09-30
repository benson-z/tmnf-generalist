"""The 12-action discrete space: {gas on/off} x {brake on/off} x {steer L/none/R}.

One action is held for a whole 50 ms step (five physics ticks). There is no
respawn action, by design.

Index layout: ``gas * 6 + brake * 3 + steer`` with steer 0 = left, 1 = none,
2 = right, so mirroring an action only ever touches the steer digit.
"""

from __future__ import annotations

import numpy as np

N_ACTIONS = 12
STEER_LEFT, STEER_NONE, STEER_RIGHT = 0, 1, 2
_STEER_NAMES = ("L", "-", "R")


def encode(gas: int, brake: int, steer: int) -> int:
    return int(gas) * 6 + int(brake) * 3 + int(steer)


def decode(action: int) -> tuple[int, int, int]:
    """(gas, brake, steer) for an action index."""
    return action // 6, (action // 3) % 2, action % 3


def keys(action: int) -> tuple[bool, bool, bool, bool]:
    """(up, down, left, right) keys for an action index."""
    gas, brake, steer = decode(action)
    return bool(gas), bool(brake), steer == STEER_LEFT, steer == STEER_RIGHT


def name(action: int) -> str:
    gas, brake, steer = decode(action)
    return f"{'G' if gas else '.'}{'B' if brake else '.'}{_STEER_NAMES[steer]}"


# Horizontal mirroring swaps left and right steer and nothing else.
MIRROR = np.array(
    [encode(g, b, 2 - s) for g, b, s in (decode(a) for a in range(N_ACTIONS))],
    dtype=np.int64,
)


def from_ticks(up: np.ndarray, down: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Label each 50 ms step by per-component majority over its 5 ticks.

    Inputs are boolean arrays of shape (steps, 5). Gas and brake are each on if
    held for at least 3 of the 5 ticks. Steer is left/right if that key alone
    was held for at least 3 ticks; a tick with both left and right held counts
    as no steer (0.02% of ticks in corpus2).
    """
    gas = up.sum(1) >= 3
    brake = down.sum(1) >= 3
    only_l = (left & ~right).sum(1)
    only_r = (right & ~left).sum(1)
    steer = np.full(up.shape[0], STEER_NONE, dtype=np.int64)
    steer[only_l >= 3] = STEER_LEFT
    steer[only_r >= 3] = STEER_RIGHT
    return gas.astype(np.int64) * 6 + brake.astype(np.int64) * 3 + steer


def per_tick(up: np.ndarray, down: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The action each individual tick corresponds to (left+right = no steer)."""
    steer = np.full(up.shape, STEER_NONE, dtype=np.int64)
    steer[left & ~right] = STEER_LEFT
    steer[right & ~left] = STEER_RIGHT
    return up.astype(np.int64) * 6 + down.astype(np.int64) * 3 + steer
