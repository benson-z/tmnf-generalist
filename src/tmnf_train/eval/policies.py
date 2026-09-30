"""What the eval harness drives with.

A :class:`Policy` hands out one :class:`Episode` per rollout. An episode sees
exactly what the model is allowed to see -- the captured frame and the speed
the HUD would show -- and returns one of the 12 actions. Nothing else from the
game reaches it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .. import actions


class Episode(Protocol):
    def act(self, frame: np.ndarray, speed_kmh: float) -> tuple[int, np.ndarray | None]:
        """frame: (240, 320, 3) uint8 RGB. Returns (action, probabilities or None)."""
        ...


class Policy(Protocol):
    id: str

    def episode(self, seed: int, temperature: float) -> Episode: ...


class RandomPolicy:
    """Independent per-component coin flips; for testing the harness only.

    Biased towards gas so a rollout moves enough to exercise checkpoints,
    timeouts and the stationary cutoff rather than always sitting still.
    """

    def __init__(self, gas_prob: float = 0.8, brake_prob: float = 0.1, id: str = "random"):
        self.id = id
        self.gas_prob = gas_prob
        self.brake_prob = brake_prob

    def episode(self, seed: int, temperature: float) -> Episode:
        return _RandomEpisode(np.random.default_rng(seed), self.gas_prob, self.brake_prob)


class _RandomEpisode:
    def __init__(self, rng: np.random.Generator, gas_prob: float, brake_prob: float):
        self.rng = rng
        g = np.array([1 - gas_prob, gas_prob])
        b = np.array([1 - brake_prob, brake_prob])
        s = np.full(3, 1 / 3)
        self.probs = (g[:, None, None] * b[None, :, None] * s[None, None, :]).reshape(-1)

    def act(self, frame: np.ndarray, speed_kmh: float) -> tuple[int, np.ndarray | None]:
        return int(self.rng.choice(actions.N_ACTIONS, p=self.probs)), self.probs
