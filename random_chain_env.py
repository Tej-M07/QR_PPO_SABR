"""Gymnasium wrapper that samples a new empirical chain on each episode reset."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pandas as pd

from empirical_env import EnvConfig, HistoricalHedgingEnv


class RandomChainEnv(gym.Env[np.ndarray, np.ndarray]):
    """Randomize episodes by sampling a contract chain from parquet data.

    The environment samples a chain at every `reset()` call and delegates `step()` to
    the underlying `HistoricalHedgingEnv`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        chain_sampler: Callable[[], Optional[pd.DataFrame]],
        config: EnvConfig | None = None,
        reward_fn: object | None = None,
    ) -> None:
        super().__init__()
        self.chain_sampler = chain_sampler
        self.config = config or EnvConfig()
        self.reward_fn = reward_fn

        # Match observation/action spaces to the underlying env.
        obs_low = np.array(
            [-5.0, 0.0, 0.0, -1.5, -20.0, -20.0, -2.0, -2.0, -20.0, 0.0, -2.0, 0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [5.0, 1.0, 4.0, 1.5, 20.0, 20.0, 2.0, 2.0, 20.0, 4.0, 2.0, 2.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._env: HistoricalHedgingEnv | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, object] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        super().reset(seed=seed)
        _ = options

        chain = self.chain_sampler()
        if chain is None or len(chain) < 3:
            raise RuntimeError("Failed to sample a valid option chain.")

        self._env = HistoricalHedgingEnv(chain=chain, config=self.config, reward_fn=self.reward_fn)
        obs, info = self._env.reset(seed=seed)
        return obs, info

    def step(self, action: np.ndarray):
        if self._env is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self._env.step(action)

