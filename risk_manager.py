"""Risk metrics and asymmetric reward shaping for deep hedging."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


def value_at_risk(pnl: np.ndarray, alpha: float = 0.05) -> float:
    """Compute Value at Risk (VaR) at alpha quantile."""
    pnl_arr = np.asarray(pnl, dtype=np.float64)
    if pnl_arr.size == 0:
        return 0.0
    return float(np.quantile(pnl_arr, alpha))


def conditional_value_at_risk(pnl: np.ndarray, alpha: float = 0.05) -> float:
    """Compute Conditional Value at Risk (CVaR), also known as Expected Shortfall."""
    pnl_arr = np.asarray(pnl, dtype=np.float64)
    if pnl_arr.size == 0:
        return 0.0
    var = value_at_risk(pnl_arr, alpha)
    tail = pnl_arr[pnl_arr <= var]
    if tail.size == 0:
        return var
    return float(np.mean(tail))


@dataclass
class AsymmetricReward:
    """Tail-aware reward function for deep hedging.

    The reward is:
        step_pnl - transaction_cost - tail_penalty * max(0, rolling_var - step_pnl)
    """

    tail_penalty: float = 8.0
    var_alpha: float = 0.05
    rolling_window: int = 252
    warmup: int = 30

    def __post_init__(self) -> None:
        self._buffer: Deque[float] = deque(maxlen=self.rolling_window)

    def reset(self) -> None:
        """Reset the internal reward history state."""
        self._buffer.clear()

    def compute(self, step_pnl: float, transaction_cost: float) -> float:
        """Compute asymmetric step reward from PnL and costs."""
        self._buffer.append(float(step_pnl))
        reward = float(step_pnl) - float(transaction_cost)

        if len(self._buffer) >= self.warmup:
            rolling_var = value_at_risk(np.asarray(self._buffer, dtype=np.float64), self.var_alpha)
            if step_pnl < rolling_var:
                reward -= self.tail_penalty * (rolling_var - step_pnl)

        return float(reward)


@dataclass
class MeanReward:
    """Expected-value reward: penalize transaction costs but ignore tail risk."""

    cost_multiplier: float = 1.0

    def reset(self) -> None:
        """No internal state."""

    def compute(self, step_pnl: float, transaction_cost: float) -> float:
        """Compute scalar reward for mean-optimized agents."""
        return float(step_pnl) - float(self.cost_multiplier) * float(transaction_cost)
