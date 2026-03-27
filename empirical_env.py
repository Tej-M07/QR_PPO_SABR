"""Gymnasium environment for empirical deep hedging over option chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from risk_manager import AsymmetricReward

REQUIRED_OPTION_COLUMNS: List[str] = [
    "symbol",
    "expiration",
    "strike",
    "date",
    "implied_volatility",
    "delta",
    "gamma",
    "vega",
    "bid",
    "ask",
    "mark",
]


def _safe_float(value: float, default: float = 0.0) -> float:
    """Convert a possibly NaN-like value to float."""
    if pd.isna(value):
        return default
    return float(value)


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for the empirical hedging environment.

    Attributes:
        contract_multiplier: Multiplier applied to option price and Greek exposures.
        delta_trade_scale: Action scaling factor for delta hedge adjustment.
        vega_trade_scale: Action scaling factor for vega hedge adjustment.
        max_abs_hedge: Clamps both delta and vega hedge positions to [-max_abs_hedge, max_abs_hedge].
        transaction_cost_bps: Transaction cost in basis points (used as a notional turnover cost component).
        pnl_norm: Normalization constant for cumulative P&L in the observation vector.
        reward_tail_penalty: Tail penalty multiplier used by the default asymmetric reward.
    """

    contract_multiplier: float = 100.0
    delta_trade_scale: float = 0.2
    vega_trade_scale: float = 0.2
    max_abs_hedge: float = 2.0
    transaction_cost_bps: float = 5.0
    pnl_norm: float = 10_000.0
    reward_tail_penalty: float = 8.0


class HistoricalHedgingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Empirical deep hedging environment.

    State (12-dim):
        - moneyness (F/K)
        - normalized DTE
        - implied volatility (IV) proxy
        - option delta
        - scaled gamma
        - scaled vega
        - current delta hedge
        - current vega hedge
        - normalized cumulative P&L
        - implied volatility (again)
        - skew proxy
        - vol-of-vol proxy

    Action (2-dim, continuous):
        - adjustments to delta and vega hedges in [-1, 1]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        chain: pd.DataFrame,
        config: EnvConfig | None = None,
        reward_fn: object | None = None,
    ) -> None:
        super().__init__()
        self.chain = chain.reset_index(drop=True).copy()
        self.config = config or EnvConfig()
        self._validate_chain()

        self.n_steps = len(self.chain) - 1
        self.current_step = 0
        self.hedge_delta = 0.0
        self.hedge_vega = 0.0
        self.cumulative_pnl = 0.0
        self.last_step_pnl = 0.0

        # Reward shaping impacts training only; P&L tracking is independent.
        self.reward_fn = reward_fn or AsymmetricReward(tail_penalty=self.config.reward_tail_penalty)

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

    def _validate_chain(self) -> None:
        missing = [col for col in REQUIRED_OPTION_COLUMNS if col not in self.chain.columns]
        if missing:
            raise ValueError(f"Chain missing required columns: {missing}")
        if len(self.chain) < 3:
            raise ValueError("Chain must contain at least 3 observations.")
        for required in ("underlying_close", "dte", "skew_proxy", "vol_of_vol_proxy"):
            if required not in self.chain.columns:
                raise ValueError(f"Chain must include '{required}' column.")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, object] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Reset environment state.

        Args:
            seed: Random seed.
            options: Unused.

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)
        _ = options

        self.current_step = 0
        self.hedge_delta = 0.0
        self.hedge_vega = 0.0
        self.cumulative_pnl = 0.0
        self.last_step_pnl = 0.0

        reset_fn = getattr(self.reward_fn, "reset", None)
        if callable(reset_fn):
            reset_fn()

        return self._get_obs(), {}

    def _row(self, idx: int) -> pd.Series:
        return self.chain.iloc[idx]

    def _get_obs(self) -> np.ndarray:
        """Construct the 12-dim observation vector for the current step."""
        row = self._row(self.current_step)

        spot = max(_safe_float(row["underlying_close"], 1.0), 1e-6)
        strike = max(_safe_float(row["strike"], 1.0), 1e-6)
        dte = np.clip(_safe_float(row["dte"], 0.0), 0.0, 252.0) / 252.0
        iv = np.clip(_safe_float(row["implied_volatility"], 0.2), 0.0, 4.0)
        delta = np.clip(_safe_float(row["delta"], 0.0), -1.5, 1.5)

        scaled_gamma = _safe_float(row["gamma"], 0.0) * (spot**2) * 0.01
        scaled_vega = _safe_float(row["vega"], 0.0) / max(spot * 0.01, 1e-6)

        cum_pnl_norm = self.cumulative_pnl / self.config.pnl_norm
        skew_proxy = np.clip(_safe_float(row["skew_proxy"], 0.0), -2.0, 2.0)
        vol_of_vol_proxy = np.clip(_safe_float(row["vol_of_vol_proxy"], 0.0), 0.0, 2.0)

        obs = np.array(
            [
                spot / strike,
                dte,
                iv,
                delta,
                scaled_gamma,
                scaled_vega,
                self.hedge_delta,
                self.hedge_vega,
                cum_pnl_norm,
                iv,
                skew_proxy,
                vol_of_vol_proxy,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _transaction_cost(self, row: pd.Series, new_delta: float, new_vega: float) -> float:
        """Compute transaction costs for changing hedge positions."""
        spread = max(_safe_float(row["ask"], 0.0) - _safe_float(row["bid"], 0.0), 0.0)
        spot = max(_safe_float(row["underlying_close"], 1.0), 1e-6)
        turnover = abs(new_delta - self.hedge_delta) + abs(new_vega - self.hedge_vega)
        notional = self.config.contract_multiplier * spot

        # Two simple cost components:
        #   1) spread proportional to option bid-ask spread
        #   2) bps proportional to notional turnover
        spread_cost = spread * self.config.contract_multiplier * 0.1 * turnover
        bps_cost = notional * (self.config.transaction_cost_bps / 10_000.0) * turnover
        return float(spread_cost + bps_cost)

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        """Advance one time step in the chain.

        Args:
            action: Array of shape (2,) containing adjustments in [-1, 1].

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        row_curr = self._row(self.current_step)
        row_next = self._row(self.current_step + 1)

        delta_adj = float(action[0]) * self.config.delta_trade_scale
        vega_adj = float(action[1]) * self.config.vega_trade_scale
        new_delta = float(
            np.clip(
                self.hedge_delta + delta_adj,
                -self.config.max_abs_hedge,
                self.config.max_abs_hedge,
            )
        )
        new_vega = float(
            np.clip(
                self.hedge_vega + vega_adj,
                -self.config.max_abs_hedge,
                self.config.max_abs_hedge,
            )
        )

        transaction_cost = self._transaction_cost(row_curr, new_delta, new_vega)

        self.hedge_delta = new_delta
        self.hedge_vega = new_vega

        mid_curr = _safe_float(row_curr["mark"], 0.0)
        mid_next = _safe_float(row_next["mark"], 0.0)
        spot_curr = _safe_float(row_curr["underlying_close"], 0.0)
        spot_next = _safe_float(row_next["underlying_close"], 0.0)
        iv_curr = _safe_float(row_curr["implied_volatility"], 0.0)
        iv_next = _safe_float(row_next["implied_volatility"], 0.0)
        vega_curr = _safe_float(row_curr["vega"], 0.0)

        option_pnl = -(mid_next - mid_curr) * self.config.contract_multiplier
        delta_hedge_pnl = self.hedge_delta * (spot_next - spot_curr) * self.config.contract_multiplier
        vega_hedge_pnl = self.hedge_vega * (iv_next - iv_curr) * vega_curr * self.config.contract_multiplier

        step_pnl = option_pnl + delta_hedge_pnl + vega_hedge_pnl
        self.last_step_pnl = float(step_pnl - transaction_cost)
        self.cumulative_pnl += self.last_step_pnl

        reward = self.reward_fn.compute(step_pnl=float(step_pnl), transaction_cost=float(transaction_cost))
        self.current_step += 1
        terminated = self.current_step >= self.n_steps
        truncated = False

        info: Dict[str, float] = {
            "step_pnl": self.last_step_pnl,
            "gross_step_pnl": float(step_pnl),
            "transaction_cost": float(transaction_cost),
            "cumulative_pnl": float(self.cumulative_pnl),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

