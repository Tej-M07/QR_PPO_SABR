"""Baselines and wrappers for naive hedging agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

from risk_manager import MeanReward


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    return norm.cdf(x)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return norm.pdf(x)


@dataclass(frozen=True)
class BlackScholesDeltaVegaHedger:
    """Static hedger that neutralizes delta and vega each step.

    This uses the observation state (which includes delta, IV, and DTE) and
    maps it to action space by driving the internal hedge positions toward
    delta- and vega-neutral targets.
    """

    delta_trade_scale: float = 0.2
    vega_trade_scale: float = 0.2
    max_abs_hedge: float = 2.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        # State layout in `HistoricalHedgingEnv._get_obs`:
        # [moneyness, dte_norm, iv, delta, scaled_gamma, scaled_vega,
        #  hedge_delta, hedge_vega, cum_pnl_norm, iv, skew_proxy, vol_of_vol_proxy]
        delta_opt = float(obs[3])
        iv = float(obs[2])
        dte_norm = float(obs[1])
        hedge_delta = float(obs[6])
        hedge_vega = float(obs[7])

        # Infer option type from the sign of delta (heuristic, but consistent with BS deltas).
        # Call deltas are typically positive; put deltas negative.
        is_call = delta_opt >= 0.0

        # Black-Scholes delta using forward moneyness F/K.
        moneyness = float(obs[0])
        t = max(dte_norm * 252.0, 1e-6) / 252.0  # convert to years-like fraction (rough).
        sigma = max(iv, 1e-6)
        ln_fk = np.log(max(moneyness, 1e-6))
        denom = sigma * np.sqrt(max(t, 1e-6))
        d1 = (ln_fk + 0.5 * sigma * sigma * t) / max(denom, 1e-12)

        if is_call:
            delta_bs = float(_norm_cdf(np.asarray(d1)))
        else:
            # Put delta (r=0): N(d1) - 1
            delta_bs = float(_norm_cdf(np.asarray(d1)) - 1.0)

        # Target hedges: neutralize delta by offsetting option delta.
        target_hedge_delta = float(-delta_bs)

        # Target hedges for vega: neutralize linear vega exposure.
        # Since the environment's vega hedge term multiplies `hedge_vega` by the
        # option's own vega scalar, setting hedge_vega=-1 neutralizes vega in the
        # same linearization used by the step P&L.
        _vega_bs = float(_norm_pdf(np.asarray(d1)) * np.sqrt(max(t, 1e-6)))
        target_hedge_vega = -1.0 if abs(_vega_bs) > 1e-12 else 0.0

        new_delta = float(np.clip(target_hedge_delta, -self.max_abs_hedge, self.max_abs_hedge))
        new_vega = float(np.clip(target_hedge_vega, -self.max_abs_hedge, self.max_abs_hedge))

        # Convert hedge targets to action adjustments.
        delta_action = (new_delta - hedge_delta) / self.delta_trade_scale
        vega_action = (new_vega - hedge_vega) / self.vega_trade_scale
        return np.clip(np.array([delta_action, vega_action], dtype=np.float32), -1.0, 1.0)


def make_mean_reward() -> MeanReward:
    """Factory for the mean-optimized reward used by naive RL."""
    return MeanReward()


def make_delta_vega_baseline_action_fn() -> Callable[[np.ndarray], np.ndarray]:
    """Create a baseline action function for evaluation."""
    hedger = BlackScholesDeltaVegaHedger()
    return lambda obs: hedger.act(obs)


class _SB3AgentBase:
    """Small wrapper around Stable-Baselines3 continuous-control agents."""

    def __init__(self) -> None:
        self.model: Any | None = None

    def train(
        self,
        env_fn: Callable[[], Any],
        total_timesteps: int,
        seed: int = 0,
        **learn_kwargs: Any,
    ) -> None:
        raise NotImplementedError

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")
        action, _state = self.model.predict(obs, deterministic=deterministic)
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim == 2 and action_arr.shape[0] == 1:
            return action_arr[0]
        return action_arr

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")
        self.model.save(path)

    def load(self, path: str) -> None:
        raise NotImplementedError


class NaivePPOAgent(_SB3AgentBase):
    """Expected-value PPO baseline (uses SB3 critic loss, i.e., MSE-like)."""

    def __init__(self, **ppo_kwargs: Any) -> None:
        super().__init__()
        self.ppo_kwargs = ppo_kwargs

    def train(
        self,
        env_fn: Callable[[], Any],
        total_timesteps: int,
        seed: int = 0,
        **learn_kwargs: Any,
    ) -> None:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec_env = DummyVecEnv([env_fn])
        self.model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0, **self.ppo_kwargs)
        self.model.learn(total_timesteps=total_timesteps, **learn_kwargs)

    def load(self, path: str) -> None:
        from stable_baselines3 import PPO

        self.model = PPO.load(path)


class NaiveSACAgent(_SB3AgentBase):
    """Expected-value SAC baseline (uses SB3 objective, i.e., MSE-like)."""

    def __init__(self, **sac_kwargs: Any) -> None:
        super().__init__()
        self.sac_kwargs = sac_kwargs

    def train(
        self,
        env_fn: Callable[[], Any],
        total_timesteps: int,
        seed: int = 0,
        **learn_kwargs: Any,
    ) -> None:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec_env = DummyVecEnv([env_fn])
        self.model = SAC("MlpPolicy", vec_env, seed=seed, verbose=0, **self.sac_kwargs)
        self.model.learn(total_timesteps=total_timesteps, **learn_kwargs)

    def load(self, path: str) -> None:
        from stable_baselines3 import SAC

        self.model = SAC.load(path)

