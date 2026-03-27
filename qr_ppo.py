"""Quantile Regression PPO (QR-PPO) implementation.

This module implements:
  - Gaussian tanh-squashed actor policy
  - Distributional critic predicting N quantiles
  - PPO-style clipped policy optimization using scalar advantages
  - Quantile Huber loss for the critic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


def _mlp(layer_sizes: List[int], activation: nn.Module | None = None) -> nn.Sequential:
    """Construct a simple MLP."""
    if activation is None:
        activation = nn.Tanh()
    layers: List[nn.Module] = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(activation)
    return nn.Sequential(*layers)


class TanhGaussianActor(nn.Module):
    """Gaussian policy with tanh squashing (for bounded continuous actions)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.backbone = _mlp([obs_dim, *hidden_sizes])
        last_hidden = hidden_sizes[-1]
        self.mean_head = nn.Linear(last_hidden, action_dim)
        self.log_std_head = nn.Linear(last_hidden, action_dim)

    def _dist(self, obs: torch.Tensor) -> Normal:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), -5.0, 2.0)
        std = torch.exp(log_std)
        return Normal(mean, std)

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        # numerically stable atanh
        x = torch.clamp(x, -0.999999, 0.999999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob)."""
        dist = self._dist(obs)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)

        # Log prob with change-of-variables term for tanh.
        # log pi(a|s) = log N(pre_tanh) - log|da/dpre_tanh|
        log_prob = dist.log_prob(pre_tanh)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log_prob of a given action."""
        dist = self._dist(obs)
        pre_tanh = self._atanh(action)
        log_prob = dist.log_prob(pre_tanh)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action using tanh of the mean."""
        dist = self._dist(obs)
        return torch.tanh(dist.mean)


class QuantileCritic(nn.Module):
    """Distributional critic returning N quantiles."""

    def __init__(
        self,
        obs_dim: int,
        n_quantiles: int = 100,
        hidden_sizes: Tuple[int, ...] = (512, 512, 256),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_quantiles = n_quantiles
        self.backbone = _mlp([obs_dim, *hidden_sizes])
        last_hidden = hidden_sizes[-1]
        self.q_head = nn.Linear(last_hidden, n_quantiles)

        taus = (2 * torch.arange(n_quantiles, dtype=torch.float32) + 1) / (2 * n_quantiles)
        self.register_buffer("taus", taus)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return quantile estimates with shape (batch, n_quantiles)."""
        features = self.backbone(obs)
        return self.q_head(features)


def quantile_huber_loss(
    pred_quantiles: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile Huber loss for scalar targets.

    pred_quantiles: (batch, N)
    target: (batch,)
    taus: (N,)
    """
    # target broadcast to (batch, N)
    target_b = target.unsqueeze(-1).expand_as(pred_quantiles)
    error = target_b - pred_quantiles
    abs_error = torch.abs(error)
    huber = torch.where(abs_error <= kappa, 0.5 * error.pow(2), kappa * (abs_error - 0.5 * kappa))
    # Indicator: I[error < 0]
    weight = torch.abs(taus.unsqueeze(0) - (error.detach() < 0.0).float())
    return (weight * huber).mean()


@dataclass(frozen=True)
class QRPPOTrajectory:
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    last_value: float
    final_obs: np.ndarray


@dataclass(frozen=True)
class QRPPoConfig:
    obs_dim: int = 12
    action_dim: int = 2
    n_quantiles: int = 100
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    device: str = "cuda"


class QRPPOAgent:
    """QR-PPO agent with quantile regression critic."""

    def __init__(self, cfg: QRPPoConfig) -> None:
        self.cfg = cfg
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.device = device
        self.actor = TanhGaussianActor(cfg.obs_dim, cfg.action_dim).to(device)
        self.critic = QuantileCritic(cfg.obs_dim, cfg.n_quantiles).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, eps=1e-5)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            action = self.actor.mean_action(obs_t)
        else:
            action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def value_mean(self, obs: torch.Tensor) -> torch.Tensor:
        quantiles = self.critic(obs)
        return quantiles.mean(dim=-1)

    @torch.no_grad()
    def collect_trajectory(self, env, obs: np.ndarray) -> QRPPOTrajectory:
        cfg = self.cfg
        obs_buf = np.zeros((cfg.rollout_steps, cfg.obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.rollout_steps, cfg.action_dim), dtype=np.float32)
        logp_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        rew_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        done_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        val_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)

        o = obs
        last_done = False
        last_value = 0.0

        for t in range(cfg.rollout_steps):
            obs_t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob = self.actor.sample(obs_t)
            value_mean = self.critic(  # distributional critic, mean for GAE
                obs_t
            ).mean(dim=-1)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _info = env.step(action_np)
            done = terminated or truncated

            obs_buf[t] = o
            act_buf[t] = action_np
            logp_buf[t] = log_prob.item()
            rew_buf[t] = reward
            done_buf[t] = float(done)
            val_buf[t] = float(value_mean.item())

            o = next_obs
            last_done = done
            if done:
                # reset env for next trajectory chunk
                next_obs, _ = env.reset()
                o = next_obs

        # Estimate value for final state.
        obs_t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
        last_value = float(self.value_mean(obs_t).item())

        return QRPPOTrajectory(
            obs=obs_buf,
            actions=act_buf,
            log_probs=logp_buf,
            rewards=rew_buf,
            dones=done_buf,
            values=val_buf,
            last_value=last_value,
            final_obs=o.astype(np.float32, copy=False),
        )

    def _gae_advantages(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation (scalar)."""
        cfg = self.cfg
        n = len(rewards)
        adv = np.zeros(n, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(n)):
            non_terminal = 1.0 - dones[t]
            next_value = last_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + cfg.gamma * next_value * non_terminal - values[t]
            last_adv = delta + cfg.gamma * cfg.gae_lambda * non_terminal * last_adv
            adv[t] = last_adv
        returns = adv + values
        return adv, returns

    def train(
        self,
        env,
        total_updates: int = 50,
    ) -> None:
        """Train QR-PPO in an on-policy manner."""
        cfg = self.cfg
        obs, _ = env.reset()

        for update in range(total_updates):
            traj = self.collect_trajectory(env, obs)
            obs = traj.final_obs

            adv, returns = self._gae_advantages(
                rewards=traj.rewards,
                dones=traj.dones,
                values=traj.values,
                last_value=traj.last_value,
            )

            # Normalize advantages for stability.
            adv_mean = float(adv.mean())
            adv_std = float(adv.std() + 1e-8)
            adv = (adv - adv_mean) / adv_std

            # Flatten to dataset.
            dataset = {
                "obs": traj.obs,
                "actions": traj.actions,
                "log_probs": traj.log_probs,
                "adv": adv,
                "returns": returns,
            }

            n = cfg.rollout_steps
            idxs = np.arange(n)
            for _epoch in range(cfg.update_epochs):
                np.random.shuffle(idxs)
                for start in range(0, n, cfg.minibatch_size):
                    mb_idx = idxs[start : start + cfg.minibatch_size]

                    obs_mb = torch.as_tensor(dataset["obs"][mb_idx], dtype=torch.float32, device=self.device)
                    act_mb = torch.as_tensor(dataset["actions"][mb_idx], dtype=torch.float32, device=self.device)
                    old_logp_mb = torch.as_tensor(dataset["log_probs"][mb_idx], dtype=torch.float32, device=self.device)
                    adv_mb = torch.as_tensor(dataset["adv"][mb_idx], dtype=torch.float32, device=self.device)
                    ret_mb = torch.as_tensor(dataset["returns"][mb_idx], dtype=torch.float32, device=self.device)

                    # Actor loss (PPO clipped objective)
                    new_logp = self.actor.log_prob(obs_mb, act_mb)
                    ratio = torch.exp(new_logp - old_logp_mb)
                    unclipped = ratio * adv_mb
                    clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv_mb
                    policy_loss = -torch.mean(torch.min(unclipped, clipped))

                    # Entropy term (optional)
                    with torch.no_grad():
                        # approximate entropy from std via Normal distribution mean/std
                        # (kept simple; avoids expensive sampling)
                        dist = self.actor._dist(obs_mb)
                        entropy = dist.entropy().sum(dim=-1).mean()

                    # Critic loss (quantile regression)
                    pred_quantiles = self.critic(obs_mb)  # (batch, N)
                    critic_loss = quantile_huber_loss(
                        pred_quantiles=pred_quantiles,
                        target=ret_mb,
                        taus=self.critic.taus,
                    )

                    total_loss = policy_loss + cfg.vf_coef * critic_loss - cfg.ent_coef * entropy

                    self.actor_opt.zero_grad(set_to_none=True)
                    self.critic_opt.zero_grad(set_to_none=True)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                    self.actor_opt.step()
                    self.critic_opt.step()

            # External evaluation hooks can be added by the caller.

    def save(self, path: str) -> None:
        """Save actor/critic and optimizers to a checkpoint file."""
        payload = {
            "cfg": self.cfg.__dict__,
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "actor_opt_state": self.actor_opt.state_dict(),
            "critic_opt_state": self.critic_opt.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        """Load actor/critic and optimizers from a checkpoint file."""
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor_state"])
        self.critic.load_state_dict(payload["critic_state"])
        self.actor_opt.load_state_dict(payload["actor_opt_state"])
        self.critic_opt.load_state_dict(payload["critic_opt_state"])

