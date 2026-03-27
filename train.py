"""Train QR-PPO, PPO, and SAC models and save checkpoints."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from baselines import NaivePPOAgent, NaiveSACAgent, make_mean_reward
from data_utils import SamplingConfig, build_chains_for_specs, sample_chain_specs
from empirical_env import EnvConfig
from qr_ppo import QRPPOAgent, QRPPoConfig
from random_chain_env import RandomChainEnv
from risk_manager import AsymmetricReward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-train-paths", type=int, default=80)
    parser.add_argument("--max-tickers-scan", type=int, default=60)
    parser.add_argument("--qrppo-total-updates", type=int, default=10)
    parser.add_argument("--qrppo-rollout-steps", type=int, default=512)
    parser.add_argument("--qrppo-update-epochs", type=int, default=4)
    parser.add_argument("--sb3-timesteps", type=int, default=25_000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    env_cfg = EnvConfig()
    rng = np.random.default_rng(args.seed)

    train_specs = sample_chain_specs(
        data_dir=args.data_dir,
        cfg=SamplingConfig(max_paths=args.num_train_paths, seed=args.seed, max_tickers_scan=args.max_tickers_scan),
    )
    train_built = build_chains_for_specs(train_specs, data_dir=args.data_dir, dte_min=21, dte_max=252)
    if not train_built:
        raise RuntimeError("No training chains built.")

    train_chains = [df for _spec, df in train_built]

    def sampler():
        return train_chains[int(rng.integers(0, len(train_chains)))]

    # Train QR-PPO (tail-aware)
    qr_agent = QRPPOAgent(
        QRPPoConfig(
            obs_dim=12,
            action_dim=2,
            n_quantiles=100,
            rollout_steps=args.qrppo_rollout_steps,
            update_epochs=args.qrppo_update_epochs,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    qr_env = RandomChainEnv(sampler, config=env_cfg, reward_fn=AsymmetricReward(tail_penalty=env_cfg.reward_tail_penalty))
    qr_agent.train(qr_env, total_updates=args.qrppo_total_updates)
    qr_agent.save(os.path.join(args.checkpoint_dir, "qrppo.pt"))

    # Train naive PPO/SAC (mean-focused)
    mean_reward = make_mean_reward()
    mean_env = RandomChainEnv(sampler, config=env_cfg, reward_fn=mean_reward)
    ppo = NaivePPOAgent()
    sac = NaiveSACAgent()
    ppo.train(lambda: mean_env, total_timesteps=args.sb3_timesteps, seed=args.seed)
    sac.train(lambda: mean_env, total_timesteps=args.sb3_timesteps, seed=args.seed)
    ppo.save(os.path.join(args.checkpoint_dir, "naive_ppo"))
    sac.save(os.path.join(args.checkpoint_dir, "naive_sac"))

    print(f"Training complete. Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()

