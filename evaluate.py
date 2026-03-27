"""Macro-evaluation of deep hedging agents on empirical option chains.

This script:
  1) samples option chains from `data/` (multiple tickers)
  2) filters to maturities between 21 and 252 DTE
  3) runs QR-PPO, naive PPO/SAC baselines, and a static Black-Scholes hedger
  4) aggregates terminal P&L and reports global CVaR_95 reduction vs baseline
"""

from __future__ import annotations

import argparse
import os
import json
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baselines import BlackScholesDeltaVegaHedger, NaivePPOAgent, NaiveSACAgent, make_mean_reward
from data_utils import ChainSpec, SamplingConfig, build_chains_for_specs, sample_chain_specs
from empirical_env import EnvConfig, HistoricalHedgingEnv
from qr_ppo import QRPPOAgent, QRPPoConfig
from random_chain_env import RandomChainEnv
from risk_manager import AsymmetricReward, conditional_value_at_risk


def _run_episode(env: HistoricalHedgingEnv, action_fn: Callable[[np.ndarray], np.ndarray]) -> float:
    """Run one episode and return terminal cumulative P&L."""
    obs, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = action_fn(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
    return float(info["cumulative_pnl"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-eval-paths", type=int, default=120)
    parser.add_argument("--num-train-paths", type=int, default=80)
    parser.add_argument("--max-tickers-scan", type=int, default=60)

    # Training budgets (kept modest so the script runs end-to-end on one machine).
    parser.add_argument("--qrppo-total-updates", type=int, default=10)
    parser.add_argument("--qrppo-rollout-steps", type=int, default=512)
    parser.add_argument("--qrppo-update-epochs", type=int, default=4)

    parser.add_argument("--sb3-timesteps", type=int, default=25_000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--save-report", action="store_true")
    parser.add_argument("--plot-tail", action="store_true")
    args = parser.parse_args()

    env_cfg = EnvConfig()

    eval_specs = sample_chain_specs(
        data_dir=args.data_dir,
        cfg=SamplingConfig(max_paths=args.num_eval_paths, seed=args.seed, max_tickers_scan=args.max_tickers_scan),
    )
    train_specs = sample_chain_specs(
        data_dir=args.data_dir,
        cfg=SamplingConfig(max_paths=args.num_train_paths, seed=args.seed + 1, max_tickers_scan=args.max_tickers_scan),
    )
    if not eval_specs:
        raise RuntimeError("No evaluation chains found. Check parquet schemas and DTE filtering.")
    if not train_specs:
        raise RuntimeError("No training chains found. Check parquet schemas and DTE filtering.")

    # Prebuild chains for repeatable training/evaluation.
    train_built = build_chains_for_specs(train_specs, data_dir=args.data_dir, dte_min=21, dte_max=252)
    eval_built = build_chains_for_specs(eval_specs, data_dir=args.data_dir, dte_min=21, dte_max=252)
    if not eval_built:
        raise RuntimeError("Failed to build any evaluation chains.")

    rng = np.random.default_rng(args.seed)

    # Chain samplers for training envs.
    train_chains = [chain_df for _spec, chain_df in train_built]

    def chain_sampler_train() -> object:
        return train_chains[int(rng.integers(0, len(train_chains)))]

    # Reward shaping for QR-PPO.
    qr_reward = AsymmetricReward(tail_penalty=env_cfg.reward_tail_penalty)
    train_env_qr = RandomChainEnv(chain_sampler=chain_sampler_train, config=env_cfg, reward_fn=qr_reward)

    # Train QR-PPO.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qr_cfg = QRPPoConfig(
        obs_dim=12,
        action_dim=2,
        n_quantiles=100,
        rollout_steps=args.qrppo_rollout_steps,
        update_epochs=args.qrppo_update_epochs,
        device=device,
    )
    qr_agent = QRPPOAgent(qr_cfg)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    qr_ckpt = os.path.join(args.checkpoint_dir, "qrppo.pt")
    ppo_ckpt = os.path.join(args.checkpoint_dir, "naive_ppo")
    sac_ckpt = os.path.join(args.checkpoint_dir, "naive_sac")

    if args.reuse_checkpoints and os.path.exists(qr_ckpt):
        qr_agent.load(qr_ckpt)
    else:
        qr_agent.train(train_env_qr, total_updates=args.qrppo_total_updates)
        qr_agent.save(qr_ckpt)

    # Train naive PPO and SAC with mean reward shaping.
    mean_reward = make_mean_reward()
    train_env_mean = RandomChainEnv(chain_sampler=chain_sampler_train, config=env_cfg, reward_fn=mean_reward)
    naive_ppo = NaivePPOAgent()
    naive_sac = NaiveSACAgent()

    if args.reuse_checkpoints and os.path.exists(ppo_ckpt + ".zip"):
        naive_ppo.load(ppo_ckpt)
    else:
        naive_ppo.train(lambda: train_env_mean, total_timesteps=args.sb3_timesteps, seed=args.seed)
        naive_ppo.save(ppo_ckpt)

    if args.reuse_checkpoints and os.path.exists(sac_ckpt + ".zip"):
        naive_sac.load(sac_ckpt)
    else:
        naive_sac.train(lambda: train_env_mean, total_timesteps=args.sb3_timesteps, seed=args.seed)
        naive_sac.save(sac_ckpt)

    # Evaluation action functions.
    bs_hedger = BlackScholesDeltaVegaHedger(
        delta_trade_scale=env_cfg.delta_trade_scale,
        vega_trade_scale=env_cfg.vega_trade_scale,
        max_abs_hedge=env_cfg.max_abs_hedge,
    )
    action_baseline = lambda obs: bs_hedger.act(obs)
    action_qr = lambda obs: qr_agent.act(obs, deterministic=True)
    action_ppo = lambda obs: naive_ppo.act(obs, deterministic=True)
    action_sac = lambda obs: naive_sac.act(obs, deterministic=True)

    agents: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "Baseline: Delta-Vega BS": action_baseline,
        "Naive RL: PPO": action_ppo,
        "Naive RL: SAC": action_sac,
        "QR-PPO (tail-aware)": action_qr,
    }

    terminal_pnls: Dict[str, List[float]] = {name: [] for name in agents.keys()}

    for _spec, chain_df in eval_built:
        # Use the same P&L dynamics for all agents; reward shaping doesn't affect P&L tracking.
        env = HistoricalHedgingEnv(chain=chain_df, config=env_cfg)
        for name, act_fn in agents.items():
            pnl = _run_episode(env=env, action_fn=act_fn)
            terminal_pnls[name].append(pnl)

    # Compute global CVaR_95 from terminal P&L.
    cvars: Dict[str, float] = {}
    for name, pnl_list in terminal_pnls.items():
        cvars[name] = conditional_value_at_risk(np.asarray(pnl_list, dtype=np.float64), alpha=0.05)

    baseline_cvar = cvars["Baseline: Delta-Vega BS"]

    report_lines: List[str] = []
    report_lines.append("=" * 60)
    report_lines.append("GLOBAL EMPIRICAL TAIL-RISK REPORT (CVaR_95)")
    report_lines.append("=" * 60)
    for name, cvar in cvars.items():
        reduction = (baseline_cvar - cvar) / abs(baseline_cvar) if baseline_cvar != 0.0 else 0.0
        report_lines.append(f"{name:30s} CVaR95={cvar:12.2f}  Reduction={reduction:+8.1%}")

    print("\n".join(report_lines))

    if args.save_report:
        os.makedirs(args.output_dir, exist_ok=True)
        rows: List[Dict[str, float | str]] = []
        for name, pnl_list in terminal_pnls.items():
            arr = np.asarray(pnl_list, dtype=np.float64)
            cvar = cvars[name]
            reduction = (baseline_cvar - cvar) / abs(baseline_cvar) if baseline_cvar != 0.0 else 0.0
            rows.append(
                {
                    "agent": name,
                    "n_paths": int(arr.size),
                    "mean_terminal_pnl": float(arr.mean()) if arr.size else 0.0,
                    "std_terminal_pnl": float(arr.std()) if arr.size else 0.0,
                    "var95": float(np.quantile(arr, 0.05)) if arr.size else 0.0,
                    "cvar95": float(cvar),
                    "tail_risk_reduction_vs_baseline": float(reduction),
                }
            )

        csv_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
        json_path = os.path.join(args.output_dir, "evaluation_metrics.json")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nSaved metrics to: {csv_path} and {json_path}")

    if args.plot_tail:
        os.makedirs(args.output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        for name, pnl_list in terminal_pnls.items():
            arr = np.asarray(pnl_list, dtype=np.float64)
            if arr.size == 0:
                continue
            # CDF-based left-tail visibility for robust small-sample plotting.
            sorted_arr = np.sort(arr)
            cdf = np.linspace(0.0, 1.0, len(sorted_arr))
            plt.plot(sorted_arr, cdf, label=f"{name} (CVaR95={cvars[name]:.1f})")
        plt.title("Terminal P&L CDF (Left-tail comparison)")
        plt.xlabel("Terminal P&L")
        plt.ylabel("CDF")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.25)
        fig_path = os.path.join(args.output_dir, "terminal_pnl_cdf.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
        print(f"Saved plot to: {fig_path}")


if __name__ == "__main__":
    main()

