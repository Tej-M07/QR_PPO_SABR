"""Evaluate saved checkpoints and export metrics/plots."""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baselines import BlackScholesDeltaVegaHedger, NaivePPOAgent, NaiveSACAgent
from data_utils import SamplingConfig, build_chains_for_specs, sample_chain_specs
from empirical_env import EnvConfig, HistoricalHedgingEnv
from qr_ppo import QRPPOAgent, QRPPoConfig
from risk_manager import conditional_value_at_risk


def run_episode(env: HistoricalHedgingEnv, action_fn: Callable[[np.ndarray], np.ndarray]) -> float:
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _r, terminated, truncated, info = env.step(action_fn(obs))
        done = terminated or truncated
    return float(info["cumulative_pnl"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-eval-paths", type=int, default=120)
    parser.add_argument("--max-tickers-scan", type=int, default=104)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--plot-tail", action="store_true")
    args = parser.parse_args()

    qr_path = os.path.join(args.checkpoint_dir, "qrppo.pt")
    ppo_path = os.path.join(args.checkpoint_dir, "naive_ppo.zip")
    sac_path = os.path.join(args.checkpoint_dir, "naive_sac.zip")
    for path in (qr_path, ppo_path, sac_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    # Load models.
    qr = QRPPOAgent(
        QRPPoConfig(obs_dim=12, action_dim=2, n_quantiles=100, device="cuda" if torch.cuda.is_available() else "cpu")
    )
    qr.load(qr_path)
    ppo = NaivePPOAgent()
    sac = NaiveSACAgent()
    ppo.load(os.path.join(args.checkpoint_dir, "naive_ppo"))
    sac.load(os.path.join(args.checkpoint_dir, "naive_sac"))

    # Build eval chains.
    specs = sample_chain_specs(
        data_dir=args.data_dir,
        cfg=SamplingConfig(max_paths=args.num_eval_paths, seed=args.seed, max_tickers_scan=args.max_tickers_scan),
    )
    built = build_chains_for_specs(specs, data_dir=args.data_dir, dte_min=21, dte_max=252)
    if not built:
        raise RuntimeError("No evaluation chains built.")

    env_cfg = EnvConfig()
    bs = BlackScholesDeltaVegaHedger(
        delta_trade_scale=env_cfg.delta_trade_scale,
        vega_trade_scale=env_cfg.vega_trade_scale,
        max_abs_hedge=env_cfg.max_abs_hedge,
    )

    agents: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "Baseline: Delta-Vega BS": lambda obs: bs.act(obs),
        "Naive RL: PPO": lambda obs: ppo.act(obs, deterministic=True),
        "Naive RL: SAC": lambda obs: sac.act(obs, deterministic=True),
        "QR-PPO (tail-aware)": lambda obs: qr.act(obs, deterministic=True),
    }

    terminal: Dict[str, List[float]] = {k: [] for k in agents}
    for _spec, chain_df in built:
        env = HistoricalHedgingEnv(chain_df, config=env_cfg)
        for name, fn in agents.items():
            terminal[name].append(run_episode(env, fn))

    cvars = {k: conditional_value_at_risk(np.asarray(v, dtype=np.float64), alpha=0.05) for k, v in terminal.items()}
    base = cvars["Baseline: Delta-Vega BS"]

    print("=" * 60)
    print("EVAL-ONLY GLOBAL EMPIRICAL TAIL-RISK REPORT (CVaR_95)")
    print("=" * 60)
    rows: List[Dict[str, float | str]] = []
    for name, cvar in cvars.items():
        red = (base - cvar) / abs(base) if base != 0.0 else 0.0
        print(f"{name:30s} CVaR95={cvar:12.2f}  Reduction={red:+8.1%}")
        arr = np.asarray(terminal[name], dtype=np.float64)
        rows.append(
            {
                "agent": name,
                "n_paths": int(arr.size),
                "mean_terminal_pnl": float(arr.mean()) if arr.size else 0.0,
                "std_terminal_pnl": float(arr.std()) if arr.size else 0.0,
                "var95": float(np.quantile(arr, 0.05)) if arr.size else 0.0,
                "cvar95": float(cvar),
                "tail_risk_reduction_vs_baseline": float(red),
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "eval_only_metrics.csv"), index=False)
    with open(os.path.join(args.output_dir, "eval_only_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if args.plot_tail:
        plt.figure(figsize=(10, 6))
        for name, pnl_list in terminal.items():
            arr = np.sort(np.asarray(pnl_list, dtype=np.float64))
            if arr.size == 0:
                continue
            cdf = np.linspace(0.0, 1.0, len(arr))
            plt.plot(arr, cdf, label=f"{name} (CVaR95={cvars[name]:.1f})")
        plt.title("Terminal P&L CDF (Eval-only)")
        plt.xlabel("Terminal P&L")
        plt.ylabel("CDF")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "eval_only_terminal_pnl_cdf.png"), dpi=180)
        plt.close()


if __name__ == "__main__":
    main()

