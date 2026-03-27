"""End-to-end experiment runner: train, evaluate, and archive outputs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime


def _run(cmd: list[str], cwd: str) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-train-paths", type=int, default=80)
    parser.add_argument("--num-eval-paths", type=int, default=120)
    parser.add_argument("--max-tickers-scan", type=int, default=104)
    parser.add_argument("--qrppo-total-updates", type=int, default=10)
    parser.add_argument("--qrppo-rollout-steps", type=int, default=512)
    parser.add_argument("--qrppo-update-epochs", type=int, default=4)
    parser.add_argument("--sb3-timesteps", type=int, default=25_000)
    parser.add_argument("--root-output-dir", type=str, default="experiments")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--source-checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.root_output_dir, f"exp_{ts}")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    result_dir = os.path.join(exp_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    cwd = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable

    if not args.skip_train:
        train_cmd = [
            py,
            "train.py",
            "--data-dir",
            args.data_dir,
            "--seed",
            str(args.seed),
            "--num-train-paths",
            str(args.num_train_paths),
            "--max-tickers-scan",
            str(args.max_tickers_scan),
            "--qrppo-total-updates",
            str(args.qrppo_total_updates),
            "--qrppo-rollout-steps",
            str(args.qrppo_rollout_steps),
            "--qrppo-update-epochs",
            str(args.qrppo_update_epochs),
            "--sb3-timesteps",
            str(args.sb3_timesteps),
            "--checkpoint-dir",
            ckpt_dir,
        ]
        _run(train_cmd, cwd=cwd)
    else:
        # Reuse existing checkpoints for evaluation-only experiments.
        src = args.source_checkpoint_dir
        required = ["qrppo.pt", "naive_ppo.zip", "naive_sac.zip"]
        missing = [name for name in required if not os.path.exists(os.path.join(src, name))]
        if missing:
            raise FileNotFoundError(f"Missing checkpoints in '{src}': {missing}")
        for name in required:
            shutil.copyfile(os.path.join(src, name), os.path.join(ckpt_dir, name))

    eval_cmd = [
        py,
        "eval_only.py",
        "--data-dir",
        args.data_dir,
        "--seed",
        str(args.seed),
        "--num-eval-paths",
        str(args.num_eval_paths),
        "--max-tickers-scan",
        str(args.max_tickers_scan),
        "--checkpoint-dir",
        ckpt_dir,
        "--output-dir",
        result_dir,
    ]
    if not args.no_plot:
        eval_cmd.append("--plot-tail")
    _run(eval_cmd, cwd=cwd)

    # Snapshot README for reproducibility context.
    readme_src = os.path.join(cwd, "README.md")
    readme_dst = os.path.join(exp_dir, "README_snapshot.md")
    if os.path.exists(readme_src):
        shutil.copyfile(readme_src, readme_dst)

    print("\nExperiment complete.")
    print(f"Output directory: {exp_dir}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()

