# QR-PPO Deep Hedging on Empirical Option Chains

Distributional reinforcement learning framework for tail-risk-aware dynamic hedging.

This repository trains and evaluates:
- `QR-PPO (tail-aware)` with asymmetric reward shaping and quantile critic
- `Naive RL baselines` (PPO and SAC; mean-oriented reward)
- `Static Black-Scholes delta-vega hedger` baseline

The goal is to reduce severe downside outcomes (left-tail risk) measured by terminal CVaR.

## Contents
- [1. What this project does](#1-what-this-project-does)
- [2. Repository structure](#2-repository-structure)
- [3. Requirements](#3-requirements)
- [4. Installation](#4-installation)
- [5. Data source and layout](#5-data-source-and-layout)
- [6. Quick start](#6-quick-start)
- [7. Detailed experiment commands](#7-detailed-experiment-commands)
- [8. Method overview](#8-method-overview)
- [9. Reproducibility](#9-reproducibility)
- [10. Outputs and metrics](#10-outputs-and-metrics)
- [11. Development workflow](#11-development-workflow)
- [12. Troubleshooting](#12-troubleshooting)
- [13. Limitations and roadmap](#13-limitations-and-roadmap)
- [14. Citation](#14-citation)
- [15. License](#15-license)

## 1. What this project does

Deep hedging is cast as a sequential decision problem:
- state: market and position context from historical option chains
- action: continuous hedge adjustment in delta and vega dimensions
- objective: maximize net hedging quality while strongly penalizing tail losses

The core research claim:
> Distributional policy learning (`QR-PPO`) plus asymmetric downside penalties can reduce CVaR-relative tail damage vs static and mean-optimized baselines.

## 2. Repository structure

```text
QR_PPO_SABR/
  baselines.py          # static hedger + naive PPO/SAC wrappers
  data_utils.py         # sampling, chain building, preprocessing
  empirical_env.py      # historical hedging Gymnasium environment
  eval_only.py          # evaluate from saved checkpoints
  evaluate.py           # train+evaluate in one script
  qr_ppo.py             # quantile-regression PPO implementation
  random_chain_env.py   # random chain training environment
  risk_manager.py       # CVaR and asymmetric reward shaping
  run_experiment.py     # one-command orchestrator with archiving
  train.py              # training script (saves checkpoints)
  requirements.txt
```

Generated runtime folders (ignored by git):
- `checkpoints/`
- `results/`
- `experiments/`

## 3. Requirements

- Python 3.10+ (3.11 recommended)
- pip 23+
- Windows, Linux, or macOS

Core libraries:
- PyTorch
- Gymnasium
- Stable-Baselines3
- NumPy, pandas, SciPy, matplotlib, pyarrow

See `requirements.txt` for the complete install list.

## 4. Installation

### Option A: virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: global Python environment

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Data source and layout

This repository does **not** include raw market data (dataset is large, ~9GB+ parquet files).

Primary source:
- US Equity Options Dataset: [`philippdubach/options-data`](https://huggingface.co/datasets/philippdubach/options-data)

Download the dataset from the source above, then organize local files as:

Expected directory structure:

```text
data/
  <ticker>/
    options.parquet
    underlying.parquet
```

Assumed columns (minimum):
- `options.parquet`: `contract_id`, `symbol`, `expiration`, `strike`, `type`, `mark`, `bid`, `ask`, `date`, `implied_volatility`, `delta`, `gamma`, `vega`
- `underlying.parquet`: `symbol`, `date`, `close`

Notes:
- Data preprocessing enforces maturity filtering (`21 <= DTE <= 252`)
- Numeric coercion and missing-value fills are applied in `data_utils.py`
- `data/` is intentionally git-ignored to keep the repository lightweight
- Step-by-step download options are in `scripts/download_data.md`
- Validate schema before training: `python scripts/verify_data.py --data-dir data`

## 6. Quick start

### 1) Train models and save checkpoints

```powershell
python train.py --num-train-paths 80 --max-tickers-scan 104 --checkpoint-dir checkpoints
```

### 2) Evaluate from checkpoints

```powershell
python eval_only.py --num-eval-paths 120 --max-tickers-scan 104 --checkpoint-dir checkpoints --output-dir results --plot-tail
```

### 3) Single command (train + eval + archive)

```powershell
python run_experiment.py --num-train-paths 80 --num-eval-paths 120 --max-tickers-scan 104
```

## 7. Detailed experiment commands

### Fast smoke test

```powershell
python evaluate.py --num-eval-paths 2 --num-train-paths 2 --max-tickers-scan 10 --qrppo-total-updates 1 --qrppo-rollout-steps 64 --qrppo-update-epochs 1 --sb3-timesteps 200
```

### Fuller training budget

```powershell
python evaluate.py --num-eval-paths 120 --num-train-paths 80 --max-tickers-scan 104 --qrppo-total-updates 40 --qrppo-rollout-steps 1024 --qrppo-update-epochs 8 --sb3-timesteps 200000
```

### Reuse existing checkpoints

```powershell
python evaluate.py --max-tickers-scan 104 --reuse-checkpoints
```

### Save machine-readable report artifacts

```powershell
python evaluate.py --reuse-checkpoints --save-report --plot-tail --output-dir results
```

## 8. Method overview

### Environment (`empirical_env.py`)
- Observation: 12-dimensional state (moneyness, DTE, IV/greeks, hedge positions, P&L context, skew/vol-of-vol proxies)
- Action: 2D continuous adjustment (delta and vega hedge changes)
- Dynamics: empirical option mark and underlying close progression
- Costs: spread-based and turnover-based transaction components

### Risk shaping (`risk_manager.py`)
- Tail utility is encoded using asymmetric reward:
  - `reward = step_pnl - cost - tail_penalty * max(0, rolling_var - step_pnl)`
- Evaluation metric for claims:
  - terminal `CVaR95` (expected loss in worst 5% outcomes)

### Agent (`qr_ppo.py`)
- Actor: Gaussian policy + tanh squashing
- Critic: quantile distribution head (`n_quantiles=100`)
- Losses: PPO clipped objective + quantile Huber critic objective

## 9. Reproducibility

- Seed controls are exposed in scripts (`--seed`)
- Checkpoints support restart and eval-only workflows
- `run_experiment.py` snapshots results into a timestamped folder and stores a `README` copy with metadata context

For strict research protocols, add:
- fixed train/validation/test splits by time
- locked package versions
- per-run config serialization for every hyperparameter

## 10. Outputs and metrics

Typical output files:
- `results/evaluation_metrics.csv`
- `results/evaluation_metrics.json`
- `results/terminal_pnl_cdf.png` (if plotting enabled)

Metric fields include:
- `mean_terminal_pnl`
- `std_terminal_pnl`
- `var95`
- `cvar95`
- `tail_risk_reduction_vs_baseline`

## 11. Development workflow

### Run local sanity checks

```powershell
python -m compileall .
```

### Suggested git workflow

```powershell
git checkout -b feat/<short-topic>
git add .
git commit -m "Describe why this change matters"
```

### Recommended pull request checklist
- [ ] Smoke test command succeeds
- [ ] Evaluation command runs and writes artifacts
- [ ] No accidental large files or secrets included
- [ ] README and usage flags remain accurate

## 12. Troubleshooting

- `No chains found / built`:
  - Verify parquet schemas and date/expiration formats
  - Confirm data exists under `data/<ticker>/`
- CUDA not detected:
  - Training automatically falls back to CPU
- SB3 import errors:
  - Reinstall dependencies in a clean virtual environment
- Very slow training:
  - Reduce `--sb3-timesteps`, `--num-train-paths`, and QR-PPO update budgets for local tests

## 13. Limitations and roadmap

Current limitations:
- Vega P&L term uses linear approximation
- Hyperparameters are fixed and not auto-tuned
- No formal unit/integration test suite yet

Planned improvements:
- walk-forward validation pipeline
- richer execution-cost and slippage models
- exposure constraints and stress scenario replay
- automated hyperparameter sweeps and tracking

## 14. Citation

If this repository contributes to your work, cite it as:

```text
QR-PPO Deep Hedging on Empirical Option Chains, 2026.
```

## 15. License

This project is licensed under the MIT License. See `LICENSE`.
