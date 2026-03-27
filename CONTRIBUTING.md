# Contributing Guide

Thanks for contributing.

## Workflow

1. Fork the repository.
2. Create a topic branch from `main`.
3. Make focused changes with clear commit messages.
4. Open a pull request with:
   - problem statement
   - approach and trade-offs
   - validation steps and outputs

## Local setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Validation before PR

- Run a smoke check:

```powershell
python evaluate.py --num-eval-paths 2 --num-train-paths 2 --max-tickers-scan 10 --qrppo-total-updates 1 --qrppo-rollout-steps 64 --qrppo-update-epochs 1 --sb3-timesteps 200
```

- Confirm generated artifacts are not committed (`checkpoints/`, `results/`, `experiments/`, `data/`).
- Update `README.md` if behavior or command flags changed.

## Coding standards

- Prefer clear, explicit names over clever abstractions.
- Keep scripts reproducible and configurable via CLI arguments.
- Maintain backward compatibility for existing command flags when possible.
- Add concise docstrings for public functions/classes.

## Commit message style

Use imperative mood and include intent:
- `add eval-only checkpoint loading`
- `fix DTE filtering when expiration parsing fails`
- `document quick-start experiment commands`
