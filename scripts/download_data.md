# Data Download Guide

This project expects local parquet files under:

```text
data/
  <ticker>/
    options.parquet
    underlying.parquet
```

Dataset source:
- US Equity Options Dataset: [`philippdubach/options-data`](https://huggingface.co/datasets/philippdubach/options-data)

The dataset is large (around 9GB+), so it is intentionally not stored in this repository.

## Option A: Hugging Face UI (simple)

1. Open the dataset page.
2. Download files/folders you need.
3. Place them under `data/<ticker>/` so each ticker has:
   - `options.parquet`
   - `underlying.parquet`

## Option B: Python download script (cross-platform)

Install dependency:

```powershell
pip install huggingface_hub
```

Run:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='philippdubach/options-data', repo_type='dataset', local_dir='data', local_dir_use_symlinks=False)"
```

## Option C: Git LFS clone (advanced, requires git-lfs)

```bash
git lfs install
git clone https://huggingface.co/datasets/philippdubach/options-data data
```

## Verify structure quickly

PowerShell:

```powershell
Get-ChildItem data -Directory | Select-Object -First 5 | ForEach-Object {
  $_.FullName
  Get-ChildItem $_.FullName
}
```

Run schema validation before training:

```powershell
python scripts/verify_data.py --data-dir data
```

## Notes

- Keep `data/` local only; it is excluded by `.gitignore`.
- If memory is limited, subset to fewer tickers first.
- Project scripts will raise errors if required parquet files or columns are missing.
