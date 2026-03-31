# aha_precedes_strategy_VMR

Code and data for the Current Biology manuscript workflow.

This repository is organized around the main manuscript analysis path. The main
entry points are:

1. `DataProcessing.ipynb`
2. `inferentialStats.ipynb`
3. `modelFittingAllParticipants.ipynb`
4. `RupturesAhaResubmitPaperFigures.ipynb`

## Setup

Use Python 3.11.

Install packages with:

```bash
pip install -r requirements.txt
```

`jupyterlab` and `ipykernel` are included in `requirements.txt` so the notebooks
can be run directly after installation.

## Main Workflow

### 1. Data processing

Run `DataProcessing.ipynb`.

This notebook reads the raw task CSVs and writes the processed data files used
by the later notebooks:

- `CGData_.csv`
- `CGDataWithOutliers_.csv`
- `BTData_.csv`

### 2. Inferential statistics

Run `inferentialStats.ipynb`.

This notebook generates the main inferential statistics from the processed CSV
files.

### 3. Model fitting

Run `modelFittingAllParticipants.ipynb`.

This notebook fits the manuscript models and writes the `.npy` fit files used
by the main figure notebook. A full rerun can take a long time.

### 4. Figures and manuscript analyses

Run `RupturesAhaResubmitPaperFigures.ipynb`.

This notebook generates the main figure panels and manuscript analyses. It also
calls the helper scripts:

- `ChangepointFitsnPlots.py`
- `ModelChangepointFits.py`
- `DingChangepoint.py`

These helper `.py` files should remain in the repository root, because the
notebooks import or execute them using the current relative-path layout.

## Full Recompute vs Fast Reproduction

There are two reasonable ways to use this repository.

### Full recompute

Run the workflow in this order:

1. `DataProcessing.ipynb`
2. `inferentialStats.ipynb`
3. `modelFittingAllParticipants.ipynb`
4. `RupturesAhaResubmitPaperFigures.ipynb`

This is the code-and-data-only route, but model fitting may take days.

### Fast reproduction

Download cached fit artifacts from OSF and place them in the repository root,
then run `RupturesAhaResubmitPaperFigures.ipynb`.

The OSF project for cached artifacts is:

- `https://osf.io/h4m9a/`

That OSF project contains the large optional `.npy` and `.pkl` files that
save long recomputation time but are not strictly required to regenerate the
results from code and data alone.

The notebooks currently expect downloaded cache files in the repository root.

## Data Notes

The essential repository contents are the CSV data files and the code.

For convenience, some processed CSVs are kept in the workflow because they are
used directly by the main notebooks. In particular, the Ding supplementary path
uses `ding_reformatted.csv`, which can also be regenerated from
`2025-12_Ding_Data.csv` via `DingChangepoint.py`.

## Legacy Materials

Older notebooks and scripts are retained for transparency, but they are not the
main workflow for this repository.

- `README.txt` contains the original repo notes for the earlier workflow.
- `docs/legacy_workflow.md` preserves those notes in Markdown form.
- `docs/repo_triage.md` records the manuscript-driven classification of core vs
  legacy materials.

## Contact

For questions about the repository, contact `max.o.b.townsend@bath.edu`.
