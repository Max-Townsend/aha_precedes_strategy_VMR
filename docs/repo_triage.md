## Repo Triage Notes

This note summarizes a manuscript-driven classification of the current workspace.
It is intended to guide a low-risk cleanup for the current paper workflow
without requiring a cell-by-cell audit of every notebook.

### Main inference

Based on the final manuscript draft and the current notebook dependencies:

- The main paper workflow appears to be centered on `ruptures`/PELT changepoint
  detection, not the older standard-deviation-breach approach.
- The Ding dataset is described in the manuscript as supplementary.
- Several overlapping notebooks appear to be variants or supersets rather
  than distinct end-to-end pipelines.

### Recommended main manuscript workflow

Keep these as the core public reproduction path:

- `DataProcessing.ipynb`
- `inferentialStats.ipynb`
- `modelFittingAllParticipants.ipynb`
- `RupturesAhaResubmitPaperFigures.ipynb`

Likely core supporting scripts for this path:

- `DataManipulation.py`
- `StepFunctionModelWithNoise.py`
- `NonGenSSMWithNoise.py`
- `NonGenDualProcessSSMWithNoise.py`
- `DualProcessKnownImplicit.py`
- `HMM.py`
- `HMMImplicit.py`
- `HMMKnownImplicit.py`
- `QLearning.py`
- `QLearningImplicit.py`
- `QLearningKnownImplicit.py`
- `ChangepointFitsnPlots.py`
- `ModelChangepointFits.py`
- `DingChangepoint.py`

Likely core data inputs:

- `oneTarBaseTask.csv`
- `eightTarBaseTask.csv`
- `oneTarSSMTask.csv`
- `eightTarSSMTask.csv`
- `demogPaper.csv`
- `CGData_.csv`
- `CGDataWithOutliers_.csv`
- `BTData_.csv`
- `BrudnerData.csv`
- `2025-12_Ding_Data.csv`
- `ding_reformatted.csv`
- `ding_modified.csv`

### Keep, but treat as optional cached artifacts

These are useful for a fast reproduction route, but should not be required for
the code-and-data-only route:

- `data.pkl`
- large fitted `.npy` files such as:
  - `HMMImpCG8AllPPs.npy`
  - `HMMVanilla8AllPPs.npy`
  - `QLearningImpCGAllPPs.npy`
  - `QLearningVanilla8AllPPs.npy`
  - `HMMDing.npy`
  - `QLearningDing.npy`
  - `SSMDing.npy`
  - `dualssmsVanilla8AllPPs.npy`
  - `ssmsAutocorrect8AllPPs.npy`
  - `HMMImplicitBTAllPPs.npy`
  - `QLearningImplicitBTAllPPs.npy`

Recommendation:

- Store these on OSF.
- Link them from the main `README.md`.
- Provide a short manifest describing what each cache file contains.

### Keep in repo, but archive/legacy rather than main path

These appear to be older alternatives, supersets, or exploratory notebooks that
do not need to be the public entry point:

- `ResubmitFigures.ipynb`
- `ResubmitPaperFigures.ipynb`
- `STDBreachAhaResubmitPaperFigures.ipynb`
- `paperFigures.ipynb`
- `ARCHIVEPaperFigure.ipynb`
- `modelFitting.ipynb`
- `previous modelling archive.py`
- older or superseded model scripts not described in the final paper

Recommended handling:

- Move them to an `archive/` or `legacy/` area.
- Do not delete unless they are clearly duplicated and no longer useful.
- Mention in the README that archived materials are retained for transparency but
  are not required for the main reproduction path.

### Generated outputs to ignore in Git

These are outputs or caches and should not live in the main Git history:

- `paperFigures/`
- `indiPlots/`
- `indiPlotsDing/`
- `indiPlotsNoModel/`
- `indiPlotsSignConfidence/`
- `dingIndiPlotsModel/`
- `phasePlots/`
- `dingPhasePlots/`
- `ahaPhasePlots/`
- `wholePPC_perPart/`
- `dingWholePPC_perPart/`
- `individualModelDistPlots/`
- `dingIndividualModelDistPlots/`
- `model_recovery_figs/`
- `param_recovery/`
- `tempFigures/` if recreated locally
- `.ipynb_checkpoints/`
- `.virtual_documents/`
- `__pycache__/`

### Practical cleanup strategy

The safest cleanup is a curation pass, not a deep refactor:

1. Keep the main execution path close to its current layout.
2. Write a new top-level `README.md` around the manuscript workflow.
3. Archive variant notebooks and older manuscript materials.
4. Ignore generated outputs and notebook checkpoints.
5. Put large cached artifacts on OSF and link them from the README.

This avoids changing save/load paths across many notebook cells before the repo
is stabilized.

### Dependency confirmation

I traced the read/write dependencies across the main workflow and did
not find any dependence on the archived notebooks.

High-confidence main workflow closure:

1. `DataProcessing.ipynb` reads the raw task CSVs and writes:
   - `CGData_.csv`
   - `BTData_.csv`
   - `CGDataWithOutliers_.csv`
2. `inferentialStats.ipynb` reads those processed CSVs for the manuscript
   statistics.
3. `modelFittingAllParticipants.ipynb` reads:
   - `CGData_.csv`
   - `BTData_.csv`
   - `BrudnerData.csv`
   - Ding processed data
   and writes the `.npy` fit files consumed by the main figure notebook.
4. `RupturesAhaResubmitPaperFigures.ipynb` reads those fit files, writes
   `data.pkl` internally, and calls:
   - `ChangepointFitsnPlots.py`
   - `ModelChangepointFits.py`
   - `DingChangepoint.py`
   to generate:
   - `changepoints.csv`
   - `model_changepoints.csv`
   - `ding_changepoints.csv`

Important caveat:

- The Ding supplementary path is the only part that prevents this from being a
  strict "four notebooks only" workflow.
- `modelFittingAllParticipants.ipynb` reads `ding_reformatted.csv`.
- `ding_reformatted.csv` is generated by `DingChangepoint.py`.

So the safest minimal reproducible set is:

- the four main notebooks
- the supporting model/helper `.py` files they import or execute
- the raw manuscript CSV inputs
- either `DingChangepoint.py` in the workflow or a committed
  `ding_reformatted.csv`

This means `data.pkl` is not required as an input, and the archived notebooks do
not appear to be required for reproducing the main manuscript workflow.
