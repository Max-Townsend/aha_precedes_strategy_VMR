# Legacy Workflow Notes

These notes are preserved from the original `README.txt`, which described an
earlier manuscript workflow rather than the main workflow used in this
repository.

The dependency note in the original README referred to the old environment
state. The current top-level `requirements.txt` has since been cleaned into a
portable install list for the main workflow.

To generate the stats and figures used in the earlier version of the paper, run
`inferentialStats.ipynb` and `paperFigures.ipynb`. The original README pointed
to `requirements.txt` for package versions. Figures are saved in the
`paperFigures` folder.

You can rerun these using `CGDataWithOutliers_.csv` to see stats and figures
without excluding any participants, or by using the `rawaim` field instead of
`aim` (and replacing other chosen fields with their `raw` counterparts) to see
stats and figures without excluding any trials.

The original data used in that workflow is `CGData_.csv`. The analysis also
uses data from Brudner et al. (2016) in `BrudnerData.csv`, and from Bond &
Taylor (2015) in `BTData_.csv`. The remaining CSV files are the raw cannon game
data in long form and the demographics file, which are fed into the data
processing code.

`DataProcessing.ipynb` contains the code for excluding trials and participants.

`modelFitting.ipynb` contains the code for fitting models. In that workflow,
`paperFigures.ipynb` generates the relevant figures using pre-fitted parameters
by default, but new parameters can also be fitted. The `.npy` files contain the
pre-fitted parameters.

`DataManipulation.py` contains many custom functions used in other scripts.

The remaining `.py` files contain the model implementations used by the fitting
notebook and by simulations within `paperFigures.ipynb`.

For queries: `max.o.b.townsend@bath.edu`
