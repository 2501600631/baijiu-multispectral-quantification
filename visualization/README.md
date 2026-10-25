# Reproducible quantitative visualization

Run `python visualization/render_all.py` from the project directory, or run any individual `plot_figure_*.py` script. Optional `--output PATH` selects an output directory. Each script reads only the CSV files in `visualization/data`; it does not fit a model. Requirements are NumPy, pandas and Matplotlib.

Outputs are vector PDF, EPS and SVG, PNG at 320 dpi and LZW-compressed TIFF at 600 dpi. Colors, markers, reference lines and direct values preserve interpretation when printed in grayscale. Fonts are embedded as TrueType in PDF. Bands and grid lines use opaque pale colors for consistent EPS rendering.

The CSV files preserve the full numeric precision of the saved-prediction analyses. Column names and analyte/strategy labels were translated to English; all numeric entries were checked for exact equality after round-trip CSV loading. The sample population comprises 139 distinct bottles, each assessed in five outer repetitions. Those repeated predictions are correlated observations.

- Figure 1 uses the fixed quadratic-feature Ridge alcohol-strength baseline and 4,000 paired bottle-cluster resamples. Intervals are conditional on the fitted predictions and do not include model refitting or repeated development.
- Figure 2 uses the adaptive marginal 90% intervals. High concentration means a reference value above the fitting-set 75th percentile. Truth-defined groups are retrospective diagnostics. The fitting sets contain 83 or 84 bottles, with 28 calibration bottles; they differ from the ordinary point-prediction fitting budget.
- Figure 3 uses 2,000 equal-count random-retention policies with common bottle-level priorities across repetitions. Shaded bands and horizontal ranges are 2.5th–97.5th percentiles of random policies, not population confidence intervals. Matching uses observed reference concentrations for diagnostic purposes. Positive gains indicate smaller observed RMSE than the concentration-matched random-policy mean.
- Figure 4 compares the ten saved ordinary point-prediction procedures. The fixed strategy selector stays primary; observed rankings do not select a new primary procedure. Macro normalized RMSE is the mean across eleven analytes of each analyte's root mean squared error divided record-wise by its fitting-set interquartile range.
- The overview is derived from the same values: ten analytes have positive gain over the fixed alcohol baseline; eleven high-concentration groups fall below 90% coverage; the gain at nominal 70% retention is 4.76479% against the unstratified random mean and 0.99908% against the concentration-matched mean.

Concentration values are expressed in the supplied g/L reporting scale. Unverified collection methods, instrument settings, physical coordinate units and concentration-reporting basis are not inferred from these files.
