# Baijiu Multispectral Quantification Toolkit

This package contains bottle-matched Raman, ultraviolet and near-infrared spectra, reference chemistry, a fixed regression library, completed internal evaluation records and reproducible diagnostic analyses.

There are 139 independent matched bottles and 11 concentration targets. The existing evaluation is retrospective internal validation after iterative development on the same cohort. Repeated cross-validation does not create additional independent bottles.

## Quick start: inspect completed results without fitting

Use Python 3.9 to 3.12. The completed source evaluation used Python 3.9.10, NumPy 1.26.4, pandas 2.2.3, SciPy 1.13.1 and scikit-learn 1.5.2. Small numerical differences can occur across interpreters and libraries.

1. Run `00_install_environment.bat` on Windows, or create a virtual environment and install `requirements.txt`.
2. Run `01_check_data.bat` to verify the English workbooks against the aligned data and inspect the fixed algorithm library. This does not fit models.
3. Run `02_verify_archives.bat` to check file hashes, split memberships and saved reference values. This does not fit models.
4. Run `03_analyze_saved_predictions.bat` to reproduce bottle-cluster uncertainty, concentration diagnostics and random-review controls from completed saved predictions. This does not fit models and can take several minutes.
5. Run the independent scripts in `visualization` to redraw each graphic from its accompanying data. See `visualization/README.md` for the exact commands.

The `analysis` directory already contains the completed diagnostic outputs. Raw evaluation tables are gzip-compressed under `archives/main`. pandas reads `.csv.gz` directly. These archives are deliberately separate from `results`: localization changes source hashes, so old checkpoints are not presented as resumable new evaluations.

## Refit the fixed evaluation, if needed

Commands below perform real model training and may take substantial time. They are optional when inspecting the completed saved results.

```text
python run.py run --profile main --mode ablation
python run.py run --profile main --mode interval
python run.py run --profile main --mode boundaries
python run.py run --profile main --mode measured_abv
python run.py verify-all --profile main
```

Run `python -m unittest discover -s tests` for synthetic execution and leakage safeguards. These tests never establish performance on additional real bottles.

## Data organization

- `data/raw`: English-localized workbooks. Header text, sheet names and descriptive category names were translated; every numeric cell was checked for exact equality with the supplied source. The historical ethyl-hexanoate spelling was corrected as confirmed by the data owner.
- `data/aligned`: bottle-ID-matched references and spectra, including supplied-ratio diagnostics. Jar numbers and unavailable acquisition dates are not inferred.
- `archives/main`: completed evaluation tables and original split memberships for each required mode.
- `archives/comparison`: fixed historical reference summaries, used as context rather than additional validation.
- `analysis`: complete diagnostic tables and a portable saved-prediction analysis script.
- `suite`: fixed preprocessing, 138 regression candidates across 20 families, training-only selection, cross-fitting, ensembles, intervals and boundaries.
- `tests`: synthetic safeguards.
- `visualization`: individual plotting scripts, exact graphic data and reproducible exports.
- `provenance`: localization, numerical-identity, archive and validation receipts.

All concentrations use the supplied g/L labels. Measured alcohol strength retains the supplied numerical scale; its unit has not been confirmed. Spectral axis numbers are retained without inventing physical units or acquisition settings. Source categorical labels were mapped one to one to English `Source_class_*` codes; these codes do not infer a quality order and are excluded from spectral predictors.

The measured-alcohol-strength reference is fixed quadratic features plus ridge regression with alpha 1. Alcohol-strength-assisted methods require that extra measured input. Their comparison with spectra-only prediction does not establish molecular specificity.

Conformal summaries report marginal and simultaneous coverage separately. Nominal overall coverage does not establish conditional coverage in high-concentration or selected subsets. Random policies matched on true concentration are retrospective diagnostics; test reference concentrations are unavailable to an operational selector. Numerical perturbations are sensitivity checks, rather than measured instrument drift or batch transfer.

## Integrity and reuse

Run `python verify_archives.py` after extraction. `CHECKSUMS.sha256` covers all distributed files except itself. Every Python source has no comments or docstrings. Standard language keywords and third-party API names are retained for execution.

No broad reuse license is assigned to owner-supplied data or code by this packaging step. The owner should choose reuse terms and confirm rights before making the repository public. No remote repository has been created or modified.
