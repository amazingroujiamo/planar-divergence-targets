# Planar Divergence Targets

Code, slice manifests, training records, and statistical analysis results for the manuscript:

**Effects of In-Plane Divergence Constraints on Machine-Learning Super-Resolution Reconstruction of Three-Dimensional Flow Slices**

Authors: Zhihao Lin, Denghui Qin, Qiaogao Huang, and Guang Pan.

## Study design

The study compares three training objectives for two-component super-resolution on planar slices of three-dimensional flows:

- **A:** velocity supervision only;
- **Z:** velocity supervision plus a zero in-plane divergence penalty;
- **T:** velocity supervision plus a penalty on the difference between predicted and DNS-derived reference in-plane divergence.

The same network, inputs, labels, training budget, discrete divergence operator, and paired seeds are used across the controlled comparisons. Velocity errors are evaluated directly against the original, unprojected high-resolution velocity fields.

## Repository contents

```text
src/pdt/                                      Model, losses, data handling, divergence operator, training, and metrics
experiments/pdt_*.py                         Acquisition, checks, training, evaluation, analysis, and reporting scripts
tests/test_pdt_analysis_reporting.py          Regression tests for statistical aggregation and reporting
artifacts/planar_divergence_targets_v1/
├── manifests/                               Split lists, source hashes, experiment matrix, and frozen settings
├── metrics/                                 Per-field, per-time, per-seed, and condition-level results
├── reports/                                 Bootstrap intervals, classifications, convergence checks, and internal report
└── runs/                                    Run metadata, summaries, and training/validation curves
```

The release includes records for 126 main models, 30 short pilot runs, and 18 continuation stages. Model checkpoints, caches, and raw HDF5 velocity fields are not included.

## Data source and slice definitions

The underlying velocity fields come from the public [Johns Hopkins Turbulence Database](https://turbulence.pha.jhu.edu/), using the `isotropic1024coarse` and `channel` datasets. Raw fields are not redistributed in this repository.

- `manifests/split_manifest.csv` defines the training, validation, gap, and original test slices.
- `manifests/test_cohort.json` defines and verifies the additional held-out native test times.
- `manifests/source_inventory.csv` records the relative file names, roles, sizes, and SHA-256 hashes of the source files.
- All JHTDB time and plane indices are **1-based**. Array indices in the code are 0-based.

The acquisition script reads a user-supplied JHTDB token from the `JHTDB_TOKEN` environment variable or the user's local `~/.env`. No credentials are included in this repository.

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The recorded training and operator checks used PyTorch 2.5.1+cu124 on an NVIDIA GeForce RTX 4090. Other PyTorch, CUDA, or hardware versions may change performance and bitwise numerical reproducibility.

## Reproduce the statistical analysis

The released per-field metrics and run records allow the statistical analysis to be regenerated without downloading the raw flow fields or retraining the models. Use a fresh working copy because these commands overwrite generated files under `artifacts/planar_divergence_targets_v1/`.

```bash
python -m unittest discover -s tests -p 'test_pdt_analysis_reporting.py' -v
python experiments/pdt_analyze.py --project-root .
python experiments/pdt_report.py --project-root .
```

The analysis uses a paired circular block bootstrap over 16 ordered test times, with block length 4 and 5000 repetitions. Block lengths 2 and 8 are sensitivity checks. The six matched-weight T–Z comparisons at the intermediate penalty weight form the primary Bonferroni-adjusted family.

## Retraining and evaluation

Full retraining requires separately obtained JHTDB HDF5 plane files arranged according to the relative paths in the manifests, sufficient disk space, and a compatible PyTorch installation. The training scripts expect a CUDA device by default. Since checkpoints are not included, models must be retrained before rerunning field-level evaluation.

Key entry points:

- `experiments/pdt_prepare.py`: source verification and training statistics;
- `experiments/pdt_operator_checks.py`: divergence-operator and loss checks;
- `experiments/pdt_run_matrix.py`: frozen training matrix and continuation runs;
- `experiments/pdt_evaluate.py`: evaluation against original high-resolution velocity truth;
- `experiments/pdt_analyze.py` and `experiments/pdt_report.py`: statistical analysis and reporting.

Run a script with `--help` for its command-line arguments.

## Result conventions

`E_v` is the normalized squared velocity error against the original high-resolution velocity. `E_d` measures the discrepancy between predicted and reference divergence, while `A_d` measures predicted divergence amplitude. A reduction in `A_d` alone is not evidence of improved velocity or divergence accuracy.

For seed-mean errors, `R_Z` and `R_T` are percentage changes relative to A. The matched-weight target-replacement effect is `R_target = 100(E_T - E_Z) / E_A = R_T - R_Z`, expressed in percentage points.

## Citation

If using this repository, please cite the associated manuscript and the Johns Hopkins Turbulence Database. Publication identifiers can be added when available.

## License

The released code and accompanying analysis files are provided under the MIT License. JHTDB data remain subject to their original terms and are not included here.
