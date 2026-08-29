# Mixed-Supervision Prostate MRI

Research code for training and evaluating prostate MRI models with mixed
supervision from dense radiologist annotations (RA), targeted-biopsy regions
(TBx), systematic-biopsy regions (SBx), and patient-level labels.

![Mixed-supervision framework](overleaf_figures/fig1_mixed_supervision_framework.png)

![Annotation-specific supervision mapping](overleaf_figures/fig2_annotation_specific_supervision_mapping.png)

## Status

This repository contains source code, tests, and reviewed schematic assets. It
intentionally excludes patient-level artifacts, datasets, checkpoints,
experiment outputs, private paths, papers, and third-party source trees.

The software is research-only and is not a medical device. Do not use it for
clinical decisions.

## Repository layout

```text
.
├── config.py                  # Experiment and path configuration
├── dataset.py                 # Dataset loading
├── model.py                   # Model definition
├── Loss_function.py           # Mixed-supervision objectives
├── train.py                   # Training entry point
├── test.py                    # Evaluation entry point
├── scripts/                   # Reproducible B/N experiment runners
├── preprocessing/             # First-party dataset preparation tools
├── tests/                     # Unit and specification tests
├── configs/                   # Public configuration examples
├── docs/assets/               # Additional reviewed schematic assets
└── overleaf_figures/          # README framework figures
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The single requirements file contains the complete training, preprocessing,
official PI-CAI evaluation, and test stack. This includes the comparatively
large optional VTK dependency used by `preprocessing/TCIA_stl2mask.py`.

## Data and paths

No medical data are distributed, downloaded, or uploaded by this repository.
The dense radiologist-annotation branch accepts a user-provided cohort that the
user is authorized to process. TCIA and PROMIS data must be obtained from their
official providers. In every case, users are responsible for the applicable
license, consent or ethics approval, data-use agreement, de-identification, and
institutional policy.

Dataset preparation now has one path-independent entry point. Start with a
preflight-only command and keep the generated workspace outside this Git
repository and outside the source input directories:

```bash
python preprocessing/run_pipeline.py \
  --workspace /absolute/path/to/rp-workspace \
  --datasets radiologist \
  --radiologist-root /absolute/path/to/RADIOLOGIST_ROOT \
  --split-mode none \
  --dry-run
```

For the user-provided dense radiologist annotations, arrange one
de-identified case per patient in this schema:

```text
RADIOLOGIST_ROOT/
├── imagesTr/
│   ├── CASE_0000.nii.gz     # T2-weighted MRI
│   ├── CASE_0001.nii.gz     # ADC map
│   └── CASE_0002.nii.gz     # DWI
├── labelsTr/
│   └── CASE.nii.gz          # dense radiologist lesion mask
└── zonesTr/
    └── CASE.nii.gz          # non-empty prostate mask
```

The adapters do not accept arbitrary MRI downloads without schema conversion.
PROMIS currently requires a separately generated 20-zone mask that is not
redistributed here. Exact input contracts, geometry requirements, processing
order, split modes, and known limitations are documented in
[`preprocessing/README.md`](preprocessing/README.md).

After a unified workspace has been built, set the training paths with
environment variables or command-line options:

```bash
export RP_PROJECT_ROOT=/path/to/mixed-supervision-prostate-mri
export RP_DATASET_ROOT=/absolute/path/to/rp-workspace
export RP_EXP_DIR=/path/to/experiment-outputs
```

See `configs/environment.example` for the public configuration template.

## Quick checks

Resolve an experiment without starting training:

```bash
python scripts/run_b_experiments.py --experiment b1 --dry-run
python scripts/run_n_experiments.py --experiment n4 --dry-run
```

Run the test suite:

```bash
python -m pytest -q
```

See `scripts/README.md` for training, ablation, and frozen-evaluation commands.

## Reproducibility scope

The repository contains the canonical B0--B4 and N1--N5 experiment runners,
TBx Dice and N4 method ablations, and the generic frozen-evaluation workflow.
It supports reproduction of the code path and reuse with compatible,
user-authorized inputs. Because the original dense radiologist-annotation
cohort is not distributed, a user-provided cohort cannot be assumed to
reproduce the original cohort composition, data distribution, or reported
numerical results. Historical, versioned, server-specific, and selected-case
scripts are documented as exclusions in `MANIFEST.md`.

## License and citation

A software license has not yet been selected. Confirm code ownership with all
relevant authors or institutions and add a `LICENSE` file before making the
GitHub repository public. A `CITATION.cff` should be added when the preferred
paper/preprint citation is final. Also confirm that the files in `docs/assets/`
and `overleaf_figures/` are first-party works or otherwise licensed for
redistribution.
