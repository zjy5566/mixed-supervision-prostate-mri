# Dataset preprocessing

`run_pipeline.py` is the supported public entry point for dataset preparation.
It replaces workstation-specific paths with explicit command-line arguments and
runs each source-specific step in the required order.

The pipeline is an adapter for the three schemas used by this project. It is
not a generic converter for arbitrary prostate MRI datasets, and it does not
download data. No medical data, clinical tables, patient-level split files, or
derived images are included in this repository.

Confirmed source records:

- [TCIA Prostate-MRI-US-Biopsy collection](https://www.cancerimagingarchive.net/collection/prostate-mri-us-biopsy/)
- [PROMIS derived dataset record](https://doi.org/10.5281/zenodo.15683922)

The source referred to as `PUB` or `RA` in this project still needs a verified
public download record and license from the maintainers; the repository does
not infer one from its internal name.

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The single `requirements.txt` covers training, preprocessing, evaluation, and
tests. TCIA STL conversion installs VTK, and reading the official Excel tables
uses openpyxl.

## What is automated

The default stages are `preprocess` followed by `unify`:

```text
PUB     NIfTI volumes -> resample/crop/normalize -> NPY files
PROMIS  NIfTI volumes -> resample/register/crop -> biopsy CSV conversion
TCIA    DICOM extraction -> STL masks -> biopsy labels -> resample/crop
all     processed cases -> Unified_Dataset -> registry -> eligible splits
```

Use a separate workspace outside both the Git repository and the downloaded
source directories. The pipeline never deletes source data and refuses a
workspace that overlaps a raw input directory.

```text
WORKSPACE/
├── intermediate/tcia_extracted/
├── processed/
│   ├── pub/
│   ├── promis/
│   └── tcia/
├── Unified_Dataset/
│   └── splits/dataset_registry.csv
└── qa/input_alignment/
```

Always run `--dry-run` first. It validates the path layout and required table
columns without writing files.

## PUB/RA input

The PUB/RA adapter expects an already aligned nnU-Net-like NIfTI layout:

```text
PUB_ROOT/
├── imagesTr/
│   ├── CASE_0000.nii.gz     # T2
│   ├── CASE_0001.nii.gz     # ADC
│   └── CASE_0002.nii.gz     # DWI
├── labelsTr/
│   └── CASE.nii.gz          # lesion annotation
└── zonesTr/
    └── CASE.nii.gz          # prostate/zone mask used for cropping
```

The code does not register PUB modalities. T2, ADC, DWI, lesion, and zone files
must already describe the same anatomy in compatible physical coordinates.
The public provenance, download instructions, and redistribution license for
the dataset called `PUB` in the research workspace are not established here;
users must supply a lawfully obtained dataset in this exact schema.

```bash
python preprocessing/run_pipeline.py \
  --workspace /absolute/path/to/rp-workspace \
  --datasets pub \
  --pub-root /absolute/path/to/PUB_ROOT \
  --split-mode none \
  --dry-run
```

Remove `--dry-run` after reviewing the plan.

## PROMIS input

The PROMIS adapter expects:

```text
PROMIS_MRI_ROOT/
└── P-*/
    ├── t2.nii.gz
    ├── adc.nii.gz
    ├── dwi.nii.gz
    ├── gland.nii.gz
    ├── gland_zone_20level_set1.nii.gz
    └── lesion_a1.nii.gz                 # optional

PROMIS_BIOPSY_ROOT/
└── P-*.csv
```

Every CSV must contain `zone_id`, `samtaken`, `zprescancer`,
`zprimgleason`, `zsecondgleason`, and `maxccisup`. The MRI case directory and
CSV stem must match, for example `P-0001/` and `P-0001.csv`.

The official archive does not by itself close this repository's current
PROMIS workflow: `gland_zone_20level_set1.nii.gz` is a project-derived 20-zone
mask. Its former generator was in a third-party source tree whose
redistribution license has not been verified, so it is intentionally not
included. The preflight check stops with a clear error when these masks are
missing. Generate them with separately licensed tooling and review the result
before continuing.

The wrapper deliberately calls full MRI preprocessing with
`lesion_only=False`; running `PROMIS_preprocessor_MRI.py` directly uses its
legacy lesion-backfill mode and is not the correct fresh-data entry point.

```bash
python preprocessing/run_pipeline.py \
  --workspace /absolute/path/to/rp-workspace \
  --datasets promis \
  --promis-mri-root /absolute/path/to/PROMIS_MRI_ROOT \
  --promis-biopsy-root /absolute/path/to/PROMIS_BIOPSY_ROOT \
  --split-mode none \
  --dry-run
```

The wrapper accepts either the direct biopsy directory or a uniquely nested
archive directory, including the historical one-level/two-level
`Template_biopsy` variants.

## TCIA input

The TCIA adapter targets the `Prostate-MRI-US-Biopsy` collection schema:

```text
TCIA_DICOM_ROOT/
└── Prostate-MRI-US-Biopsy-*/
    └── .../*.dcm

TCIA_STL_ROOT/
└── Prostate-MRI-US-Biopsy-*-seriesUID-*.STL

TCIA-Biopsy-Data.xlsx
```

The biopsy table must contain the patient number, MRI Series Instance UID,
core label, primary/secondary Gleason values, and tip/base MRI coordinates.
UCLA/PIRADS fallback labelling is disabled by the public wrapper, so target
supervision remains biopsy verified. The DICOM and STL arguments may point at
direct directories or at a wrapper directory containing one unambiguous
matching directory.

```bash
python preprocessing/run_pipeline.py \
  --workspace /absolute/path/to/rp-workspace \
  --datasets tcia \
  --tcia-dicom-root /absolute/path/to/TCIA_DICOM_ROOT \
  --tcia-stl-root /absolute/path/to/TCIA_STL_ROOT \
  --tcia-biopsy-table /absolute/path/to/TCIA-Biopsy-Data.xlsx \
  --split-mode auto \
  --dry-run
```

The current TCIA converter still uses collection-specific folder-name clues to
identify T2, ADC, and DWI and chooses a deterministic first ADC/DWI candidate
when more than one is present. Inspect the extraction summary and run alignment
QA before training. A preflight pass is not a substitute for visual geometry,
STL overlap, registration, label-range, or clinical-table review.

## Run all sources

After all inputs have been prepared, one command enforces the source-specific
order and builds the unified dataset:

```bash
python preprocessing/run_pipeline.py \
  --workspace /absolute/path/to/rp-workspace \
  --datasets pub promis tcia \
  --pub-root /absolute/path/to/PUB_ROOT \
  --promis-mri-root /absolute/path/to/PROMIS_MRI_ROOT \
  --promis-biopsy-root /absolute/path/to/PROMIS_BIOPSY_ROOT \
  --tcia-dicom-root /absolute/path/to/TCIA_DICOM_ROOT \
  --tcia-stl-root /absolute/path/to/TCIA_STL_ROOT \
  --tcia-biopsy-table /absolute/path/to/TCIA-Biopsy-Data.xlsx \
  --split-mode experiment \
  --dry-run
```

Remove `--dry-run` only after every preflight check passes.

## Stages and split modes

Run preprocessing first and integration later with the same workspace:

```bash
python preprocessing/run_pipeline.py [source arguments] --stages preprocess
python preprocessing/run_pipeline.py [source arguments] --stages unify
```

`--stages unify` reads the canonical `WORKSPACE/processed/*` directories, so
raw source arguments are no longer required on the second command. The split
modes are:

- `auto`: generate the original experiment CSVs only if the registry contains
  a TCIA cohort with both TBx and SBx supervision; otherwise keep the registry
  and explain why experiment splits were skipped. It also skips experiment
  splits if multiple derived MRI studies map to the same original TCIA patient,
  because row-level splitting would risk patient leakage.
- `experiment`: require the compatible joint-TCIA cohort and fail if the
  original mixed-supervision splits cannot be generated safely.
- `none`: write only `dataset_registry.csv`, suitable for a single-source run
  or a custom downstream split policy.

The full B/N experiments require more than successful image preprocessing.
They depend on the original supervision combination, including joint TCIA
TBx+SBx cases; PROMIS is held out as external validation.

## Alignment QA

After unification, run the read-only multimodal alignment audit:

```bash
python preprocessing/run_pipeline.py \
  --workspace /absolute/path/to/rp-workspace \
  --datasets pub promis tcia \
  --stages qa \
  --qa-workers 4 \
  --qa-visuals suspect
```

The audit writes only below `WORKSPACE/qa/input_alignment`. Review its CSV
summary and, when explicitly enabled, suspect-case figures before training.
`--qa-visuals none` is the wrapper default to minimize case-level artifacts.
These generated files can contain patient identifiers and absolute paths and
must not be committed to Git.

## Existing-output and overwrite behavior

By default, the wrapper refuses non-empty derived output directories. Pass
`--allow-existing-output` only after inspecting an interrupted source-specific
preprocessing run. Existing low-level preprocessors can overwrite derived
files and do not transactionally remove stale case directories, so this option
is not a true resumable or atomic workflow.

`Unified_Dataset` and the QA output directory must always be fresh, even when
`--allow-existing-output` is present. This prevents an old experiment split or
case figure from surviving a run that skipped or changed its corresponding
stage. Use a new workspace when rebuilding integration, splits, or QA, and
whenever preprocessing rules or source files change.

For training, point `RP_DATASET_ROOT` at the workspace that contains
`Unified_Dataset`:

```bash
export RP_DATASET_ROOT=/absolute/path/to/rp-workspace
export RP_EXP_DIR=/absolute/path/to/experiment-outputs
python scripts/run_b_experiments.py --experiment b1 --dry-run
```

Always comply with each source dataset's license, citation requirements,
consent/ethics terms, and institutional data-governance policy. Treat raw and
derived clinical tables, registry CSVs, split CSVs, logs, masks, and case
figures as research data rather than source code.
