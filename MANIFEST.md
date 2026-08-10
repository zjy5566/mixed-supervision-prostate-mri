# Public-release manifest

This repository was assembled on 2026-08-08 from the current working files in
the private `Research_Project` workspace. It starts with fresh Git history.

## Included source

- Core: `config.py`, `dataset.py`, `model.py`, `Loss_function.py`, `utils.py`,
  `train.py`, and `test.py`.
- Stable workflows: the shared experiment runner, B/N experiment families,
  TBx Dice and N4 method ablations, and generic frozen evaluation.
- First-party preprocessing: the 14 top-level Python tools under
  `preprocessing/`, including the path-independent `run_pipeline.py` wrapper.
- Tests: loss/metrics, test artifacts, top-k checkpoints, grouped splitting,
  preprocessing preflights, B/N specifications, and N4 ablation specifications.
- Assets: `methodology_example.svg` and
  `figure2_label_supervision_mapping_v2_sampling_sectors.png` only.

## Deliberately excluded

- Existing Git history and commit metadata.
- Virtual environments, caches, editor state, local dependency trees, and
  temporary files.
- Datasets, patient/case identifiers, medical images, split tables, predictions,
  per-case metrics, checkpoints, logs, and generated experiment outputs.
- Papers, reports, posters, student records, institutional templates, fonts,
  and office documents.
- Historical/versioned analysis scripts, selected-case workflows, server-bound
  manifests, and report-specific rendering scripts.
- `preprocessing/PROMIS-curated-main` and historical zone-segmentation code,
  models, papers, and sample images pending independent license review.

Adding an excluded category requires a separate privacy, provenance, size, and
license review.
