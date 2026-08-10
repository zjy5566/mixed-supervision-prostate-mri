# Experiment runners

All commands below are run from the repository root. Dataset and output paths
can be supplied through `RP_DATASET_ROOT` and `RP_EXP_DIR`, or with the shared
`--dataset-root` and `--exp-dir` options.

## Dry runs

Dry runs resolve the experiment configuration without importing the training
stack or starting optimisation.

```bash
python scripts/run_b_experiments.py --experiment b1 --dry-run
python scripts/run_n_experiments.py --experiment n4 --dry-run
python scripts/run_tbx_dice_ablation.py --experiment t1 --dry-run
python scripts/run_n4_method_ablation.py --experiment n4abl_ref --dry-run
```

Use `--help` on each runner to list its current experiment choices.

## Training

```bash
export RP_DATASET_ROOT=/path/to/datasets
export RP_EXP_DIR=/path/to/experiment-outputs

python scripts/run_b_experiments.py --experiment b1
python scripts/run_n_experiments.py --experiment n4
```

The shell wrappers run the corresponding experiment families sequentially and
accept the same environment variables.

## Frozen evaluation

The generic frozen workflow selects thresholds on validation data once, then
reuses the frozen artifact for internal and external tests.

```bash
python scripts/run_frozen_evaluation.py \
  --checkpoint /path/to/model.pth \
  --validation-csv /path/to/validation.csv \
  --internal-csv /path/to/internal_test.csv \
  --external-csv /path/to/external_test.csv \
  --output-root /path/to/evaluation-output \
  --experiment-mode N4_MIXED_CLEAN
```

Checkpoints and evaluation outputs are intentionally ignored by Git.
