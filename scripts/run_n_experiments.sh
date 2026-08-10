#!/usr/bin/env bash
set -euo pipefail

# Run the redesigned N1--N5 matrix. N1--N4 exclude patient supervision; N5
# adds biopsy-confirmed P to the full D+T+S model. A is disabled in all runs.
#
# Examples:
#   bash scripts/run_n_experiments.sh
#   bash scripts/run_n_experiments.sh n2 n3 n4 n5
#   DRY_RUN=1 bash scripts/run_n_experiments.sh
#   RUN_MODE=slurm SEED=43 bash scripts/run_n_experiments.sh n4 n5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-local}"

if [[ "$#" -gt 0 ]]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=(n1 n2 n3 n4 n5)
fi

build_args_array() {
  local exp="$1"
  RUN_ARGS=("${SCRIPT_DIR}/run_n_experiments.py" --experiment "$exp")
  if [[ -n "${RP_BASE_DIR:-}" ]]; then RUN_ARGS+=(--base-dir "$RP_BASE_DIR"); fi
  if [[ -n "${RP_DATASET_ROOT:-}" ]]; then RUN_ARGS+=(--dataset-root "$RP_DATASET_ROOT"); fi
  if [[ -n "${RP_EXP_DIR:-}" ]]; then RUN_ARGS+=(--exp-dir "$RP_EXP_DIR"); fi
  if [[ -n "${EPOCHS:-}" ]]; then RUN_ARGS+=(--epochs "$EPOCHS"); fi
  if [[ -n "${SEED:-}" ]]; then RUN_ARGS+=(--seed "$SEED"); fi
  if [[ -n "${LR:-}" ]]; then RUN_ARGS+=(--lr "$LR"); fi
  if [[ -n "${POS_WEIGHT:-}" ]]; then RUN_ARGS+=(--pos-weight "$POS_WEIGHT"); fi
  if [[ -n "${SYS_POS_WEIGHT:-}" ]]; then RUN_ARGS+=(--sys-pos-weight "$SYS_POS_WEIGHT"); fi
  if [[ -n "${TBX_DICE_WEIGHT:-}" ]]; then RUN_ARGS+=(--tbx-dice-weight "$TBX_DICE_WEIGHT"); fi
  if [[ -n "${DROPOUT_RATE:-}" ]]; then RUN_ARGS+=(--dropout-rate "$DROPOUT_RATE"); fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then RUN_ARGS+=(--dry-run); fi
  if [[ "${FORCE:-0}" == "1" ]]; then RUN_ARGS+=(--force); fi
}

quote_args() {
  printf '%q ' "$@"
}

cd "$PROJECT_DIR"

for exp in "${EXPERIMENTS[@]}"; do
  exp_upper="$(printf '%s' "$exp" | tr '[:lower:]' '[:upper:]')"
  echo "=== Launching ${exp_upper} redesigned RA-anchored experiment ==="
  build_args_array "$exp"
  if [[ "$RUN_MODE" == "slurm" ]]; then
    cmd="cd $(printf '%q' "$PROJECT_DIR") && $(printf '%q' "$PYTHON_BIN") $(quote_args "${RUN_ARGS[@]}")"
    sbatch_args=(--job-name "RP_${exp_upper}_BEST")
    if [[ -n "${SLURM_PARTITION:-}" ]]; then sbatch_args+=(--partition "$SLURM_PARTITION"); fi
    sbatch_args+=(--gres "gpu:${SLURM_GPUS:-1}")
    sbatch_args+=(--cpus-per-task "${SLURM_CPUS:-8}")
    sbatch_args+=(--mem "${SLURM_MEM:-48G}")
    sbatch_args+=(--time "${SLURM_TIME:-2-00:00:00}")
    sbatch_args+=(--output "${PROJECT_DIR}/slurm-%x-%j.out")
    sbatch "${sbatch_args[@]}" --wrap "$cmd"
  else
    "$PYTHON_BIN" "${RUN_ARGS[@]}"
  fi
done
