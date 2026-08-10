#!/usr/bin/env bash
set -euo pipefail

# T0--T2 run by default. Set INCLUDE_T3=1 or pass t3 explicitly to include
# the optional weight-1.0 condition.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-local}"

if [[ "$#" -gt 0 ]]; then
  EXPERIMENTS=("$@")
elif [[ "${INCLUDE_T3:-0}" == "1" ]]; then
  EXPERIMENTS=(t0 t1 t2 t3)
else
  EXPERIMENTS=(t0 t1 t2)
fi

build_args_array() {
  local exp="$1"
  RUN_ARGS=("${SCRIPT_DIR}/run_tbx_dice_ablation.py" --experiment "$exp")
  if [[ -n "${RP_BASE_DIR:-}" ]]; then RUN_ARGS+=(--base-dir "$RP_BASE_DIR"); fi
  if [[ -n "${RP_DATASET_ROOT:-}" ]]; then RUN_ARGS+=(--dataset-root "$RP_DATASET_ROOT"); fi
  if [[ -n "${RP_EXP_DIR:-}" ]]; then RUN_ARGS+=(--exp-dir "$RP_EXP_DIR"); fi
  if [[ -n "${EPOCHS:-}" ]]; then RUN_ARGS+=(--epochs "$EPOCHS"); fi
  if [[ -n "${SEED:-}" ]]; then RUN_ARGS+=(--seed "$SEED"); fi
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
  echo "=== Launching ${exp_upper} TBx Dice ablation ==="
  build_args_array "$exp"
  if [[ "$RUN_MODE" == "slurm" ]]; then
    cmd="cd $(printf '%q' "$PROJECT_DIR") && $(printf '%q' "$PYTHON_BIN") $(quote_args "${RUN_ARGS[@]}")"
    sbatch_args=(--job-name "RP_${exp_upper}_TBX_DICE")
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
