#!/usr/bin/env bash
set -euo pipefail

# Five unique jobs cover three comparisons. The shared reference is reused as
# LME, patient-loss-off, and staged-curriculum control.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-local}"

if [[ "$#" -gt 0 ]]; then
  REQUESTED=("$@")
else
  REQUESTED=(all)
fi

EXPANDED=()
for item in "${REQUESTED[@]}"; do
  case "$item" in
    all)
      EXPANDED+=(n4abl_ref n4abl_sbx_mean n4abl_sbx_max n4abl_patient n4abl_all_e1)
      ;;
    remaining)
      # The joint-TBx/SBx cohort and plain-LME endpoint invalidate the older
      # reference/mean runs. Keep this compatibility alias safe by rerunning the
      # complete five-condition matrix under the new protocol tag.
      EXPANDED+=(n4abl_ref n4abl_sbx_mean n4abl_sbx_max n4abl_patient n4abl_all_e1)
      ;;
    sbx_pooling)
      EXPANDED+=(n4abl_ref n4abl_sbx_mean n4abl_sbx_max)
      ;;
    patient_supervision)
      EXPANDED+=(n4abl_ref n4abl_patient)
      ;;
    curriculum)
      EXPANDED+=(n4abl_ref n4abl_all_e1)
      ;;
    n4abl_ref|n4abl_sbx_mean|n4abl_sbx_max|n4abl_patient|n4abl_all_e1)
      EXPANDED+=("$item")
      ;;
    *)
      echo "Unknown ablation/group: $item" >&2
      exit 2
      ;;
  esac
done

EXPERIMENTS=()
for candidate in "${EXPANDED[@]}"; do
  duplicate=0
  for existing in "${EXPERIMENTS[@]:-}"; do
    if [[ "$candidate" == "$existing" ]]; then
      duplicate=1
      break
    fi
  done
  if [[ "$duplicate" == "0" ]]; then
    EXPERIMENTS+=("$candidate")
  fi
done

build_args_array() {
  local exp="$1"
  RUN_ARGS=("${SCRIPT_DIR}/run_n4_method_ablation.py" --experiment "$exp")
  if [[ -n "${RP_BASE_DIR:-}" ]]; then RUN_ARGS+=(--base-dir "$RP_BASE_DIR"); fi
  if [[ -n "${RP_DATASET_ROOT:-}" ]]; then RUN_ARGS+=(--dataset-root "$RP_DATASET_ROOT"); fi
  if [[ -n "${RP_EXP_DIR:-}" ]]; then RUN_ARGS+=(--exp-dir "$RP_EXP_DIR"); fi
  if [[ -n "${EPOCHS:-}" ]]; then RUN_ARGS+=(--epochs "$EPOCHS"); fi
  if [[ -n "${SEED:-}" ]]; then RUN_ARGS+=(--seed "$SEED"); fi
  if [[ -n "${LR:-}" ]]; then RUN_ARGS+=(--lr "$LR"); fi
  if [[ -n "${POS_WEIGHT:-}" ]]; then RUN_ARGS+=(--pos-weight "$POS_WEIGHT"); fi
  if [[ -n "${SYS_POS_WEIGHT:-}" ]]; then RUN_ARGS+=(--sys-pos-weight "$SYS_POS_WEIGHT"); fi
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
  echo "=== Launching ${exp_upper} N4 method ablation ==="
  build_args_array "$exp"
  if [[ "$RUN_MODE" == "slurm" ]]; then
    cmd="cd $(printf '%q' "$PROJECT_DIR") && $(printf '%q' "$PYTHON_BIN") $(quote_args "${RUN_ARGS[@]}")"
    sbatch_args=(--job-name "RP_${exp_upper}")
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
