#!/usr/bin/env bash

set -euo pipefail

MANIFEST_PATH="${1:?manifest path required}"
PYTHON_BIN="${SRGAN_HPC_PYTHON:-python}"
RUN_DIR="$(dirname "${MANIFEST_PATH}")"

if [[ -n "${SRGAN_HPC_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  IFS=',' read -r -a MODULE_LIST <<< "${SRGAN_HPC_MODULES}"
  for module_name in "${MODULE_LIST[@]}"; do
    module load "${module_name}"
  done
fi

if [[ -n "${SRGAN_HPC_CONDA_ENV:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${SRGAN_HPC_CONDA_ENV}"
  else
    source activate "${SRGAN_HPC_CONDA_ENV}"
  fi
fi

exec "${PYTHON_BIN}" -m deployment.srgan_hpc.cli collect --run-dir "${RUN_DIR}"
