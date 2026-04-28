#!/usr/bin/env bash

set -euo pipefail

MANIFEST_PATH="${1:?manifest path required}"
PYTHON_BIN="${SRGAN_HPC_PYTHON:-python}"

if [[ -n "${SRGAN_HPC_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  IFS=',' read -r -a MODULE_LIST <<< "${SRGAN_HPC_MODULES}"
  for module_name in "${MODULE_LIST[@]}"; do
    module load "${module_name}"
  done
fi

if [[ -n "${SRGAN_HPC_CONDA_ENV:-}" ]]; then
  source activate "${SRGAN_HPC_CONDA_ENV}"
fi

exec "${PYTHON_BIN}" -m deployment.srgan_hpc.cli run task --manifest "${MANIFEST_PATH}"
