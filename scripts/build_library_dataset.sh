#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"
python3 src/data_pipeline/dataset_builder.py "$@"
