#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"
download_dir="${DOWNLOAD_DIR:-data/package_cache}"

mkdir -p "$download_dir"
"$python_bin" -m pip download -r requirements-data-libs.txt -d "$download_dir"

echo "Downloaded data library packages to $download_dir"
