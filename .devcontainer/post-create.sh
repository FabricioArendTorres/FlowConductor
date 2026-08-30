#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y --no-install-recommends curl g++

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="${HOME}/.local/bin:${PATH}"

# uv owns this environment. Official pytorch/pytorch images are for CI, not
# the editor. CPU torch keeps the editor install small; CUDA is tested in CI.
VENV="${VIRTUAL_ENV:-${HOME}/.venv}"
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cpu}"

uv venv --clear "${VENV}"
uv pip install --python "${VENV}" -e ".[dev,examples,docs]"
"${VENV}/bin/pre-commit" install
