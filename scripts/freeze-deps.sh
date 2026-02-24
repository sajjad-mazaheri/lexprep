#!/usr/bin/env bash
# Regenerate requirements.lock from a clean virtual environment.
# Run this after updating pyproject.toml or web/requirements.txt.
#
# Usage:
#   bash scripts/freeze-deps.sh
#
# The output file (requirements.lock) should be committed to the repo.
# During deploy, use: pip install -c requirements.lock ...

set -euo pipefail

LOCK_FILE="requirements.lock"
VENV_DIR=$(mktemp -d)
trap 'rm -rf "$VENV_DIR"' EXIT

echo "Creating temporary virtualenv in $VENV_DIR ..."
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing project with all extras ..."
pip install --upgrade pip --quiet
pip install -e ".[fa,en,ja,dev]" --quiet
pip install -r web/requirements.txt --quiet

echo "Freezing dependencies ..."
pip freeze --exclude-editable | sort > "$LOCK_FILE"

deactivate

echo "Wrote $(wc -l < "$LOCK_FILE") packages to $LOCK_FILE"
echo "Done. Commit $LOCK_FILE to the repository."
