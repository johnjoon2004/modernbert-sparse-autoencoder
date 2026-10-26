#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # ~/.../sae/src/frontend
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"  # ~/.../sae

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

streamlit run src/frontend/main.py "$@"