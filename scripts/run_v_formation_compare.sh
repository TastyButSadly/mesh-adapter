#!/usr/bin/env bash
set -euo pipefail

uv run python -m examples.firedrake.v_formation_compare "$@"
