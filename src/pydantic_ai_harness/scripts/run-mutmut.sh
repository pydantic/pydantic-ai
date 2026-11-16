#!/usr/bin/env bash
# One-off mutation testing runner.
#
# mutmut is intentionally not a project dev dependency: it pulls in a large
# tree and is only needed when validating test quality. Install it ephemerally
# via `uv run --with` and invoke it as a subcommand.
#
# Config (paths_to_mutate, tests_dir, also_copy, pytest_add_cli_args) lives in
# [tool.mutmut] in pyproject.toml — mutmut v3 reads it from CWD by default.
#
# Usage:
#   scripts/run-mutmut.sh                # run all mutants
#   scripts/run-mutmut.sh results        # show pass/fail summary
#   scripts/run-mutmut.sh show <mutant>  # inspect a specific mutant
#   scripts/run-mutmut.sh --max-children 4 run   # any mutmut flag works
#
# Pair with `make testcov` to keep coverage at 100% — surviving mutants usually
# indicate missing test cases for boundary conditions.

set -euo pipefail

cd "$(dirname "$0")/.."

uv run --with "mutmut>=3.5.0" -- mutmut "$@"
