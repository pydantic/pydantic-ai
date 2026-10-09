#!/usr/bin/env bash
# Builds what `/update` installs on the `main` channel into OUT_DIR, from the checked-out commit:
# an sdist of CLAI and of each workspace package it pins exactly, named `<package>-<commit>.tar.gz`,
# and `clai2-bleeding.json`, which names the commit. The `clai2-bleeding` workflow publishes them to
# the `clai2-bleeding` GitHub release. To try them locally, serve OUT_DIR over HTTP and start CLAI
# with `CLAI_BLEEDING_URL` pointing at it (see "Updating" in the README).
#
# The version comes from Git tags and history, so the checkout needs both. Each sdist records it,
# with exact pins between the packages, so installing needs no Git.
set -euo pipefail

out=${1:?usage: build_bleeding.sh OUT_DIR}
mkdir -p "$out"
out=$(cd "$out" && pwd)
cd "$(git rev-parse --show-toplevel)"
commit=$(git rev-parse HEAD)
staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT

# Keep in sync with `PACKAGES` in `pydantic_clai2/cli/self_update.py`; a test compares them.
for package in pydantic-clai2 pydantic-ai-harness pydantic-ai-slim pydantic-graph; do
  uv build --sdist --no-sources --package "$package" --out-dir "$staging"
  mv "$staging"/*.tar.gz "$out/${package//-/_}-$commit.tar.gz"
done
printf '{"commit": "%s"}\n' "$commit" > "$out/clai2-bleeding.json"
