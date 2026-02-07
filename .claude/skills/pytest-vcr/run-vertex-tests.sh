#!/bin/bash
# Run pytest with Vertex AI auth
# Usage: .claude/skills/pytest-vcr/run-vertex-tests.sh [pytest args...]

set -e

# Verify gcloud auth
if ! gcloud auth application-default print-access-token &>/dev/null; then
    echo "ERROR: gcloud auth not configured. Run:"
    echo "  gcloud auth application-default login"
    exit 1
fi

# Detect project
PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT" ]; then
    echo "ERROR: no gcloud project configured. Run:"
    echo "  gcloud config set project <your-project-id>"
    exit 1
fi

# Force Vertex auth path (not API key)
unset GOOGLE_API_KEY GEMINI_API_KEY

export GOOGLE_PROJECT="$PROJECT"
export GOOGLE_CLOUD_PROJECT="$PROJECT"
export GOOGLE_LOCATION="${GOOGLE_LOCATION:-global}"

echo "Vertex AI: project=$PROJECT location=$GOOGLE_LOCATION"

ENABLE_VERTEX=1 uv run pytest "$@"
