#!/bin/bash
# Run pytest with Vertex AI configuration
# Usage: .claude/skills/pytest-vcr/run-vertex-tests.sh [pytest args...]

set -e

# Check gcloud auth
if ! gcloud auth application-default print-access-token &>/dev/null; then
    echo "ERROR: gcloud auth not configured"
    echo "Run: gcloud auth application-default login"
    exit 1
fi

# Get project from gcloud config
PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT" ]; then
    echo "ERROR: No gcloud project configured"
    echo "Run: gcloud config set project <your-project-id>"
    exit 1
fi

# Update .env.vertex with detected project
sed -i.bak "s/^export GOOGLE_PROJECT=.*/export GOOGLE_PROJECT=$PROJECT/" .env.vertex && rm -f .env.vertex.bak
sed -i.bak "s/^export GOOGLE_CLOUD_PROJECT=.*/export GOOGLE_CLOUD_PROJECT=$PROJECT/" .env.vertex && rm -f .env.vertex.bak


# Source the updated .env.vertex
source .env.vertex

echo "Using Vertex AI project: $PROJECT"
echo "Location: ${GOOGLE_CLOUD_LOCATION:-global}"
echo ""

ENABLE_VERTEX=1 uv run pytest "$@"
