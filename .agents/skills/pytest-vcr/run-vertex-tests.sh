#!/bin/bash
# Run pytest with Vertex AI auth
# Usage: .claude/skills/pytest-vcr/run-vertex-tests.sh [pytest args...]

set -e

# Force Vertex auth path (not API key)
unset GOOGLE_API_KEY GEMINI_API_KEY

if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    # Service account credentials — no gcloud needed
    if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        echo "ERROR: GOOGLE_APPLICATION_CREDENTIALS file not found: $GOOGLE_APPLICATION_CREDENTIALS"
        exit 1
    fi
    # Extract project from credentials JSON if not already set
    if [ -z "$GOOGLE_PROJECT" ] && [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
        PROJECT=$(python3 -c "import json; print(json.load(open('$GOOGLE_APPLICATION_CREDENTIALS')).get('project_id', ''))" 2>/dev/null)
        if [ -n "$PROJECT" ]; then
            export GOOGLE_PROJECT="$PROJECT"
            export GOOGLE_CLOUD_PROJECT="$PROJECT"
        fi
    fi
else
    # Fall back to gcloud
    GCLOUD="$(command -v gcloud 2>/dev/null || echo "$HOME/projects/google-cloud-sdk/bin/gcloud")"
    if [ ! -x "$GCLOUD" ]; then
        echo "ERROR: No GOOGLE_APPLICATION_CREDENTIALS set and gcloud not found."
        echo "Either set GOOGLE_APPLICATION_CREDENTIALS or install Google Cloud SDK."
        exit 1
    fi

    if ! "$GCLOUD" auth application-default print-access-token &>/dev/null; then
        echo "ERROR: gcloud auth not configured. Run:"
        echo "  gcloud auth application-default login"
        exit 1
    fi

    PROJECT=$("$GCLOUD" config get-value project 2>/dev/null)
    if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
        echo "ERROR: no gcloud project configured. Run:"
        echo "  gcloud config set project <your-project-id>"
        exit 1
    fi

    export GOOGLE_PROJECT="$PROJECT"
    export GOOGLE_CLOUD_PROJECT="$PROJECT"
fi

export GOOGLE_LOCATION="${GOOGLE_LOCATION:-global}"

echo "Vertex AI: project=${GOOGLE_PROJECT:-${GOOGLE_CLOUD_PROJECT:-unset}} location=$GOOGLE_LOCATION"

ENABLE_VERTEX=1 uv run pytest "$@"
