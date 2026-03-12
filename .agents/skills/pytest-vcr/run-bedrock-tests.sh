#!/bin/bash
# Run tests with AWS bedrock profile (assumes bedrock-test-role via ~/.aws/config).
# Usage: .claude/skills/pytest-vcr/run-bedrock-tests.sh [pytest args...]
set -e

export AWS_PROFILE=bedrock
export AWS_DEFAULT_REGION=us-east-1
unset AWS_BEARER_TOKEN_BEDROCK AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

source .env
exec uv run pytest "$@"
