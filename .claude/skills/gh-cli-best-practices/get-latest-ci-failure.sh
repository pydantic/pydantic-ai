#!/bin/bash
# get-latest-ci-failure.sh

REPO="pydantic/pydantic-ai"
BRANCH=$(git branch --show-current)

# Get PR number
PR_NUM=$(gh pr list --head "$BRANCH" --repo "$REPO" --json number -q '.[0].number')

if [ -z "$PR_NUM" ]; then
    echo "No PR found for branch: $BRANCH"
    exit 1
fi

echo "PR #$PR_NUM"

# Get the failed test check link and extract run ID from it
FAILED_LINK=$(gh pr checks "$PR_NUM" --repo "$REPO" --json name,state,link -q '.[] | select(.state == "FAILURE" and (.name | startswith("test"))) | .link' | head -1)

if [ -z "$FAILED_LINK" ]; then
    echo "No failed test checks"
    exit 0
fi

# Extract run ID from link (format: .../runs/12345678/job/...)
RUN_ID=$(echo "$FAILED_LINK" | grep -oE 'runs/[0-9]+' | cut -d'/' -f2)

echo "Failed run ID: $RUN_ID"
echo "Link: $FAILED_LINK"
echo ""

# Get logs and extract Summary of Failures section, clean up the output
# Format is: job_name<TAB>step_name<TAB>timestamp content
gh run view "$RUN_ID" --repo "$REPO" --log-failed 2>&1 \
    | grep -A 15 "Summary of Failures" \
    | grep -v "^check" \
    | awk -F'\t' '{
        # Get the last field (timestamp + content), then strip the timestamp
        content = $NF
        sub(/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.Z]+[ ]?/, "", content)
        print content
    }' \
    | grep -v "##\[error\]"
