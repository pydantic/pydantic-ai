#!/bin/bash
# Sync script for pulling Claude Code Mobile reports from fork

set -e

echo "🔄 Syncing reports from Claude Code Mobile..."

# Fetch latest changes from fork
git fetch origin

# Check if there are changes in claude/ directory
if git diff --quiet HEAD origin/main -- claude/; then
    echo "✅ Already up to date - no new reports from Mobile"
else
    echo "📥 New reports found, pulling changes..."
    git pull origin main --no-edit
    echo "✅ Sync complete! New reports from Mobile are now available."
fi
