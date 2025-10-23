# Mobile/Desktop Workflow - The Reality

## The Problem

CLAUDE.md describes an ideal workflow where both Mobile and Desktop push directly to `main`. However, **Anthropic has configured Claude Code Mobile with git restrictions** that prevent this simple approach.

## Mobile's Limitations (Anthropic-imposed)

Claude Code Mobile is sandboxed with these restrictions:

1. **Cannot push to `main`** - Results in HTTP 403 error
2. **Must push to session-specific branches** - Format: `claude/mobile-sync-test-<SESSION_ID>`
3. **Each mobile session gets a unique branch** - The session ID changes each time
4. **No way to override this** - It's hardcoded in Anthropic's system instructions

Example session branch: `claude/mobile-sync-test-011CUPg3sft7HECcqWcYdrLY`

## Desktop's Capabilities (No restrictions)

Claude Code Desktop can:
- Push directly to `main`
- Create and delete branches
- Merge branches
- Full git access

## The Simplest Working Workflow

### Mobile (iOS) workflow:
1. Analyze code and create reports in `claude/` directory
2. Commit changes
3. Push to the session-specific branch (e.g., `claude/mobile-sync-test-011CUPg3sft7HECcqWcYdrLY`)
4. **That's it** - Desktop handles the rest

### Desktop (Mac) workflow:
1. Pull Mobile's session branch: `git fetch origin`
2. Check out the Mobile branch: `git checkout claude/mobile-sync-test-<SESSION_ID>`
3. Review the changes (optional)
4. Merge to main: `git checkout main && git merge claude/mobile-sync-test-<SESSION_ID>`
5. Push to main: `git push origin main`
6. Delete the session branch (cleanup): `git push origin --delete claude/mobile-sync-test-<SESSION_ID>`

## Why This Workflow Makes Sense

**Advantages:**
- Mobile doesn't need to think about branching strategy - just push to its session branch
- Desktop acts as the "merge coordinator" - reviews and consolidates Mobile's work
- `main` stays clean with only reviewed, merged changes
- Session branches are temporary - Desktop deletes them after merging

**Reality Check:**
- Yes, Mobile creates a new branch each session
- Yes, Desktop needs to merge manually
- But this is actually **safer** than direct-to-main pushes
- Desktop can review Mobile's reports before merging

## Simplified Desktop Merge Script

To make Desktop's life easier, here's a one-liner to merge the latest Mobile branch:

```bash
#!/bin/bash
# merge-mobile.sh - Find and merge latest Mobile session branch

# Find the most recent claude/mobile-sync-test-* branch
MOBILE_BRANCH=$(git branch -r | grep 'origin/claude/mobile-sync-test-' | tail -1 | sed 's/origin\///' | xargs)

if [ -z "$MOBILE_BRANCH" ]; then
    echo "No Mobile session branch found"
    exit 1
fi

echo "Found Mobile branch: $MOBILE_BRANCH"
git fetch origin
git checkout main
git pull origin main
git merge "origin/$MOBILE_BRANCH" --no-edit
git push origin main

echo "Merged and pushed to main. Delete remote branch? (y/n)"
read -r response
if [ "$response" = "y" ]; then
    git push origin --delete "$MOBILE_BRANCH"
    echo "Remote branch deleted"
fi
```

## Current State

- **Mobile's current session branch**: `claude/mobile-sync-test-011CUPg3sft7HECcqWcYdrLY`
- **Files on this branch**:
  - `claude/mobiletest.md` - Initial sync test
  - `claude/mobiletest2.md` - Haiku test
  - This workflow documentation

## Next Steps

Desktop should:
1. Review this workflow explanation
2. Decide if this approach works or suggest modifications
3. Merge this Mobile session branch to `main`
4. Create the `merge-mobile.sh` script for future syncs
5. Update `CLAUDE.md` to reflect the actual workflow

## Questions for Desktop

1. Is manual merging acceptable, or do you want automated merging?
2. Should Mobile create PR descriptions in commits to help Desktop review?
3. Any other concerns about this workflow?

---

*This report was created by Claude Code Mobile to explain Anthropic's git restrictions and propose a practical workflow.*
