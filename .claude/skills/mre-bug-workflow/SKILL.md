---
name: mre-bug-workflow
description: Enforces MRE-first bug fixing. When given an issue link for a bug fix, automatically create minimal reproducible example scripts before implementing any fix. Uses UV inline dependencies to test against both PyPI release and local branch.
---

# MRE-Based Bug Fix Workflow

When given a GitHub issue link for a bug fix, follow this workflow AUTOMATICALLY.

## Trigger Conditions

Activate this workflow when:
- User shares a GitHub issue link
- Issue describes a bug (not a feature request)
- User wants to fix the bug

## Phase 1: Create MRE (BEFORE any implementation)

1. **Read the issue** via `gh issue view` or `gh api`
2. **Extract or create MRE** - distill to minimal reproduction
3. **Create `local-notes/mre/` folder** if it doesn't exist
4. **Create two scripts**:

### `mre_release.py` (tests against PyPI)

```python
# /// script
# dependencies = ['pydantic-ai']
# ///
"""MRE for issue #XXXX - [brief description]
Expected: [what should happen]
Actual: [what happens instead]
"""
# Minimal reproduction code
```

### `mre_branch.py` (tests against local branch)

```python
# /// script
# dependencies = ['pydantic-ai @ file:///PATH/TO/WORKTREE']
# ///
"""MRE for issue #XXXX - tests fix from branch [branch-name]"""
# Same code as mre_release.py
```

## Phase 2: Verify Bug Exists

```bash
uv run local-notes/mre/mre_release.py
```

**STOP if bug doesn't reproduce** - investigate or ask user for clarification.

## Phase 3: Implement Fix

Proceed with normal implementation.

## Phase 4: Verify Fix Works

```bash
# Shows bug (release version)
uv run local-notes/mre/mre_release.py

# Shows fix (local branch)
uv run local-notes/mre/mre_branch.py
```

## Notes

- UV inline deps (`# /// script`) run with specific versions without affecting env
- `file://` syntax points to local source for unreleased changes
- Keep MRE minimal - just enough to demonstrate the bug
- Reference these scripts in PR description as verification
