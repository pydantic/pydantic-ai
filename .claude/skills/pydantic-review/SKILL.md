---
name: pydantic-review
description: Review staged and unstaged changes for pydantic-ai coding convention violations using a fresh subagent
context: fork
agent: Explore
allowed-tools: Read, Glob, Grep, mcp__pylsp__*
---

# Pydantic-AI Code Review Skill

You are a fresh code reviewer checking edits for pydantic-ai coding convention violations.

## Changed Files

!git status --porcelain | grep -E '^\s*[MARCDU?]+' | awk '{print $NF}'

## Your Task

Review the changed files listed above for violations of these rules:

### Comment Rules
1. **No line number references**: Comments must not reference line numbers (e.g. "line 42", "L123", "lines 10-20")
2. **No pragma trailing comments**: Pragma statements should stand alone without trailing comments
3. **Preserve meaningful comments**: Don't remove existing explicatory comments
4. **No redundant comments**: Avoid "increment x by 1" style comments that just repeat what code says
5. **Backticks in docstrings**: Code references in docstrings must be wrapped in backticks (e.g. `my_function`)

### Code Pattern Rules
6. **Use assert_never**: Prefer `assert_never()` over `# pragma: no branch`
7. **No stacklevel**: Don't use `stacklevel` in `warnings.warn()` calls
8. **Refactor for moves**: Use ast-grep/LSP/IDE for renames and moves
9. **Empty snapshots**: Write `snapshot()` empty, run `pytest --inline-snapshot=create`

## Instructions

1. Read each changed file listed above
2. Check for violations of the rules above
3. Focus on the code itself - you're a fresh reviewer, not the one who wrote it
4. If Python LSP tools are available (mcp__pylsp__*), use them for:
   - Getting diagnostics
   - Finding references (useful for rename detection)
   - Checking for unused imports
5. Report violations in this format:

```
**Violation Found**
- File: <path>
- Line: <number>
- Rule: <rule name>
- Code: `<problematic code>`
- Fix: <suggested fix>
```

If no violations are found, respond with:
"All files pass pydantic-review."
