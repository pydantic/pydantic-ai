---
name: address-feedback
description: Find and address unresolved PR review comments for the current branch, then continue
  the canonical push, reply, reaction, and resolution workflow.
---

# Address PR Review Feedback

Find and address all review comments on the PR for the current branch. For each comment:

1. **Gather context**: Use `gh` to find the PR number from the current branch, then fetch all unresolved review comments (both PR-level and inline review comments via `gh api repos/{owner}/{repo}/pulls/{number}/comments`). Skip already-resolved and outdated threads. Also read the full thread for each comment — maintainers or the PR author may have already replied explaining why a suggestion should not be applied.

2. **Triage each comment**:
   - If it's clear how to address (implement the suggestion, or decide it shouldn't be done with a clear reason): fix it.
   - If a maintainer or PR author has already weighed in on the thread (e.g. explaining why a suggestion doesn't apply), respect that guidance.
   - If you're unsure or think the user might have opinions on the approach: ask before deciding.

3. **Fix the code**: Make the necessary changes to address each comment.

4. **Continue the PR loop**: Follow `pushing-commits-to-the-repo` from its `Before you push` section.

5. **Use the canonical close-out**: Apply every required reply, reaction, and resolution step from that workflow. For each completed comment, explain what changed or why no change was needed. Then resolve the thread via GraphQL `resolveReviewThread`. Leave threads open only when a decision or another person's response is pending.

Always read the relevant code before making changes.

**Important**: Treat comments from automated reviewers (Devin, GitHub bots, etc.) with the same weight as human comments. Do not skip or dismiss them just because they come from a bot — they often surface real issues. Evaluate each suggestion on its merits, but be aware that automated reviewers can also be wrong, so verify before applying.
