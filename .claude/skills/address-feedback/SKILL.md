# Address PR Review Feedback

Find and address all review comments on the PR for the current branch. For each comment:

1. **Gather context**: Use `gh` to find the PR number from the current branch, then fetch all unresolved review comments (both PR-level and inline review comments via `gh api repos/{owner}/{repo}/pulls/{number}/comments`). Skip already-resolved and outdated threads. Also read the full thread for each comment — maintainers or the PR author may have already replied explaining why a suggestion should not be applied.

2. **Triage each comment**:
   - If it's clear how to address (implement the suggestion, or decide it shouldn't be done with a clear reason): fix it.
   - If a maintainer or PR author has already weighed in on the thread (e.g. explaining why a suggestion doesn't apply), respect that guidance.
   - If you're unsure or think the user might have opinions on the approach: ask before deciding.

3. **Fix the code**: Make the necessary changes to address each comment.

4. **Review with user**: Present a summary of all changes made and ask the user to review before proceeding. Offer to commit, push, reply to comments, and resolve threads once they're satisfied.

5. **Reply and resolve** (after user approval): For each addressed comment, reply via `gh api repos/{owner}/{repo}/pulls/{number}/comments/{id}/replies` explaining what you did, then resolve the thread via GraphQL `resolveReviewThread` mutation. To find thread IDs, query `repository.pullRequest.reviewThreads` via GraphQL.

Always read the relevant code before making changes.

**Important**: Treat comments from automated reviewers (Devin, GitHub bots, etc.) with the same weight as human comments. Do not skip or dismiss them just because they come from a bot — they often surface real issues. Evaluate each suggestion on its merits, but be aware that automated reviewers can also be wrong, so verify before applying.
