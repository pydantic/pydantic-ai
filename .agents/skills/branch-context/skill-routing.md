# Skill routing

Pick the canonical entry point for the user's intent. **One arrow per intent.** If two skills look right, the one listed here wins.

| Intent | Entry point |
|--------|-------------|
| Start work on a new issue | `/initialize-worktree` |
| Pick up an existing PR (yours or someone else's) | `/adopt-pr` |
| Re-sync `issue-brief.md` after new issue comments | `/refresh-issue-brief` |
| Hand off / clear and continue | EPEH in `branch-context/SKILL.md` (its `/handoff` alias for the persist steps alone) |
| Review the branch diff before pushing | `/pre-push-review` |
| Open a PR, or push a commit to one, and carry it to green | `/pushing-commits-to-the-repo` |
| Address unresolved PR review comments | `/address-feedback` |
| A submitted patch fixes only a narrow symptom of the reported pain | `/complete-partial-pr` |
| Add support for a newly-released model id | `/add-new-model` |
| Wire a provider API capability (caching, thinking, service tier, …) through the library | `/adding-a-provider-api-feature` |
| Second opinion on a big new-feature PR, grounded in what the underlying API requires | `/poweruser-feature-audit` |
| Discovered real but out-of-scope work on a complete, stable PR | `/file-followup-issue` |
| Record / re-record / debug VCR cassettes | `/testing-skill` |

## When two intents collide

- **Reviewing before a push** → `/pre-push-review` is the review; `/pushing-commits-to-the-repo` dispatches it inside the push loop, at most three times per PR during one task. Going through the push skill keeps that count honest.
- **"Look at the PR feedback and check CI"** → `/address-feedback` owns the comments; CI belongs to the push loop in `/pushing-commits-to-the-repo`. Address the feedback first, then let the push loop carry the result to green.
- **Bug fix on a PR you did not start** → `/adopt-pr` first, so the fix has a brief and a decisions log to record against, then fix.
- **A contributor patch that looks too narrow** → `/complete-partial-pr` before reviewing the diff line by line. Scope comes first; a line-level review of the wrong scope is wasted.
