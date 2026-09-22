---
name: address-feedback
description: Find and address unresolved PR review comments for the current branch, then continue
  the canonical push, reply, reaction, and resolution workflow.
---

# Address PR Review Feedback

Find and address all review comments on the PR for the current branch. For each comment:

1. **Gather context**: Use `gh` to find the PR number from the current branch, then fetch all unresolved review comments (both PR-level and inline review comments via `gh api repos/{owner}/{repo}/pulls/{number}/comments`). Skip already-resolved and outdated threads. Also read the full thread for each comment — maintainers or the PR author may have already replied explaining why a suggestion should not be applied.

2. **Triage each comment**: A review comment — bot or human — is evidence to weigh, never an acceptance criterion. This PR's acceptance criteria are its linked issue, the repository instructions, and settled maintainer decisions. Sort each comment into *fix*, *decline*, *escalate* or *file*; one that clears the gates below is a *fix*.
   - Read the thread first. If a maintainer has already weighed in, that settles it; your own earlier reply does not.
   - **If the finding claims a defect, reproduce it before you write the fix.** Run a script, a failing test, or a snippet. A plausible reading is not a reproduction, and a severity label is the reviewer's guess. If it doesn't reproduce, reply with what you ran and why that run covers the path the finding names, then react 👎. A repro you could not run — no credentials, no worker, no cassette — is not a refutation: ask the user driving you instead. A finding about docs, tests, naming, or API shape has nothing to reproduce; weigh it against the repository instructions.
   - **List the production files the fix would touch.** A file outside the ones this PR already changes for its issue — a shared core module especially (`pydantic_ai/_agent_graph.py`, `pydantic_ai/_run_context.py`, `pydantic_ai/messages.py`, `pydantic_ai/tools.py`, a base class, a serialized dataclass) — means the finding has outgrown the PR's stated scope. Implement it here only on the root `AGENTS.md` bar for the case at hand — a sibling field, provider or model needs the same defect reproduced there; a shared protocol, helper or abstraction needs the narrow fix to be unavailable, or the refactor to be itself the confirmed fix — and only when including it neither explodes scope nor delays an already mergeable PR. Otherwise escalate it per `pushing-commits-to-the-repo` ("Escalate real trade-offs, don't guess") when it needs a maintainer decision, or file it as its own issue when it is real but belongs elsewhere. Tests and documentation this change already owes are in scope, not an expansion.
   - **Name who else moves.** If the users whose observable behavior changes are a wider set than the users who hit the reported bug, the finding breaks the root `AGENTS.md` requirement to "leave behavior unchanged for users who aren't hitting the problem you are solving". Re-scope or escalate; never implement and document.
   - Ask the user driving you when the call is a preference they may hold. A finding can be wrong on the merits wherever it sits: decline that one with code evidence, per `pushing-commits-to-the-repo` ("Invalid"), rather than filing it.

3. **Fix the code**: Make the changes for each comment triage sent to *fix*.

4. **Continue the PR loop**: Follow `pushing-commits-to-the-repo` from its `Before you push` section.

5. **Use the canonical close-out**: Apply every required reply, reaction, and resolution step from that workflow. For each completed comment, explain what changed or why no change was needed. Then resolve the thread via GraphQL `resolveReviewThread`. Leave threads open only when a decision or another person's response is pending.

Always read the relevant code before making changes.

**Important**: Automated reviewers surface real issues and are never skipped — every finding gets a reply and a reaction. But a bot cannot approve a scope expansion, and its severity label carries no authority. A `HIGH` on a defect that does not reproduce is a 👎. Refuting one leaves `CI Review`'s `REQUEST_CHANGES` standing; a later push re-runs the review, and the verdict clears only if that run stops finding the `HIGH`. Say so when you hand the PR back.
