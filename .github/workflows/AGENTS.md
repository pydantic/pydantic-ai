# Workflows in this directory

## Agentic workflows (`gh-aw`)

The `pydantic-ai-*` workflows in this directory are [agentic workflows](https://github.com/githubnext/gh-aw) authored as human-editable `<name>.md` sources (frontmatter + prompt) that **compile** to a generated `<name>.lock.yml`. GitHub Actions runs the `.lock.yml`, never the `.md`.

- **Never hand-edit a `*.lock.yml`.** It is generated — the header says so. Manual edits are silently overwritten on the next recompile, and until then the running workflow diverges from its source.
- **After editing a workflow `*.md` source, recompile and commit the regenerated `*.lock.yml` in the same change** — a `*.md` edit without its recompiled lock is incomplete, and source and lock drift apart:

  ```
  gh aw compile
  ```

- **Recompilation is required for anything the lock bakes in:** a source's frontmatter (`on:` triggers, `permissions`, `tools`, `safe-outputs`, jobs, path/`detect` filters) and its `imports:` shared fragments (`shared/*.md`) are inlined into the lock at compile time.
- **Exception — runtime-resolved prompts need no recompile.** Agent prompts under `shared/prompts/` are fetched at run time (via the `fetch-dynamic-prompt` action / a Logfire-managed variable), not baked into the lock, so editing one takes effect on the next run without recompiling.

## The `auto-review` label (`bots.yml`)

`bots.yml` has a `review` job that a maintainer triggers by adding the **`auto-review`** label to a PR. Its lifecycle is the part agents get wrong:

1. a maintainer adds `auto-review`,
2. the `review` job runs and posts its review,
3. the job removes the label again — `if: always()`, so it goes even when the review fails.

**A PR's current labels therefore say nothing about whether it was auto-reviewed.** Every PR that has ever been auto-reviewed carries no `auto-review` label today. To find out whether — and when — a PR was reviewed, read the label *events*, not the label set:

```
gh api repos/pydantic/pydantic-ai/issues/<number>/events --paginate --jq '.[] | select(.label.name == "auto-review")'
```

Filtering on the current label (`gh pr view --json labels`) matches nothing and returns an empty result rather than an error.

Two consequences:

- `pydantic-ai-pr-review` reads the labels once, when its `opened`/`synchronize`/`ready_for_review` event fires, and skips if `auto-review` is already present — so the two reviewers usually don't review the same push. The label bot is the maintainer-triggered escalation; the agentic workflow is the automatic pass. Because the label is stripped after the run, a *later* push to the same PR is reviewed by the agentic workflow again. There is a race: labelling a PR while an agentic run for the same head SHA is already in flight starts a second review — `bots.yml` doesn't cancel the running one — so wait for an in-flight agentic run to finish (or cancel it) before adding the label.
- The label is safe on external-contributor PRs: a maintainer applies it, the checkout does not persist credentials, and the job refuses to run on **any** PR — fork or same-repo branch — whose diff touches `AGENTS.md`, `CLAUDE.md`, or `.claude/`, since the review loads the head's config into its own prompt. A consequence worth knowing: a PR that edits agent config can never be auto-reviewed, and its `Category Classify` check fails by design rather than by breakage.
