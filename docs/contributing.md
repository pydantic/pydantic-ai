We'd love you to contribute to Pydantic AI!

## How we work — the short version

Pydantic AI is maintained by a small team. We set our own priorities based on what benefits the most users, and we work through issues and PRs in that order — not in the order they arrive.

- **Found a bug?** Open an issue with a clear description and a minimal reproducible example. Including a [Logfire](https://logfire.pydantic.dev/) trace link helps us debug dramatically faster.
- **Want a feature or API change?** Open an issue describing the problem you're solving. Do not start with code.
- **Want to help build a feature?** Comment on the issue explaining why you need it and what context you bring. We call this being a "champion" — more on that below.
- **Have a fix or code to share?** Make sure a maintainer has agreed to the approach on the issue and assigned you. Then open a PR.

The rest of this page explains why we work this way and what to expect.

## Before you write code

For anything non-trivial, align with a maintainer on the approach before writing code. A pre-aligned PR is much faster to land than one we're seeing cold.

### Trivial fixes

Typos, broken links, small doc improvements, obvious one-line fixes: just open a PR. No issue needed.

### Bug fixes

If the fix could reasonably go more than one way, or you're unsure it's actually a bug: open an issue first. Include a minimal reproducible example and ideally a [Logfire trace link](https://logfire.pydantic.dev/) showing the problem. For well-scoped bugs, we may generate a fix internally — the most valuable thing you can do is file a clear report and then validate that the fix works for your use case.

### Features, integrations, or API changes

Before writing code, ask whether the change needs to live in core at all. Most new agent behaviors belong in [**Pydantic AI Harness**](https://github.com/pydantic/pydantic-ai-harness), the official capability library — not in this repo. Pydantic AI core is for the agent loop, model providers, and capabilities that require model-specific support or are fundamental to the agent experience. Standalone capabilities — guardrails, memory, context management, file system access, etc. — belong in the harness, where they can iterate faster. See [What goes where?](harness.md#what-goes-where) for the full distinction.

**If your idea is a capability**, open an issue on [pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness/issues) instead. You can also publish capabilities as your own package using the `pydantic-ai-<name>` convention — see [Publishing capability packages](extensibility.md#publishing-capability-packages). Once a capability has real users and a stable shape, we can talk about upstreaming to harness or core.

If it does belong in core:

1. **Search first.** If an existing issue covers your need, comment there. If the closest match is only related, open a new issue and link it.
2. **Describe the problem, not just the solution.** Tell us what you're building, what's blocking you, and what you've tried. This context matters more than code.
3. **Propose a plan before building.** Post the shape of the solution on the issue, or open a draft PR with just a `PLAN.md`. For larger features, we do short video calls with contributors to iterate on the design — a 20-minute call often saves weeks of async review cycles.
4. **Wait for assignment.** A maintainer needs to agree on the approach and assign the issue to you before you open a PR. Unassigned PRs may be auto-closed.

!!! warning
    Writing a large feature PR without prior alignment is the most common way for a contribution to stall or be closed.

## Champions

A "champion" is someone who needs a feature, has context on the problem, and is willing to invest time to help us get it right. If you want to champion a feature:

- Comment on the issue explaining: what you're building, why you need this, and what you can contribute (domain knowledge, testing, validation).
- We prioritize features where one or more champions with production use cases have stepped up. A feature with no champion stays in the backlog until either we prioritize it ourselves or someone with real context shows up.
- Being a champion doesn't mean writing the code. It means shaping the plan and validating the result. For significant features, we'll set up a call to iterate on the design together.

Champions are credited as co-authors when the feature ships.

## What to expect during review

### We review PRs in our priority order, not submission order

We do not automatically triage every new PR. PRs on issues we have not pre-aligned on are not in our review queue, regardless of how well written they are. If no maintainer has agreed to the change on an issue and assigned it to you, assume we have not seen your PR.

Even for PRs with code we've previously engaged with: we treat all contributed code as a starting point, not a finished product. We review and prioritize PRs based on the feature's importance to the project, not on how much effort went into the code. This is a change from how open source traditionally worked, and we'd rather be honest about it than leave PRs sitting with no signal.

**If you want to know where your PR stands**, the best thing to do is ping `#pydantic-ai` on [Pydantic Slack](https://logfire.pydantic.dev/docs/join-slack/).

### We may rewrite or supersede your code

We treat contributed code as illustrative: a starting point that shows the shape of the change and proves the approach works, not the final form we merge. The most useful thing you can give us for a non-trivial change is a plan plus a working example — not a polished, merge-ready implementation.

On any PR, we may push commits to your branch, open a follow-up PR that supersedes yours, or rewrite from scratch. For security reasons, we lean toward rewriting contributed code rather than merging as-is. You will still be credited as the original author.

Please don't spend effort chasing green CI, addressing every automated review comment, or rebasing for merge conflicts on a PR we haven't pre-aligned on. If we take the change forward, that polish gets thrown away when we rewrite. Get the approach working, then stop and ping us on Slack.

### Automated review is advisory, not a gate

PRs are automatically reviewed by Devin and our own tooling. These reviews are advisory:

- A bot approval does not mean your PR is ready to merge. Only a human maintainer's review counts.
- A bot finding does not mean you must act on it. If you disagree, say so.
- If automated review is generating noise on your PR, tell us. We use that feedback to retune the tooling.

### Priority

We receive far more contributions than we can review, and we focus where it has the most impact. We cannot promise to get to every PR, even good ones, and we'd rather say so up front than leave your work open indefinitely with no signal.

How we weigh priorities:

- **User demand** -- features that more users need get priority. Champion-backed features with production use cases outrank speculative additions.
- **Provider significance** -- work that affects frontier providers (Anthropic, OpenAI, Google) or providers we know are heavily used gets priority. A model integration for a niche provider will wait; a fix for Anthropic won't.
- **Roadmap alignment** -- features that align with our current focus areas get priority. Right now that includes the capabilities/hooks API, provider-adaptive tools, and the [Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness) capability library.
- **Capabilities over core** -- features that could live as a [capability](capabilities.md) should go to [Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness) or ship as your own package — that's often the fastest path. Once it has traction, come back and we can talk about upstreaming.

## If your PR or issue has gone quiet

1. Ping `#pydantic-ai` on [Pydantic Slack](https://logfire.pydantic.dev/docs/join-slack/) with a link.
2. Say what you need: "Can you take a look?", "I'm blocked — is this on your radar?", or "Should I close this?" are all fine.
3. If you've been waiting weeks without any human response, flag it. That's a process failure on our side and we want to know.

## Installation and Setup

Clone your fork and cd into the repo directory

```bash
git clone git@github.com:<your username>/pydantic-ai.git
cd pydantic-ai
```

Install `uv` (version 0.4.30 or later) and `pre-commit`:

- [`uv` install docs](https://docs.astral.sh/uv/getting-started/installation/)
- [`pre-commit` install docs](https://pre-commit.com/#install)

To install `pre-commit` you can run the following command:

```bash
uv tool install pre-commit
```

Install `pydantic-ai`, all dependencies and pre-commit hooks

```bash
make install
```

## Running Tests etc.

We use `make` to manage most commands you'll need to run.

For details on available commands, run:

```bash
make help
```

To run code formatting, linting, static type checks, and tests with coverage report generation, run:

```bash
make
```

## Documentation Changes

To run the documentation page locally, run:

```bash
uv run mkdocs serve
```

## Rules for adding new models to Pydantic AI {#new-model-rules}

To avoid an excessive workload for the maintainers of Pydantic AI, we can't accept all model contributions, so we're setting the following rules for when we'll accept new models and when we won't. This should hopefully reduce the chances of disappointment and wasted work.

- To add a new model with an extra dependency, that dependency needs > 500k monthly downloads from PyPI consistently over 3 months or more
- To add a new model which uses another models logic internally and has no extra dependencies, that model's GitHub org needs > 20k stars in total
- For any other model that's just a custom URL and API key, we're happy to add a one-paragraph description with a link and instructions on the URL to use
- For any other model that requires more logic, we recommend you release your own Python package `pydantic-ai-xxx`, which depends on [`pydantic-ai-slim`](install.md#slim-install) and implements a model that inherits from our [`Model`][pydantic_ai.models.Model] ABC

If you're unsure about adding a model, please [create an issue](https://github.com/pydantic/pydantic-ai/issues).
