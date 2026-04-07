We'd love you to contribute to Pydantic AI!

## How review works here, and why this page exists

Pydantic AI is maintained by a small team, and the volume of issues, PRs, and GitHub notifications this project generates has grown faster than we can keep up with. We read more than we respond to, and things do get lost when a thread is buried under newer notifications. We know that's been frustrating for some contributors, and we're sorry. This page is our attempt to be honest about how review actually works here and what you can do to get a change landed.

## Before you write code

For anything non-trivial, please align with a maintainer on the approach before you write the code. A pre-aligned PR is much faster to land than one we're seeing cold.

### Trivial fixes

Typos, broken links, small doc improvements, obvious bug fixes, docstring clarifications: just open a PR. No issue needed.

### Bug fixes with a judgment call

If the fix could reasonably be done more than one way, or if you're not sure the behaviour is actually a bug: open an issue, or comment on an existing one, and describe what you plan to do.

### Features, new integrations, or public API changes

Please don't start with code. Instead:

1. Search existing issues and PRs first. If a maintainer has already weighed in on a similar proposal, that's your source of truth.
2. Propose the shape of the solution before building it. You can do this as a comment on the relevant issue, or as a draft PR containing only a `PLAN.md` file.
3. For larger features, new provider integrations, or anything that adds public API surface, reach out in [Slack](https://logfire.pydantic.dev/docs/join-slack/) first. We've started doing short plan and spec review calls with contributors on bigger work, because iterating on design in GitHub comments has turned out to be slow. A 20-minute call often saves weeks of review cycles.

!!! warning
    Writing a large feature PR without prior alignment is the most common way for a contribution to stall.

## What to expect during review

### Silence is rarely rejection

There are very few maintainers on this project and well over a hundred PRs open at any given time. A PR sitting without a response usually does not mean we've looked at it and decided not to act; more often it means it arrived during a week we were focused elsewhere and then got buried. If you need our attention, you may need to ask for it more than once, and Slack is the most reliable channel.

### We may take over small and medium PRs

For smaller fixes and medium-sized changes, we sometimes push commits directly to your branch or open a follow-up PR that supersedes yours, rather than iterating over multiple review rounds. This is a time trade-off on our end; you'll still be credited as the original author. If you'd rather drive the changes yourself, say so on the PR and we'll leave it to you.

### Automated review is advisory, not a gate

PRs in this repo are automatically reviewed by Devin and by our own tooling. Treat these reviews like a linter, not a merge blocker:

* A "no issues found" comment from a bot does not mean your PR is ready to merge. Only a human maintainer's comment counts as review.
* A bot finding does not mean you have to act on it. If you disagree, say so on the PR.
* If automated review is generating noise rather than signal on your PR, please tell us. We use that to retune the tooling.

### Priority

When we work through the PR queue, we tend to prioritise in roughly this order: security and correctness fixes, regressions, changes that unblock roadmap work, contributions from people we've collaborated with before, then everything else.

## If your PR or issue has gone quiet

1. Ping us in `#pydantic-ai` on [Pydantic Slack](https://logfire.pydantic.dev/docs/join-slack/) with a link.
2. Say what you'd like to happen. "Can you take a look?", "I'm blocked on this and want to know if it's on your radar", and "Do you want me to close this?" are all fine.
3. If you've been waiting weeks without any human response, please flag it. That's a process failure on our side.

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
