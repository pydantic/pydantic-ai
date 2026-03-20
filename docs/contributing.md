We'd love you to contribute to Pydantic AI!

## ⚠️ Before You Start

**All PRs must be linked to an existing, approved GitHub issue.** PRs submitted without a linked issue, or linked to an issue without maintainer input, will be closed without review.

This is not bureaucracy for its own sake. We receive a very high volume of pull requests, and experience has shown that PRs submitted without prior discussion are overwhelmingly unlikely to be mergeable, regardless of their quality. The issue-first process protects everyone's time: yours, ours, and the community's.

### The Process

1. **Search existing issues** to see if your bug/feature has already been reported or requested.
2. **Open a new issue** using the appropriate [issue template](https://github.com/pydantic/pydantic-ai/issues/new/choose) if one doesn't exist.
3. **Wait for maintainer feedback** on the issue before starting work. A maintainer will either:
   - Confirm the issue and indicate it's ready for a PR, or
   - Ask clarifying questions or suggest a different approach, or
   - Decline the issue with an explanation.
4. **Comment on the issue** to indicate you'd like to work on it, so others don't duplicate effort.
5. **Submit your PR** following the PR template, referencing the issue with `Closes #<number>`.

### Exceptions

The following types of changes may be submitted without a prior issue, but must still follow the PR template:

- **Typo fixes** in documentation (single-word or punctuation corrections)
- **Broken link fixes** in documentation

Everything else, including documentation rewrites, refactors, "cleanup" PRs, and new features, requires an issue first.

## PR Limits for New Contributors

To ensure quality reviews and prevent overload, we limit simultaneous open PRs for newer contributors:

- **First-time contributors** (0 merged PRs): 1 open PR at a time
- **1 merged PR**: up to 2 open PRs
- **2 merged PRs**: up to 3 open PRs
- **3+ merged PRs**: no limit

PRs that exceed your limit will be closed automatically. Please wait for your existing PRs to be reviewed and merged before opening new ones.

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

## Quality Standards

All contributions must meet the following standards. PRs that don't meet these will not be merged:

### Code Quality
- Modern, idiomatic Python (3.10+)
- Full type safety, no unnecessary `cast()` or `Any` annotations
- Follow existing code patterns and conventions in the module you're modifying
- Run `make format` and `make lint` before submitting

### Testing
- 100% test coverage for new code paths
- Run `make test` to verify all tests pass
- Run `make typecheck` to verify type safety
- Prefer integration tests with recorded API responses over unit tests with mocks
- Use `inline-snapshot` for complex assertion values

### Documentation
- Update relevant docs for any behavior changes
- Add docstrings for new public APIs
- Run `make docs-serve` to verify documentation renders correctly
- Code examples in docs must be testable (avoid `test="skip"` unless truly necessary)

### PR Requirements
- Fill in the PR template completely. PRs with incomplete templates will be closed
- PR title should be suitable for a release changelog entry
- Keep PRs focused. One logical change per PR
- No breaking changes per the [version policy](version-policy.md)

## Documentation Changes

To run the documentation page locally, run:

```bash
uv run mkdocs serve
```

## AI-Assisted Contributions

We welcome contributions that use AI tools as part of the development process. However:

- **You are responsible for every line of code you submit.** "The AI wrote it" is not an explanation for bugs, style violations, or missing tests.
- **Check the "AI generated code" box** in the PR template if any AI tools were used. This must be done manually by the human author, not by the AI tool.
- **AI-generated PRs that show no evidence of human understanding** (e.g., no meaningful commit messages, no response to review feedback, generic descriptions) will be closed.
- **Read and follow the full contribution process above.** Using an AI tool does not exempt you from the issue-first requirement.

## Rules for adding new models to Pydantic AI {#new-model-rules}

To avoid an excessive workload for the maintainers of Pydantic AI, we can't accept all model contributions, so we're setting the following rules for when we'll accept new models and when we won't. This should hopefully reduce the chances of disappointment and wasted work.

- To add a new model with an extra dependency, that dependency needs > 500k monthly downloads from PyPI consistently over 3 months or more
- To add a new model which uses another models logic internally and has no extra dependencies, that model's GitHub org needs > 20k stars in total
- For any other model that's just a custom URL and API key, we're happy to add a one-paragraph description with a link and instructions on the URL to use
- For any other model that requires more logic, we recommend you release your own Python package `pydantic-ai-xxx`, which depends on [`pydantic-ai-slim`](install.md#slim-install) and implements a model that inherits from our [`Model`][pydantic_ai.models.Model] ABC

If you're unsure about adding a model, please [create an issue](https://github.com/pydantic/pydantic-ai/issues).
