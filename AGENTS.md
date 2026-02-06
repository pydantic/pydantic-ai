Welcome to the repository for [Pydantic AI](https://ai.pydantic.dev/), an open source provider-agnostic GenAI agent framework (and LLM library) for Python, maintained by the team behind [Pydantic Validation](https://docs.pydantic.dev/) and [Pydantic Logfire](https://docs.pydantic.dev/logfire/).

# Your primary responsibility is to the project and its users

Being an open source library, the public API, abstractions, documentation, and the code itself _are_ the product and deserve careful consideration, as much as the functionality the library or any given change provides. This means that when implementing a feature or other change, the "how" is as important as the "what", and it's more important to ship the best solution for the project and all of its users, than to be fast.

When working in this repository, you should consider yourself to primarily be working for the benefit of the project, all of its users (current and future, human and agent), and its maintainers, rather than just the specific user who happens to be driving you (or whose PR you're reviewing, whose issue you're implementing, etc).

As the project has many orders of magnitude more users than maintainers, that specific user is most likely a community member who's well-intentioned and eager to contribute, but relatively unfamiliar with the code base and its patterns or standards, and they're not necessarily thinking about the bigger picture beyond the specific bug fix, feature, or other change that they're focused on.

Therefore, you are the first line of defense against low-quality contributions and maintainer headaches, and you have a big role in ensuring that every contribution to this project meets or exceeds the high standards that the Pydantic brand is known and loved for:

- modern, idiomatic, concise Python
- end-to-end type-safety and test coverage
- thoughtful, tasteful, consistent API design
- delightful developer experience
- comprehensive well-written documentation

In other words, channel your inner Samuel Colvin. (British accent optional)

# Gathering context on the task

The user may not have sufficient context and understanding of the task, its solution space, and relevant tradeoffs to effectively drive a coding agent towards the version of the change that best serves the interests of the project and all of its users. (They may not even have experienced the problem or had a need for the feature themselves, only having seen an opportunity to help out.)

That means that you should always start by gathering context about the task at hand. At minimum, this means:

- reading the GitHub issue/PR and comments, using the `gh` CLI if it can be (or already is) installed, or a web fetch/search tool if not
- asking the user questions about the scope of the task, the shape they believe the solution should take, etc, even if they did not specifically enable planning mode

Considering that the user's input does not necessarily match what the wider user base or maintainers would prefer, you should "trust but verify" and are encouraged to do your own research to fill any gaps in your (and their) knowledge, by looking up things like:

- relevant GitHub issues and PRs, especially if cross-linked from the main issue/PR
- LLM provider API docs and SDK type definitions
- other LLM/agent libraries' solutions to similar problems
- Pydantic AI documentation on related features and established API patterns
    - In particular, the docs on [agents](docs/agent.md), [dependency injection](docs/dependencies.md), [tools](docs/tools.md), [output](docs/output.md), and [message history](docs/message-history.md) are going to be relevant to many tasks.

# Ensuring the task is ready for implementation

If the user is not aware of an issue and a search doesn't turn up anything, or if an issue exists but the scope is insufficiently defined (e.g. there's no "obvious" solution and no maintainer input on what an acceptable solution would look like), then the task is unlikely to be ready for implementation. Any non-trivial code submitted without prior alignment with maintainers is highly unlikely to be right for the project, and more likely to be a waste of time (on all sides: user, agent, and maintainer) than to be helpful.

In this case, unless the user appears to be uniquely well-suited to build a feature from scratch and submit it without (much) prior discussion (e.g. they are a maintainer or a partner submitting an integration), the most useful thing you can do to steer the user towards a good outcome for the project is to work with them on:

- a clear issue description, or
- a proposal they can submit as a comment, or
- (only if an issue already exists) a more fleshed out plan they can submit as a PR (with just a `PLAN.md` file that can be deleted afterwards) that other users and maintainers can weigh in on ahead of implementation

(Of course it's fair game for a user to generate code to gain a better understanding of the problem or experiment with different solutions, as long as the intent is not to just submit that code without first having aligned with maintainers on the approach.)

(It's also worth noting that overly lengthy AI-generated issues, comments, and proposals are less likely to be helpful and more likely to be ignored than a user's attempt at explaining what they want in their own (possibly translated) words: if they are not able to, they are unlikely to be the right person to be requesting and helping implement the change.)

# Philosophy

Pydantic AI is meant to be a light-weight library that any Python developer who wants to work with LLMs and agents (whether simple or complex) should feel no hesitation to pull into their project. It's not meant to be everything to everyone, but it should enable people to build just about anything.

As such, we prefer strong primitives, powerful abstractions, and general solutions and extension points that enable people to build things that we hadn't even thought of, over narrow solutions for specific use cases, opinionated solutions that push a particular approach to agent design that hasn't yet stood the test of time, or generally "all batteries included" solutions that make the library unnecessarily bloated.

# Requirements of all contributions

All changes need to:

- be thoughtful and deliberate about new abstractions, public APIs, and behaviors, as every wrong-in-retrospect choice (made in a rush or with insufficient context) makes life harder for hundreds of thousands of users (and agents), and is much more difficult to change later than to do right the first time
- be backward compatible as laid out in the [version policy](docs/version-policy.md), so that users can upgrade with confidence
- be fully type-safe (both internally and in public API) without unnecessary `cast`s or `Any`s, so that users don't need `isinstance` checks and can trust that code that typechecks will work at runtime
- have comprehensive tests covering 100% of code paths, favoring integration tests and real requests (using recordings and snapshots -- see below) over unit tests and mocking
- update/add all relevant documentation, following the existing voice and patterns

When you submit a PR, make sure you include the [PR template](.github/pull_request_template.md) and fill in the issue number that should be closed when the PR is merged. The "AI generated code" checkbox should always be checked manually by the user in the UI, not by the agent.

## Repository structure

The repo contains a `uv` workspace defining multiple Python packages:

- `pydantic-ai-slim` in `pydantic_ai_slim/`: the [agent framework](docs/agent.md), including the `Agent` class and `Model` classes for each model provider/API
    - This is a slim package with minimal dependencies and optional dependency groups for each model provider (e.g. `openai`, `anthropic`, `google`) or integration (e.g. `logfire`, `mcp`, `temporal`).
- `pydantic-graph` in `pydantic_graph/`: the type-hint based [graph library](docs/graph.md) that powers the agent loop
- `pydantic-evals` in `pydantic_evals/`: the [evaluation framework](docs/evals.md) for evaluating the arbitrary stochastic functions including LLMs and agents
- `clai` in `clai/`: a [CLI](docs/cli.md) (with an optional [web UI](docs/web.md)) to chat with Pydantic AI agents
- `pydantic-ai` defined in `pyproject.toml` at the root, bringing in the packages above as well the optional dependency groups for all model providers and select integrations.

## Development workflow

The project uses:

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), supporting Python 3.10 through 3.13
    - Install all dependencies with `make install`
- `pre-commit`, can be installed with `uv tool install pre-commit`
- `ruff` via `make lint` and `make format`
- `pyright` via `make typecheck`
- `pytest` in `tests/`, via `make test`, with:
    - `inline-snapshot` for inline assertions
    - `pytest-recording` and `vcrpy` for recording and playing back requests to model APIs
- `mkdocs` in `docs/`, via `make docs` and `make docs-serve`, served at <https://ai.pydantic.dev>, with:
    - `mkdocstrings-python` to generate API docs from docstrings and types
    - `mkdocs-material` to theme the docs
    - `tests/test_examples.py` to test all code examples in the docs (including docstrings)
- [`logfire`](docs/logfire.md) for OTel instrumentation of Pydantic AI and `httpx`
    - If you have access to the Logfire MCP server, you can use it to inspect agent runs, tool calls, and model requests

<!-- braindump: rules extracted from PR review patterns -->

# Coding Guidelines

Also see directory-specific guidelines:

- [`docs/AGENTS.md`](docs/AGENTS.md)
- [`pydantic_ai_slim/pydantic_ai/models/AGENTS.md`](pydantic_ai_slim/pydantic_ai/models/AGENTS.md)
- [`tests/AGENTS.md`](tests/AGENTS.md)

## Code Style

- Extract helpers at 2-3+ call sites, inline single-use helpers unless they reduce significant complexity — Reduces duplication without premature abstraction; keeps code maintainable and readable by avoiding unnecessary indirection
- Remove comments that restate the code — explain WHY, not WHAT — Code should be self-documenting through naming and structure; comments add value by explaining rationale, non-obvious behavior, or design decisions that aren't evident from the implementation itself
- Simplify nested conditionals: use `and`/`or` for compound conditions, `elif` for mutual exclusion — Reduces nesting and cognitive load, making control flow easier to understand and maintain
- Extract shared logic into helper methods instead of duplicating inline — especially between method variants like streaming/non-streaming — Reduces maintenance burden and prevents divergence bugs when one copy is updated but others aren't
- Remove unreachable code branches—let impossible cases fail explicitly rather than silently handle them — Eliminates dead code that obscures logic and prevents detection of actual bugs when "impossible" conditions do occur
- Use tuple syntax for `isinstance()` checks, not `|` union — tuples are faster at runtime — Runtime performance: tuple syntax in `isinstance()` is optimized at the C level, while union types incur extra overhead
- Prefer list comprehensions over empty list + loop append — More Pythonic, concise, and often faster than initializing empty lists and appending in loops
- Place model profiles in `profiles/{company}.py`, provider routing in `providers/` — separates model metadata from provider-specific logic — Keeps architectural boundaries clear: model characteristics are company-specific facts, while routing/fallback logic is provider implementation detail
- Extract profile logic to `profiles/` only when shared across providers — inline provider-specific logic in `model_profile()` methods — Avoids unnecessary indirection and keeps code close to usage when there's no reuse, while enabling consistency when multiple providers support the same model family
- Use `set` for unique collections; convert to `list` only for API boundaries — Prevents accidental duplicates, makes membership checks O(1), and clarifies intent when order doesn't matter
- Eliminate duplicate validation logic — extract repeated checks into shared helpers or reuse existing parent/utility validations — Prevents inconsistencies when validation logic changes and reduces maintenance burden by keeping validation logic in one place (DRY principle)
- Use walrus operator (`:=`) to combine assignment with conditional checks — reduces redundancy and keeps related logic together — Eliminates separate assignment lines when values are immediately tested, improving code clarity and reducing variable scope leakage
- Use `else` instead of `elif` when remaining cases are exhaustively covered — Reduces redundancy and makes code intent clearer by avoiding unnecessary explicit conditions that are logically implied
- Use `any()` with generator expressions instead of `for` loops with `break` for existence checks — More concise, idiomatic Python that clearly expresses intent and avoids mutable flag variables

## Documentation

- Document provider features in 3 places: `Supported by:` in docstrings (IDE hints), compatibility notes in generic docs (selection), detailed provider sections with links to official docs (deep dive) — Coordinated multi-location documentation ensures users discover limitations early (docstrings), choose the right provider (feature docs), and find authoritative details without duplication (provider docs with external links)
- Sync provider docs with implementation in same PR — verify features against official API docs — Prevents documentation drift and false claims about provider capabilities that would mislead users and cause integration failures.
- Use consistent terminology across code, docs, APIs, and errors — avoid synonyms for the same concept — Prevents confusion and cognitive load when developers encounter multiple terms for the same concept across different parts of the system
- Use latest stable model versions in docs and examples (e.g., `openai:gpt-5.2` not `gpt-4o`) — Outdated models in documentation confuse users and make examples less relevant; fictional future models break code when users copy-paste.
- Wrap all code elements in backticks in docstrings, docs, and error messages — improves readability and distinguishes code from prose — Backticks create visual distinction between code identifiers and natural language, making documentation scannable and reducing ambiguity about what's a code reference versus plain text.
- Update `README.md`, `docs/index.md`, `docs/models/overview.md`, and `docs/api/providers.md` when adding a provider — Keeps provider listings consistent across all documentation surfaces so users can discover new providers
- Register new doc pages in `mkdocs.yml` nav — undiscoverable pages won't appear in site navigation — Without registration, new documentation pages exist but are invisible to users browsing the site
- Write docs from user perspective — describe what users can do, not internal mechanics — Users need to understand capabilities and behavior, not implementation details; keeps docs maintainable when internals change
- Update or remove docstrings/comments when code changes — stale docs mislead maintainers and cause bugs — Outdated documentation creates confusion, wastes debugging time, and leads to incorrect assumptions about behavior
- Match documentation depth and style across related API elements — prevents user confusion and signals equal importance — Consistent documentation patterns help users understand that similar parameters/methods have equal status and provide predictable learning across the API surface
- Comment non-obvious implementation choices — explain why the code deviates from simpler/intuitive approaches — Helps maintainers understand tradeoffs and prevents "why not just..." refactorings that reintroduce bugs or performance issues
- Update API reference pages when adding cross-reference links in docs — prevents broken links and incomplete documentation — Cross-references create dependencies between narrative docs and API references; updating both together prevents broken links that frustrate users

## API Design

- Prefix internal helpers with `_` — prevents accidental public API surface expansion — Functions, classes, and modules not intended for external use should be marked private to prevent users from depending on implementation details that may change
- Use `*` to make optional params keyword-only — keeps 1-2 essential args positional, rest keyword-only — Prevents breakage when reordering parameters and makes call sites self-documenting
- Use dedicated typed fields for provider settings, not generic dicts like `extra_body` — Provides type safety, autocomplete, and prevents merge conflicts when multiple providers extend the same base class
- Keep provider-specific features inside provider classes, not in generic interfaces — Prevents leaking implementation details into the public API, maintains clean abstractions, and avoids coupling generic code to specific provider quirks
- Don't pass data separately if it's already in a context object — reduces redundancy and prevents sync issues — Avoids parameter duplication, prevents caller/callee inconsistencies when context attributes change, and keeps function signatures cleaner
- Remove provider-specific settings fields when equivalent exists in base `ModelSettings`/`EmbeddingSettings` — Prevents duplication and inconsistency between base and provider classes, ensuring settings are defined once and inherited uniformly across all providers
- Implement new provider features for ≥2 providers upfront — validates abstraction is provider-agnostic — Prevents designing APIs that are accidentally coupled to one provider's specifics, catching abstraction issues early before they become breaking changes
- Use provider-agnostic terminology in public APIs and messages — reserve vendor terms only for direct provider interactions — Prevents vendor lock-in, keeps the API portable across providers, and maintains consistent user-facing terminology (e.g., `file_store_ids` not `vector_store_ids`, `thinking` not `reasoning`)

## Type System

- Use `assert_never()` in `else` clause when handling union types — catches unhandled variants at type-check time — Ensures exhaustive handling of union types so new variants added later will trigger type errors rather than silent bugs
- Use `TypedDict` instead of `dict[str, Any]` for structured data with known fields — Enables static type checking of dict keys/values and prevents runtime `cast()` workarounds
- Remove `# type: ignore` comments once underlying type issues are fixed — Stale ignore comments hide real type errors and prevent type-checker from catching new bugs
- Use `TYPE_CHECKING` imports for optional deps instead of `Any` — preserves type safety without runtime import errors — Enables precise type hints for optional dependencies while avoiding import failures at runtime, giving users better IDE support without requiring all extras to be installed.

## Error Handling

- Raise explicit errors for unsupported inputs/parameters — prevents silent failures and makes contract violations obvious — Explicit failures expose bugs immediately instead of allowing invalid data to propagate through the system silently
- Catch specific exception types, not broad `Exception` — identifies actual failure modes and prevents masking unexpected errors — Broad exception handlers hide bugs by catching unexpected errors (like `KeyboardInterrupt` or `SystemExit`) and make debugging harder by obscuring the actual failure type

## General

- Place imports at module level; use inline imports only for circular dependencies or optional deps wrapped in `try`/`except ImportError` with install instructions — Follows PEP 8, improves readability, and graceful optional dependency handling prevents cryptic errors by directing users to install the right extras group
- Write tests for reachable code; reserve `pragma: no cover` only for untestable branches (platform-specific, type-constrained, defensive unreachable) — Coverage pragmas hide gaps in test suites — conditional logic and public APIs need real tests to prevent regressions
- Omit redundant context from names when clear from class/module/types/call site — Reduces noise and improves readability—`URL._infer_media_type()` is clearer than `URL._infer_url_media_type()` since the URL context is already obvious

<!-- /braindump -->
