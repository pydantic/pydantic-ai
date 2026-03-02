<!-- braindump: rules extracted from PR review patterns -->

# Coding Guidelines

## Code Style

<!-- rule:409 -->
- Keep PRs focused on their stated purpose — exclude unrelated changes even if conceptually related — Simplifies review, prevents unintended side effects, and makes rollbacks cleaner when each PR has a single clear objective
<!-- rule:910 -->
- Wrap code identifiers in backticks in user-facing messages (errors, warnings, logs) — Improves readability and clearly distinguishes code elements from prose, making error messages easier to parse and debug
<!-- rule:193 -->
- Centralize validation at one layer — removes redundancy and establishes single source of truth — Prevents validation drift when requirements change and reduces maintenance burden by avoiding duplicate validation logic across the call chain
<!-- rule:2 -->
- Extract duplicated logic into shared helpers after 2+ occurrences — refactor existing code rather than creating parallel implementations — Prevents bugs from inconsistent implementations, reduces maintenance burden, and creates single sources of truth for validation, transformation, and schema handling
<!-- rule:341 -->
- Remove commented-out code, unused definitions, and superseded implementations — Version control preserves history; dead code creates confusion about intent, control flow, and which implementation is actually active
<!-- rule:559 -->
- Consolidate duplicate logic across conditional branches using combined conditions, extracted variables, or hoisted shared code — Reduces duplication, makes changes easier to maintain in one place, and clarifies that behavior is intentionally shared across branches
<!-- rule:14 -->
- Inline single-use helpers that only wrap property access or delegation — reduces nesting and cognitive load without sacrificing clarity — Eliminates unnecessary indirection that forces readers to jump between methods to understand simple operations, making code more direct and maintainable
<!-- rule:21 -->
- Extract model profile logic into dedicated `{provider}_model_profile()` functions in `profiles/{provider}.py` rather than inline in provider classes — Separates profile definitions from provider implementation, making profiles testable in isolation and easier to maintain across providers
<!-- rule:263 -->
- Extract repeated logic into helper methods or top-level functions when patterns recur (e.g., streaming vs non-streaming handlers, serialization, part types, message mappings, model adapters) — Prevents duplication bugs and makes changes easier to apply consistently across all code paths (like both streaming and non-streaming handlers)
<!-- rule:176 -->
- Scope helpers and constants to their single usage site — define inline or within the class/function that uses them, not at module level — Reduces namespace pollution, clarifies intent, and prevents accidental reuse of implementation details not designed for broader use
<!-- rule:345 -->
- Extract duplicated logic (validation, types, activity definitions, transformations) to parent classes or shared utilities — prevents drift and reduces maintenance burden across implementations — Keeping shared code in one place (like `_call_tool_in_activity` in `TemporalWrapperToolset`) prevents inconsistencies when logic evolves across multiple implementations (`TemporalFunctionToolset`, `TemporalMCPServer`, etc.)
<!-- rule:284 -->
- Use `model_dump()` for Pydantic model serialization; reserve `TypeAdapter` with `mode='json'` for collections or external SDKs needing JSON-compatible primitives — Prevents manual dictionary construction errors and ensures consistent serialization; `TypeAdapter.dump_python(mode='json')` guarantees primitive types (dicts/lists/strings) instead of `BaseModel` instances when required by external systems
<!-- rule:499 -->
- Compile static regex patterns at module level as constants — avoids recompilation overhead on repeated calls — Prevents performance degradation when regex-using functions are called frequently, as pattern compilation is expensive

## Type System

<!-- rule:0 -->
- Use `isinstance()` for type checking, not `hasattr()`, `getattr()`, `type(obj).__name__`, or discriminator field checks like `part_kind` — Enables proper type narrowing for static analysis and prevents fragile string-based comparisons that break during refactoring
<!-- rule:142 -->
- Use `Literal` types instead of plain `str` for fixed string value sets in parameters, fields, and return types — Makes valid values explicit in type signatures, enabling static type checkers to catch invalid strings at compile time and improving IDE autocomplete
<!-- rule:809 -->
- Create type aliases for complex types (3+ union branches, `dict[str, Any] | Callable` patterns, multi-value `Literal`s) or types used 2+ times — skip aliases for simple one-off internal types — Reduces duplication and improves readability for complex types while avoiding unnecessary abstraction that obscures simple inline hints
<!-- rule:95 -->
- Use `if TYPE_CHECKING:` blocks for optional dependency types with quoted hints — keeps package installable without all deps while preserving type safety — Prevents runtime import errors when optional dependencies aren't installed while maintaining proper type annotations instead of falling back to `Any`
<!-- rule:513 -->
- Type signatures to match runtime reality — if control flow (e.g., `match`/`case`, API contracts) guarantees only specific types reach a code path, narrow the annotation to exclude impossible types from unions — Prevents confusion, enables better type checking, and documents actual behavior rather than overly permissive signatures that suggest unreachable code paths
<!-- rule:46 -->
- Fix type errors properly instead of using `# type: ignore` or `# pyright: ignore` — use type annotations, narrowing, or `cast()` with explanatory comments — Prevents masking real type errors and makes code safer; when suppressions are genuinely needed (complex generics, tool limits), document with error codes and justification so reviewers understand the safety reasoning
<!-- rule:479 -->
- Remove redundant runtime checks when types already constrain the value — prevents noise and maintains type system trust — Redundant assertions (`assert x is not None` for non-`Optional` types, duplicate `isinstance()` checks, etc.) add visual clutter and imply the type system can't be trusted, making code harder to maintain
<!-- rule:469 -->
- Fix type definitions instead of using `cast()` — adjust generics or remove unnecessary unions to match runtime reality — Prevents masking structural type mismatches that indicate design problems; only use `cast()` when runtime logic guarantees safety but static analysis cannot narrow (e.g., after literal checks or known invariants)
<!-- rule:494 -->
- Don't add `| None` to `TypedDict` fields marked `total=False` or `NotRequired` — optionality is already expressed — Prevents redundant type declarations and makes it clear that omission (not None) is the intended optional behavior
<!-- rule:196 -->
- Remove `| None` from type annotations when values are guaranteed to be initialized or always provided — Prevents false optionality in types, making the API clearer and avoiding unnecessary None-checks that can never trigger

## Error Handling

<!-- rule:895 -->
- Raise `ModelRetry` for recoverable tool errors (timeouts, validation failures, missing params) — enables automatic retry with corrected input instead of terminal failure — Distinguishes transient/fixable errors from hard failures, allowing the agent to self-correct rather than propagating error messages to users
<!-- rule:400 -->
- Use `assert` for invariants that should never fail, not `RuntimeError('Internal error')` or `pragma: no cover` — Asserts document assumptions and fail fast in development; `RuntimeError` obscures programming errors as runtime issues and `pragma: no cover` hides untested branches
<!-- rule:32 -->
- Use `!r` format specifier for identifiers in error messages (e.g., `f'Tool {name!r}'` not `f'Tool `{name}`'`) — Provides consistent, unambiguous quoting that clearly delimits values and handles edge cases like empty strings or special characters.
<!-- rule:353 -->
- Fail fast on explicit user config conflicts; gracefully fallback on internal/auto setting conflicts — Catching user mistakes early with clear errors prevents debugging confusion, while internal fallbacks enable cross-provider compatibility and system resilience when constraints are automatically inferred or propagated
<!-- rule:337 -->
- Inherit new exception types from existing base exceptions like `UnexpectedModelBehavior` when semantically appropriate — Maintains backward compatibility so user code catching parent exceptions continues to work when new exception types are introduced
<!-- rule:320 -->
- Catch specific exception types instead of bare `except Exception` when failure modes are known — Prevents catching unexpected errors that should propagate, makes debugging easier, and documents expected failure cases
<!-- rule:1104 -->
- Validate input parameters before expensive operations — fail fast to avoid wasted computation — Prevents unnecessary resource consumption and provides faster feedback when invalid inputs are detected
<!-- rule:130 -->
- Trust validated invariants and use defaults over assertions — reduces brittle failures and improves resilience — Assertions crash on unexpected states; defaults and graceful handling keep the system operational when assumptions don't hold, while trusting earlier validation stages avoids redundant defensive checks.

## Naming

<!-- rule:280 -->
- Drop redundant prefixes when context is clear — prefer `ToolConfig.description` over `ToolConfig.tool_description`, `MCPServerTool.label` over `MCPServerTool.server_label` — Reduces noise and improves readability since the class/module name already provides context (e.g., `tool_config.description` is clearer than `tool_config.tool_description`)
<!-- rule:198 -->
- Rename methods/functions when their behavior changes — names must reflect actual scope, return values, and abstraction level — Prevents confusion and bugs when implementation evolves (e.g., `_call_function_tool` handling output tools should become `_call_tool_traced`)
<!-- rule:321 -->
- Use specific parameter/variable names that convey semantic meaning — prefer `toolset_id`, `memory_id`, `config_data` over generic `id`, `name`, `data` — Improves code readability and prevents confusion when multiple IDs or data objects are in scope
<!-- rule:488 -->
- Avoid redundant type suffixes (`Value`, `Type`, `Class`, `_dict`, `_list`, `_str`) when type is clear from annotations or context — Reduces noise and improves readability since Python's type system already documents the type explicitly
<!-- rule:770 -->
- Use `UPPER_CASE` for module constants; prefix with `_` if internal (`_MAX_RETRIES`) — Distinguishes public API from internal implementation details and signals immutability

## Imports

<!-- rule:464 -->
- Place all imports at the top of the file, not inline within functions or test bodies — Ensures imports are visible at module load time, prevents hidden dependencies, and follows Python conventions for clarity and consistency
<!-- rule:77 -->
- Handle optional dependencies: (1) import inside functions to defer requirements, OR (2) use `try`/`except ImportError` at module level with helpful errors directing to install groups like `[web]`, `[bedrock]` — Keeps the package installable without all dependencies while providing clear guidance when optional features are used
<!-- rule:141 -->
- Remove unused imports — reduces dependency bloat and keeps the module namespace clean — Prevents accidental dependencies, reduces cognitive load when reading code, and avoids circular import issues
<!-- rule:223 -->
- Remove duplicate imports — keep only one declaration per imported item — Prevents confusion, reduces file size, and avoids potential issues if imports have side effects

## Testing

<!-- rule:432 -->
- Remove tests when redundant, obsolete, or duplicative — each test should verify distinct, valuable behavior that currently exists — Reduces maintenance burden and keeps test suite focused on actual behavior; prevents false confidence from tests covering non-existent code paths or duplicating coverage without verifying edge cases
<!-- rule:97 -->
- Avoid `# pragma: no cover` — write tests instead. Only use for truly untestable code (defensive guards, platform branches, optional deps unavailable in CI) — Coverage pragmas hide gaps in test coverage; proper tests prevent regressions and document expected behavior, while pragmas should only mark code paths that cannot be executed in testing environments

## Documentation

<!-- rule:132 -->
- Use latest/frontier models (e.g., `'gpt-5'` not `'gpt-4o'`) in docs and examples — Shows users current best practices and prevents outdated examples from becoming cargo-culted into production code
<!-- rule:390 -->
- Use provider-prefixed model identifiers (`{provider}:{model}`) and platform-specific formats (e.g., AWS Bedrock requires `us.anthropic.claude-{model}-{version}:0`) — Prevents misconfiguration and API errors by matching exact identifier formats required by each platform, ensures consistency across docs and code

## General

<!-- rule:-2 -->
- Use latest frontier models (e.g. `openai:gpt-5.2`, `anthropic:claude-opus-4-6`) in `docs/examples` — Outdated model references make our product look unmaintained and reduce user trust
<!-- rule:449 -->
- Use `make install` to regenerate lock files (e.g., `uv.lock`) after dependency changes — Ensures reproducible builds and keeps lock file diffs minimal. Update the package manager (uv, npm, pip-tools) to latest first and start from clean state. If diffs are unexpectedly large, reset to base branch and regenerate to isolate actual changes — prevents spurious conflicts and version drift.
<!-- rule:717 -->
- Override profile properties in model/provider classes, not in shared profile functions — Prevents provider-specific logic from leaking into shared utilities like `anthropic_model_profile()` that multiple providers (OpenAI, Bedrock, etc.) depend on — keeps profiles reusable and avoids cross-provider bugs

## Topic Guides

Check these when working in specific areas:

- **[Code Simplification & Idioms](agent_docs/code-simplification.md)**: When refactoring code for clarity or looking to simplify complex patterns
- **[Documentation](agent_docs/documentation.md)**: When writing or updating documentation, comments, or docstrings
- **[API Design & Interfaces](agent_docs/api-design.md)**: When designing or modifying public APIs, parameters, or class interfaces
<!-- /braindump -->
