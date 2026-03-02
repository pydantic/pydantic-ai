<!-- braindump: rules extracted from PR review patterns -->

# Coding Guidelines

## Type System

<!-- rule:185 -->
- End union type `isinstance()` or `if`/`elif` chains with `else: assert_never()` — catches missing variants at type-check time when unions expand — Type checkers flag unhandled union members before runtime, preventing silent bugs when new types are added to the union
<!-- rule:60 -->
- Avoid `Any` types — use `Union`, `Protocol`, `TypedDict`, or `TypeVar` with bounds; import optional types under `if TYPE_CHECKING:` with quoted hints — Precise types catch bugs at type-check time, improve IDE autocomplete, and document expected structures without requiring docstrings
<!-- rule:882 -->
- Extract complex (3+ union branches, nested generics) or repeated (2+ uses) type annotations into named type aliases — Improves readability, ensures consistency across related fields, and makes refactoring type definitions safer
<!-- rule:142 -->
- Use `Literal` types instead of plain `str` for parameters/fields/returns with fixed string values — Makes valid values explicit in type signatures, enables better IDE autocomplete, and catches invalid string values at type-check time instead of runtime
<!-- rule:0 -->
- Use `isinstance()` for type checking, not `hasattr()`, `type(obj).__name__`, or discriminator field comparisons like `part_kind == 'text'` — Enables type narrowing, prevents breakage during refactoring, and allows type checkers to understand your code flow
<!-- rule:469 -->
- Avoid `cast()` by fixing type definitions to match runtime — prefer narrowing unions, correcting generics, or improving type guards over casting to bypass errors — Prevents masking structural type problems and ensures static analysis catches real bugs; only use `cast()` when runtime guarantees safety but static analysis can't infer it (e.g., after literal checks or known invariants)
<!-- rule:46 -->
- Fix type errors with proper annotations/narrowing/casting instead of `# type: ignore` — when suppression is unavoidable, add specific error code and justification inline — Prevents masking real type safety issues; inline suppressions with error codes (e.g., `# type: ignore[attr-defined]`) make future debugging easier and signal intentional decisions rather than laziness
<!-- rule:635 -->
- Remove or justify type system workarounds (`# type: ignore`, redundant `isinstance()`, unnecessary `is not None` checks) — improves type safety and prevents silently masking real type errors — Type workarounds often hide real type issues or become outdated as the codebase evolves; removing them ensures the type system catches actual bugs and maintains accurate type information
<!-- rule:494 -->
- Don't add `| None` to TypedDict fields marked `total=False` or `NotRequired` — optionality is already expressed — Redundant None unions create confusion about whether a field can be omitted vs. must be present with a None value; TypedDict mechanisms already handle optionality correctly

## Naming

<!-- rule:280 -->
- Drop redundant prefixes/suffixes when context (class name, module, type annotation) already conveys meaning — Reduces noise and improves readability — `ToolConfig.description` is clearer than `ToolConfig.tool_description`, and type annotations eliminate the need for `_str`/`_dict`/`_list` suffixes
<!-- rule:198 -->
- Rename methods when implementation scope changes — prevents misleading names that hide actual behavior — Accurate names prevent bugs by making it clear what code actually does; renaming `_call_function_tool` to `_call_tool_traced` when it starts handling both function and output tools signals the change to maintainers and prevents incorrect assumptions about behavior.
<!-- rule:321 -->
- Use specific parameter/variable names like `toolset_id`, `memory_id`, `config_data` instead of generic `name`, `id`, `data` — improves code clarity and prevents ambiguity — Descriptive names convey semantic meaning across method signatures, error messages, and docs, reducing cognitive load and preventing bugs from mistaken identity
<!-- rule:770 -->
- Name module constants `UPPER_CASE`; prefix with `_` if internal — signals API boundaries — Distinguishes public API constants from internal implementation details, preventing accidental dependencies on private constants.
<!-- rule:556 -->
- Use parallel naming patterns for equivalent fields across providers — e.g., `send_back_thinking_parts` in Bedrock becomes `openai_chat_send_back_thinking_parts` in OpenAI — Consistent naming across provider implementations signals common purpose and makes the codebase predictable for users working with multiple providers

## Code Style

<!-- rule:409 -->
- Keep PRs focused on their stated purpose — exclude unrelated code, docs, or formatting changes — Makes reviews faster, reduces merge conflicts, and isolates changes for easier debugging and rollback
<!-- rule:910 -->
- Wrap code elements (variables, functions, classes, types, keywords) in backticks in user-facing messages — Improves readability and clearly distinguishes code from prose in error messages, warnings, and logs
<!-- rule:176 -->
- Place single-use helpers and constants in function/method scope, not module-level — reduces namespace pollution and clarifies usage scope — Keeps module namespace clean and makes it immediately clear which code depends on which helpers, improving maintainability and reducing cognitive load when reading the code
<!-- rule:120 -->
- Colocate utilities with their primary types — place factory methods, helpers, and context managers in the module with the types/ContextVars they operate on, preferably as class methods — Keeps related code together, improves discoverability, and makes dependencies explicit — critical for maintaining consistency across CLI, temporal, models, messages, UI, and agent graph modules

## Documentation

<!-- rule:325 -->
- Use latest frontier models in docs/examples (e.g., `claude-sonnet-4-5`, `gpt-4o`) — users copy-paste this code and expect current best practices — Outdated model references like `gpt-3.5-turbo` mislead users and hurt adoption of better-performing models
<!-- rule:102 -->
- When adding a provider, update `docs/models/{provider}.md`, `docs/api/models/{provider}.md`, `docs/api/providers.md`, `docs/models/overview.md`, `docs/index.md`, `README.md`, `mkdocs.yml`, and feature tables in `builtin-tools.md`, `thinking.md`, `input.md` — Ensures users discover new providers across all documentation entry points and understand feature support
<!-- rule:50 -->
- Update examples to show current patterns, not deprecated ones — examples are teaching tools, not compatibility contracts — Keeps documentation accurate and prevents developers from learning outdated patterns that may be less safe, less idiomatic, or harder to maintain
<!-- rule:390 -->
- Use provider-prefixed model identifiers (`{provider}:{model}`) and platform-specific formats (e.g., `bedrock:us.anthropic.claude-3-5-sonnet-20241022:0`) — Ensures model references work correctly across different platforms and prevents confusion when the same model has different identifiers per provider

## API Design

<!-- rule:146 -->
- Prefix internal-only functions, methods, classes, and modules with underscore; exclude from `__all__` — makes API boundaries clear and prevents accidental external dependencies — Clear API boundaries prevent external code from depending on implementation details that may change, while allowing easy promotion to public API when legitimate external needs emerge
<!-- rule:589 -->
- Store model capabilities (`supports_*` flags, builtin tools) as properties in `ModelProfile`, set conditionally in profile definitions or `model_profile()` methods — Centralizes capability metadata in profiles instead of scattering `isinstance()` checks at usage sites, making capabilities discoverable and reducing coupling between model detection and feature logic.
<!-- rule:265 -->
- Use `'provider:model'` format (e.g., `'openai:gpt-4'`) with `infer_model()` for instantiation — Provides a consistent, unified interface across all code and documentation, preventing fragmentation of model instantiation patterns and making the API predictable for users
<!-- rule:717 -->
- Override profile properties in model/provider classes, not in shared profile functions — Shared profile functions like `anthropic_model_profile()` are used by multiple providers (e.g., OpenAI and Bedrock); modifying them breaks compatibility for other consumers—use `dataclasses.replace()` in provider-specific `.profile` properties instead

## Error Handling

<!-- rule:895 -->
- Raise `ModelRetry` for recoverable tool errors (timeouts, validation failures, invalid params) — enables automatic retry with corrected input instead of hard failure — This is a core framework recovery mechanism that allows the agent to self-correct rather than failing or returning error messages in tool responses
<!-- rule:400 -->
- Use `assert` for invariants that should never fail in correct code — not `RuntimeError('Internal error')`, `'Unreachable code'`, or `pragma: no cover` — Distinguishes programming errors (bugs) from runtime errors, making bugs immediately visible in development while keeping production stacktraces clean and meaningful
<!-- rule:32 -->
- Use `!r` format specifier for identifiers in error messages (e.g., `f'Tool {name!r}'` not `f'Tool `{name}`'`) — Ensures consistent, unambiguous representation of names/values in user-facing strings and prevents quoting inconsistencies

## Imports

<!-- rule:155 -->
- Place imports at module level; group optional deps in single `try`/`except ImportError` with install instructions — Avoids import overhead on every call, makes dependencies explicit at file top, and provides clear error messages directing users to the correct install group (e.g., `[anthropic]`, `[web]`)
<!-- rule:77 -->
- Import optional dependencies inside functions at point of use, not at module level — Defers dependency requirements until functionality is actually needed, keeping the package installable without all optional deps
<!-- rule:223 -->
- Remove duplicate imports — keep only one declaration per imported item — Prevents confusion, reduces file size, and avoids potential issues if imports have side effects

## Testing

<!-- rule:432 -->
- Remove tests when their code paths are eliminated or superseded by better coverage — Integration tests covering realistic usage provide more valuable validation than isolated unit tests of the same paths; duplicate tests create maintenance burden without adding safety
<!-- rule:97 -->
- Write tests instead of using `# pragma: no cover` — reserve pragmas only for truly untestable code (defensive errors, platform branches, optional deps unavailable in CI); remove pragmas when tests are added or CI shows execution — Maintains accurate coverage metrics and prevents lazy testing practices; pragmas should reflect actual untestability, not developer convenience

## General

<!-- rule:449 -->
- Use `make install` to regenerate lock files after dependency changes — ensures consistent tooling versions and isolates actual dependency changes from lock file churn — Prevents accidental lock file corruption from version mismatches and makes PR diffs reviewable by showing only intentional changes.
<!-- rule:29 -->
- Export commonly-used types and classes from top-level `pydantic_ai` package — simplifies imports and prevents user coupling to internal module structure — Users shouldn't need to know internal submodule paths; top-level exports improve discoverability and allow refactoring internals without breaking imports

## Topic Guides

Check these when working in specific areas:

- **[Code Style & Simplification](code-style-simplification.md)**: When writing loops, conditionals, variable assignments, or refactoring code for clarity
- **[Documentation & Comments](documentation.md)**: When writing or updating documentation, docstrings, comments, or user-facing guides
- **[API Design & Public Interfaces](api-design.md)**: When designing new APIs, adding parameters to public functions, or deciding what to expose publicly

<!-- /braindump -->
