# API Design & Interfaces

> Rules for designing public APIs, managing visibility, backward compatibility, and API patterns

**When to check**: When designing or modifying public APIs, parameters, or class interfaces

## Rules

### Compatibility impacts

- Treat previously valid user code that now needs modification as a compatibility impact.
- Preserve compatibility unless the bridge retains incorrect behavior, creates a permanent legacy-only API, or makes the contract contradictory.
- When no valid bridge exists, identify the exact exception in `docs/version-policy.md`.
- Without an applicable exception, deprecate the old behavior before removal or wait for a major release.
- Treat non-underscore symbols on public modules and classes as public by default. A missing docstring alone does not make a symbol internal.
- Treat additions to released public unions as compatibility-relevant changes. Inspect attributes common to prior arms and exhaustive consumers.
- Do not add semantically false fields only to preserve a union's previous structural shape.
- For every permitted compatibility impact, add the `compatibility impact` label and a release note.
- Add a `[!WARNING]` PR section that names affected code, the policy exception, and the migration.
- Add an exact API-check waiver when the deterministic compatibility gate reports the permitted change.

<!-- rule:146 -->
- Prefix implementation details with underscore (`_`) and exclude from `__all__` — prevents accidental API surface expansion and signals internal-only usage — Keeps the public API surface minimal and clearly separates internal implementation from stable public interfaces, preventing backward compatibility obligations for internal code.
<!-- rule:29 -->
- Export commonly-used types and classes from top-level `pydantic_ai` package — hides internal structure and simplifies user imports — Makes the public API easier to use and allows internal refactoring without breaking user code that would otherwise depend on specific submodule paths
<!-- rule:124 -->
- Use `_: KW_ONLY` marker before optional fields in dataclasses/Pydantic models — Prevents breakage when adding parameters — callers can't accidentally pass defaults positionally, ensuring backward compatibility when fields are added or reordered
<!-- rule:56 -->
- Prefer instance methods when accessing `self` attributes or enabling polymorphism; use module-level functions when no instance state is needed — Reduces unnecessary coupling and parameter passing while enabling proper polymorphism; extract shared logic to private top-level helpers to avoid duplication across classes
<!-- rule:491 -->
- Keep old names as deprecated aliases when renaming public API elements — prevents breaking existing code — Maintains backward compatibility so users can migrate gradually rather than experiencing immediate breakage when upgrading
<!-- rule:775 -->
- Return new collections from transform functions instead of mutating inputs — prevents surprising side effects and makes code easier to reason about (exceptions: performance-critical paths or functions named `update_*`/`*_inplace`) — Immutable transforms prevent surprising side effects and make code easier to reason about, improving maintainability across the codebase
<!-- rule:367 -->
- Don't access or modify private attributes (`_prefixed`) — use public APIs, properties, or constructor parameters — Prevents breakage when internal implementation changes and ensures compatibility with library updates
<!-- rule:72 -->
- Promote settings to base classes (`ModelSettings`, embedding settings) when 2-3+ providers support them; maintain backward compatibility with automatic mapping from new common fields to legacy provider-prefixed fields — Prevents API duplication across provider-specific subclasses (e.g., `OpenAIEmbeddingSettings`, `CohereEmbeddingSettings`) while preserving backward compatibility when refactoring provider-prefixed parameters (e.g., `cohere_`, `openai_`) to shared fields
- Never change a shared default, public signature, or abstraction to accommodate one new feature — if a new type only behaves correctly once an existing public default is rewritten, the new type's design is wrong — Fix the new code instead, or split the shared change into its own PR with maintainer sign-off, so one caller's needs don't silently change behavior for every existing user
<!-- rule:302 -->
- Keep `NativeToolReturnPart.content` flat and non-redundant — avoid duplicating part fields, repeating `return_value` data, single-key wrappers, or unnecessary lists — Reduces API surface area, prevents inconsistencies between duplicate fields, and simplifies consumption for both users and AI assistants
<!-- rule:265 -->
- Use `'provider:model'` format (e.g., `'openai:gpt-4'`, `'anthropic:claude-3'`) and `infer_model()` for instantiation — Ensures consistent model reference syntax across code, docs, and CLI; provides unified instantiation interface that prevents fragmentation
