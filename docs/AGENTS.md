<!-- braindump: rules extracted from PR review patterns -->

# docs/ Guidelines

## Documentation

<!-- rule:113 -->
- Reference API elements as `[ElementName][module.path.ElementName]` in docs — enables consistent linking and IDE navigation — This markdown reference-style format with full module paths creates maintainable cross-references that documentation tools can process and validate automatically
<!-- rule:232 -->
- Hyperlink concepts/features to their dedicated doc pages on first mention — improves discoverability and reduces reader friction — Prevents users from having to search for definitions and ensures they can quickly access detailed explanations when encountering unfamiliar terms
<!-- rule:772 -->
- Link API elements to their reference docs using `[`text`][fully.qualified.name]` syntax — Makes documentation navigable and helps users discover the full API surface without hunting for definitions
<!-- rule:481 -->
- Maintain single canonical source for each doc topic — link to it rather than duplicating content across READMEs, API docs, and guides — Prevents documentation drift and conflicting information when content is updated in one location but not others
<!-- rule:941 -->
- Avoid `test="skip"` in docs code examples; use mocks/fixtures when possible — ensures examples stay accurate and runnable — Untested examples drift from actual API behavior over time; only skip when truly unavoidable (external services, credentials), and add expected output as comments when you do.
<!-- rule:132 -->
- Use `'gpt-5'` as the default model in documentation examples, not `'gpt-4o'` — Keeps docs consistent with current recommended model and prevents users from copying outdated patterns
<!-- rule:93 -->
- Omit implementation details (SDK deps, internal classes, provider API mappings, auto-normalization) from user docs unless they affect user decisions — Keeps documentation focused on user-facing behavior rather than internal mechanics, reducing maintenance burden and avoiding confusion when implementation changes
<!-- rule:714 -->
- Omit deprecated features from user docs — document only current recommended approaches — Prevents users from learning outdated patterns and reduces maintenance burden of documenting multiple approaches
<!-- rule:152 -->
- In `docs/`, lead with the recommended approach, then introduce alternatives with explicit relational language (e.g., "As an alternative to...", "In addition to...") using specific feature names — Prevents users from adopting legacy/suboptimal patterns first, ensures clear hierarchy of solutions
<!-- rule:82 -->
- Write project name as 'Pydantic AI' (two words, space) in documentation — maintains consistent branding — Consistent branding across all documentation and prose prevents confusion and maintains professional presentation for users and contributors
<!-- rule:34 -->
- In `docs/models/`, link to provider's official model catalog instead of maintaining inline model lists — include 1-2 examples max — Prevents documentation drift when providers update their model offerings, and reduces maintenance burden
<!-- rule:359 -->
- Place explanatory context before code examples, behavioral notes after — improves learning flow — Readers understand examples better when context (purpose, when to use) comes first, and caveats come after the code rather than mixed with reference material
<!-- rule:808 -->
- Document user-facing APIs, behaviors, and workflows — not implementation details or exhaustive method listings — Keeps user guides task-oriented and discoverable; reserve comprehensive API coverage and internal logic (private methods, call chains, resolution logic) for API reference docs
<!-- rule:396 -->
- Avoid redundant content in documentation — don't duplicate sections, repeat explanations around code examples, or re-list requirements/constraints/fields already shown in examples — Reduces maintenance burden and improves readability by keeping documentation concise and single-source
<!-- rule:301 -->
- Consolidate examples showing parameter variations into one code block with notes — separate examples only for mutually exclusive params or distinct use cases — Reduces cognitive load and makes documentation scannable by grouping related options together instead of fragmenting simple variations across multiple examples
<!-- rule:67 -->
- Place provider-specific config/features in `docs/models/{provider}.md` and `docs/api/models/{provider}.md`; centralize provider-agnostic features (like builtin tools) in dedicated dirs (e.g., `docs/builtin-tools/`) — prevents duplication and keeps provider docs focused — Keeps documentation maintainable by avoiding duplication across provider pages while ensuring users find provider-specific details (like Azure token constraints or OpenAI's `connector_id`) in one authoritative location
<!-- rule:485 -->
- Structure feature docs with: (1) conceptual intro (what/why), (2) capabilities overview, (3) standalone examples before integrated ones, (4) configuration details, (5) limitations — prevents users from guessing feature scope or missing critical caveats — Consistent documentation structure helps users quickly assess feature relevance, understand complete capabilities upfront, and learn progressively from simple to complex usage patterns
<!-- rule:500 -->
- Document cross-cutting features (structured outputs, validators, tool args, streaming) fully in each relevant section with explicit context lists and cross-links — Strategic duplication ensures users discover behavioral differences in context without hunting through separate sections, reducing missed edge cases and integration errors
<!-- rule:298 -->
- Document provider-specific constraints in a 'Notes' column in feature support tables, not in config sections — Centralizes provider variations (required fields, special values, limitations) in one scannable location instead of scattering them across config examples or parenthetical notes, making cross-provider comparison easier
<!-- rule:1112 -->
- Document default behavior and motivation for config options — helps users decide if they need to override — Users can't make informed configuration choices without knowing what the default does and when alternatives are appropriate
<!-- rule:52 -->
- Structure docs from concrete to abstract: examples first, then concepts, then advanced patterns — improves learnability — Starting with concrete examples helps users understand basics before encountering complex abstractions, reducing cognitive load and making documentation more accessible to new users.
<!-- rule:508 -->
- Verify all docs links with `make docs-serve` before committing — catches broken references from refactoring or typos — Prevents broken internal/external links that frustrate users and make documentation unreliable, especially after code refactoring
<!-- rule:135 -->
- Use real, currently available model names in examples (e.g., `'gpt-4o'`), not hypothetical ones (e.g., `'gpt-5'`) — Prevents user confusion and ensures examples are copy-paste ready with actual working model identifiers
<!-- rule:283 -->
- Use MkDocs admonition blocks (`!!! note`, `!!! warning`) for callouts — not blockquotes (`>`) or GitHub alerts (`> [!NOTE]`) — Ensures proper rendering in MkDocs without cluttering the table of contents, and maintains consistency across documentation
<!-- rule:54 -->
- Use fence-level `{test="skip" lint="skip"}` over inline `# noqa` in doc examples — keeps code clean and reader-focused — Doc examples should model clean code for readers; fence annotations hide necessary skips without cluttering the example itself
<!-- rule:58 -->
- Show realistic use cases in `docs/` examples — avoid placeholder scenarios, type introspection, or complexity from unrelated features — Documentation examples should demonstrate when and why to use a feature, not just prove it works; this helps users understand the feature's value proposition and apply it correctly
<!-- rule:618 -->
- Nest doc subtopics, examples, and config details within parent sections — omit redundant parent context from nested headers — Improves doc navigation and readability by making hierarchy clear from structure rather than repetitive headings
<!-- rule:68 -->
- In provider-agnostic docs: show one example (if behavior identical) OR exhaustive examples (if demonstrating compatibility matters); explicitly list which providers/models support each feature and distinguish universal vs provider-specific options — Prevents user confusion about feature support and reduces ambiguity when different providers have different capabilities or defaults
<!-- rule:634 -->
- In provider feature support tables, use explicit categories (`Full feature support`, `Limited parameter support`) and separate `Unsupported` columns — avoids ambiguous exceptions in checkmarks — Clear categorization prevents confusion about what actually works and makes feature gaps immediately scannable for users choosing providers.
<!-- rule:168 -->
- When documenting alternatives, explain tradeoffs: limitations, requirements, benefits, use-cases, and warn about conflicts when combining approaches — Helps users make informed decisions and avoid hidden pitfalls when choosing between or mixing API patterns

<!-- /braindump -->
