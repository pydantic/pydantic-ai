<!-- braindump: rules extracted from PR review patterns -->

# docs/ Guidelines

## Documentation

<!-- rule:232 -->
- Link all concepts, features, and API elements to their docs/reference pages using anchor fragments (`#section-name`) for specific sections — Improves discoverability and reduces user friction by providing direct navigation to relevant documentation context
<!-- rule:66 -->
- Use reference-style links for API elements: `[ElementName][module.path.ElementName]` — enables hover docs and navigation in mkdocs — Provides interactive documentation features like tooltips and jump-to-definition that plain backticks cannot support
<!-- rule:714 -->
- Omit deprecated features from user-facing docs — document only current approaches — Prevents users from learning outdated patterns and reduces confusion about the recommended way forward
<!-- rule:82 -->
- Write project name as `Pydantic AI` (two words) in docs — not `Pydantic-AI`, `PydanticAI`, or `pAI` — Maintains consistent brand identity and prevents confusion across documentation
<!-- rule:359 -->
- Structure code examples as: context/intro → code block → caveats/details (never code before context) — Ensures readers understand purpose and usage before seeing code, making docs more learnable and preventing confusion
<!-- rule:93 -->
- Hide implementation details from user docs unless they affect user decisions — focus on what users can control, not how it works internally — Keeps documentation clean and maintainable by separating user-facing APIs from implementation that may change
<!-- rule:52 -->
- Structure docs with progressive disclosure: concept → capabilities → examples (standalone first) → config → edge cases — Helps readers build mental models incrementally, reducing cognitive load and making features easier to adopt
<!-- rule:152 -->
- In docs, show the **recommended approach first**, then introduce alternatives with explicit relational language ("In addition to...", "As an alternative to...") using specific feature names — Prevents users from adopting legacy or suboptimal patterns by ensuring they encounter the best practice first
<!-- rule:109 -->
- Remove docs content describing features "working as expected" — focus only on integration-specific concerns, limitations, or deviations — Reduces cognitive load and maintenance burden by eliminating noise; prevents documentation staleness from trivial statements
<!-- rule:67 -->
- Keep provider-specific config/features in `docs/models/{provider}.md` and `docs/api/models/{provider}.md`; general docs stay provider-agnostic with one minimal example + links — Prevents duplication, keeps general feature docs clean and maintainable, ensures users find provider-specific details in one canonical location rather than scattered across multiple pages
<!-- rule:941 -->
- Avoid `test="skip"` in code examples unless unavoidable (external services, credentials, non-deterministic behavior) — use mocks or fixtures instead — Testable documentation examples prove the code works and prevent docs from drifting out of sync with actual behavior
<!-- rule:727 -->
- Link to canonical sources rather than duplicating lists or summaries maintained elsewhere — Prevents docs from becoming outdated when the source of truth changes
<!-- rule:808 -->
- Focus docs on user tasks and public APIs, defer implementation details to docstrings — Task-oriented guides help users accomplish goals faster, while keeping advanced/internal details in API reference prevents overwhelming users with complexity when sensible defaults exist
<!-- rule:301 -->
- In docs, consolidate examples showing parameter variations into one block with notes — split only for mutually exclusive params or distinct use cases — Reduces cognitive load and makes docs more scannable by avoiding repetitive boilerplate for simple parameter alternatives
<!-- rule:58 -->
- In docs examples, demonstrate realistic use cases that show *why* the feature matters — prevents misleading users with toy scenarios or debugging code that obscure actual value — Well-crafted examples help users understand when to apply features and avoid implementing unnecessary patterns for problems solvable with simpler approaches
<!-- rule:54 -->
- Use fence-level `{test="skip" lint="skip"}` instead of inline suppressions in doc examples — keeps code clean and reader-focused — Documentation code should model best practices; fence-level skip directives separate tooling concerns from the example itself, while inline `# noqa` or `# type: ignore` pollutes pedagogical code with implementation details
<!-- rule:151 -->
- Cross-reference alternatives and explain trade-offs when documenting overlapping features — Prevents users from missing better-suited options or implementing duplicate functionality when multiple approaches exist (e.g., `UsageLimits` vs rate-limiting, provider-specific implementations)
<!-- rule:1112 -->
- Document default behavior and use cases for all configurable features — helps users decide when to override defaults — Users can't make informed configuration choices without knowing what happens by default and when alternatives are appropriate
<!-- rule:135 -->
- Use actual, currently available model names in documentation examples — prevents user confusion and copy-paste errors with non-existent models — Ensures users can run documentation examples without modification and avoids frustration from referencing models that don't exist yet or are hypothetical
<!-- rule:508 -->
- Verify all doc links with `make docs-serve` before committing — catches broken internal/external references early — Prevents documentation drift and broken links from reaching users, especially after code refactoring
<!-- rule:283 -->
- Use MkDocs admonitions (`!!! note`, `!!! warning`) for callouts, not blockquotes (`>`) or GitHub alerts (`> [!NOTE]`) — Ensures consistent rendering in MkDocs and prevents callouts from cluttering the table of contents
<!-- rule:618 -->
- Nest subtopics, examples, and config details within parent sections — improves discoverability and reduces redundant context — Hierarchical organization makes documentation easier to navigate and understand by grouping related content together rather than scattering it across top-level sections or separate files.
<!-- rule:298 -->
- In provider feature support tables, use a `Notes` or `Provider Support Notes` column for variations, limitations, and special values — keeps table structure clean and constraints discoverable — Centralizes provider-specific exceptions in one scannable location instead of scattering them across config examples or inline parentheticals, making cross-provider differences easier to find
<!-- rule:634 -->
- In provider feature tables, use standard labels (`Full feature support`, `Limited parameter support`) and move unsupported variants to `Unsupported` column, not inline exceptions — Ensures consistent, scannable documentation structure where users can quickly identify exact support boundaries across providers
<!-- rule:168 -->
- When documenting alternative approaches, explain tradeoffs (limitations, requirements, benefits, use-cases) and warn about conflicts when combining them — Helps users make informed decisions and avoid subtle bugs from conflicting configurations

<!-- /braindump -->
