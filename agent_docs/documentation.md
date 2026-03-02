# Documentation & Comments

> Rules for writing clear, accurate, and maintainable documentation including docstrings, comments, markdown formatting, code element backticks, terminology consistency, and keeping docs synchronized with code changes.

**When to check**: When writing or updating documentation, docstrings, comments, or user-facing guides

## Rules

<!-- rule:272 -->
- Wrap all code identifiers in docs with backticks — parameter names, variables, functions, classes, types, fields, API terms — Ensures consistent markdown formatting and makes code elements instantly scannable in documentation
<!-- rule:339 -->
- Remove comments that restate obvious code — explain non-obvious intent, reasoning, edge cases, or external constraints instead — Reduces noise and maintenance burden while focusing documentation on information that isn't self-evident from the code itself
<!-- rule:150 -->
- Add comments to non-obvious logic: edge case conditionals, operations depending on prior transformations, and protocol-specific workarounds (name the protocol and why) — Prevents future developers from misunderstanding fragile logic dependencies, state assumptions, or external API limitations
<!-- rule:138 -->
- Update all docs (docstrings, external docs, inline comments) in the same PR as implementation changes — Prevents documentation drift that misleads users and developers about current behavior, capabilities, and limitations
<!-- rule:420 -->
- Update docstrings, API docs, and integration docs when modifying `AbstractToolset`, `TemporalRunContext`, or other public context/integration classes — keeps documentation in sync with code changes — Prevents documentation drift that confuses users and makes the API harder to use correctly
<!-- rule:107 -->
- Register new docs in `mkdocs.yml` nav section — makes them visible in the site — Without registration, documentation files exist but are inaccessible to users browsing the generated site
<!-- rule:824 -->
- Preserve comments explaining non-obvious behavior, design decisions, and edge cases during refactoring — Documentation loss during refactoring erases critical context about system-specific differences and rationale, making future maintenance harder and bugs more likely
<!-- rule:644 -->
- Use heading syntax (`##`, `###`, `####`) not bold (`**Text:**`) for sections — enables proper document structure and navigation — Markdown headings create semantic structure for parsers, table-of-contents generation, and screen readers, while bold text is invisible to these tools
<!-- rule:31 -->
- Use consistent terminology across code, errors, and docs — pick one variant and stick with it (e.g., `freeform` not `free-form` or `free form`) — Prevents user confusion and makes the codebase searchable and maintainable when the same concept is always named identically
<!-- rule:312 -->
- Link to official provider docs for API-specific features — prevents outdated inline details and directs users to authoritative sources — Linking to primary documentation (e.g., OpenAI guides, AWS docs) instead of embedding extensive details inline ensures docs stay accurate as provider APIs evolve and reduces maintenance burden
<!-- rule:27 -->
- Document workarounds with: (1) expected behavior, (2) why it fails (API limitation/bug/spec issue), (3) tradeoffs introduced; add `TODO` links for fixable issues — Future maintainers need context to know when workarounds can be removed and what costs they impose, preventing premature removal or perpetual technical debt
<!-- rule:801 -->
- Keep docs concise — focus on essential info, not implementation details or edge cases — Reduces maintenance burden and keeps documentation readable; comprehensive details often go stale and create noise
<!-- rule:614 -->
- Use consistent terminology across code, comments, and docs — don't mix 'API Formats'/'API Flavors', 'API Type'/'Provider ID', or 'OpenAI format' variants for the same concept — Prevents user confusion and makes the codebase more maintainable by ensuring readers don't have to mentally map multiple terms to the same entity
<!-- rule:623 -->
- Avoid line numbers in comments/docstrings — reference function/class names instead — Line numbers become outdated when code changes, breaking documentation references; function/class names remain stable
<!-- rule:76 -->
- Use `TODO:`/`FIXME:` only for tracked, actionable work with clear removal conditions — remove when conditions are met — Prevents codebase clutter with stale comments and ensures future work is actually tracked and actionable, not just wishful thinking
<!-- rule:656 -->
- Document user-facing features in dedicated sections where users encounter them, not just docstrings — Ensures discoverability — users learning about features like file uploads in `input.md` or `defer_loading` in Tools docs won't miss critical functionality buried in API references
