# Documentation

> Rules for writing docstrings, comments, user-facing documentation, and maintaining documentation accuracy

**When to check**: When writing or updating documentation, comments, or docstrings

## Rules

<!-- rule:272 -->
- Wrap all code identifiers in docstrings with single backticks — parameters, variables, functions, classes, types, fields, API terms — Ensures consistent docstring formatting and makes code elements visually distinct for better readability
<!-- rule:339 -->
- Remove comments that restate obvious code — explain non-obvious intent, edge cases, or constraints instead — Reduces noise and maintenance burden while preserving information that can't be inferred from self-documenting code
<!-- rule:35 -->
- Use Markdown heading syntax (`##`, `###`, `####`) instead of bold text for sections — preserves semantic structure and document hierarchy — Proper heading levels enable navigation, accessibility, and consistent documentation structure across all `.md` files
<!-- rule:396 -->
- Establish one canonical source per topic and link to it — prevents inconsistency and maintenance burden — Documentation that's duplicated across locations becomes outdated differently, creating contradictions and requiring multiple updates for every change
<!-- rule:34 -->
- Link to official provider/project docs instead of duplicating model lists, features, or setup details — prevents stale documentation and reduces maintenance burden — Exhaustive inline lists become outdated quickly; authoritative external sources stay current and reduce maintenance
<!-- rule:138 -->
- Update all related docs in the same PR when changing functionality, APIs, or capabilities — includes docstrings, comments, external docs (e.g., `ai.pydantic.dev`), and API references — Prevents documentation drift that misleads users about actual behavior, limitations, or API contracts
<!-- rule:107 -->
- Register new `docs/` files in `mkdocs.yml` nav section — Ensures documentation files are discoverable in the generated site navigation; orphaned files won't appear in the docs site
<!-- rule:31 -->
- Use consistent terminology across code, docs, comments, and errors (e.g., `freeform` vs `free-form`, `messages` vs `last message`) — prevents user confusion and makes codebase searchable — Inconsistent terminology fragments documentation searches, confuses users trying to map concepts between docs and code, and signals poor API design quality
<!-- rule:76 -->
- Prefix future work with `TODO:` and link workarounds to upstream/internal issues — enables tracking and cleanup when conditions change — Explicit markers with tracking links prevent abandoned workarounds and make technical debt actionable and removable when upstream fixes land
<!-- rule:386 -->
- Keep docs and implementation in sync — when they conflict, explicitly decide which to update and fix it — Prevents user confusion and wasted debugging time when documented behavior doesn't match actual behavior; applies to params, config options, component characteristics, and especially test docstrings which must describe what's actually validated
<!-- rule:106 -->
- Document provider feature support with 'Supported by:' sections — link each provider, list supported/unsupported, distinguish variants (Google Gemini vs Vertex AI), include syntax/config/permissions — Prevents users from attempting unsupported features and enables quick evaluation of provider capabilities without trial-and-error testing across multiple providers
<!-- rule:750 -->
- Document all defaults comprehensively: explicit values, fallback chains, compatibility tradeoffs (backward/forward), and implicit/conditional defaults from parameter interactions — Prevents API confusion and misuse by making fallback behavior, override precedence, and compatibility constraints discoverable in docstrings rather than requiring code archaeology
<!-- rule:150 -->
- Comment non-obvious conditionals — explain edge cases, error handling, and state-based logic — Intent isn't always clear from code alone; comments prevent confusion during maintenance and debugging
<!-- rule:368 -->
- Document what code does now, not what it used to do — skip historical references like "original", "old", "legacy", or "this used to do X" — Historical comments become outdated noise that confuses future readers; focus on current behavior and rationale
<!-- rule:801 -->
- Keep documentation concise—focus on essential information, not implementation details or edge cases — Reduces maintenance burden and improves readability by avoiding unnecessary detail that becomes outdated or clutters the docs
<!-- rule:313 -->
- Document workarounds with (1) expected behavior, (2) why it fails, and (3) that it compensates for external constraints — Prevents future developers from "fixing" workarounds that address API limitations, spec non-compliance, or external bugs — these look like code smells but are actually intentional compensations
<!-- rule:623 -->
- Avoid line numbers in comments/docstrings — use function/class names instead — Line numbers become stale immediately when code changes, breaking the reference and misleading readers
<!-- rule:656 -->
- Document new user-facing features in dedicated sections where users naturally encounter them, not just in docstrings — Users discover features through conceptual docs and guides, not API references — ensures feature discoverability and proper context
