<!-- braindump: rules extracted from PR review patterns -->

# docs/ Guidelines

## Documentation

- Document all providers that support a feature with examples — ensures complete coverage and discoverability for users — Users may miss provider support if docs only show partial examples, leading to incorrect assumptions about feature availability
- Link to provider docs for model lists and features — prevents stale info when providers update their offerings — External provider catalogs change frequently; linking to authoritative sources keeps docs accurate without maintenance burden
- Link to existing docs/API refs instead of re-explaining concepts — reduces duplication and keeps info in sync — Prevents documentation drift and outdated explanations by maintaining a single source of truth for each concept
- Keep feature docs provider-agnostic; put provider-specific details in dedicated provider sections — Prevents duplication and makes docs easier to maintain when shared parameters change across providers
- Link to canonical docs rather than duplicating content — prevents drift and maintenance burden — Consolidating documentation into existing files with cross-references keeps information consistent and reduces the effort needed to update multiple locations when changes occur.
- Document only public APIs and user-facing behavior — exclude internals, framework abstractions, and implementation plumbing — Users need actionable documentation on what they can use, not confusing details about internal mechanics they can't control
- Explain before showing — place explanatory text before code examples, not after — Users need context to understand code examples; "explain then show" improves comprehension and reduces confusion
- Use 'Pydantic AI' (with space) in prose — ensures consistent brand identity — Maintains professional brand consistency across all documentation and prevents confusion with variations like 'Pydantic-AI', 'PydanticAI', or 'pAI'
- Create dedicated pages for substantial features — ensures discoverability and comprehensive coverage vs. fragmented mentions — Prevents users from missing features when they approach from different contexts (CLI vs. API) and allows features to be documented holistically rather than buried in subsections.
- Avoid `# ruff: noqa` or `# type: ignore` in doc examples — ensures examples stay correct and runnable — Skip directives hide bugs and type errors in documentation code that users will copy, leading to broken examples in the wild
- Explicitly mark parameters/features as 'optional' in docs, even when types show it — reduces cognitive load for readers — Users shouldn't need to parse type signatures to understand optionality; explicit labels make documentation scannable and accessible to all skill levels
- Remove documentation sections explaining standard behavior that "just works" — keeps docs focused on actionable, non-obvious information — Users don't need explanations of things that work automatically; documentation should focus on configuration requirements, edge cases, and non-obvious behaviors that affect usage decisions
- Strip boilerplate from docs examples — show only the feature being demonstrated — Reduces cognitive load and helps readers focus on the specific API or pattern being taught without distraction from scaffolding code.
- Always include provider prefixes in model name examples (e.g., `openai:gpt-5.2`, `anthropic:claude-opus-4-5`) — Teaches users the correct format and prevents runtime errors from missing provider context

<!-- /braindump -->
