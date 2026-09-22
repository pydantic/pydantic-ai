<!-- braindump: rules extracted from PR review patterns -->

# `docs/` guidelines

Follow the general [documentation guidance](../agent_docs/documentation.md). These rules cover published Markdown under `docs/`.

## Links and structure

- Use reference-style links for API elements: `[ElementName][module.path.ElementName]`. They provide hover documentation and API navigation on the published site.
- Write the project name as `Pydantic AI`.
- Use admonitions (`!!! note`, `!!! warning`) for callouts, not blockquotes or GitHub alerts.
- Keep provider-specific configuration and behavior in `docs/models/{provider}.md` and `docs/api/models/{provider}.md`. General guides use a minimal provider-agnostic example and link to the provider page.
- In provider feature tables, use a `Notes` or `Provider Support Notes` column for variations, limitations, and special values. Use the standard labels `Full feature support` and `Limited parameter support`, and put unsupported variants in the `Unsupported` column.

## Examples

- Keep code examples executable unless they require external services, credentials, or non-deterministic behavior. Use mocks or fixtures when they keep the example representative.
- Put example-level exclusions on the fence, such as `{test="skip" lint="skip"}`, rather than adding tooling suppressions to pedagogical code.
- Combine parameter variations when one example plus notes preserves every meaningful difference. Split examples when use cases, prerequisites, or constraints differ.
- Use examples that demonstrate a credible user task or decision without introducing complexity unrelated to the feature.

## Review

- Render documentation in a unified-docs preview before merging.

<!-- /braindump -->

# Front pages: `docs/index.md` and `README.md`

The docs index and repository README tell the same story on two surfaces. Keep their shared wording and code examples synchronized while preserving the markup each renderer needs:

- `docs/index.md` uses relative links, tabs (`=== "..."`), numbered annotations (`(1)!`), and MkDocs-only markup.
- `README.md` uses absolute documentation links, `###` sections instead of tabs, and plain one-line `#` comments instead of annotations.
- Mirrored code examples remain code-identical; only comments, annotations, link forms, and fence attributes may differ.
- README snippets that cannot run in the documentation test environment are excluded by `tests/test_examples.py`, not by fence attributes, so README fences remain compatible with GitHub rendering.
- When the shared tagline or Harness framing changes, check the Harness repository's `docs/index.md` and `README.md` too.
