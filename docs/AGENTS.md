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
- Put example-level exclusions on the fence, such as `{test="skip" lint="skip"}`, rather than adding tooling suppressions to pedagogical code. Examples that are linted are also type-checked with pyright; opt one out with `typecheck="skip - <reason>"` only when no natural fix exists, such as a deliberate type error or an uninstalled dependency.
- Use `{gateway_toggle="skip"}` to suppress rendered Gateway variants for a provider-only example. Mark provider-only examples even when an explicit model constructor currently prevents automatic Gateway toggles.
- Check Google examples against Google Cloud support before allowing Gateway variants. Mark Gemini API-only examples with `{gateway_toggle="skip"}`. Link to [Google model guidance](models/google.md) when documenting Gateway support.
- Keep `gateway_toggle` separate from `test` and `lint` metadata. The flag controls rendered Gateway variants and does not suppress example checks.
- Combine parameter variations when one example plus notes preserves every meaningful difference. Split examples when use cases, prerequisites, or constraints differ.
- Use examples that demonstrate a credible user task or decision without introducing complexity unrelated to the feature.

## Review

- Use `trigger:docs-preview` on a Pydantic AI pull request to dispatch navigation validation to unified-docs. This workflow does not render an HTML preview.
- Read the unified-docs result comment and verify its reported SHA matches the current pull request head.
- The workflow removes `trigger:docs-preview` after dispatch. Re-add the label to request validation for the current pull request head.
- Ask a maintainer to arrange a pinned-source preview for the current Pydantic AI pull request head.
- Render the current PR head in a unified-docs preview before merging.
- Verify the preview's reported SHA matches that head.
- Treat navigation success and rendered-preview success as separate gates.
- Unified-docs renderer pull requests use their separate `trigger:docs` label. Docs Self-Preview builds the site and posts its rendered URL.

<!-- /braindump -->

# Front pages: `docs/index.md` and `README.md`

The docs index and repository README tell the same story on two surfaces. Keep their shared wording and code examples synchronized while preserving the markup each renderer needs:

- `docs/index.md` uses relative links, tabs (`=== "..."`), numbered annotations (`(1)!`), and MkDocs-only markup.
- `README.md` uses absolute documentation links, `###` sections instead of tabs, and plain one-line `#` comments instead of annotations.
- Mirrored code examples remain code-identical; only comments, annotations, link forms, and fence attributes may differ.
- README snippets that cannot run in the documentation test environment are excluded by `tests/test_examples.py`, not by fence attributes, so README fences remain compatible with GitHub rendering.
- When the shared tagline or Harness framing changes, check `docs/harness/index.md` and `src/pydantic_ai_harness/README.md` too.
