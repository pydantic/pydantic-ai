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
- Use `{gateway_toggle="skip"}` for examples that use features unavailable through Pydantic AI Gateway. Keep the marker on unsupported examples even when the renderer does not automatically toggle explicit provider constructors; see the [renderer contract](https://github.com/pydantic/unified-docs/blob/877c3a80f635bd100b86599c76e1873f3983ec57/README.md#gateway-toggle-examples).
- Check Google examples against Google Cloud support before allowing Gateway variants. Mark Gemini API-only examples with `{gateway_toggle="skip"}`. Link to [Google model guidance](models/google.md) when documenting Gateway support.
- Keep `gateway_toggle` separate from `test` and `lint` metadata. The flag controls rendered Gateway variants and does not suppress example checks.
- Combine parameter variations when one example plus notes preserves every meaningful difference. Split examples when use cases, prerequisites, or constraints differ.
- Use examples that demonstrate a credible user task or decision without introducing complexity unrelated to the feature.

## Review

- On a Pydantic AI pull request, apply `trigger:docs-preview` to dispatch navigation validation to unified-docs. Dispatch removes the label; re-add it after source changes.
- Read the result and verify its reported SHA matches the current Pydantic AI pull request head. This workflow validates navigation; it does not render HTML.
- Before merging, have a maintainer arrange a rendered preview pinned to the current Pydantic AI source head. Verify the preview's reported SHA. Navigation validation and the rendered preview are separate gates.
- On a unified-docs renderer pull request, use its separate `trigger:docs` label for Docs Self-Preview. It builds the site and posts its rendered URL.

<!-- /braindump -->

# Front pages: `docs/index.md` and `README.md`

The docs index and repository README tell the same story on two surfaces. Keep their shared wording and code examples synchronized while preserving the markup each renderer needs:

- `docs/index.md` uses relative links, tabs (`=== "..."`), numbered annotations (`(1)!`), and MkDocs-only markup.
- `README.md` uses absolute documentation links, `###` sections instead of tabs, and plain one-line `#` comments instead of annotations.
- Mirrored code examples remain code-identical; only comments, annotations, link forms, and fence attributes may differ.
- README snippets that cannot run in the documentation test environment are excluded by `tests/test_examples.py`, not by fence attributes, so README fences remain compatible with GitHub rendering.
- When the shared tagline or Harness framing changes, check `docs/harness/index.md` and `src/pydantic_ai_harness/README.md` too.

# Model names

- Name models with an alias from [`model_aliases/aliases.json`](model_aliases/aliases.json), never a model ID. Chat models use `<provider>:<tier>`, with the tiers `small`, `medium`, `large`, and `xlarge`; every chat provider defines all four. Realtime, image generation, and embedding models use `<provider>:realtime`, `<provider>:image`, and `<provider>:embedding`.
- Pick a chat tier by the model's role in the example: `medium` for a generic example, `small` for a cheap side task such as a simple judge, and `large` or `xlarge` when the example needs a more capable model.
- `{{ openai:medium }}` resolves to the full model string, as in `Agent('{{ openai:medium }}')` or `gateway/{{ openai:medium }}`. `{{ openai:medium.name }}` resolves to the bare model name, as in `OpenAIResponsesModel('{{ openai:medium.name }}')`, `openai-chat:{{ openai:medium.name }}`, or `model_name='{{ openai:medium.name }}'` in example output.
- Aliases stand only for models served under their provider's own IDs. Third-party catalogs, such as OpenRouter, Snowflake, or Heroku, keep their own model IDs.
- Don't name or describe the aliased model in prose, labels, or comments, since the alias can move. Write "an OpenAI model", not the model's name.
- The exceptions are pages about specific models or versions, such as provider pages under `models/` or the changelog. `tests/test_docs_model_aliases.py` fails on model IDs anywhere else; register an exception in its `EXEMPT_PAGES` or `EXEMPT_DIRS`, with the reason.
- To move the docs to a new model, edit only `model_aliases/aliases.json`, following [`model_aliases/AGENTS.md`](model_aliases/AGENTS.md), then run `make update-examples` to refresh example output that depends on the model, such as costs; it keeps the aliases. The docs site and `tests/test_examples.py` resolve the aliases. The READMEs that `tests/test_docs_parity.py` and `tests/harness/test_docs_parity.py` compare with docs pages keep concrete model IDs, since GitHub can't resolve aliases; update them when those tests fail.
