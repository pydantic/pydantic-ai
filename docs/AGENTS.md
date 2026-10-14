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
- Combine parameter variations when one example plus notes preserves every meaningful difference. Split examples when use cases, prerequisites, or constraints differ.
- Use examples that demonstrate a credible user task or decision without introducing complexity unrelated to the feature.

## Model names

- Always name models with an alias from [`model_aliases.py`](model_aliases.py), never a concrete model ID. An alias is `<provider>:<tier>`, with the tiers `small`, `medium`, and `large`.
- `{{ openai:medium }}` resolves to the full model string, as in `Agent('{{ openai:medium }}')` or `gateway/{{ openai:medium }}`. `{{ openai:medium.name }}` resolves to the bare model name, as in `OpenAIChatModel('{{ openai:medium.name }}')`, `openai-chat:{{ openai:medium.name }}`, or `model_name='{{ openai:medium.name }}'` in example output.
- To move the docs to a new model, edit only `model_aliases.py`. The docs site and `tests/test_examples.py` resolve the aliases. Then run `make update-examples` to refresh example output that depends on the model, such as costs; it keeps the aliases. `tests/test_docs_parity.py` and `tests/harness/test_docs_parity.py` flag the READMEs that mirror a docs page with concrete names, since GitHub can't resolve aliases.
- Name a model directly only on a page about that model or version, such as a provider page or the changelog. `tests/test_docs_model_aliases.py` fails on concrete names elsewhere; register a deliberate exception in its `EXEMPT_PAGES`, with the reason.

## Review

- Render documentation in a unified-docs preview before merging.

<!-- /braindump -->

# Front pages: `docs/index.md` and `README.md`

The docs index and repository README tell the same story on two surfaces. Keep their shared wording and code examples synchronized while preserving the markup each renderer needs:

- `docs/index.md` uses relative links, tabs (`=== "..."`), numbered annotations (`(1)!`), and MkDocs-only markup.
- `README.md` uses absolute documentation links, `###` sections instead of tabs, and plain one-line `#` comments instead of annotations.
- Mirrored code examples remain code-identical; only comments, annotations, link forms, and fence attributes may differ.
- README snippets that cannot run in the documentation test environment are excluded by `tests/test_examples.py`, not by fence attributes, so README fences remain compatible with GitHub rendering.
- When the shared tagline or Harness framing changes, check `docs/harness/index.md` and `src/pydantic_ai_harness/README.md` too.
