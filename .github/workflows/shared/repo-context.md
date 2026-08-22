---
# Shared repository context for Pydantic AI gh-aw prompts.
# gh-aw imports this file; the markdown below (after the closing ---) is
# appended to the agent's task prompt at runtime via {{#runtime-import}}.
---

You are working in the **Pydantic AI** repository
([pydantic.dev/docs/ai](https://pydantic.dev/docs/ai/)), a provider-agnostic GenAI agent
framework for Python. It is a `uv` workspace: `pydantic_ai_slim/` (the agent
framework), `pydantic_graph/`, `pydantic_evals/`, `clai/`, with tests in
`tests/`.
