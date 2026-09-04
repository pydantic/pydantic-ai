# GitHub Copilot

[GitHub Copilot](https://docs.github.com/en/copilot) serves Anthropic, OpenAI, Google, xAI and MoonshotAI models through an OpenAI-compatible Chat Completions API, metered in AI credits drawn from your Copilot subscription at published per-model token rates.

!!! note "This is not GitHub Models"
    [`GitHubProvider`][pydantic_ai.providers.github.GitHubProvider] and the `github:` prefix served [GitHub Models](openai.md#github-models), which was retired in July 2026. Copilot is a different API with a different host, its own model ids, and its own credentials.

## Install

To use [`GitHubCopilotModel`][pydantic_ai.models.github_copilot.GitHubCopilotModel], you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

Copilot authenticates with a bearer token. An OAuth user token — what `gh auth token` prints, or what the Copilot CLI stores after `copilot login` — works directly against the inference API; no token exchange is needed.

| Token type | Status |
| --- | --- |
| OAuth user token (`gho_`) | Works. |
| Copilot API token (`tid=…`) | Works, for plans that issue one. |
| Fine-grained PAT (`github_pat_`) with **Copilot Requests** | Listed by [GitHub's Copilot SDK docs](https://docs.github.com/copilot/how-tos/copilot-sdk/authenticate-copilot-sdk/authenticate-copilot-sdk), but rejected with `401 unauthorized` on the Individual plan we tested. |
| Classic PAT (`ghp_`) | Not supported by GitHub. |

## Environment variable

```bash
export GITHUB_COPILOT_API_KEY='your-copilot-token'
```

`GITHUB_COPILOT_API_TOKEN` and `COPILOT_GITHUB_TOKEN` are read as fallbacks, since GitHub's own tooling uses those names. The general-purpose `GITHUB_TOKEN`, `GH_TOKEN` and `GITHUB_API_KEY` variables are deliberately **not** read, so a token you set for the GitHub API is never sent to Copilot.

You can then use [`GitHubCopilotModel`][pydantic_ai.models.github_copilot.GitHubCopilotModel] by name:

```python
from pydantic_ai import Agent

agent = Agent('github-copilot:claude-haiku-4.5')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.github_copilot import GitHubCopilotModel

model = GitHubCopilotModel('gpt-5.4')
agent = Agent(model)
...
```

Or pass the token explicitly through [`GitHubCopilotProvider`][pydantic_ai.providers.github_copilot.GitHubCopilotProvider]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.github_copilot import GitHubCopilotModel
from pydantic_ai.providers.github_copilot import GitHubCopilotProvider

model = GitHubCopilotModel(
    'claude-haiku-4.5',
    provider=GitHubCopilotProvider(api_key='your-copilot-token'),
)
agent = Agent(model)
...
```

## Model ids depend on your plan

Copilot's catalog varies by subscription and changes often, so Pydantic AI ships no fixed list — any id is accepted and sent to Copilot exactly as you wrote it, dots included. List the ids your own plan serves with:

```bash
curl -H "Authorization: Bearer $GITHUB_COPILOT_API_KEY" https://api.githubcopilot.com/models
```

Two `400` responses tell you why an id didn't work:

- `model_not_supported` — your plan doesn't include that model. `claude-sonnet-4.5`, for instance, is unavailable on an Individual plan.
- `unsupported_api_for_model` — the model exists but isn't served on Chat Completions. Pydantic AI does not yet speak Copilot's Responses API, so these ids are unreachable for now.

## Thinking

Reasoning models reachable on Chat Completions — the GPT, Gemini, Grok and Kimi ids whose catalog entry lists `reasoning_effort` — take the unified [`thinking`][pydantic_ai.settings.ModelSettings.thinking] setting:

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

agent = Agent(
    'github-copilot:gpt-5.4',
    model_settings=ModelSettings(thinking='high'),
)
...
```

Copilot's Anthropic models are the exception. They think, but only through an API Pydantic AI doesn't speak yet: Copilot's Chat Completions endpoint rejects `reasoning_effort` for them outright, so requesting `thinking` on a `claude-` id raises a [`UserError`][pydantic_ai.exceptions.UserError] rather than silently returning an answer with no reasoning. `thinking=False` is accepted and sends nothing, since that is what it asks for.

## Custom endpoints

Copilot Enterprise hosts, GitHub Enterprise Server, and local proxies speak the same API on a different host. Point the provider at one with `base_url`, or with the `GITHUB_COPILOT_BASE_URL`, `COPILOT_API_URL` or `GITHUB_COPILOT_API_BASE` environment variable:

```python
from pydantic_ai import Agent
from pydantic_ai.models.github_copilot import GitHubCopilotModel
from pydantic_ai.providers.github_copilot import GitHubCopilotProvider

model = GitHubCopilotModel(
    'claude-haiku-4.5',
    provider=GitHubCopilotProvider(
        api_key='your-copilot-token',
        base_url='https://copilot.example.com',
    ),
)
agent = Agent(model)
...
```

## Not supported

Copilot's Responses (`/responses`) and Messages (`/v1/messages`) APIs, embeddings, and realtime are not implemented. Cost and context-window data are also unavailable: [genai-prices](https://github.com/pydantic/genai-prices) has no `github-copilot` entry yet, tracked in [genai-prices#681](https://github.com/pydantic/genai-prices/issues/681).
