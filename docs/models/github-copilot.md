# GitHub Copilot

[GitHub Copilot](https://docs.github.com/en/copilot) serves Anthropic, OpenAI, Google, xAI and MoonshotAI models, metered in AI credits drawn from your Copilot subscription at published per-model token rates. Pydantic AI talks to Copilot's OpenAI-compatible Chat Completions API, which reaches only the ids Copilot exposes there: Claude, Gemini and Kimi ids and `gpt-5.4` at the time of writing, while every xAI Grok id and most other GPT ids are served on the Responses API alone and are out of reach until Pydantic AI speaks it. See [Model ids depend on your plan](#model-ids-depend-on-your-plan) to check yours.

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
curl -H "Authorization: Bearer $GITHUB_COPILOT_API_KEY" \
     -H "Copilot-Integration-Id: vscode-chat" \
     https://api.githubcopilot.com/models
```

Each entry's `supported_endpoints` says which API serves it; Pydantic AI needs `/chat/completions` in that list.

The listing depends on the `Copilot-Integration-Id` header, which `GitHubCopilotProvider` sends on every request, so an `Authorization`-only call returns fewer ids than the provider can actually reach — the Gemini ids, at the time of writing.

Two `400` responses tell you why an id didn't work:

- `model_not_supported` — your plan doesn't include that model. `claude-sonnet-4.5`, for instance, is unavailable on an Individual plan.
- `unsupported_api_for_model` — the model exists but isn't served on Chat Completions. Pydantic AI does not yet speak Copilot's Responses API, so these ids — every xAI Grok id, at the time of writing — are unreachable for now.

## Thinking

Reasoning models reachable on Chat Completions — such as `gpt-5.4`, `claude-sonnet-5` and `gemini-3.8-flash` at the time of writing — take the unified [`thinking`][pydantic_ai.settings.ModelSettings.thinking] setting:

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

agent = Agent(
    'github-copilot:gpt-5.4',
    model_settings=ModelSettings(thinking='high'),
)
...
```

Which ids surface their reasoning depends on the family. Copilot returns Anthropic and Google reasoning in a `reasoning_text` field rather than in either of the field names OpenAI-compatible providers usually use, and Pydantic AI knows that field for `claude-` and `gemini-` ids, so their reasoning arrives as a [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] on both the streamed and non-streamed paths and goes back in the same field on later turns. Copilot returns a `reasoning_opaque` signature alongside it, which Pydantic AI does not carry; Copilot accepts follow-up turns without it. The OpenAI and MoonshotAI ids are the other case: `gpt-5.4` and `kimi-k3` reason on the effort you give them — `kimi-k3` bills reasoning tokens for it — but Copilot returns no reasoning text at all, so those ids never produce a `ThinkingPart`.

The Claude ids also reason *adaptively*: the effort you set is a ceiling rather than an instruction, so Copilot may answer an easy question with no reasoning at any effort, and a `ThinkingPart` is not guaranteed on every response.

`thinking` is forwarded as `reasoning_effort` and Copilot decides what it accepts, per model: ask for a level an id doesn't list and it answers `400 invalid_reasoning_effort` naming the levels it does. Three cases worth knowing:

- `thinking=False` becomes `reasoning_effort='none'`, which only some ids offer. The `claude-` and `gemini-` ids list `low` upwards and no `none`, so it `400`s there rather than silently doing nothing.
- `thinking=True` becomes `reasoning_effort='medium'`, which `kimi-k3` does not list (its levels are `low`, `high`, `max`), so pick an explicit level for that one.
- An id whose entry carries no `reasoning_effort` key at all, such as `claude-haiku-4.5`, rejects every value.

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

Copilot's Responses (`/responses`) and Messages (`/v1/messages`) APIs and realtime are not implemented. Neither are embeddings, but because `github-copilot` counts as an OpenAI-chat-compatible provider, `Embedder('github-copilot:...')` still builds an [`OpenAIEmbeddingModel`](../embeddings.md) rather than raising — it points at the gateway's `/embeddings`, which answers `400`. Cost and context-window data are also unavailable: [genai-prices](https://github.com/pydantic/genai-prices) gained a `github-copilot` entry in [genai-prices#683](https://github.com/pydantic/genai-prices/pull/683), but no published release carries it yet.
