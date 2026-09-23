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

## Device login

Use [`GitHubCopilotOAuthFlow`][pydantic_ai.providers.github_copilot.GitHubCopilotOAuthFlow] to obtain a token through GitHub's device flow:

```python {test="skip"}
import os
import sys

import anyio

from pydantic_ai import Agent
from pydantic_ai.models.github_copilot import GitHubCopilotModel
from pydantic_ai.providers.github_copilot import (
    GitHubCopilotOAuthFlow,
    GitHubCopilotProvider,
)


async def main() -> None:
    flow = GitHubCopilotOAuthFlow(client_id=os.environ['GITHUB_OAUTH_CLIENT_ID'])
    authorization = await flow.start()
    sys.stdout.write(f'Open {authorization.verification_uri}\nEnter code: {authorization.user_code}\n')
    sys.stdout.flush()
    credentials = await flow.wait_for_authorization()

    provider = GitHubCopilotProvider(api_key=credentials.access_token)
    async with provider:
        agent = Agent(GitHubCopilotModel('claude-haiku-4.5', provider=provider))
        result = await agent.run('Explain Python context managers in two sentences.')
    sys.stdout.write(f'{result.output}\n')


anyio.run(main)
```

[`GitHubCopilotOAuthFlow`][pydantic_ai.providers.github_copilot.GitHubCopilotOAuthFlow] implements GitHub.com's [device authorization flow](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#device-flow). Register an OAuth application and enable device flow in its settings. `GITHUB_OAUTH_CLIENT_ID` is application configuration used by this example, not a variable Pydantic AI reads automatically.

!!! warning "Only approve device codes from your own login attempt"
    Use device flow for constrained clients such as CLI or headless applications. Applications that can receive browser redirects should use authorization code with PKCE instead. Anyone can initiate a device grant with a public client ID; entering a code supplied by an attacker authorizes their client, not yours. Tell users to approve only codes shown by the application they are signing into, not codes received in messages. See [RFC 8628's phishing guidance](https://www.rfc-editor.org/rfc/rfc8628.html#section-5.4).

The helper requires your application's `client_id`; it does not borrow another application's identity. It requests no scopes by default. Pass `scope=` if your application needs GitHub permissions.

!!! warning "GitHub authorization does not establish Copilot access"
    A successful login returns a GitHub OAuth token, not proof that Copilot accepts that application's token or that the account has an eligible subscription. Verify both for your application. The existing-token examples below remain available when you already have working credentials.

`start()` returns a [`GitHubCopilotDeviceAuthorization`][pydantic_ai.providers.github_copilot.GitHubCopilotDeviceAuthorization]. Display its `user_code` and `verification_uri`, then await `wait_for_authorization()`. Polling respects GitHub's interval and increases it when GitHub responds with `slow_down`. The local deadline is `expires_in` seconds after the device-code response arrives; time spent displaying the code still counts toward it. Expiry, denial, and invalid responses raise [`UserError`][pydantic_ai.exceptions.UserError]; transport errors propagate unchanged. Cancellation stops polling. Call `start()` again to retry after any outcome, and use a separate flow instance for each concurrent login.

Your application owns browser opening and credential storage. [`GitHubCopilotCredentials`][pydantic_ai.providers.github_copilot.GitHubCopilotCredentials] excludes tokens from `repr`, but serialized credentials still contain secrets. Keep them in your application's credential store, not logs. An injected `http_client` stays caller-owned; without one, the helper closes its temporary clients after each request. OAuth redirects are not followed, and this helper supports GitHub.com, not enterprise authorization hosts.

!!! note "Credential renewal stays application-owned"
    Some OAuth applications issue expiring access tokens. The result preserves `refresh_token`, `expires_in`, and `refresh_token_expires_in` when GitHub supplies them; durations are seconds from issuance. Persist the issuance time with these values if you save the credentials. `GitHubCopilotProvider(api_key=...)` does not renew tokens. Your application must refresh them through GitHub's documented flow or ask the user to sign in again.

## Environment variable

If you already have a token, set `GITHUB_COPILOT_API_KEY` before starting your application:

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

Copilot's Claude ids inherit Anthropic's sampling restriction. On Opus 4.7, Opus 4.8, Opus 5, Sonnet 5, Fable 5 and Mythos 5 — and on any id whose name starts with one of those — `temperature` and `top_p` are dropped from the request rather than forwarded, silently, exactly as they are when you reach the same models through the [Anthropic API](anthropic.md). Only those two keys go: `top_k` has no Chat Completions equivalent to drop, and a `temperature` you pass in `extra_body`, or any `openai_*` setting, is still sent. Every other id — `claude-haiku-4.5` among them — forwards both unchanged.
