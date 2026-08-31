# OpenAI Codex

Use your [ChatGPT/Codex subscription](https://chatgpt.com/codex) with any Pydantic AI agent: the `openai-codex` provider authenticates against the Codex backend using the same OAuth flow as the official [Codex CLI](https://developers.openai.com/codex/cli/), distinct from the API-key-based [`openai` provider](openai.md). Requests go to the Codex backend with OAuth credentials instead of an API key. Your use of the Codex backend is governed by your agreement with OpenAI; check the applicable [usage policies](https://openai.com/policies/) for your subscription.

## Install

To use the Codex provider, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Acquiring credentials

Codex credentials come from a browser login against your ChatGPT account. There are two ways to get them.

### Use the Codex CLI

Run `codex login` once with the official [Codex CLI](https://developers.openai.com/codex/cli/), then it just works:

```python
from pydantic_ai import Agent

agent = Agent('openai-codex:gpt-5.6-luna')
...
```

The `'openai-codex:'` prefix resolves to [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] backed by [`OpenAICodexProvider`][pydantic_ai.providers.openai_codex.OpenAICodexProvider], which loads the CLI's credentials **read-only**: it honors `CODEX_HOME`, never writes the file, and never falls back to `OPENAI_API_KEY`. Constructing [`OpenAICodexProvider`][pydantic_ai.providers.openai_codex.OpenAICodexProvider] without `credentials` performs the same load.

### Run your own login flow

Applications that log users in themselves (including multi-tenant ones) use [`OpenAICodexOAuthFlow`][pydantic_ai.providers.openai_codex.OpenAICodexOAuthFlow] for the authorization-code + PKCE handshake.

The public Codex client pins its redirect URI to exactly `http://localhost:1455/auth/callback`, so every login completes on the user's own machine. [`exchange_code_from_callback()`][pydantic_ai.providers.openai_codex.OpenAICodexOAuthFlow.exchange_code_from_callback] occupies that port for a moment to catch the redirect, then exchanges the code. A hosted web app cannot receive the callback directly; it runs this same flow from a component on the user's machine (or a tunnel to it) and sends the resulting credentials to the backend.

Credentials outlive the login, so acquire them only when you have none stored:

```python {title="codex_login.py" test="skip - opens a browser and requires user login"}
import webbrowser

from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexOAuthFlow,
)


async def credentials_for(user_id: str) -> OpenAICodexCredentials:
    if (stored := await load_from_your_store(user_id)) is not None:
        return stored

    flow = OpenAICodexOAuthFlow()
    webbrowser.open(flow.authorization_url())
    # Serves localhost:1455 until the browser redirect arrives, then exchanges the code.
    credentials = await flow.exchange_code_from_callback()
    await save_to_your_store(user_id, credentials)
    return credentials


async def load_from_your_store(user_id: str) -> OpenAICodexCredentials | None: ...


async def save_to_your_store(user_id: str, credentials: OpenAICodexCredentials) -> None: ...
```

Store them wherever your app keeps secrets, not in `~/.codex`, which belongs to the Codex CLI. To embed login in an existing web server or a tunnel, serve the redirect yourself and pass the code to [`exchange_code()`][pydantic_ai.providers.openai_codex.OpenAICodexOAuthFlow.exchange_code] instead.

## Storing credentials

Access tokens expire and refresh tokens rotate, so the provider refreshes automatically. Where the rotated set goes depends on how you construct the provider.

### In memory

Passing `credentials` directly keeps everything in memory: fine for a script or a CLI, but the next process starts from the original set again.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexProvider,
)


async def agent_for_user(user_id: str) -> Agent:
    provider = OpenAICodexProvider(credentials=await credentials_for(user_id))
    return Agent(OpenAIResponsesModel('gpt-5.6-luna', provider=provider))


async def credentials_for(user_id: str) -> OpenAICodexCredentials:
    ...  # the credentials you acquired above
```

### In your own storage

For anything longer-lived, give the provider an [`OpenAICodexCredentialSource`][pydantic_ai.providers.openai_codex.OpenAICodexCredentialSource]: your storage, two methods, no lifecycle logic.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexCredentialSource,
    OpenAICodexProvider,
)


class DatabaseCredentialSource(OpenAICodexCredentialSource):
    """Sketch of a store-backed source; the storage internals are yours."""

    def __init__(self, user_id: str):
        self.user_id = user_id

    async def load(self) -> OpenAICodexCredentials: ...

    async def save(self, credentials: OpenAICodexCredentials) -> None: ...


provider = OpenAICodexProvider(credential_source=DatabaseCredentialSource('user-123'))
agent = Agent(OpenAIResponsesModel('gpt-5.6-luna', provider=provider))
...
```

The provider calls `load()` on first use and `save()` after every rotation. That also makes stateless multi-replica deployments safe: because refresh tokens rotate, two replicas refreshing the same stored grant would invalidate each other, so before refreshing, the provider re-reads storage and adopts a peer's newer credentials instead of spending its own. Implementations wanting stricter mutual exclusion can hold a per-user lock inside `save()`.

If `save()` raises, the refreshed credentials stay live in memory and a [`CredentialsPersistenceError`][pydantic_ai.providers.openai_codex.CredentialsPersistenceError] surfaces rather than pretending durability succeeded. Both it and [`CredentialsRefreshError`][pydantic_ai.providers.openai_codex.CredentialsRefreshError] subclass [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] treats an unusable grant like any other provider failure and moves on to the next model. One provider instance carries one user's credentials; construct one per user rather than sharing globally.

## Session affinity and prompt caching

The official Codex client keys prompt-cache affinity off a stable session ID: its root thread keeps `session-id`, `thread-id`, and `x-client-request-id` equal and stable across turns, and defaults the body `prompt_cache_key` to the session ID. Pydantic AI mirrors this per conversation: when messages carry a [`conversation_id`](../message-history.md) (every agent run does), all three headers and the default `prompt_cache_key` carry that conversation identity, since runs continuing shared message history are turns on the same root thread. Runs that share message history therefore share cache affinity automatically, and separate conversations stay isolated.

An explicit `openai_prompt_cache_key` model setting, or explicitly supplied `extra_headers`, always win over the derived values (the cache key only affects the request body; it is never copied into headers). Callers modeling Codex-style child threads, which inherit the session but get a fresh thread ID, can override `thread-id` through `extra_headers`.

## Limitations

- The Codex backend is streaming-only; for non-streaming runs the library transparently drains a stream, so `agent.run_sync()` and friends work as usual.
- The backend rejects some request settings, so they are dropped before sending: `max_tokens`, `temperature`, `top_p`, `openai_top_logprobs`, `openai_truncation`, and `openai_user`.
- `count_tokens()` raises [`UserError`][pydantic_ai.exceptions.UserError]: the input-tokens endpoint is not served under subscription auth.
- There is no device flow: the authorization-code + PKCE redirect flow above is the only login flow the public client supports.
