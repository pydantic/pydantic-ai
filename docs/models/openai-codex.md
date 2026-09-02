# OpenAI Codex

Use your [ChatGPT/Codex subscription](https://chatgpt.com/codex) with Pydantic AI instead of a pay-per-token API key. The `openai-codex` provider logs in with the same OAuth flow as the official [Codex CLI](https://developers.openai.com/codex/cli/); for API keys, use the [`openai` provider](openai.md) instead. Your use of the Codex backend is governed by your agreement with OpenAI; check the applicable [usage policies](https://openai.com/policies/) for your subscription.

## Install

To use the Codex provider, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Usage

Run `codex login` once with the [Codex CLI](https://developers.openai.com/codex/cli/), then use the `openai-codex:` prefix:

```python
from pydantic_ai import Agent

agent = Agent('openai-codex:gpt-5.6-luna')
...
```

This resolves to [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] backed by [`OpenAICodexProvider`][pydantic_ai.providers.openai_codex.OpenAICodexProvider], which reads the CLI's credentials from `~/.codex/auth.json` (or `$CODEX_HOME/auth.json`). The file is never written to; refreshed tokens live in memory for the rest of the process.

## Logging in without the Codex CLI

If you don't want to depend on the Codex CLI, [`OpenAICodexOAuthFlow`][pydantic_ai.providers.openai_codex.OpenAICodexOAuthFlow] runs the same browser login. The Codex client pins its redirect URI to `http://localhost:1455/auth/callback`, so [`exchange_code_from_callback()`][pydantic_ai.providers.openai_codex.OpenAICodexOAuthFlow.exchange_code_from_callback] listens on that port until the browser redirects there, then exchanges the code for credentials:

```python {title="codex_login.py" test="skip - opens a browser and requires user login"}
import webbrowser

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_codex import OpenAICodexOAuthFlow, OpenAICodexProvider


async def main():
    flow = OpenAICodexOAuthFlow()
    webbrowser.open(flow.authorization_url())
    credentials = await flow.exchange_code_from_callback()

    provider = OpenAICodexProvider(credentials=credentials)
    agent = Agent(OpenAIResponsesModel('gpt-5.6-luna', provider=provider))
    result = await agent.run('Where does "hello world" come from?')
    print(result.output)
```

Passing `credentials` keeps them in memory only, so the next process has to log in again. To log in once, persist them as described below.

## Persisting credentials

The provider refreshes expired tokens automatically, and refresh tokens are single-use, so the stored copy has to keep up. Give the provider an [`OpenAICodexCredentialSource`][pydantic_ai.providers.openai_codex.OpenAICodexCredentialSource] and it calls `load()` on first use and `save()` after every refresh:

```python
import json
from dataclasses import asdict
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexCredentialSource,
    OpenAICodexProvider,
)


class FileCredentialSource(OpenAICodexCredentialSource):
    def __init__(self, path: Path):
        self.path = path

    async def load(self) -> OpenAICodexCredentials:
        return OpenAICodexCredentials(**json.loads(self.path.read_text()))

    async def save(self, credentials: OpenAICodexCredentials) -> None:
        self.path.write_text(json.dumps(asdict(credentials)))


source = FileCredentialSource(Path('codex-credentials.json'))
provider = OpenAICodexProvider(credential_source=source)
agent = Agent(OpenAIResponsesModel('gpt-5.6-luna', provider=provider))
...
```

Seed the store with the credentials from the login flow above, and keep it wherever your app keeps secrets, not in `~/.codex`, which belongs to the Codex CLI.

If `save()` raises, the refreshed credentials stay live in memory and a [`CredentialsPersistenceError`][pydantic_ai.providers.openai_codex.CredentialsPersistenceError] is raised. Both it and [`CredentialsRefreshError`][pydantic_ai.providers.openai_codex.CredentialsRefreshError] subclass [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] treats an unusable login like any other provider failure.

## Prompt caching

The Codex backend keys prompt caching off a stable session identity, sent as the `session-id`, `thread-id`, and `x-client-request-id` headers and the `prompt_cache_key` request field. Pydantic AI derives all four from the [`conversation_id`](../message-history.md) of the message history, so runs continuing the same conversation share the cache and separate conversations stay isolated. An explicit `openai_prompt_cache_key` model setting or explicitly supplied `extra_headers` always win over the derived values.

## Limitations

- The Codex backend is streaming-only; for non-streaming runs the library transparently drains a stream, so `agent.run_sync()` and friends work as usual.
- The backend rejects some request settings, so they are dropped before sending: `max_tokens`, `temperature`, `top_p`, `openai_top_logprobs`, `openai_truncation`, and `openai_user`.
- The backend requires `store=false`, so every request is sent with it and an explicit `openai_store=True` is silently overridden: responses are never persisted server-side. Consequently, resuming a suspended run raises [`UserError`][pydantic_ai.exceptions.UserError], since there is no stored response to continue from.
- `count_tokens()` raises [`UserError`][pydantic_ai.exceptions.UserError]: the input-tokens endpoint is not served under subscription auth.
- There is no device flow: the browser login above is the only login flow the Codex client supports.
