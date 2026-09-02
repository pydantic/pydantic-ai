# OpenAI Codex

Use your [ChatGPT/Codex subscription](https://chatgpt.com/codex) with Pydantic AI instead of a pay-per-token API key. The `openai-codex` provider authenticates against the Codex backend using the same OAuth flow as the official [Codex CLI](https://developers.openai.com/codex/cli/), and is distinct from the API-key-based [`openai` provider](openai.md). Your use of the Codex backend is governed by your agreement with OpenAI; check the applicable [usage policies](https://openai.com/policies/) for your subscription.

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

This resolves to [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] backed by [`OpenAICodexProvider`][pydantic_ai.providers.openai_codex.OpenAICodexProvider], which reads the CLI's credentials from `~/.codex/auth.json` (or `$CODEX_HOME/auth.json`). The file is only ever read, never written: when the provider refreshes expired tokens, the refreshed set lives in memory for the rest of the process. Pydantic AI never falls back to `OPENAI_API_KEY` here.

## Logging in without the Codex CLI

If you don't want to depend on the Codex CLI, [`OpenAICodexOAuthFlow`][pydantic_ai.providers.openai_codex.OpenAICodexOAuthFlow] runs the same browser login. The Codex client pins its redirect URI to `http://localhost:1455/auth/callback`, so [`exchange_code_from_callback()`][pydantic_ai.providers.openai_codex.OpenAICodexOAuthFlow.exchange_code_from_callback] listens on that port until the browser redirects there, then exchanges the code for [`OpenAICodexCredentials`][pydantic_ai.providers.openai_codex.OpenAICodexCredentials].

You only want to do that once, and since access tokens expire and refresh tokens are single-use, the credentials change over time. So give the provider an [`OpenAICodexCredentialSource`][pydantic_ai.providers.openai_codex.OpenAICodexCredentialSource] that wraps your storage: the provider calls `load()` on first use and `save()` after every refresh, and the login flow only runs when there is nothing stored yet.

```python {title="codex_login.py" test="skip - opens a browser and requires user login"}
import json
import webbrowser
from dataclasses import asdict
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexCredentialSource,
    OpenAICodexOAuthFlow,
    OpenAICodexProvider,
)


class FileCredentialSource(OpenAICodexCredentialSource):
    def __init__(self, path: Path):
        self.path = path

    async def load(self) -> OpenAICodexCredentials:
        return OpenAICodexCredentials(**json.loads(self.path.read_text()))

    async def save(self, credentials: OpenAICodexCredentials) -> None:
        self.path.write_text(json.dumps(asdict(credentials)))


async def main():
    source = FileCredentialSource(Path('codex-credentials.json'))
    if not source.path.exists():
        flow = OpenAICodexOAuthFlow()
        webbrowser.open(flow.authorization_url())
        await source.save(await flow.exchange_code_from_callback())

    provider = OpenAICodexProvider(credential_source=source)
    agent = Agent(OpenAIResponsesModel('gpt-5.6-luna', provider=provider))
    result = await agent.run('Where does "hello world" come from?')
    print(result.output)
```

Store the file wherever your app keeps secrets, not in `~/.codex`, which belongs to the Codex CLI. If you'd rather not persist anything, pass `credentials=` to the provider instead of a source: refreshed tokens then live in memory only, and the next process has to log in again.

If `save()` raises, the refreshed credentials stay live in memory and a [`CredentialsPersistenceError`][pydantic_ai.providers.openai_codex.CredentialsPersistenceError] is raised so you know the store is out of date. Both it and [`CredentialsRefreshError`][pydantic_ai.providers.openai_codex.CredentialsRefreshError] subclass [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] treats an unusable login like any other provider failure and moves on to the next model.

## Prompt caching

The Codex backend keys prompt caching off a stable session identity, sent as the `session-id`, `thread-id`, and `x-client-request-id` headers and the `prompt_cache_key` request field. Pydantic AI derives all four from the [`conversation_id`](../message-history.md) of the message history, so runs continuing the same conversation share the cache and separate conversations stay isolated. An explicit `openai_prompt_cache_key` model setting or explicitly supplied `extra_headers` always win over the derived values.

## Limitations

- The Codex backend is streaming-only; for non-streaming runs the library transparently drains a stream, so `agent.run_sync()` and friends work as usual.
- The backend rejects some request settings, so they are dropped before sending: `max_tokens`, `temperature`, `top_p`, `openai_top_logprobs`, `openai_truncation`, and `openai_user`.
- The backend requires `store=false`, so every request is sent with it and an explicit `openai_store=True` is silently overridden: responses are never persisted server-side. Consequently, resuming a suspended run raises [`UserError`][pydantic_ai.exceptions.UserError], since there is no stored response to continue from.
- `count_tokens()` raises [`UserError`][pydantic_ai.exceptions.UserError]: the input-tokens endpoint is not served under subscription auth.
- There is no device flow: the browser login above is the only login flow the Codex client supports.
