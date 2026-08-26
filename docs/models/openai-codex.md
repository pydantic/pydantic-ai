# OpenAI Codex

Use your [ChatGPT/Codex subscription](https://chatgpt.com/codex) with any Pydantic AI agent: the `openai-codex` provider authenticates against the Codex backend using the same OAuth flow as the official [Codex CLI](https://developers.openai.com/codex/cli/), distinct from the API-key-based [`openai` provider](openai.md). Requests go to the Codex backend with OAuth credentials instead of an API key. Your use of the Codex backend is governed by your agreement with OpenAI; check the applicable [usage policies](https://openai.com/policies/) for your subscription.

## Install

To use the Codex provider, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Local development

Run `codex login` once with the official [Codex CLI](https://developers.openai.com/codex/cli/), then it just works:

```python
from pydantic_ai import Agent

agent = Agent('openai-codex:gpt-5.6-luna')
...
```

The `'openai-codex:'` prefix resolves to [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] backed by [`OpenAICodexProvider`][pydantic_ai.providers.openai_codex.OpenAICodexProvider], which loads the CLI's credentials **read-only**: it honors `CODEX_HOME`, never writes the file, and never falls back to `OPENAI_API_KEY`. The same load is available explicitly as [`OpenAICodexProvider.from_codex_cli()`][pydantic_ai.providers.openai_codex.OpenAICodexProvider.from_codex_cli].

## Application-owned credentials

Applications (including multi-tenant ones) own login and storage themselves: obtain credentials once per user (see [below](#build-your-own-login)), persist them your way, and inject them per user:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexProvider,
)


async def save_credentials(credentials: OpenAICodexCredentials) -> None:
    ...  # persist to your per-user store


provider = OpenAICodexProvider(
    credentials=OpenAICodexCredentials(
        access_token='...', refresh_token='...', account_id='...'
    ),
    on_credentials_refresh=save_credentials,
)
agent = Agent(OpenAIResponsesModel('gpt-5.6-luna', provider=provider))
...
```

Tokens are refreshed automatically; rotated credentials are passed to `on_credentials_refresh` so your store stays current (if the callback raises, the in-memory credentials are still updated and a [`CredentialsPersistenceError`][pydantic_ai.providers.openai_codex.CredentialsPersistenceError] surfaces). One provider instance carries one user's credentials; construct one per user rather than sharing globally.

## Build your own login

[`OpenAICodexOAuthFlow`][pydantic_ai.providers.openai_codex.OpenAICodexOAuthFlow] provides the authorization-code + PKCE primitives without any interactive machinery, so you can embed login anywhere. A complete CLI-style flow:

```python {title="codex_login.py" test="skip - opens a browser and requires user login"}
import asyncio
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_codex import OpenAICodexOAuthFlow, OpenAICodexProvider

flow = OpenAICodexOAuthFlow()  # defaults to the pinned http://localhost:1455/auth/callback
code: list[str] = []


class Callback(BaseHTTPRequestHandler):
    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        if params.get('state', [None])[0] == flow.state:
            code.append(params['code'][0])
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Logged in - you can close this tab.')


server = HTTPServer(('localhost', 1455), Callback)
webbrowser.open(flow.authorization_url())
while not code:
    server.handle_request()  # serve until the callback with a valid state arrives

credentials = asyncio.run(flow.exchange_code(code[0]))
# persist wherever your app keeps secrets - not ~/.codex, which belongs to the Codex CLI

provider = OpenAICodexProvider(credentials=credentials)
agent = Agent(OpenAIResponsesModel('gpt-5.6-luna', provider=provider))
print(agent.run_sync('hello from my own login flow').output)
```

!!! note
    The public Codex client pins its redirect URI to exactly `http://localhost:1455/auth/callback`, so login always completes on the user's machine: a hosted web app can never receive the callback directly. Web apps run this same flow from a component on the user's machine (or a tunnel to it) and send the resulting credentials to the backend.

## Session affinity and prompt caching

The official Codex client keys prompt-cache affinity off a stable session ID: it sends `session-id` and `thread-id` headers on every request and defaults the body `prompt_cache_key` to the session ID. Pydantic AI mirrors this per conversation: when messages carry a [`conversation_id`](../message-history.md) (every agent run does), the conversation is the session and the run is the thread, and `prompt_cache_key` defaults to the conversation ID. Runs that share message history therefore share cache affinity automatically, and separate conversations stay isolated.

An explicit `openai_prompt_cache_key` model setting, or explicitly supplied `extra_headers`, always win over the derived values.

## Limitations

- The Codex backend is streaming-only; for non-streaming runs the library transparently drains a stream, so `agent.run_sync()` and friends work as usual.
- The backend rejects some request settings, so they are dropped before sending: `max_tokens`, `temperature`, `top_p`, `openai_top_logprobs`, `openai_truncation`, and `openai_user`.
- `count_tokens()` raises [`UserError`][pydantic_ai.exceptions.UserError]: the input-tokens endpoint is not served under subscription auth.
- There is no device flow: the authorization-code + PKCE redirect flow above is the only login flow the public client supports.
