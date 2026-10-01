---
description: "Use Sign in with ChatGPT to authorize Pydantic AI agents with a user's ChatGPT plan through the public Responses API."
---

# Sign in with ChatGPT

The `openai-chatgpt` provider uses [Sign in with ChatGPT](https://developers.openai.com/siwc/token-sharing-open-source) to access the public Responses API with the signed-in user's ChatGPT plan. It is separate from the [OpenAI API-key provider](openai.md) and the [Codex CLI integration](openai-codex.md): it does not read their credentials or send Codex-specific headers.

!!! warning "Preview and eligibility"
    This is an opt-in integration with OpenAI's preview API. Check OpenAI's [eligibility and limitations](https://developers.openai.com/siwc/token-sharing-open-source/limitations) before enabling it. Configuring an OAuth client, including a client secret, does not itself authorize ChatGPT plan usage for a remotely hosted application.

## Install

Install the `openai-chatgpt` optional group, including when using the full `pydantic-ai` package:

```bash
pip/uv-add "pydantic-ai[openai-chatgpt]"
```

For a slim installation:

```bash
pip/uv-add "pydantic-ai-slim[openai-chatgpt]"
```

## Local sign-in

Choose and persist a stable host identifier before the first login. Reuse it for the same host, not for every sign-in attempt. The application name should be your app's actual name, consistent across installations.

[`OpenAIChatGPTOAuthFlow`][pydantic_ai.providers.openai_chatgpt.OpenAIChatGPTOAuthFlow] implements PKCE and verifies the ID token's signature, issuer, audience, expiry, and nonce. Its loopback callback uses `127.0.0.1`, not `localhost`.

```python {title="chatgpt_login.py" test="skip - requires browser login and ChatGPT plan authorization"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.openai_chatgpt import OpenAIChatGPTModel
from pydantic_ai.providers.openai_chatgpt import (
    OpenAIChatGPTOAuthFlow,
    OpenAIChatGPTProvider,
)


async def main():
    flow = OpenAIChatGPTOAuthFlow(
        ext_agent_host_id='urn:uuid:74a2cbb6-9561-46e9-9cd9-dc9535b0f2bb',
        agent_name='My Research Assistant',
    )
    # Open this URL in your system browser once the script is waiting below.
    print(flow.authorization_url())
    credentials = await flow.exchange_code_from_callback()

    provider = OpenAIChatGPTProvider(credentials)
    agent = Agent(OpenAIChatGPTModel('gpt-6.1-sol', provider=provider))
    async with agent:
        result = await agent.run('Where does "hello world" come from?')
        print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

The example keeps credentials in memory; see [Persisting credentials](#persisting-credentials) for reuse after restart. Pick a model slug from the selected account's [model catalog](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference), not from the Platform API-key model inventory.

First-time login starts with `dynamic_agent_client`. The callback returns an issued client ID, which is retained in [`OpenAIChatGPTCredentials`][pydantic_ai.providers.openai_chatgpt.OpenAIChatGPTCredentials]. For reauthorization, pass the saved credentials to the flow; it reuses the issued ID and account hints and checks that the validated account identity has not changed. Only the callback port may change between OSS sign-ins; keep its scheme, host, and path unchanged.

An application with its own redirect handler should call [`exchange_callback()`][pydantic_ai.providers.openai_chatgpt.OpenAIChatGPTOAuthFlow.exchange_callback] with the complete callback URL. Keep the pending flow server-side and associate it with the user's application session. Initial registration cannot use `exchange_code()` alone because it also needs the issued client ID. Each attempt expires after ten minutes and can exchange a callback only once.

## Persisting credentials

Keep each registration separate by its issued client ID, verified subject, and host ID. Store the complete credential record in protected application storage, including the callback URI, token expiry, granted scopes, and ID token for returning sign-in hints. Tokens are omitted from `repr`, not from serialization. Do not log callback URLs, authorization URLs containing account hints, token request bodies, or token responses. Avoid HTTP body capture in tracing unless you scrub these secrets.

The provider refreshes credentials shortly before expiry and makes one refresh/replay attempt on HTTP 401. A 401 is not proof of expiry; if the replay still fails, the error propagates. Refresh tokens rotate, so publishing the whole replacement record is part of refresh, not a later best-effort save.

For durable storage, implement [`OpenAIChatGPTCredentialSource`][pydantic_ai.providers.openai_chatgpt.OpenAIChatGPTCredentialSource]:

- `load()` returns the application's selected registration on first use.
- `rotate(expected, refresh)` must serialize all refreshes of that registration, including across provider instances and processes. Under that exclusion, reload the stored record. If it differs from `expected`, return it without refreshing; otherwise call `refresh`, atomically persist the complete result, then return it.
- The application owns locking, storage protection, atomic replacement, and recovery after a lost exchange response or persistence failure. Do not retry a possibly spent refresh token.

Pass the source as `OpenAIChatGPTProvider(credential_source=source)`, then pass that provider to `OpenAIChatGPTModel` as above. Bind a provider instance to one async event loop. Within an instance, concurrent requests share one refresh.

!!! warning "Recovering a failed rotation"
    After a dispatched refresh fails or is cancelled, the provider will not spend the same token again. Recover the selected registration through application storage, or sign in again, and create a new provider with the recovered credentials/source. Recreating a provider with a stale refresh token is not recovery. With memory-only credentials, a successful rotation is available as `provider.credentials` but is not saved automatically.

## Provisioned HTTPS clients

An OpenAI-provisioned client can use its exact registered HTTPS callback and configured token-endpoint authentication method:

```python {title="chatgpt_provisioned.py" test="skip - requires an OpenAI-provisioned client"}
import os

from pydantic_ai.providers.openai_chatgpt import (
    OpenAIChatGPTClient,
    OpenAIChatGPTOAuthFlow,
)

client = OpenAIChatGPTClient(
    client_id=os.environ['CHATGPT_CLIENT_ID'],
    redirect_uri='https://my-app.example/auth/chatgpt/callback',
    token_endpoint_auth_method='client_secret_basic',
    client_secret=os.environ['CHATGPT_CLIENT_SECRET'],
)
flow = OpenAIChatGPTOAuthFlow(
    ext_agent_host_id='my-persisted-host-id',
    agent_name='My Research Assistant',
    client=client,
)
# Redirect the browser to flow.authorization_url(). In your app's callback handler,
# await flow.exchange_callback(callback_url) for this user's pending flow.
```

Use `client_secret_post` only when provisioned for that method. Public clients use `none` and omit the secret. Keep confidential-client secrets on the backend. Pass the same `client` to `OpenAIChatGPTProvider` for refresh. HTTPS callbacks are handled by your application, not the built-in loopback listener. See OpenAI's [website integration guide](https://developers.openai.com/siwc/website) for provisioning; identity sign-in and authorization to spend a ChatGPT plan are distinct grants.

## Request behavior and limitations

- Both `agent.run()` and `agent.run_stream()` use SSE with `store=false`. Ordinary runs drain the stream transparently. Explicit `openai_store=True` is overridden; stored-response continuation is unavailable.
- Generic `max_tokens` (mapped to `max_output_tokens`), `temperature`, and `top_p` are omitted. Explicit `openai_*` settings use the normal Responses adapter and API errors report unsupported combinations.
- Local function/custom tools use a stable leading `additional_tools` input item. Web search is the supported normalized native tool; other native tools are rejected. Responses `tool_search` deferral is disabled; local tool discovery remains available.
- `count_tokens()` is unavailable under this authorization. Use the usage returned by inference.
- A run is successful only after `response.completed`. Failed, incomplete, explicit-error, or interrupted streams raise [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], including failures after output has started.
- This integration does not enable unsupported SIWC endpoints or modalities. Consult OpenAI's [current limitations](https://developers.openai.com/siwc/token-sharing-open-source/limitations) before using audio, video, files, or provider-native tools.
