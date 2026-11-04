---
description: "Let the users of your open-source or local app run it on their own ChatGPT plan instead of an API key, through OpenAI's Sign in with ChatGPT."
---

# OpenAI ChatGPT Plan Usage

[Sign in with ChatGPT](https://developers.openai.com/siwc/token-sharing-open-source) lets the users of your open-source or locally run app use their ChatGPT plan for its requests instead of an API key. The `openai-chatgpt` provider runs the sign-in, sends Responses API requests with the signed-in user's token, and refreshes it as it expires. Offering plan usage in a paid or remotely hosted app needs OpenAI's approval first, through its [interest form](https://openai.com/form/sign-in-with-chatgpt-interest/). Your use is governed by OpenAI's [terms and policies](https://openai.com/policies/).

This is different from the [`openai-codex` provider](openai-codex.md), which signs in as the Codex CLI to use your own subscription against the Codex backend. With Sign in with ChatGPT, each user approves your app by name, requests go to the public OpenAI API, and users can see and limit your app's usage in [ChatGPT settings](https://chatgpt.com/settings/usage).

## Install

To use the ChatGPT provider, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Signing in

[`OpenAIChatGPTOAuthFlow`][pydantic_ai.providers.openai_chatgpt.OpenAIChatGPTOAuthFlow] runs the browser sign-in. OpenAI requires a callback on `http://127.0.0.1:PORT/auth/callback`, so [`exchange_code_from_callback()`][pydantic_ai.providers.openai_chatgpt.OpenAIChatGPTOAuthFlow.exchange_code_from_callback] listens on that port (1455 unless you pass another `redirect_uri`) until the browser redirects there, then exchanges the code for credentials:

```python {title="chatgpt_sign_in.py" test="skip - opens a browser and requires user login"}
import uuid
import webbrowser

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_chatgpt import OpenAIChatGPTOAuthFlow, OpenAIChatGPTProvider

HOST_ID = uuid.uuid4().urn  # create once per installation and persist it


async def main():
    flow = OpenAIChatGPTOAuthFlow(app_name='My App', host_id=HOST_ID)
    webbrowser.open(flow.authorization_url())
    credentials = await flow.exchange_code_from_callback()

    provider = OpenAIChatGPTProvider(credentials=credentials)
    agent = Agent(OpenAIResponsesModel('gpt-6.1-sol', provider=provider))
    result = await agent.run('Where does "hello world" come from?')
    print(result.output)
```

The provider needs your app's credentials, so build the model with it as shown, rather than passing an `'openai-chatgpt:...'` model name string to the agent.

`app_name` is the name the user sees when approving your app. Use the same name on every installation. `host_id` is an opaque identifier for one installation of your app, such as a laptop or a VM; create it once and reuse it for every sign-in on that installation. OpenAI asks apps to send it, but it's optional here because sign-ins that send it have been reported to time out; leave it out if yours do.

The first sign-in to a ChatGPT account registers your app with it, and OpenAI issues the credentials' `client_id` for that registration. The user can decline to let your app use their plan: the sign-in then still returns credentials, so you can save the registration, but [`plan_usage_granted`][pydantic_ai.providers.openai_chatgpt.OpenAIChatGPTCredentials.plan_usage_granted] is `False` and the provider refuses them.

## Persisting credentials

Passing `credentials` keeps them in memory only, so the next process has to sign in again. The provider refreshes expired tokens automatically, and refresh tokens are single-use, so the stored copy has to keep up. Give the provider an [`OpenAIChatGPTCredentialSource`][pydantic_ai.providers.openai_chatgpt.OpenAIChatGPTCredentialSource] and it calls `load()` on first use and `save()` after every refresh:

```python {title="chatgpt_persist.py" test="skip - opens a browser and requires user login"}
import json
import uuid
import webbrowser
from dataclasses import asdict
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_chatgpt import (
    OpenAIChatGPTCredentials,
    OpenAIChatGPTCredentialSource,
    OpenAIChatGPTOAuthFlow,
    OpenAIChatGPTProvider,
)


class FileCredentialSource(OpenAIChatGPTCredentialSource):
    def __init__(self, path: Path):
        self.path = path

    async def load(self) -> OpenAIChatGPTCredentials:
        return OpenAIChatGPTCredentials(**json.loads(self.path.read_text()))

    async def save(self, credentials: OpenAIChatGPTCredentials) -> None:
        self.path.write_text(json.dumps(asdict(credentials)))


async def main():
    host_id_file = Path('chatgpt-host-id')
    if not host_id_file.exists():
        host_id_file.write_text(uuid.uuid4().urn, encoding='utf-8')
    source = FileCredentialSource(Path('chatgpt-credentials.json'))
    if not source.path.exists():
        flow = OpenAIChatGPTOAuthFlow(app_name='My App', host_id=host_id_file.read_text(encoding='utf-8'))
        webbrowser.open(flow.authorization_url())
        await source.save(await flow.exchange_code_from_callback())

    provider = OpenAIChatGPTProvider(credential_source=source)
    agent = Agent(OpenAIResponsesModel('gpt-6.1-sol', provider=provider))
    result = await agent.run('Where does "hello world" come from?')
    print(result.output)
```

!!! tip "Protect the credentials file"
    The file holds a refresh token that grants access to the user's ChatGPT plan, so treat it like any other secret: keep it out of source control and logs, and restrict it to the current user, for example with `self.path.chmod(0o600)` after writing.

If `save()` raises, the refreshed credentials stay live in memory and a [`CredentialsPersistenceError`][pydantic_ai.providers.openai_chatgpt.CredentialsPersistenceError] is raised. A rejected refresh raises [`CredentialsRefreshError`][pydantic_ai.providers.openai_chatgpt.CredentialsRefreshError]; when its message says the grant was rejected, the user has to sign in again. Both subclass [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] treats an unusable sign-in like any other provider failure.

## Signing in again

To sign the same ChatGPT account in again, for example after its refresh token expired, pass the saved `client_id` so the sign-in reuses the registration instead of creating another one, and the saved `id_token` so the user doesn't have to pick the account:

```python {title="chatgpt_sign_in_again.py" test="skip - opens a browser and requires user login"}
import webbrowser

from pydantic_ai.providers.openai_chatgpt import OpenAIChatGPTCredentials, OpenAIChatGPTOAuthFlow


async def sign_in_again(saved: OpenAIChatGPTCredentials, host_id: str) -> OpenAIChatGPTCredentials:
    flow = OpenAIChatGPTOAuthFlow(
        app_name='My App', host_id=host_id, client_id=saved.client_id, id_token_hint=saved.id_token
    )
    # `prompt=consent` asks a user who declined plan usage again.
    extra_params = None if saved.plan_usage_granted else {'prompt': 'consent'}
    webbrowser.open(flow.authorization_url(extra_params=extra_params))
    return await flow.exchange_code_from_callback()
```

With `id_token_hint`, the sign-in fails if the user completes it with a different ChatGPT account. The authorization URL then carries the ID token, so keep it out of logs.

A provider checks plan usage before every request and keeps refusing credentials without it, so after a new sign-in, give the new credentials to a new provider.

## Errors

A sign-in the token endpoint rejects, or whose ID token fails validation, raises [`CredentialsRefreshError`][pydantic_ai.providers.openai_chatgpt.CredentialsRefreshError]. Start the sign-in again with a new flow, passing `client_id=flow.client_id`: the failed flow keeps the client id issued for it, so the retry reuses that registration instead of creating another. A sign-in that completes for a different account than `id_token_hint` names raises [`UserError`][pydantic_ai.exceptions.UserError], and its credentials are discarded.

When the user's plan can't serve a request, the request fails with a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] whose body carries an error code such as `subscription_sharing_usage_limit_exceeded`. OpenAI never moves the request to another billing method. [Errors and recovery](https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery) lists the codes and what to tell the user; for usage limits, point them to [ChatGPT settings](https://chatgpt.com/settings/usage).

## Limitations

ChatGPT plan usage is a preview with the [limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations) OpenAI documents:

- Requests are streamed; for non-streaming runs the library transparently drains a stream, so `agent.run_sync()` and friends work as usual.
- Every request is sent with `store=false`, and an explicit `openai_store=True` is silently overridden: responses are never persisted server-side. Consequently, resuming a suspended run raises [`UserError`][pydantic_ai.exceptions.UserError].
- System prompts and instructions are sent as developer messages, since system messages are rejected.
- Unsupported generic settings (`max_tokens`, `service_tier`, `temperature`, and `top_p`) are dropped before sending. Explicit `openai_*` settings are sent unchanged, so ones the route rejects, such as `openai_truncation` and `openai_user`, surface as errors.
- Of the [native tools](../native-tools.md), only [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] is supported. [Tool search](../tools-advanced.md#tool-search) runs locally instead of through OpenAI's native tool search.
- The prompt caching settings have no effect, and `count_tokens()` raises [`UserError`][pydantic_ai.exceptions.UserError].
- The callback reaches the computer running the browser, so for an app on a remote VM, sign in locally and copy the credentials to it, as described in [Self-hosted VMs](https://developers.openai.com/siwc/token-sharing-open-source/self-hosted-vms).
