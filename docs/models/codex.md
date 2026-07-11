# Codex

Pydantic AI can use models available through a ChatGPT Codex subscription. This is separate from the [OpenAI Platform API](openai.md): it uses ChatGPT-managed sign-in, subscription limits, and the Codex model catalog rather than `OPENAI_API_KEY` and Platform billing.

## Sign in

The recommended setup is to sign in once with `clai`, then use the `codex:` model prefix:

```bash
clai auth login codex
```

Browser login uses an authorization-code flow with PKCE and a short-lived loopback callback. For a remote or headless machine, use device authorization instead:

```bash
clai auth login codex --method device
```

Device authorization may need to be enabled in your ChatGPT account or workspace settings.

After signing in, you can construct an agent without an API key:

```python {test="skip"}
from pydantic_ai import Agent

agent = Agent('codex:gpt-5.5')
result = agent.run_sync('Explain the most important risk in this patch.')
print(result.output)
```

`gpt-5.5` is a model included in the pinned official Codex client used to verify this integration. Availability depends on your account and can change independently of Pydantic AI. Consult the [Codex model documentation](https://developers.openai.com/codex/models/) for current availability.

## Manage authentication

The auth commands use the same core lifecycle as [`CodexProvider`][pydantic_ai.providers.codex.CodexProvider]:

```bash
clai auth status codex
clai auth status codex --json
clai auth refresh codex
clai auth logout codex
```

Logout attempts upstream token revocation and always removes the local record. Use `--local-only` to skip revocation:

```bash
clai auth logout codex --local-only
```

Status output never includes tokens or the full ChatGPT account identifier.

## Use core authentication directly

Applications can use [`CodexAuth`][pydantic_ai.auth.codex.CodexAuth] without importing CLI code. For example, an application can supply its own browser interaction and then pass the same credential source to the provider:

```python {test="skip"}
import webbrowser

from pydantic_ai.auth.codex import CodexAuth
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.codex import CodexProvider


async def create_model() -> OpenAIResponsesModel:
    codex_auth = CodexAuth()
    await codex_auth.login_browser(webbrowser.open)

    provider = CodexProvider(credential_source=codex_auth)
    return OpenAIResponsesModel('gpt-5.5', provider=provider)
```

The returned [`CodexCredentials`][pydantic_ai.auth.codex.CodexCredentials] object uses secret-redacted fields. Do not unwrap or display them outside narrowly scoped authentication and persistence code.

### Application-owned credentials and persistence

Multi-user services should not share the default local credential file. Instead, implement [`CodexCredentialSource`][pydantic_ai.auth.codex.CodexCredentialSource] for request-time credentials, or implement [`CodexCredentialStore`][pydantic_ai.auth.codex.CodexCredentialStore] and pass it to [`CodexAuth`][pydantic_ai.auth.codex.CodexAuth].

A credential source must return one coherent access-token/account snapshot and honor `force_refresh=True` with `rejected_revision` for unauthorized recovery. If the current credential revision no longer matches the rejected revision, it should return the newer snapshot instead of rotating the refresh token again. A store must provide exclusive rotation ownership plus conditional replacement, so two workers cannot reuse the same rotating refresh token.

## Credential storage and security

By default, credentials are stored in plaintext at `~/.pydantic-ai/auth.json`. Pydantic AI:

- creates its default directory, or a missing custom parent directory, for the current user only where POSIX permissions are supported, without changing permissions on an existing custom parent;
- serializes refresh-token rotation with a process lock;
- writes complete records through atomic replacement;
- validates that refresh does not switch ChatGPT accounts.

Treat this file like a password: do not commit, paste, share, back it up to an untrusted location, or mount it into a multi-user service. Use an application-owned credential source or store when filesystem isolation is not appropriate.

## Codex and OpenAI Platform differences

The `codex:` prefix selects [`CodexProvider`][pydantic_ai.providers.codex.CodexProvider] with provider identity `codex`. This keeps Codex response IDs, encrypted reasoning data, and message-history semantics separate from `openai:` requests.

The provider uses the existing [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel], but Codex requests use the Codex backend, ChatGPT bearer/account headers, and `store=False`. The Codex backend requires streaming responses, so ordinary `run()` and `run_sync()` calls stream internally and return the locally aggregated response. Codex subscription limits, billing, model names, feature availability, and deprecation schedules can differ from the OpenAI Platform API.

This integration provides model requests and authentication. It does not embed the Codex coding-agent harness, sandbox, repository editing, or local-shell execution behavior.
