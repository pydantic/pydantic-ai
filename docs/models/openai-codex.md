# OpenAI Codex

Pydantic AI can use models available through OpenAI Codex, which is included with a ChatGPT subscription. This is separate from the [OpenAI Platform API](openai.md): it uses ChatGPT-managed sign-in, subscription limits, and the Codex model catalog rather than `OPENAI_API_KEY` and Platform billing.

## Install

Install `pydantic-ai` or `pydantic-ai-slim` with the `openai-codex` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai-codex]"
```

Alongside the OpenAI client this pulls in `filelock`, which the default credential store uses to keep concurrent processes from rotating the same refresh token twice. An application that supplies its own [`OpenAICodexCredentialStore`][pydantic_ai.auth.openai_codex.OpenAICodexCredentialStore] never reaches that code.

## Sign in

The recommended setup is to sign in once with `clai`, then use the `openai-codex:` model prefix. The `clai` command comes from its own [`clai` package](../cli.md#installation), which carries these dependencies too:

```bash
clai auth login openai-codex
```

Browser login uses an authorization-code flow with PKCE and a short-lived loopback callback. For a remote or headless machine, use device authorization instead:

```bash
clai auth login openai-codex --method device
```

Device authorization may need to be enabled in your ChatGPT account or workspace settings.

After signing in, you can construct an agent without an API key:

```python
from pydantic_ai import Agent

agent = Agent('openai-codex:gpt-5.5')
result = agent.run_sync('Explain the most important risk in this patch.')
print(result.output)
#> The refresh token is written before the account check.
```

`gpt-5.5` is a model included in the pinned official Codex client used to verify this integration. Availability depends on your account and can change independently of Pydantic AI. Consult the [Codex model documentation](https://developers.openai.com/codex/models/) for current availability.

## Manage authentication

The auth commands use the same core lifecycle as [`OpenAICodexProvider`][pydantic_ai.providers.openai_codex.OpenAICodexProvider]:

```bash
clai auth status openai-codex
clai auth status openai-codex --json
clai auth refresh openai-codex
clai auth logout openai-codex
```

Logout attempts upstream token revocation and always removes the local record. Use `--local-only` to skip revocation:

```bash
clai auth logout openai-codex --local-only
```

Status output never includes tokens or the full ChatGPT account identifier.

## Use core authentication directly

Applications can use [`OpenAICodexAuth`][pydantic_ai.auth.openai_codex.OpenAICodexAuth] without importing CLI code. For example, an application can supply its own browser interaction and then pass the same credential source to the provider:

```python
import webbrowser

from pydantic_ai.auth.openai_codex import OpenAICodexAuth
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_codex import OpenAICodexProvider


async def create_model() -> OpenAIResponsesModel:
    openai_codex_auth = OpenAICodexAuth()
    await openai_codex_auth.login_browser(webbrowser.open)

    provider = OpenAICodexProvider(credential_source=openai_codex_auth)
    return OpenAIResponsesModel('gpt-5.5', provider=provider)
```

The returned [`OpenAICodexCredentials`][pydantic_ai.auth.openai_codex.OpenAICodexCredentials] object uses secret-redacted fields. Do not unwrap or display them outside narrowly scoped authentication and persistence code.

### Application-owned credentials and persistence

Multi-user services should not share the default local credential file. Instead, implement [`OpenAICodexCredentialSource`][pydantic_ai.auth.openai_codex.OpenAICodexCredentialSource] for request-time credentials, or implement [`OpenAICodexCredentialStore`][pydantic_ai.auth.openai_codex.OpenAICodexCredentialStore] and pass it to [`OpenAICodexAuth`][pydantic_ai.auth.openai_codex.OpenAICodexAuth].

A credential source must return one coherent access-token/account snapshot and honor `force_refresh=True` with `rejected_revision` for unauthorized recovery. If the current credential revision no longer matches the rejected revision, it should return the newer snapshot instead of rotating the refresh token again. A store must provide exclusive rotation ownership plus conditional replacement, so two workers cannot reuse the same rotating refresh token.

## Credential storage and security

By default, credentials are stored in plaintext at `~/.pydantic-ai/auth.json`. Pydantic AI:

- creates its default directory, or a missing custom parent directory, for the current user only where POSIX permissions are supported, without changing permissions on an existing custom parent;
- serializes refresh-token rotation with a process lock;
- writes complete records through atomic replacement;
- validates that refresh does not switch ChatGPT accounts.

Treat this file like a password: do not commit, paste, share, back it up to an untrusted location, or mount it into a multi-user service. Use an application-owned credential source or store when filesystem isolation is not appropriate.

## Differences from the OpenAI Platform API

The `openai-codex:` prefix selects [`OpenAICodexProvider`][pydantic_ai.providers.openai_codex.OpenAICodexProvider] with provider identity `openai-codex`. This keeps Codex response IDs, encrypted reasoning data, and message-history semantics separate from `openai:` requests.

The provider uses the existing [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel], but Codex requests use the Codex backend, ChatGPT bearer/account headers, and `store=False`. The Codex backend requires streaming responses, so ordinary `run()` and `run_sync()` calls stream internally and return the locally aggregated response. Codex subscription limits, billing, model names, feature availability, and deprecation schedules can differ from the OpenAI Platform API.

Four consequences are worth planning around:

- **Some generic settings are dropped.** The Codex backend answers `400 Unsupported parameter` for `max_tokens`, `temperature`, and `top_p`, so a [`ModelSettings`][pydantic_ai.settings.ModelSettings] carrying any of them has that setting silently omitted rather than failing every request. `openai_`-prefixed settings are passed through unchanged, so a backend rejection surfaces as an error there.
- **Responses are not resumable by id.** `store=False` is a backend requirement, so nothing is retained server-side. The `provider_response_id` on a Codex response cannot be used with `openai_previous_response_id`, background continuation, or retrieval by id; carry conversation state as [message history](../message-history.md) instead.
- **Token counting ahead of a request is unavailable.** The Codex backend does not serve the endpoint the OpenAI Platform API uses to count input tokens, so [`UsageLimits(count_tokens_before_request=True)`][pydantic_ai.usage.UsageLimits] raises a [`UserError`][pydantic_ai.exceptions.UserError] naming the limitation. Usage reported *after* a request is unaffected.
- **Reported cost is a Platform-equivalent estimate, not a bill.** `usage.cost` is priced at OpenAI Platform per-token rates, the only published price list for these models. A Codex subscription is charged as a flat fee against a quota, so the figure is what the same tokens *would* have cost on the Platform API, not an amount you are charged. [`UsageLimits(cost_limit=...)`][pydantic_ai.usage.UsageLimits] caps that estimate on the same terms.

## Not signed in

Every Codex request resolves credentials first, so a missing or unusable sign-in fails before anything reaches the backend, with a [`OpenAICodexLoginRequiredError`][pydantic_ai.auth.openai_codex.OpenAICodexLoginRequiredError] naming the command that fixes it:

```bash
clai auth login openai-codex
```

That error is raised as itself rather than reported as a connection failure, so it stays actionable at the boundary where you see it. Failures that *are* transport problems — an unreachable `auth.openai.com` during a token refresh, for example — keep surfacing as ordinary model API errors, which is what lets [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] route around them.

This integration provides model requests and authentication. It does not embed the Codex coding-agent harness, sandbox, repository editing, or local-shell execution behavior.
