"""Current subscription usage per account, for `/accounts`.

CLAI fetches usage for the sign-ins it owns: ChatGPT/Codex through the core provider's own client,
so stale tokens refresh and persist exactly as they do for model requests, and GitHub Copilot with
the saved GitHub login. A plugin account reports its own through `PluginLogin.usage`.
"""

from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, date, datetime, time
from typing import TYPE_CHECKING

from anyio import to_thread
from pydantic import BaseModel, ValidationError

from pydantic_ai.exceptions import UserError
from pydantic_clai2.models.accounts import Account
from pydantic_clai2.plugins import AccountUsage, PluginLogin, UsageWindow

if TYPE_CHECKING:
    import httpx2

    from pydantic_ai.providers.openai_codex import OpenAICodexProvider

UsageFetch = Callable[[], Awaitable[AccountUsage]]

CODEX_USAGE_URL = 'https://chatgpt.com/backend-api/wham/usage'
"""What the Codex CLI's `/status` reads; the core provider's auth signs requests to this host."""

COPILOT_USAGE_URL = 'https://api.github.com/copilot_internal/user'
"""What Copilot's editors read for the plan's quotas."""

TIMEOUT = 10
"""Seconds before a usage request gives up; the menu shows the account without usage."""


def usage_fetcher(
    item: Account,
    *,
    codex: Callable[[str], 'OpenAICodexProvider'],
    plugins: Mapping[str, PluginLogin],
) -> UsageFetch | None:
    """How to fetch `item`'s usage, or `None` when its provider does not report any."""
    if item.plugin_login is not None:
        plugin = plugins.get(item.plugin_login)
        usage = plugin.usage if plugin is not None else None
        if usage is None:
            return None
        profile = item.profile
        return lambda: usage(profile)
    if item.provider == 'openai-codex':
        return lambda: codex_usage(codex(item.login))
    if item.provider == 'github-copilot':
        return lambda: copilot_usage(item.login)
    return None


class _CodexWindow(BaseModel):
    used_percent: float
    limit_window_seconds: int
    reset_at: int | None = None


class _CodexLimit(BaseModel):
    primary_window: _CodexWindow | None = None
    secondary_window: _CodexWindow | None = None


class _CodexUsage(BaseModel):
    plan_type: str | None = None
    rate_limit: _CodexLimit | None = None


async def codex_usage(provider: 'OpenAICodexProvider') -> AccountUsage:
    """The ChatGPT plan's rolling windows, such as five hours and a week."""
    from openai import APIError

    client = provider.client.with_options(max_retries=0, timeout=TIMEOUT)
    try:
        body = await client.get(CODEX_USAGE_URL, cast_to=object)
    except APIError as exc:
        raise UserError(f'Codex usage unavailable: {exc.message}') from None
    try:
        usage = _CodexUsage.model_validate(body)
    except ValidationError:
        raise UserError('Codex returned usage CLAI cannot read.') from None
    limit = usage.rate_limit or _CodexLimit()
    windows = tuple(
        UsageWindow(
            label=window_label(window.limit_window_seconds),
            used_percent=window.used_percent,
            resets_at=None if window.reset_at is None else datetime.fromtimestamp(window.reset_at, UTC),
        )
        for window in (limit.primary_window, limit.secondary_window)
        if window is not None
    )
    return AccountUsage(windows=windows, plan=usage.plan_type)


def window_label(seconds: int) -> str:
    """`5h` for five hours, `7d` for a week: whole days when the window is a multiple of one."""
    hours = max(1, round(seconds / 3600))
    return f'{hours // 24}d' if hours % 24 == 0 else f'{hours}h'


class _CopilotQuota(BaseModel):
    percent_remaining: float
    unlimited: bool = False


class _CopilotUser(BaseModel):
    copilot_plan: str | None = None
    quota_reset_date_utc: datetime | None = None
    quota_reset_date: date | None = None
    quota_snapshots: dict[str, _CopilotQuota] = {}


_COPILOT_LABELS = {'premium_interactions': 'premium', 'chat': 'chat', 'completions': 'completions'}


async def copilot_usage(account: str, *, transport: 'httpx2.AsyncBaseTransport | None' = None) -> AccountUsage:
    """The Copilot plan's metered quotas, premium requests first; unlimited ones are left out."""
    import httpx2

    from pydantic_clai2.models import github_copilot

    token = await to_thread.run_sync(github_copilot.token, account)
    headers = {'Authorization': f'token {token}', 'Accept': 'application/json'}
    async with httpx2.AsyncClient(transport=transport, timeout=TIMEOUT, follow_redirects=False) as client:
        try:
            response = await client.get(COPILOT_USAGE_URL, headers=headers)
        except httpx2.HTTPError:
            raise UserError('Copilot usage unavailable: could not reach GitHub.') from None
    if response.status_code != 200:
        raise UserError(f'Copilot usage unavailable: GitHub answered {response.status_code}.')
    try:
        user = _CopilotUser.model_validate_json(response.content)
    except ValidationError:
        raise UserError('GitHub returned Copilot usage CLAI cannot read.') from None
    resets_at = user.quota_reset_date_utc or (
        None if user.quota_reset_date is None else datetime.combine(user.quota_reset_date, time(), UTC)
    )
    windows = tuple(
        UsageWindow(label=label, used_percent=100 - quota.percent_remaining, resets_at=resets_at)
        for key, label in _COPILOT_LABELS.items()
        if (quota := user.quota_snapshots.get(key)) is not None and not quota.unlimited
    )
    return AccountUsage(windows=windows, plan=user.copilot_plan)
