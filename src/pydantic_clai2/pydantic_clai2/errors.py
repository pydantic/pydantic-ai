"""Actionable terminal messages without changing exceptions delivered to plugins."""

import re
from typing import cast

_DETAIL = 200
"""Characters of each failed model's error to show; provider error bodies can be long."""


_POLICY = re.compile(
    r'Spending policy(?: `(?P<policy>[^`]+)`)? (?P<window>daily|weekly|monthly|total) limit'
    r'(?: of \$(?P<limit>[0-9]+(?:\.[0-9]+)?))?(?: for model `(?P<model>[^`]+)`)? exhausted'
)
_SCOPED = re.compile(r'\b(?P<scope>User|Project|Organization|Session) limit exceeded')
_KEY = 'Spending limit exceeded'
_WINDOWS = {'daily': 'today', 'weekly': 'this week', 'monthly': 'this month'}
_SCOPES = {'User': 'your', 'Project': 'the project', 'Organization': 'the organization', 'Session': "this sign-in's"}
_ADMIN = 'Ask your Logfire admin.'


def budget_message(text: str) -> str | None:
    """A Pydantic AI Gateway spend-limit rejection, said plainly; `None` for any other error.

    The gateway answers with plain text: `Spending policy `NAME` monthly limit of $50 exhausted` (429) for a
    spending policy, `User limit exceeded` and its project, organization and session siblings (429), or
    `Forbidden - Spending limit exceeded` (403) once a key is blocked. It says the limit, never the amount spent.
    """
    if (policy := _POLICY.search(text)) is not None:
        window = policy['window']
        when = f' for {_WINDOWS[window]}' if window in _WINDOWS else ''
        details = [
            *([f'limit ${policy["limit"]}'] if policy['limit'] else []),
            *([f'for {policy["model"]}'] if policy['model'] else []),
            *([f'policy {policy["policy"]}'] if policy['policy'] else []),
        ]
        detail = f' ({", ".join(details)})' if details else ''
        total = ' total' if window == 'total' else ''
        return f"Your organization's{total} AI budget{when} is used up{detail}. {_ADMIN}"
    if (scoped := _SCOPED.search(text)) is not None:
        whose = _SCOPES[scoped['scope']]
        return f"Your organization's AI budget is used up ({whose} spending limit). {_ADMIN}"
    if _KEY in text:
        return f"Your organization's AI budget is used up. {_ADMIN}"
    return None


def error_message(error: BaseException) -> str:
    """Recognize Codex refresh failures even when the SDK wraps them as connection errors.

    A fallback chain or `PROVIDER@*` that failed on every model lists why each one failed, in order.
    """
    from pydantic_ai.providers.openai_codex import CredentialsRefreshError

    if isinstance(error, BaseExceptionGroup):
        group = cast('BaseExceptionGroup[BaseException]', error)
        lines = [str(group).splitlines()[0]]
        for number, failure in enumerate(group.exceptions, start=1):
            detail = ' '.join(error_message(failure).split())
            if len(detail) > _DETAIL:
                detail = detail[: _DETAIL - 3] + '...'
            lines.append(f'  {number}. {type(failure).__name__}: {detail}')
        return '\n'.join(lines)
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (budget := budget_message(str(current))) is not None:
            return budget
        if isinstance(current, CredentialsRefreshError):
            return (
                # The error does not say which account failed, so point to every way back in.
                'Could not refresh your Codex login. Sign in again with /login openai-codex, or press Enter on '
                'the account in /accounts, then retry your message.'
            )
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return str(error)
