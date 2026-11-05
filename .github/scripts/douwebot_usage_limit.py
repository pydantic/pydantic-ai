"""Recognize explicit Claude subscription quota rejections in SDK execution files.

The pinned SDK emits rate_limit_event messages for Claude subscription users.
Only rejected five-hour or seven-day limits trigger the fallback; missing or
unknown signals fail closed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

_MESSAGES: TypeAdapter[list[dict[str, object]]] = TypeAdapter(list[dict[str, object]])
_OBJECT: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
_SUBSCRIPTION_WINDOWS = {'five_hour', 'seven_day', 'seven_day_opus', 'seven_day_sonnet'}


def is_usage_limit(payload: object) -> bool:
    """Return whether the last rate-limit event rejects a subscription window."""
    try:
        messages = _MESSAGES.validate_python(payload, strict=True)
    except ValidationError:
        return False
    events = [message for message in messages if message.get('type') == 'rate_limit_event']
    if not events:
        return False
    try:
        info = _OBJECT.validate_python(events[-1].get('rate_limit_info'), strict=True)
    except ValidationError:
        return False
    if info.get('status') != 'rejected':
        return False
    window = info.get('rateLimitType')
    return isinstance(window, str) and window in _SUBSCRIPTION_WINDOWS


def is_usage_limit_file(path: str) -> bool:
    """Return whether the execution file at `path` is a subscription-limit rejection."""
    if not path:
        return False
    try:
        text = Path(path).read_text(encoding='utf-8')
    except OSError:
        return False
    try:
        payload: object = json.loads(text)
    except json.JSONDecodeError:
        return False
    return is_usage_limit(payload)


def main() -> None:
    """Write `fallback=true` only for a subscription-limit execution file."""
    if not is_usage_limit_file(os.environ.get('EXECUTION_FILE', '')):
        print(
            '::error::Claude Code failed without the subscription-limit signature; not falling back to OpenCode.',
            file=sys.stderr,
        )
        raise SystemExit(1)
    output = os.environ.get('GITHUB_OUTPUT')
    if not output:
        print('::error::GITHUB_OUTPUT is not set.', file=sys.stderr)
        raise SystemExit(1)
    with Path(output).open('a', encoding='utf-8') as handle:
        handle.write('fallback=true\n')


if __name__ == '__main__':
    main()
