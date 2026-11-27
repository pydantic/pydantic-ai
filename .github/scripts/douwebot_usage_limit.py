"""Recognize explicit Claude subscription quota rejections in SDK execution files.

The pinned SDK emits rate_limit_event messages for Claude subscription users.
Only rejected five-hour or seven-day limits trigger the fallback; missing or
unknown signals fail closed. The no-signature failure annotation names the
API-error fields the pinned SDK recorded on the file's result message, when
it recorded any.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TypeGuard

_SUBSCRIPTION_WINDOWS = {'five_hour', 'seven_day', 'seven_day_opus', 'seven_day_sonnet'}
_API_ERROR_FIELDS = ('api_error_status', 'api_error', 'terminal_reason')


def _is_container(value: object) -> TypeGuard[list[object] | dict[object, object]]:
    return isinstance(value, (list, dict))


def is_usage_limit(payload: object) -> bool:
    """Return whether the last rate-limit event rejects a subscription window."""
    if not _is_container(payload) or not isinstance(payload, list):
        return False
    events: list[dict[object, object]] = []
    for message in payload:
        if not _is_container(message) or not isinstance(message, dict):
            return False
        if not all(isinstance(key, str) for key in message):
            return False
        if message.get('type') == 'rate_limit_event':
            events.append(message)
    if not events:
        return False
    info = events[-1].get('rate_limit_info')
    if not _is_container(info) or not isinstance(info, dict):
        return False
    if not all(isinstance(key, str) for key in info):
        return False
    if info.get('status') != 'rejected':
        return False
    window = info.get('rateLimitType')
    return isinstance(window, str) and window in _SUBSCRIPTION_WINDOWS


def _read_execution_file(path: str) -> object:
    """Return the parsed execution file, or `None` when it is missing, unreadable, or unparsable."""
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding='utf-8')
    except OSError:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def is_usage_limit_file(path: str) -> bool:
    """Return whether the execution file at `path` is a subscription-limit rejection."""
    return is_usage_limit(_read_execution_file(path))


def _annotationize(text: str) -> str:
    """Escape `text` for a single-line GitHub Actions error annotation.

    Annotations must stay on one line: line breaks become `%0A` and percent
    signs become `%25`, matching the workflow command's percent-escaping rules.
    """
    return text.replace('%', '%25').replace('\r', '').replace('\n', '%0A')


def _stringify(value: object) -> str:
    """Render one API-error diagnostic value as annotation text."""
    if isinstance(value, str):
        return value
    if value is True:
        return 'true'
    if value is False:
        return 'false'
    if value is None:
        return 'null'
    return json.dumps(value)


def _no_signature_annotation(payload: object) -> str:
    """Build the no-signature error annotation, including API details when present.

    The SDK result message records API errors in `api_error_status`, `api_error`,
    and `terminal_reason`. It does not always record them: the failure that
    motivated this script surfaced no diagnostic fields, so that shape reports
    the always-present result fields instead of an API error that was never
    captured.
    """
    result: dict[object, object] = {}
    if _is_container(payload) and isinstance(payload, list):
        for message in payload:
            if _is_container(message) and isinstance(message, dict) and message.get('type') == 'result':
                result = message
    parts = ['Claude Code failed without the subscription-limit signature; not falling back to OpenCode.']
    present = [field for field in _API_ERROR_FIELDS if field in result]
    if present:
        parts.extend(f'{field}={_stringify(result[field])}' for field in present)
    else:
        parts.append('session failed; no API-error fields were recorded in the execution file')
        parts.extend(f'{field}={_stringify(result[field])}' for field in ('is_error', 'num_turns') if field in result)
    return '::error::' + _annotationize(' '.join(parts))


def main() -> None:
    """Write `fallback=true` only for a subscription-limit execution file."""
    execution_file = os.environ.get('EXECUTION_FILE', '')
    if not is_usage_limit_file(execution_file):
        print(_no_signature_annotation(_read_execution_file(execution_file)), file=sys.stderr)
        raise SystemExit(1)
    output = os.environ.get('GITHUB_OUTPUT')
    if not output:
        print('::error::GITHUB_OUTPUT is not set.', file=sys.stderr)
        raise SystemExit(1)
    with Path(output).open('a', encoding='utf-8') as handle:
        handle.write('fallback=true\n')


if __name__ == '__main__':
    main()
