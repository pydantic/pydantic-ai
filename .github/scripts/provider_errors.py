"""Every CI agent runner's classes of provider failure, and the provider errors in each.

The OpenCode chain in `douwebot_opencode_run.py` reads OpenCode's error type, and the gh-aw shim reads Z.AI's error
code. Both decide by `ErrorClass` whether a run can move to another provider or account.

It lives outside the shim package, whose `__init__` imports `pydantic_ai`, which the OpenCode runners' `python3` lacks,
and imports only the standard library. `stage-gh-aw-agent.sh` copies it beside the staged shim.
"""

from __future__ import annotations

from typing import Literal

ErrorClass = Literal['plan_limit', 'go_limit', 'free_limit', 'rate_limit', 'other']
"""`plan_limit`: an account or plan quota, which cannot clear within a run. `go_limit`: a spent OpenCode Go usage
window. `free_limit`: a spent OpenCode Zen free allowance. `rate_limit`: a request-rate limit or overload, which can
clear within a run. `other`: any other failure."""

# OpenCode 2.2.5 types a spent OpenCode Go usage window `provider.go-limit`, and a quota it cannot name
# `provider.quota`. It also types any 402, and a Z.AI 429 whose message names a plan, quota or balance limit, such as
# codes 1308, 1309 and 1310, `provider.quota`. It retries any other 429, such as 1302 rate limits and 1305 overload,
# and types one that outlasts its retries `provider.rate-limit`: `classify` in
# https://github.com/anomalyco/opencode/blob/v2.2.5/packages/ai/src/provider-error.ts
# Zen reports a spent free allowance as `FreeUsageLimitError`, which OpenCode 2.2.5 types `provider.free-limit`:
# `toSessionError` in https://github.com/anomalyco/opencode/blob/v2.2.5/packages/core/src/session/to-session-error.ts
OPENCODE_ERROR_CLASSES: dict[str, ErrorClass] = {
    'provider.quota': 'plan_limit',
    'provider.go-limit': 'go_limit',
    'provider.free-limit': 'free_limit',
    'provider.rate-limit': 'rate_limit',
}

# Z.AI 429 codes for account and plan limits, which cannot clear inside the shim's retry window. Other 429s, such as
# `1302` request-rate limits and `1305` overload, are transient.
# Keep these codes aligned with https://docs.z.ai/api-reference/api-code.
ZAI_PLAN_LIMIT_CODES = frozenset(
    {'1113', '1308', '1309', '1310', '1311', '1313', '1314', '1315', '1316', '1317', '1318', '1319', '1320', '1321'}
)


def opencode_error_class(error_type: str | None) -> ErrorClass:
    """The class of the OpenCode error type `error_type`."""
    return OPENCODE_ERROR_CLASSES.get(error_type or '', 'other')


def zai_rate_limit_class(code: str | None) -> ErrorClass:
    """The class of a Z.AI 429 whose body carries error code `code`."""
    return 'plan_limit' if code in ZAI_PLAN_LIMIT_CODES else 'rate_limit'
