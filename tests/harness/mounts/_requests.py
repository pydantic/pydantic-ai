"""The JSON request the mount helper reads on stdin, as the tests that inspect what a sandbox was sent read it."""

from __future__ import annotations

import json

from pydantic_ai_harness._mounts._helper import Keys, Request, Wanted

__all__ = ('Keys', 'Request', 'Wanted', 'parse_request')


def parse_request(data: str | bytes) -> Request:
    # The helper's own types: Pydantic can't validate them on Python 3.11, where it needs `typing_extensions`'
    # `TypedDict`, but the helper uses the standard library's. The tests snapshot the whole request instead.
    request: Request = json.loads(data)
    return request
