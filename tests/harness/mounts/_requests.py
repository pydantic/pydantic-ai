"""The JSON request the mount helper reads on stdin, typed, for the tests that inspect what a sandbox was sent."""

from __future__ import annotations

from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import TypedDict

# Unknown keys fail too, so a field added to the request shows up in the tests that pin it.
_STRICT = ConfigDict(extra='forbid')


@with_config(_STRICT)
class Keys(TypedDict):
    access_key_id: str
    secret_access_key: str
    session_token: str


@with_config(_STRICT)
class Wanted(TypedDict):
    """One mount the helper is asked to make or refresh."""

    path: str
    fingerprint: str
    remote: dict[str, str]
    root: str
    read_only: bool
    keys: Keys
    expires_at: float


@with_config(_STRICT)
class Request(TypedDict):
    working_dir: str
    mounts: list[Wanted]


_REQUEST = TypeAdapter(Request)


def parse_request(data: str | bytes) -> Request:
    """Validate a request's JSON, so a renamed or retyped field fails the test that reads it."""
    return _REQUEST.validate_json(data)
