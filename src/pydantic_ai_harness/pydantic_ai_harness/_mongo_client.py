"""Shared construction of the `AsyncMongoClient` the MongoDB-backed stores own.

When a store is given `db_url=` it creates and owns its client. That client identifies
this library to the server in the connection handshake (`client.driver` metadata, see the
[MongoDB handshake spec](https://github.com/mongodb/specifications/blob/master/source/mongodb-handshake/handshake.md)),
so MongoDB logs and Atlas show `PyMongo|pydantic-ai-harness`. A caller-supplied `client=` is
left untouched.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pymongo import AsyncMongoClient
from pymongo.driver_info import DriverInfo

_DRIVER_NAME = 'pydantic-ai-harness'


def _harness_version() -> str | None:
    try:
        return version(_DRIVER_NAME)
    except PackageNotFoundError:  # pragma: no cover
        return None


def create_async_mongo_client(db_url: str) -> AsyncMongoClient[dict[str, object]]:
    """Create an `AsyncMongoClient` for `db_url` that reports pydantic-ai-harness in the handshake."""
    return AsyncMongoClient[dict[str, object]](
        db_url, driver=DriverInfo(name=_DRIVER_NAME, version=_harness_version())
    )
