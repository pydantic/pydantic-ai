"""MongoDB handshake metadata shared by the stores that own an `AsyncMongoClient`.

A store given `db_url=` creates its own client and passes `harness_driver_info()` so the server sees
`PyMongo|...|pydantic-ai-harness` in the connection handshake's `client.driver` metadata (see the
[handshake spec](https://github.com/mongodb/specifications/blob/master/source/mongodb-handshake/handshake.md)).
A caller-supplied `client=` is left alone.
"""

from __future__ import annotations

from functools import cache
from importlib.metadata import PackageNotFoundError, version

from pymongo.driver_info import DriverInfo

_NAME = 'pydantic-ai-harness'


@cache
def harness_driver_info() -> DriverInfo:
    try:
        return DriverInfo(name=_NAME, version=version(_NAME))
    except PackageNotFoundError:  # pragma: no cover
        return DriverInfo(name=_NAME)
