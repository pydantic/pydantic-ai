"""PyMongo handshake metadata for the `AsyncMongoClient`s the Mongo stores create from `db_url=`."""

from __future__ import annotations

from functools import cache
from importlib.metadata import PackageNotFoundError, version

from pymongo.driver_info import DriverInfo


@cache
def harness_driver_info() -> DriverInfo:
    """Name `pydantic-ai-harness` and its version in the connection handshake's `client.driver` metadata.

    PyMongo appends both to its own name and version, as MongoDB's handshake spec asks of wrapping libraries.
    """
    try:
        harness_version = version('pydantic-ai-harness')
    except PackageNotFoundError:  # pragma: no cover
        harness_version = 'unknown'
    return DriverInfo(name='pydantic-ai-harness', version=harness_version)
