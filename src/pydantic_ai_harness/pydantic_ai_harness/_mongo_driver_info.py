"""PyMongo handshake metadata for the `AsyncMongoClient`s the Mongo stores create from `db_url=`."""

from __future__ import annotations

from pymongo.driver_info import DriverInfo

from pydantic_ai_harness._version import HARNESS_DISTRIBUTION, harness_version


def harness_driver_info() -> DriverInfo:
    """Name `pydantic-ai-harness` and its version in the connection handshake's `client.driver` metadata.

    PyMongo appends both to its own name and version, as MongoDB's handshake spec asks of wrapping libraries.
    """
    return DriverInfo(name=HARNESS_DISTRIBUTION, version=harness_version())
