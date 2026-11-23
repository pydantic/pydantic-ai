"""The file, SQLite and in-memory step stores load back a drawn history that dumps to the bytes the saved one did.

The Postgres and Mongo stores are left out: their tests run on stand-ins (a SQLite pool, `mongomock-motor`), and
`test_dict_key_order_round_trips` in `test_postgres.py` and `test_mongo.py` pins key order on one hand-built history.
The histories come from `tests/history_strategies.py`, and the property compares
`ModelMessagesTypeAdapter.dump_json` of the history saved and of the history loaded back. Equal history bytes mean
every model maps the two to the same request body, so the property needs no model mappers, unlike the UI round trip
in `tests/test_ui_history_round_trip.py`. The bytes include dict key order, which a store keeping message JSON as text
has to keep for the next request to hit the provider's prompt cache
(https://github.com/pydantic/pydantic-ai/pull/11137).

The file and SQLite stores externalize every payload the media store takes, so the property covers the
externalized path as well as the inline one, which every dict and list in a history still takes.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import given

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    InMemoryStepStore,
    SqliteStepStore,
    StepStore,
)

from ...history_strategies import IGNORE_RANDOM_STATE_NOTE, PROPERTY_SETTINGS, histories

pytestmark = IGNORE_RANDOM_STATE_NOTE


@dataclass(frozen=True)
class StoreCase:
    id: str
    build: Callable[[Path], StepStore]
    """The store, keeping whatever it writes under the directory it is given."""


STORE_CASES = [
    StoreCase('file', lambda directory: FileStepStore(directory, media_threshold_bytes=1)),
    StoreCase('sqlite', lambda directory: SqliteStepStore(database=directory / 'runs.db', media_threshold_bytes=1)),
    StoreCase('in-memory', lambda _: InMemoryStepStore()),
]


async def _save_and_load(store: StepStore, history: list[ModelMessage]) -> list[ModelMessage]:
    await store.save_snapshot(ContinuableSnapshot(run_id='run', step_index=0, messages=history))
    snapshot = await store.latest_snapshot(run_id='run')
    assert snapshot is not None
    return snapshot.messages


class TestStepStoreRoundTrip:
    # Sync, with a fresh directory per example: Hypothesis runs every example in one call of the test, so a
    # function-scoped `tmp_path` would be shared by all of them, and an async test breaks on the `BaseException` it
    # ends a draw with.
    @pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in STORE_CASES])
    @PROPERTY_SETTINGS
    @given(history=histories())
    def test_loaded_history_dumps_to_the_saved_bytes(self, case: StoreCase, history: list[ModelMessage]) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded = asyncio.run(_save_and_load(case.build(Path(directory)), history))
        assert ModelMessagesTypeAdapter.dump_json(loaded) == ModelMessagesTypeAdapter.dump_json(history)
