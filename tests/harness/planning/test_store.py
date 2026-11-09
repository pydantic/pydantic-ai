import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

from pydantic_ai_harness.planning._store import SqlitePlanStore


def test_sqlite_plan_store_closes_connection_when_schema_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = Mock()
    connection.execute.side_effect = sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(sqlite3, 'connect', lambda *_args, **_kwargs: connection)

    store = SqlitePlanStore(database=str(tmp_path / 'plan.db'))
    with pytest.raises(sqlite3.OperationalError, match='database is locked'):
        store._connect()

    connection.close.assert_called_once_with()
