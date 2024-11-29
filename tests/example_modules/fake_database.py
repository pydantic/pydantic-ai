from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Any


class FakeTable:
    def get(self, name: str) -> int | None:
        if name == 'John Doe':
            return 123


@dataclass
class DatabaseConn:
    users: FakeTable = field(default_factory=FakeTable)
    _forecasts: dict[int, str] = field(default_factory=dict)

    async def execute(self, query: str) -> list[dict[str, Any]]:
        return [{'id': 123, 'name': 'John Doe'}]

    async def store_forecast(self, user_id: int, forecast: str) -> None:
        self._forecasts[user_id] = forecast

    async def get_forecast(self, user_id: int) -> str | None:
        return self._forecasts.get(user_id)


class QueryError(RuntimeError):
    pass
