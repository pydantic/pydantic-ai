import re
from datetime import datetime
from typing import Any

import pytest
from dirty_equals import IsDatetime, IsInstance, IsStr
from inline_snapshot.plugin import customize
from pydantic_ai.usage import RequestUsage


class InlineSnapshotPlugin:
    @customize(tryfirst=True)
    def nondeterministic_values(self, value: object, builder: Any) -> Any:  # pragma: no cover
        if isinstance(value, datetime):
            return builder.create_call(IsDatetime)
        if isinstance(value, RequestUsage):
            return builder.create_call(IsInstance, [RequestUsage])
        if isinstance(value, str) and re.fullmatch(r'01[a-z0-9]{6}(?:-[a-z0-9]{4}){3}-[a-z0-9]{12}', value):
            return builder.create_call(IsStr)
        if isinstance(value, str) and re.search(r'01[a-z0-9]{6}-', value):
            normalized = re.sub(r'01[a-z0-9]{6}(?:-[a-z0-9]{4}){3}-[a-z0-9]{12}', 'RUN_ID', value)
            pattern = re.escape(normalized).replace('RUN_ID', r'01[a-z0-9]{6}(?:\-[a-z0-9]{4}){3}\-[a-z0-9]{12}')
            return builder.create_call(IsStr, [], {'regex': pattern})


@pytest.fixture(scope='module')
def vcr_config() -> dict[str, Any]:
    return {
        'filter_headers': [
            ('authorization', 'REDACTED'),
            ('x-api-key', 'REDACTED'),
        ],
        # `safe_download` connects to a resolved IP address, which may differ
        # between recording and replay for the same URL.
        'match_on': ['method', 'path', 'query'],
    }


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'
