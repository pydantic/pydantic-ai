import re
from datetime import datetime
from typing import Any

import pytest
from dirty_equals import IsDatetime, IsInstance, IsStr
from inline_snapshot.plugin import customize

from pydantic_ai.usage import RequestUsage
from tests import cassette_hooks


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


_IP_LITERAL_HOST = re.compile(r'://(?:(?:\d{1,3}\.){3}\d{1,3}|\[[0-9a-fA-F:]+\])(?::\d+)?/')


def _normalize_resolved_host(uri: str) -> str:
    # `safe_download` connects to a resolved IPv4 or IPv6 address, which may differ
    # between recording and replay for the same URL.
    return _IP_LITERAL_HOST.sub('://RESOLVED_IP/', cassette_hooks.normalize_uri(uri))


@pytest.fixture(scope='module')
def vcr_config(vcr_config: dict[str, Any]) -> dict[str, Any]:
    return {**vcr_config, 'uri_normalizer': _normalize_resolved_host}
