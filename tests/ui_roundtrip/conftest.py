"""Runs the real JavaScript UI clients for the round-trip tests in this directory.

`driver.mjs` wraps the AI SDK's `AbstractChat` and `@ag-ui/client`'s `AbstractAgent`, pinned in
`package.json`. Install them with `npm ci --ignore-scripts` in this directory. Without `node` or the
installed packages, tests marked `ui_roundtrip` skip, unless `UI_ROUNDTRIP_REQUIRED` is set, as the
`UI round trip` CI job does, which makes them fail instead.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).parent
MARKER = 'ui_roundtrip'


def _unavailable_reason() -> str | None:
    if shutil.which('node') is None:
        return '`node` is not installed'
    if not (HERE / 'node_modules').is_dir():
        return f'run `npm ci --ignore-scripts` in {HERE} to install the JavaScript UI clients'
    return None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line('markers', f'{MARKER}: runs the real JavaScript UI clients in tests/ui_roundtrip')


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if (reason := _unavailable_reason()) is None:
        return
    if os.environ.get('UI_ROUNDTRIP_REQUIRED'):
        raise pytest.UsageError(f'UI_ROUNDTRIP_REQUIRED is set, but {reason}')
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker(MARKER) is not None:
            item.add_marker(skip)


class JSDriver:
    """A `node driver.mjs` process, sent one JSON request at a time."""

    def __init__(self) -> None:
        self._process = subprocess.Popen(
            ['node', str(HERE / 'driver.mjs')],
            cwd=HERE,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding='utf-8',
        )
        self._next_id = 0

    def request(self, op: str, **params: Any) -> Any:
        assert self._process.stdin is not None and self._process.stdout is not None
        self._next_id += 1
        self._process.stdin.write(json.dumps({'id': self._next_id, 'op': op, **params}) + '\n')
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f'the UI driver exited with code {self._process.wait()}')
        response = json.loads(line)
        assert response['id'] == self._next_id
        if 'error' in response:
            raise RuntimeError(f'{op} failed in the UI driver:\n{response["error"]}')
        return response['result']

    def close(self) -> None:
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.close()
        self._process.wait(timeout=10)
        self._process.stdout.close()


@pytest.fixture(scope='session')
def js_driver() -> Iterator[JSDriver]:
    driver = JSDriver()
    try:
        yield driver
    finally:
        driver.close()
