"""Coder keeps the filesystem change-request protocol: a listener can refuse a write or edit."""

from pathlib import Path

import pytest
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event

from pydantic_ai_harness.filesystem import FileChangeRequestEvent

from .test_tools import call

pytestmark = pytest.mark.anyio


class Listener(AbstractCapability[None]):
    @on_event(FileChangeRequestEvent)
    async def requested(self, ctx: RunContext[None], event: FileChangeRequestEvent) -> None:
        event.cancel('not approved')


class TestCoder:
    @pytest.mark.parametrize('tool', ['write_file', 'edit_file'])
    async def test_refusal(self, tmp_path: Path, tool: str) -> None:
        path = tmp_path / 'file'
        path.write_text('old')
        arguments: dict[str, object] = {
            'path': 'file',
            **(
                {'content': 'new'}
                if tool == 'write_file'
                else {
                    'old_text': 'old',
                    'new_text': 'new',
                }
            ),
        }
        result = await call(tmp_path, tool, arguments, capabilities=[Listener()])
        assert 'not approved' in result
        assert path.read_text() == 'old'
