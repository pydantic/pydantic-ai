from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event

from pydantic_ai_harness.filesystem import FileChangeRequestEvent

from .test_tools import call

pytestmark = pytest.mark.anyio


@dataclass(kw_only=True)
class Listener(AbstractCapability[None]):
    path: Path
    mode: Literal['cancel', 'modify', 'delete']

    @on_event(FileChangeRequestEvent)
    async def requested(self, ctx: RunContext[None], event: FileChangeRequestEvent) -> None:
        if self.mode == 'cancel':
            event.cancel('not approved')
        elif self.mode == 'modify':
            self.path.write_text('concurrent')
        else:
            self.path.unlink()


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
        result = await call(tmp_path, tool, arguments, capabilities=[Listener(path=path, mode='cancel')])
        assert 'not approved' in result
        assert path.read_text() == 'old'

    @pytest.mark.parametrize('mode', ['modify', 'delete'])
    async def test_concurrent_change_retries(self, tmp_path: Path, mode: Literal['modify', 'delete']) -> None:
        path = tmp_path / 'file'
        path.write_text('old')
        result = await call(
            tmp_path,
            'edit_file',
            {'path': 'file', 'old_text': 'old', 'new_text': 'new'},
            capabilities=[Listener(path=path, mode=mode)],
        )
        assert 'Cannot edit' in result
        if mode == 'modify':
            assert path.read_text() == 'concurrent'
        else:
            assert not path.exists()
