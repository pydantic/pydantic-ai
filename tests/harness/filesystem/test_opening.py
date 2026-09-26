"""Custom file streams retain filesystem tool behavior without OS descriptors."""

import io
import os
from pathlib import Path
from typing import BinaryIO

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.filesystem import FileSystem, FileSystemToolset

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class BufferedWrite(io.BytesIO):
    def __init__(self, path: Path, content: bytes) -> None:
        super().__init__(content)
        self.path = path

    def close(self) -> None:
        if not self.closed:
            self.path.write_bytes(self.getvalue())
        super().close()


class StreamToolset(FileSystemToolset[None]):
    def __init__(self, root: Path) -> None:
        super().__init__(
            root_dir=root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=100,
            max_list_results=100,
            max_search_results=100,
            max_find_results=100,
        )
        self.streams: list[io.BytesIO] = []
        self.opens: list[tuple[bool, bool]] = []

    def open_read(self, resolved: Path) -> BinaryIO:
        source = io.BytesIO(resolved.read_bytes())
        self.streams.append(source)
        return source

    def open_write(self, resolved: Path, *, read_back: bool, create: bool) -> tuple[BinaryIO, bool]:
        self.opens.append((read_back, create))
        created = False
        if create:
            try:
                with resolved.open('xb'):
                    pass
                created = True
            except FileExistsError:
                pass
        source = BufferedWrite(resolved, resolved.read_bytes())
        self.streams.append(source)
        return source, created


class StreamFileSystem(FileSystem[None]):
    def __init__(self, toolset: StreamToolset) -> None:
        super().__init__()
        self.toolset = toolset

    def get_toolset(self) -> StreamToolset:
        return self.toolset


def reject_descriptor(*args: object, **kwargs: object) -> int:
    raise AssertionError('Custom streams must not require OS descriptors')  # pragma: no cover


class TestFileSystemToolsetOpening:
    async def test_registered_write_and_edit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / 'file.txt'
        target.write_text('before')
        toolset = StreamToolset(tmp_path)
        monkeypatch.setattr(os, 'open', reject_descriptor)
        monkeypatch.setattr(os, 'fstat', reject_descriptor)

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('write_file', {'path': 'file.txt', 'content': 'after'})])
            if len(messages) == 3:
                assert target.read_text() == 'after'
                assert len(toolset.streams) == 2
                return ModelResponse(
                    parts=[ToolCallPart('edit_file', {'path': 'file.txt', 'old_text': 'after', 'new_text': 'edited'})]
                )
            return ModelResponse(parts=[TextPart('done')])

        await Agent(FunctionModel(respond), capabilities=[StreamFileSystem(toolset)], deps_type=type(None)).run(
            'Write and edit'
        )
        assert target.read_text() == 'edited'
        assert len(toolset.streams) == 4
        assert toolset.opens == [(True, True), (True, False)]
        assert all(source.closed for source in toolset.streams)

    async def test_create_overwrite_and_conflict(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        toolset = StreamToolset(tmp_path)
        monkeypatch.setattr(os, 'open', reject_descriptor)
        monkeypatch.setattr(os, 'fstat', reject_descriptor)
        await toolset.write_file('file.txt', 'first', expected_hash='ignored-for-new-file')
        await toolset.write_file('file.txt', 'second')
        with pytest.raises(ModelRetry, match='Conflict'):
            await toolset.write_file('file.txt', 'stale', expected_hash='wrong')
        assert (tmp_path / 'file.txt').read_text() == 'second'
        assert toolset.opens == [(True, True), (False, True), (True, True)]
        assert all(source.closed for source in toolset.streams)
        for source in toolset.streams:
            source.close()

    async def test_missing_edit_does_not_create(self, tmp_path: Path) -> None:
        toolset = StreamToolset(tmp_path)
        with pytest.raises(FileNotFoundError):
            toolset.open_write(tmp_path / 'missing', read_back=True, create=False)
        assert not (tmp_path / 'missing').exists()
