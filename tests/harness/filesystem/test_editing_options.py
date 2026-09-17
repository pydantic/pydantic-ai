"""`FileSystem` options added for single-writer coding agents: `cwd`, `content_hashes`, and batch edits."""

import os
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.filesystem import FileSystem, FileSystemToolset, Replacement

from .._tool_calls import call_tool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def toolset(root: Path, **settings: object) -> FileSystemToolset[None]:
    capability = FileSystem[None](root_dir=root, **settings)  # pyright: ignore[reportArgumentType]
    built = capability.get_toolset()
    assert isinstance(built, FileSystemToolset)
    return built


async def call(root: Path, name: str, arguments: dict[str, object], **settings: object) -> str:
    capability = FileSystem[None](root_dir=root, **settings)  # pyright: ignore[reportArgumentType]
    return await call_tool([capability], name, arguments)


class TestContentHashes:
    async def test_schema_omits_expected_hash(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[FileSystem(root_dir=tmp_path, content_hashes=False)]).run('Inspect')
        assert model.last_model_request_parameters is not None
        schemas = {t.name: t.parameters_json_schema for t in model.last_model_request_parameters.function_tools}
        assert 'expected_hash' not in schemas['write_file']['properties']
        assert 'expected_hash' not in schemas['edit_file']['properties']
        assert 'replacements' in schemas['edit_file']['properties']

        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[FileSystem(root_dir=tmp_path)]).run('Inspect')
        assert model.last_model_request_parameters is not None
        schemas = {t.name: t.parameters_json_schema for t in model.last_model_request_parameters.function_tools}
        assert 'expected_hash' in schemas['write_file']['properties']
        assert 'expected_hash' in schemas['edit_file']['properties']

    async def test_results_omit_hashes(self, tmp_path: Path) -> None:
        written = await call(tmp_path, 'write_file', {'path': 'f.txt', 'content': 'one\ntwo\n'}, content_hashes=False)
        assert written == 'Wrote 8 chars (2 lines) to f.txt.'
        edited = await call(
            tmp_path, 'edit_file', {'path': 'f.txt', 'old_text': 'one', 'new_text': 'uno'}, content_hashes=False
        )
        assert edited == 'Edited f.txt.'
        read = await call(tmp_path, 'read_file', {'path': 'f.txt'}, content_hashes=False)
        assert read.startswith('[f.txt | 2 lines]\n')
        assert 'hash' not in read
        assert (tmp_path / 'f.txt').read_text() == 'uno\ntwo\n'


class TestReplacements:
    async def test_batch_is_sequential(self, tmp_path: Path) -> None:
        path = tmp_path / 'f.txt'
        path.write_bytes(b'one\r\ntwo\r\n')
        result = await call(
            tmp_path,
            'edit_file',
            {
                'path': 'f.txt',
                'replacements': [
                    {'old_text': 'one', 'new_text': 'three'},
                    {'old_text': 'three', 'new_text': 'four'},
                ],
            },
        )
        assert result.startswith('Edited f.txt. [hash:')
        assert path.read_bytes() == b'four\r\ntwo\r\n'

    @pytest.mark.parametrize(
        'arguments,message',
        [
            ({'replacements': []}, 'non-empty replacements list'),
            ({}, 'Provide old_text and new_text'),
            ({'old_text': 'one'}, 'Provide old_text and new_text'),
            (
                {'old_text': 'one', 'new_text': 'two', 'replacements': [{'old_text': 'one', 'new_text': 'two'}]},
                'not both',
            ),
            (
                {'replacements': [{'old_text': 'one', 'new_text': 'two'}, {'old_text': 'missing', 'new_text': '3'}]},
                'replacement 2 not found in f.txt. No changes were written.',
            ),
            ({'replacements': [{'old_text': 'x', 'new_text': 'y'}]}, 'old_text found 2 times'),
            ({'old_text': '', 'new_text': 'two'}, 'old_text is empty'),
            ({'old_text': 'x', 'new_text': 'two'}, 'old_text found 2 times'),
        ],
    )
    async def test_invalid_batch_leaves_file_unchanged(
        self, tmp_path: Path, arguments: dict[str, object], message: str
    ) -> None:
        path = tmp_path / 'f.txt'
        path.write_text('one x x')
        assert message in await call(tmp_path, 'edit_file', {'path': 'f.txt', **arguments})
        assert path.read_text() == 'one x x'

    async def test_binary_files_are_not_edited(self, tmp_path: Path) -> None:
        path = tmp_path / 'blob.bin'
        path.write_bytes(b'a\0b')
        assert 'binary file' in await call(
            tmp_path, 'edit_file', {'path': 'blob.bin', 'old_text': 'a', 'new_text': 'c'}
        )
        assert path.read_bytes() == b'a\0b'

    async def test_direct_method_keeps_single_pair(self, tmp_path: Path) -> None:
        (tmp_path / 'f.txt').write_text('one')
        assert (await toolset(tmp_path).edit_file('f.txt', 'one', 'two')).startswith('Edited f.txt.')
        assert (tmp_path / 'f.txt').read_text() == 'two'
        assert Replacement(old_text='a', new_text='b').new_text == 'b'


class TestCwd:
    async def test_relative_paths_resolve_from_cwd(self, tmp_path: Path) -> None:
        project = tmp_path / 'project'
        project.mkdir()
        (tmp_path / 'shared.txt').write_text('outside the project')
        built = toolset(tmp_path, cwd=project)
        await built.write_file('local.txt', 'inside')
        assert (project / 'local.txt').read_text() == 'inside'
        assert 'outside the project' in await built.read_file('../shared.txt')
        assert 'outside the project' in await built.read_file(str(tmp_path / 'shared.txt'))
        assert await built.list_directory('.') == await built.list_directory('../project')

    @pytest.mark.skipif(os.name == 'nt', reason='POSIX symlinks')
    async def test_file_info_reports_the_symlink_at_cwd(self, tmp_path: Path) -> None:
        project = tmp_path / 'project'
        project.mkdir()
        (tmp_path / 'target.txt').write_text('shared')
        (project / 'link.txt').symlink_to(tmp_path / 'target.txt')
        (tmp_path / 'link.txt').write_text('a regular file at the root with the same name')
        info = await toolset(tmp_path, cwd=project).file_info('link.txt')
        assert 'symlink' in info and 'target.txt' in info

    def test_cwd_must_be_inside_root(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='outside root_dir'):
            toolset(tmp_path / 'root', cwd=tmp_path)

    async def test_traversal_is_still_bounded_by_root(self, tmp_path: Path) -> None:
        project = tmp_path / 'root' / 'project'
        project.mkdir(parents=True)
        result = await call(tmp_path / 'root', 'read_file', {'path': '../../outside.txt'}, cwd=project)
        assert 'outside the root directory' in result
