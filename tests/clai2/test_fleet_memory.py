"""Hackathon fleet memory: personal notebooks on this machine, read-only repo notes from Logfire, proposals."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_clai2.builtin_plugins.fleet_memory import (
    MAX_NOTE_BYTES,
    PROPOSED,
    MemoryNotes,
    RepoNote,
    RepoNotesStore,
    personal_memory,
    repo_memory,
    repo_notes,
)

NOTE = RepoNote(
    path='MEMORY.md',
    content='- Run `make test` before pushing.',
    proposed_by='bob@example.com',
    accepted_by='alice@example.com',
    accepted_at='2026-10-08T12:00:00Z',
    applies_to={'repos': ['acme/*']},
)


def _prompt_text(messages: list[ModelMessage]) -> str:
    request = messages[-1]
    assert isinstance(request, ModelRequest)
    return '\n'.join(
        str(item.content if hasattr(item, 'content') else item)
        for part in request.parts
        if isinstance(part, UserPromptPart)
        for item in (part.content if isinstance(part.content, list) else [part.content])
    )


async def test_repo_notes_are_injected_with_provenance_and_cannot_be_written() -> None:
    notes = [NOTE]
    proposals: list[tuple[str, str, str]] = []

    def propose(path: str, content: str, why: str) -> str:
        proposals.append((path, content, why))
        return PROPOSED

    seen: dict[str, object] = {}

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if 'tools' not in seen:
            seen['tools'] = sorted(tool.name for tool in info.function_tools)
            seen['prompt'] = _prompt_text(messages)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'repo_propose_memory',
                        {'path': 'testing.md', 'content': 'Use pytest -x.', 'why': 'Everyone reruns slowly.'},
                    )
                ]
            )
        seen['result'] = messages[-1].parts[0].content  # pyright: ignore[reportAttributeAccessIssue]
        return ModelResponse(parts=[TextPart('ok')])

    agent = Agent(FunctionModel(model), capabilities=[repo_memory(lambda: notes, propose=propose)])
    await agent.run('hi')
    assert seen['tools'] == ['repo_propose_memory', 'repo_read_memory', 'repo_search_memory']
    prompt = str(seen['prompt'])
    assert '## Repo notes (shared via Logfire)' in prompt
    assert 'Run `make test` before pushing.' in prompt
    assert 'accepted by alice@example.com on 2026-10-08, proposed by bob@example.com' in prompt
    assert proposals == [('testing.md', 'Use pytest -x.', 'Everyone reruns slowly.')]
    assert seen['result'] == PROPOSED


async def test_the_read_only_store_refuses_writes_and_searches_notes() -> None:
    store = RepoNotesStore(lambda: [NOTE])
    with pytest.raises(PermissionError, match='read-only'):
        await store.write('repo/MEMORY.md', 'x', expected_version=None)
    with pytest.raises(PermissionError, match='read-only'):
        await store.delete('repo/MEMORY.md', expected_version=None)
    assert await store.list_paths('repo/', limit=10) == ['repo/MEMORY.md']
    assert await store.get_operation(object()) is None  # pyright: ignore[reportArgumentType]
    found = await store.search('repo/', 'make test', limit=5, max_files=5, max_chars=500, max_file_chars=500)
    assert [match.path for match in found.matches] == ['repo/MEMORY.md']
    empty = await store.search('repo/', '  ', limit=5, max_files=5, max_chars=500, max_file_chars=500)
    assert empty.matches == []
    read = await store.read('repo/MEMORY.md', max_chars=5)
    assert read is not None and read.truncated and read.content == '- Run'
    assert await store.read('repo/missing.md', max_chars=5) is None


def test_repo_notes_keep_valid_in_scope_notes_within_limits() -> None:
    def note(path: str, **extra: object) -> RepoNote:
        return RepoNote.model_validate({'path': path, 'content': 'x', **extra})

    notes = MemoryNotes(
        files=[
            note('MEMORY.md'),
            note('MEMORY.md', content='duplicate'),
            note('team.md', scope='team'),
            note('other.md', applies_to={'repos': ['other/*']}),
            note('../escape.md'),
            note('nested/dir.md'),
            note('big.md', content='x' * (MAX_NOTE_BYTES + 1)),
            *(note(f'n{index}.md') for index in range(30)),
        ]
    )

    def applies(scope: object) -> bool:
        return scope is None

    kept = repo_notes(notes, applies)
    assert [n.path for n in kept][:2] == ['MEMORY.md', 'n0.md']
    assert kept[0].content == 'x'
    assert len(kept) == 20


async def test_proposals_are_checked_before_they_are_recorded() -> None:
    capability = repo_memory(lambda: [], propose=lambda path, content, why: PROPOSED)
    toolset = capability.get_toolset()
    assert toolset is not None
    replies: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'repo_propose_memory', {'path': 'a/b.md', 'content': 'x', 'why': 'y'}, tool_call_id='a'
                    ),
                    ToolCallPart(
                        'repo_propose_memory',
                        {'path': 'big.md', 'content': 'x' * (MAX_NOTE_BYTES + 1), 'why': 'y'},
                        tool_call_id='b',
                    ),
                ]
            )
        replies.extend(str(part.content) for part in messages[-1].parts)  # pyright: ignore[reportAttributeAccessIssue]
        return ModelResponse(parts=[TextPart('ok')])

    await Agent(FunctionModel(model), capabilities=[capability]).run('hi')
    assert replies == [
        'Use a plain file name ending in .md, such as MEMORY.md or testing.md.',
        f'Keep a note under {MAX_NOTE_BYTES:,} bytes; split it or trim it.',
    ]


async def test_personal_notebooks_are_per_repository_plus_global(tmp_path: Path) -> None:
    slug: list[str | None] = ['acme/widgets']
    calls = iter(
        [
            ToolCallPart('write_memory', {'content': 'repo fact'}, tool_call_id='1'),
            ToolCallPart('global_write_memory', {'content': 'global fact'}, tool_call_id='2'),
        ]
    )

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        call = next(calls, None)
        return ModelResponse(parts=[call] if call else [TextPart('ok')])

    agent = Agent(FunctionModel(model), capabilities=personal_memory(tmp_path, repo=lambda: slug[0]))
    await agent.run('hi')
    assert (tmp_path / 'repos/acme/widgets/personal/MEMORY.md').read_text().strip() == 'repo fact'
    assert (tmp_path / 'global/personal/MEMORY.md').read_text().strip() == 'global fact'

    slug[0] = None
    calls = iter([ToolCallPart('write_memory', {'content': 'dir fact'}, tool_call_id='3')])
    await agent.run('hi')
    [dir_notebook] = list((tmp_path / 'dirs').glob('*/personal/MEMORY.md'))
    assert dir_notebook.read_text().strip() == 'dir fact'
