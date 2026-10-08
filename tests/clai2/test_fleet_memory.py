"""Hackathon fleet memory: personal notebooks on this machine, read-only repo notes from Logfire, proposals."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_clai2.builtin_plugins.fleet_memory import (
    MAX_NOTE_BYTES,
    MemoryNotes,
    PendingNotes,
    RepoNote,
    RepoNotesStore,
    personal_memory,
    proposed,
    repo_memory,
    repo_notes,
    with_pending,
)

PROPOSED = 'Proposed.'

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


def test_a_proposal_is_live_for_its_proposer_until_published_or_dismissed(tmp_path: Path) -> None:
    pending = PendingNotes(tmp_path / 'pending.json')
    pending.add(repo='acme/widgets', path='testing.md', content='Use pytest -x.', why='speed')
    pending.add(repo='acme/widgets', path='MEMORY.md', content='- New fact.', why='')
    pending.add(repo='other/repo', path='MEMORY.md', content='- Elsewhere.', why='')
    mine = pending.for_repo('acme/widgets')
    assert [(note.path, note.pending) for note in mine] == [('testing.md', True), ('MEMORY.md', True)]
    notes = with_pending([NOTE], mine)
    assert [note.path for note in notes] == [
        'testing.md',
        'MEMORY.md',
    ]  # The pending MEMORY.md replaces the shared one.
    store = RepoNotesStore(lambda: notes)
    assert store._files()['repo/testing.md'].endswith('_(pending review: only you see this)_')  # pyright: ignore[reportPrivateUsage]

    # Published (the shared file now has this content) or dismissed (same repo, file and content): dropped.
    shared = [RepoNote(path='testing.md', content='Use pytest -x.', accepted_by='alice@example.com')]
    proposals = [
        {
            'kind': 'memory',
            'status': 'dismissed',
            'repo_slug': 'acme/widgets',
            'path': 'MEMORY.md',
            'content': '- New fact.',
        },
        {'kind': 'memory', 'status': 'dismissed', 'repo_slug': 'acme/widgets', 'path': 'x.md', 'content': 'other'},
    ]
    told = pending.reconcile(repo='acme/widgets', shared=shared, proposals=proposals)
    assert told == ["Your note MEMORY.md wasn't accepted by your team's admins."]
    assert pending.for_repo('acme/widgets') == []
    assert pending.reconcile(repo='acme/widgets', shared=shared, proposals=proposals) == []
    assert [note.path for note in pending.for_repo('other/repo')] == ['MEMORY.md']
    assert pending.reconcile(repo=None, shared=[], proposals=[]) == []


def test_the_reply_to_a_proposal_says_how_it_will_be_shared() -> None:
    assert proposed('review', 'acme/widgets').startswith("Proposed to your team's admins in Logfire for review")
    assert proposed('corroborate', 'acme/widgets').startswith("Proposed; it will be shared once a teammate's agent")
    assert proposed('auto', 'acme/widgets').startswith('Shared with everyone in acme/widgets')


async def test_memory_command_lists_opens_edits_forgets_and_proposes(tmp_path: Path) -> None:
    from pydantic_clai2.builtin_plugins.fleet_memory import PendingNote, personal_dirs
    from pydantic_clai2.builtin_plugins.memory_command import MemoryCommand

    pending = PendingNotes(tmp_path / 'pending.json')
    proposals: list[tuple[str, str, str]] = []
    withdrawn: list[PendingNote] = []
    edited: list[str] = []

    def propose(path: str, content: str, why: str) -> str:
        proposals.append((path, content, why))
        pending.add(repo='acme/widgets', path=path, content=content, why=why)
        return 'Proposed.'

    async def edit(text: str, title: str) -> str | None:
        edited.append(title)
        return None if title == 'cancel' else '- Prefer uv.\n'

    command = MemoryCommand(
        directory=tmp_path / 'memory',
        repo=lambda: 'acme/widgets',
        notes=lambda: [NOTE],
        mode=lambda: 'review',
        pending=pending,
        propose=propose,
        withdraw=withdrawn.append,
        edit=edit,
        link='https://logfire.example/acme/clai2/agents/clai2/configure/edit#memory',
    )
    here, everywhere = personal_dirs(tmp_path / 'memory', 'acme/widgets')
    assert await command(['edit']) == f'Saved {here / "MEMORY.md"}. The agent sees it from the next model request.'
    await command(['edit', 'global'])
    assert edited == ['Your notes (acme/widgets)', 'Your notes (every repository)']
    assert await command(['edit']) == f'No changes to {here / "MEMORY.md"}.'
    (here / 'testing.md').write_text('Use pytest -x.\n', encoding='utf-8')

    overview = await command([])
    assert 'Shared notes are published by review' in overview
    assert 'Personal, acme/widgets' in overview and 'testing.md  15 B' in overview
    assert '- Prefer uv.' in overview
    assert 'accepted by alice@example.com on 2026-10-08, proposed by bob@example.com' in overview
    assert 'Edit them in Logfire: https://logfire.example/acme/clai2/agents/clai2/configure/edit#memory' in overview

    assert (await command(['open', 'testing.md'])).endswith('Use pytest -x.\n')
    assert (await command(['open', 'global/MEMORY.md'])).endswith('- Prefer uv.\n')
    assert (await command(['open', 'MEMORY.md'])).startswith(str(here / 'MEMORY.md'))

    assert await command(['propose', 'testing.md']) == 'Proposed.'
    assert proposals[0][:2] == ('testing.md', 'Use pytest -x.\n')
    assert 'Pending, only you see these' in await command([])
    assert (await command(['open', 'testing.md'])).startswith(str(here))  # Personal first.

    assert (
        await command(['forget', 'testing.md'])
        == 'Withdrew your proposal testing.md; it no longer applies to your sessions.'
    )
    assert [note.path for note in withdrawn] == ['testing.md']
    assert (await command(['forget', 'testing.md'])) == f'Deleted {here / "testing.md"}.'
    command.notes = lambda: [RepoNote(path='shared.md', content='x')]
    with pytest.raises(ValueError, match='shared repo note'):
        await command(['forget', 'shared.md'])
    with pytest.raises(ValueError, match='No personal memory file or pending note'):
        await command(['forget', 'missing.md'])
    with pytest.raises(ValueError, match='No personal memory file'):
        await command(['propose', 'missing.md'])
    with pytest.raises(ValueError, match='Usage: /memory'):
        await command(['nonsense'])
    with pytest.raises(ValueError, match='No memory file'):
        await command(['open', 'missing.md'])


async def test_personal_memory_works_on_a_fresh_machine(tmp_path: Path) -> None:
    """The notebook folder doesn't exist before the first write; searching it must not fail the turn."""
    directory = tmp_path / 'config/pydantic-clai2/memory'
    calls = iter([ToolCallPart('search_memory', {'query': 'bits'}, tool_call_id='1')])

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        call = next(calls, None)
        return ModelResponse(parts=[call] if call else [TextPart('ok')])

    agent = Agent(FunctionModel(model), capabilities=personal_memory(directory, repo=lambda: 'acme/widgets'))
    result = await agent.run('hi')
    assert result.output == 'ok'
    assert directory.is_dir()
