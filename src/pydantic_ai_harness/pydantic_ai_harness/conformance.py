"""Conformance data and suites for Pydantic AI Harness capabilities.

`CALLS` holds, by capability class name, the tool calls that run each Harness capability's tools offline, the
counterpart of [`pydantic_ai.capabilities.conformance.CALLS`][pydantic_ai.capabilities.conformance.CALLS], and
`HarnessDurableEngineSuite` runs them under a durable engine with the core cases. Requires pytest and the anyio
pytest plugin.

`CALLS`, `HarnessDurableEngineSuite` and the case names are stable in 2.x, as the core suite's are. Minor releases
can add cases. The stand-ins a `build` uses are internal: what they are and their names can change.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar, ParamSpec, TypeVar

import pytest

from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, Toolset
from pydantic_ai.capabilities.conformance import (
    _ECHO,  # pyright: ignore[reportPrivateUsage]
    CALLS as _CORE_CALLS,
    Calls,
    CapabilityBuilder,
    _echo_toolset,  # pyright: ignore[reportPrivateUsage]
    _with_echo,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.durable_exec.conformance import (
    _NESTED as _SUITE_COUNTS,  # pyright: ignore[reportPrivateUsage]
    DurableEngineSuite,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolSelector

from .background_tools import BackgroundTools
from .capability_creation import CapabilityCreation, CapabilityStore
from .code_mode import CodeMode
from .coder import Coder
from .compaction import CompactOnRequest, DeduplicateFileReads, PruneToolResults, SlidingWindowCompaction
from .conversation_search import ConversationSearch, SnapshotHistorySource
from .file_system import FileSystem
from .guardrails import ToolGuardrail
from .macroscope import Macroscope
from .memory import InMemoryMemoryStore, Memory
from .permissions import PermissionRule, Permissions
from .planning import Planning, SqlitePlanStore
from .pydantic_ai_docs import PydanticAIDocs
from .repair_tool_arguments import RepairToolArguments
from .repo_context import RepoContext
from .report_tool_errors import ReportToolErrors
from .shell import Shell
from .skills import Skills
from .step_persistence import InMemoryStepStore, StepPersistence
from .tool_bridge import ToolBridge
from .tool_budgets import ToolBudgets

__all__ = ('CALLS', 'HarnessDurableEngineSuite')

P = ParamSpec('P')
T = TypeVar('T')


def _workspace(build: Callable[[], AbstractCapability[None]]) -> CapabilityBuilder:
    """The capability plus a local workspace in the directory, for a capability that needs one."""
    return lambda directory: [build(), LocalWorkspace(directory)]


def _skills(directory: Path) -> list[AbstractCapability[None]]:
    """`Skills` over a library with one `weather` skill, in a local workspace."""
    skill = directory / 'skills' / 'weather'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\ndescription: Report the weather.\n---\n\nSay sunny.\n')
    return [Skills(directory / 'skills'), LocalWorkspace(directory)]


def _lowered_skills(directory: Path) -> list[AbstractCapability[None]]:
    # Below `'operator'`, so the skill is held back from the instructions and `list_skills` and `load_skill` offer it.
    return [_skills(directory)[0], LocalWorkspace(directory, authority='none')]


def _coder_repo(directory: Path) -> list[AbstractCapability[None]]:
    """`Coder` in a repo of its own.

    Its `.git` ends the walk up for the repo root, which would otherwise make a workspace call per parent directory,
    so the units its calls record depend on how deep the directory is.
    """
    (directory / '.git').mkdir()
    return [Coder(), LocalWorkspace(directory)]


def _repo_context(directory: Path) -> list[AbstractCapability[None]]:
    """`RepoContext` over a repo with an `AGENTS.md`, with a file system to read it through."""
    (directory / 'AGENTS.md').write_text('Keep answers short.\n')
    return [RepoContext(), FileSystem(), LocalWorkspace(directory)]


def _nested_repo_context(directory: Path) -> list[AbstractCapability[None]]:
    """`_repo_context`, over a repo with an instruction file nested in `src`."""
    (directory / 'src').mkdir()
    (directory / 'src' / 'AGENTS.md').write_text('Type every function.\n')
    (directory / 'src' / 'app.py').write_text('print("hi")\n')
    return _repo_context(directory)


def _permissions(directory: Path) -> list[AbstractCapability[None]]:
    """`Permissions` in `'auto'` mode, with a rule that denies `git push`, over a file system and a shell."""
    rules = [PermissionRule[None]('deny', command='git push', reason='Pushing is for people.')]
    return [Permissions('auto', rules=rules), FileSystem(), Shell(), LocalWorkspace(directory)]


def _read_file_key(call: ToolCallPart) -> str | None:
    args = call.args_as_dict()
    return str(args['path']) if call.tool_name == 'read_file' else None


def _deduplicate_file_reads(directory: Path) -> list[AbstractCapability[None]]:
    """`DeduplicateFileReads`, keyed by `read_file`'s path, over a file system with one file."""
    (directory / 'notes.md').write_text('Sunny.\n')
    return [DeduplicateFileReads(file_key=_read_file_key), FileSystem(), LocalWorkspace(directory)]


def _pydantic_ai_docs(directory: Path) -> list[AbstractCapability[None]]:
    """`PydanticAIDocs` over a local copy of the docs with one page, uncached."""
    (directory / 'docs').mkdir()
    (directory / 'docs' / 'hooks.md').write_text('# Hooks\n')
    return [PydanticAIDocs(local_docs_path=directory / 'docs', cache=False), LocalWorkspace(directory)]


def _macroscope(directory: Path) -> list[AbstractCapability[None]]:
    """`Macroscope` running a stand-in CLI that reports a finished review with no issues, on stderr as it does."""
    command = directory / 'macroscope'
    command.write_text("#!/bin/sh\nprintf 'review_id=review\\nissue_status=completed\\n' >&2\n")
    command.chmod(0o755)
    return [Macroscope(command=str(command), base='main'), LocalWorkspace(directory)]


def _capability_creation(directory: Path) -> list[AbstractCapability[None]]:
    """`CapabilityCreation` storing what it authors in the directory's workspace."""
    return [CapabilityCreation(directory / 'capabilities'), LocalWorkspace(directory)]


_AUTHORED = """
from pydantic_ai.capabilities import AbstractCapability


class Marker(AbstractCapability):
    def get_instructions(self):
        return 'marker'
"""
"""A capability for `CapabilityCreation` to author."""


def _background_echo(_: Path) -> list[AbstractCapability[None]]:
    """`BackgroundTools` with its default selector, beside an `echo` tool marked to run in the background.

    Selecting by metadata rather than by name keeps the tool selected under a prefix.
    """
    return [BackgroundTools(), Toolset(_echo_toolset(background=True))]


def _prompt_injection_defender() -> AbstractCapability[None]:
    # Imported here, as `stackone_defender` is an optional dependency. It declares durable operations and no default
    # id, so a durable agent needs one given.
    from .prompt_injection_defender import PromptInjectionDefender

    return PromptInjectionDefender(id='prompt_injection_defender')


def _ipython_code_mode(tools: ToolSelector[None] = ()) -> AbstractCapability[None]:
    # Imported here, as `jupyter_client` is an optional dependency.
    from .ipython_code_mode import IPythonCodeMode

    return IPythonCodeMode(tools=tools)


_FILE_EDITS = (
    ('write_file', {'path': 'notes.md', 'content': 'Sunny.\n'}),
    ('edit_file', {'path': 'notes.md', 'old_text': 'Sunny.', 'new_text': 'Rainy.'}),
    ('read_file', {'path': 'notes.md'}),
)
"""A file written, edited, then read back."""


CALLS: Mapping[str, Calls] = {
    'BackgroundTools': Calls(_ECHO, build=_background_echo, follow_ups=1),
    'CapabilityCreation': Calls(
        (
            ('author_capability', {'name': 'marker', 'code': _AUTHORED}),
            ('list_authored_capabilities', {}),
            ('disable_authored_capability', {'name': 'marker'}),
        ),
        build=_capability_creation,
    ),
    'CodeMode': Calls(
        # The program calls `echo` through the sandbox, which dispatches it to the host tool.
        (('run_code', {'code': "await echo(text='hi')"}),),
        build=_with_echo(CodeMode),
    ),
    'Coder': Calls(_FILE_EDITS, build=_coder_repo),
    'CompactOnRequest': Calls(
        (('compact_context', {}),),
        build=lambda _: [CompactOnRequest(SlidingWindowCompaction(max_messages=1, keep_messages=2))],
    ),
    'ConversationSearch': Calls(
        (('search_conversation_history', {'query': 'tabs'}),),
        build=lambda _: [
            ConversationSearch(SnapshotHistorySource(InMemoryStepStore()), scope='all', add_instructions=False)
        ],
    ),
    'DeduplicateFileReads': Calls(
        (('read_file', {'path': 'notes.md'}), ('read_file', {'path': 'notes.md'})), build=_deduplicate_file_reads
    ),
    'FileSystem': Calls(_FILE_EDITS, build=_workspace(FileSystem)),
    'IPythonCodeMode': Calls(
        (('run_code', {'code': "echo(text='hi')"}),), build=_with_echo(lambda: _ipython_code_mode('all'))
    ),
    'Macroscope': Calls((('run_macroscope_review', {}),), build=_macroscope),
    'Memory': Calls(
        (
            ('write_memory', {'content': 'The user prefers tabs.', 'file': 'style.md'}),
            # A missing file asks the model to retry, which the operation records as data.
            ('read_memory', {'file': 'missing.md'}),
            ('search_memory', {'query': 'tabs'}),
            ('delete_memory', {'file': 'missing.md'}),
        ),
        build=lambda _: [Memory(store=InMemoryMemoryStore())],
    ),
    'Permissions': Calls(
        # `'auto'` runs the write and the plain command, and the rule denies the push without running it.
        (
            ('write_file', {'path': 'notes.md', 'content': 'Sunny.\n'}),
            ('run_command', {'command': 'echo hi'}),
            ('run_command', {'command': 'git push'}),
        ),
        build=_permissions,
    ),
    'Planning': Calls(
        (
            ('write_plan', {'items': [{'id': 'first', 'content': 'Write the migration'}]}),
            ('add_task', {'content': 'Run the tests'}),
        ),
        build=lambda directory: [Planning(store=SqlitePlanStore(database=str(directory / 'plan.db')))],
    ),
    'PromptInjectionDefender': Calls(_ECHO, build=_with_echo(_prompt_injection_defender)),
    'PruneToolResults': Calls(
        # The first is refused, as no previous step returned `missing`; the second prunes the first's result, which
        # the calls' scripted model gives the ID `call_0`.
        (
            ('prune_tool_result', {'tool_call_id': 'missing', 'summary': 'Quiet.'}),
            ('prune_tool_result', {'tool_call_id': 'call_0', 'summary': 'Quiet.'}),
        ),
        build=lambda _: [PruneToolResults()],
    ),
    'PydanticAIDocs': Calls((('read_pyai_docs', {'topic': 'hooks'}),), build=_pydantic_ai_docs),
    'RepairToolArguments': Calls(_ECHO, build=_with_echo(RepairToolArguments)),
    # Reading a file in `src` also states the instruction file nested there. Its inventory tool steps aside for
    # `read_file`, which the nested instructions need.
    'RepoContext': Calls((('read_file', {'path': 'src/app.py'}),), build=_nested_repo_context),
    'ReportToolErrors': Calls(_ECHO, build=_with_echo(ReportToolErrors)),
    'Shell': Calls((('run_command', {'command': 'echo hi'}),), build=_workspace(Shell)),
    'Skills': Calls((('list_skills', {}), ('load_skill', {'name': 'weather'})), build=_lowered_skills),
    'StepPersistence': Calls(_ECHO, build=_with_echo(StepPersistence)),
    'ToolBridge': Calls(
        # The model finds `echo` behind the bridge, reads its schema, and calls it.
        (
            ('tool_search', {'queries': ['echo']}),
            ('tool_describe', {'name': 'echo'}),
            ('tool_call', {'name': 'echo', 'args_json': '{"text": "hi"}'}),
        ),
        build=_with_echo(ToolBridge),
    ),
    'ToolBudgets': Calls(_ECHO, build=_with_echo(ToolBudgets)),
    # It declares durable operations and no default id, so a durable agent needs one given.
    'ToolGuardrail': Calls(_ECHO, build=_with_echo(lambda: ToolGuardrail(id='tool_guardrail'))),
}
"""The calls that run each Pydantic AI Harness capability's tools offline, keyed by capability class name."""


_COUNTED_METHODS: tuple[tuple[str, type, str], ...] = (
    ('store.write', CapabilityStore, 'write'),
    ('store.list_all', CapabilityStore, 'list_all'),
    ('store.disable', CapabilityStore, 'disable'),
    # A memory write first looks up its idempotency key; the write is what recovery must not repeat.
    ('memory.write', InMemoryMemoryStore, 'write'),
    # The plan tools read the store many times; a write is what recovery must not repeat.
    ('plan.set_items', SqlitePlanStore, 'set_items'),
    ('plan.add_item', SqlitePlanStore, 'add_item'),
    ('history.list_runs', SnapshotHistorySource, 'list_runs'),
)
"""The store methods `HarnessDurableEngineSuite` counts, each under its key: the requests a recovered run must not
make again."""


def _counting(key: str, method: Callable[P, T]) -> Callable[P, T]:
    """`method`, counting each call in the suite's own counts under `key`, when its request starts."""

    def counting(*args: P.args, **kwargs: P.kwargs) -> T:
        _SUITE_COUNTS[key] += 1
        return method(*args, **kwargs)

    return counting


class HarnessDurableEngineSuite(DurableEngineSuite):
    """`DurableEngineSuite` with the Pydantic AI Harness cases added to the core ones.

    Subclass it instead of [`DurableEngineSuite`][pydantic_ai.durable_exec.conformance.DurableEngineSuite] to check an
    engine against both: everything else is the same.
    """

    cases: ClassVar[Mapping[str, Calls]] = {**_CORE_CALLS, **CALLS}
    """The core and Harness cases, by capability class name."""

    @pytest.fixture(autouse=True)
    def _count_stores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Count the writes and reads of the stores the Harness cases' stand-ins use, beside the suite's own counts.

        They're counted in this process, whatever `count_requests` returns, so an override that counts stand-ins of
        its own keeps them.
        """
        for key, owner, method in _COUNTED_METHODS:
            monkeypatch.setattr(owner, method, _counting(key, getattr(owner, method)))
