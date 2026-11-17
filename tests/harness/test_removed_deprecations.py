"""The compatibility shims deprecated during 0.x are gone in 1.0 (#12997)."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import pytest

from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.compaction import ReportContextUsage
from pydantic_ai_harness.exa import ExaSearch
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.macroscope import Macroscope
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.spend_limits import SpendLimits
from pydantic_ai_harness.sub_agents import SubAgents
from pydantic_ai_harness.system_reminders import LLMReminder, SystemReminders
from pydantic_ai_harness.warn_on_cache_busts import WarnOnCacheBusts
from pydantic_ai_harness.youdotcom import YouSearch


@pytest.mark.parametrize(
    'module',
    [
        *(
            f'pydantic_ai_harness.experimental.{name}'
            for name in (
                'authoring',
                'compaction',
                'context',
                'docs',
                'dynamic_workflow',
                'media',
                'overflow',
                'planning',
                'step_persistence',
                'subagents',
            )
        ),
        *(
            f'pydantic_ai_harness.{name}'
            for name in (
                'cache_stability',
                'context',
                'docker_sandbox',
                'docs',
                'filesystem',
                'overflowing_tool_output',
                'runtime_authoring',
                'spend',
                'subagents',
            )
        ),
        'pydantic_ai_harness.step_persistence.naming',
    ],
)
def test_moved_module_paths_are_gone(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize(
    ('module', 'name'),
    [
        ('pydantic_ai_harness', 'DockerSandbox'),
        ('pydantic_ai_harness', 'InputGuard'),
        ('pydantic_ai_harness.bubblewrap_sandbox', 'BubblewrapSandbox'),
        ('pydantic_ai_harness.compaction', 'ContextUsageEvent'),
        ('pydantic_ai_harness.compaction', 'LimitWarner'),
        ('pydantic_ai_harness.compaction', 'SlidingWindow'),
        ('pydantic_ai_harness.guardrails', 'GuardResult'),
        ('pydantic_ai_harness.guardrails', 'InputGuardFunc'),
        ('pydantic_ai_harness.guardrails', 'OutputGuard'),
        ('pydantic_ai_harness.guardrails', 'OutputGuardFunc'),
        ('pydantic_ai_harness.media', 'DiskMediaStore'),
        ('pydantic_ai_harness.memory', 'FileStore'),
        ('pydantic_ai_harness.memory', 'InMemoryStore'),
        ('pydantic_ai_harness.planning', 'EventCallback'),
        ('pydantic_ai_harness.planning', 'PlanEvent'),
        ('pydantic_ai_harness.planning', 'PlanEventEmitter'),
        ('pydantic_ai_harness.planning', 'PlanEventType'),
        ('pydantic_ai_harness.spend_limits', 'SpendStore'),
        ('pydantic_ai_harness.step_persistence', 'SnapshotSaved'),
        ('pydantic_ai_harness.tool_output_limits', 'LocalFileStore'),
        ('pydantic_ai_harness.tool_output_limits', 'WorkspaceStore'),
    ],
)
def test_renamed_aliases_are_gone(module: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize('name', ['AcpFileSystemToolset', 'AcpSession', 'AcpSessionConfig', 'AcpTerminalToolset'])
def test_renamed_acp_aliases_are_gone(name: str) -> None:
    pytest.importorskip('acp')
    assert not hasattr(importlib.import_module('pydantic_ai_harness.experimental.acp'), name)


@pytest.mark.parametrize(
    'construct',
    [
        pytest.param(lambda: Shell('.'), id='Shell'),  # pyright: ignore[reportCallIssue]
        pytest.param(lambda: FileSystem('.', ['*.py']), id='FileSystem'),  # pyright: ignore[reportCallIssue]
        pytest.param(lambda: RepoContext('./repo'), id='RepoContext'),  # pyright: ignore[reportCallIssue]
        pytest.param(lambda: ReportContextUsage(None, 200_000), id='ReportContextUsage'),  # pyright: ignore[reportCallIssue]
    ],
)
def test_a_removed_positional_argument_cannot_shift_onto_the_next_field(construct: Callable[[], object]) -> None:
    with pytest.raises(TypeError, match='positional argument'):
        construct()


@pytest.mark.parametrize(
    ('construct', 'keyword'),
    [
        (lambda: Coder(workspace='.'), 'workspace'),  # pyright: ignore[reportCallIssue]
        (lambda: FileSystem(cwd='.'), 'cwd'),  # pyright: ignore[reportCallIssue]
        (lambda: FileSystem(protected_patterns=[]), 'protected_patterns'),  # pyright: ignore[reportCallIssue]
        (lambda: Shell(cwd='.'), 'cwd'),  # pyright: ignore[reportCallIssue]
        (lambda: Macroscope(cwd='.'), 'cwd'),  # pyright: ignore[reportCallIssue]
        (lambda: RepoContext(workspace_dir='.'), 'workspace_dir'),  # pyright: ignore[reportCallIssue]
        (lambda: RepoContext(traversal_tool_names=frozenset()), 'traversal_tool_names'),  # pyright: ignore[reportCallIssue]
        (lambda: SubAgents(inherit_tools=True), 'inherit_tools'),  # pyright: ignore[reportCallIssue]
        (lambda: SpendLimits(on_spend=print), 'on_spend'),  # pyright: ignore[reportCallIssue]
        (lambda: ReportContextUsage(on_usage=print), 'on_usage'),  # pyright: ignore[reportCallIssue]
        (lambda: SystemReminders(on_fire=print), 'on_fire'),  # pyright: ignore[reportCallIssue]
        (lambda: SystemReminders(cache_ttl='5m'), 'cache_ttl'),  # pyright: ignore[reportCallIssue]
        (lambda: LLMReminder(model=TestModel(), cache_ttl='5m'), 'cache_ttl'),  # pyright: ignore[reportCallIssue]
        (lambda: Planning(cache_ttl='5m'), 'cache_ttl'),  # pyright: ignore[reportCallIssue]
        (lambda: ExaSearch(num_results=3), 'num_results'),  # pyright: ignore[reportCallIssue]
        (lambda: YouSearch(num_results=3), 'num_results'),  # pyright: ignore[reportCallIssue]
        (lambda: CodeMode(monty_sandbox_url='http://x'), 'monty_sandbox_url'),  # pyright: ignore[reportCallIssue]
        (lambda: WarnOnCacheBusts(collapse_ratio=0.5), 'collapse_ratio'),  # pyright: ignore[reportCallIssue]
        (lambda: WarnOnCacheBusts(min_prefix_tokens=1), 'min_prefix_tokens'),  # pyright: ignore[reportCallIssue]
        (lambda: WarnOnCacheBusts(cache_ttl_seconds=60), 'cache_ttl_seconds'),  # pyright: ignore[reportCallIssue]
    ],
)
def test_a_removed_keyword_argument_is_rejected(construct: Callable[[], object], keyword: str) -> None:
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{keyword}'"):
        construct()
