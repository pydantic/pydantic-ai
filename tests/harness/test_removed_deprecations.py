"""The compatibility shims deprecated during 0.x are gone in 1.0 (#12997)."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import pytest

from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.compaction import ReportContextUsage
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.macroscope import Macroscope
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.sub_agents import SubAgents


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
        ('pydantic_ai_harness.peers', 'PeerNetwork'),
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
        pytest.param(lambda: Macroscope('main', 'macroscope', '.'), id='Macroscope'),  # pyright: ignore[reportCallIssue]
        pytest.param(lambda: SubAgents((), {}, None, {}, None, True, False), id='SubAgents'),  # pyright: ignore[reportCallIssue]
    ],
)
def test_a_removed_positional_argument_cannot_shift_onto_the_next_field(construct: Callable[[], object]) -> None:
    with pytest.raises(TypeError, match='positional argument'):
        construct()


@pytest.mark.parametrize(
    ('module', 'name', 'kwargs'),
    [
        ('advisor', 'Advisor', {'model': 'test', 'mode': 'native'}),
        ('coder', 'Coder', {'workspace': '.'}),
        ('file_system', 'FileSystem', {'cwd': '.'}),
        ('file_system', 'FileSystem', {'protected_patterns': ['*.py']}),
        ('shell', 'Shell', {'cwd': '.'}),
        ('macroscope', 'Macroscope', {'cwd': '.'}),
        ('repo_context', 'RepoContext', {'workspace_dir': '.'}),
        ('repo_context', 'RepoContext', {'traversal_tool_names': frozenset({'read_file'})}),
        ('sub_agents', 'SubAgents', {'inherit_tools': True}),
        ('spend_limits', 'SpendLimits', {'on_spend': print}),
        ('stackone', 'StackOne', {'account_id': '1', 'api_key': 'key'}),
        ('compaction', 'ReportContextUsage', {'on_usage': print}),
        ('system_reminders', 'SystemReminders', {'on_fire': print}),
        ('system_reminders', 'SystemReminders', {'cache_ttl': '5m'}),
        ('system_reminders', 'LLMReminder', {'model': TestModel(), 'cache_ttl': '5m'}),
        ('planning', 'Planning', {'cache_ttl': '5m'}),
        ('exa', 'ExaSearch', {'num_results': 3}),
        ('youdotcom', 'YouSearch', {'num_results': 3}),
        ('code_mode', 'CodeMode', {'monty_sandbox_url': 'http://x'}),
        ('modal_sandbox', 'ModalSandbox', {'sandbox_id': 'sb-1'}),
        ('modal_sandbox', 'ModalSandbox', {'workdir': '/root'}),
        ('warn_on_cache_busts', 'WarnOnCacheBusts', {'collapse_ratio': 0.5}),
        ('warn_on_cache_busts', 'WarnOnCacheBusts', {'min_prefix_tokens': 1}),
        ('warn_on_cache_busts', 'WarnOnCacheBusts', {'cache_ttl_seconds': 60}),
    ],
)
def test_a_removed_keyword_argument_is_rejected(module: str, name: str, kwargs: dict[str, object]) -> None:
    capability: Callable[..., object] = getattr(
        pytest.importorskip(f'pydantic_ai_harness.{module}', exc_type=ImportError), name
    )
    removed = list(kwargs)[-1]
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{removed}'"):
        capability(**kwargs)
