"""The compatibility shims deprecated during 0.x are gone in 1.0 (#12997)."""

from __future__ import annotations

import importlib

import pytest


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
        ('pydantic_ai_harness.compaction', 'SlidingWindow'),
        ('pydantic_ai_harness.guardrails', 'GuardResult'),
        ('pydantic_ai_harness.media', 'DiskMediaStore'),
        ('pydantic_ai_harness.memory', 'FileStore'),
        ('pydantic_ai_harness.planning', 'PlanEventEmitter'),
        ('pydantic_ai_harness.spend_limits', 'SpendStore'),
        ('pydantic_ai_harness.step_persistence', 'SnapshotSaved'),
        ('pydantic_ai_harness.tool_output_limits', 'LocalFileStore'),
    ],
)
def test_renamed_aliases_are_gone(module: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module), name)
