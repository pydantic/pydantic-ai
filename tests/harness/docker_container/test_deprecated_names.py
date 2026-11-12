"""The old `pydantic_ai_harness.docker_sandbox` names still import, with a `HarnessDeprecationWarning`, and old specs still load."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass

import pytest

from pydantic_ai import Agent
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.docker_container import DockerContainer, DockerContainerBackend


def test_shim_warns_and_aliases_old_names() -> None:
    # Import once quietly so the deprecation fires on the reload we assert on, even if
    # another test already imported the shim.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        shim = importlib.import_module('pydantic_ai_harness.docker_sandbox')

    with pytest.warns(
        HarnessDeprecationWarning, match=r'renamed to `pydantic_ai_harness\.docker_container`.*security sandbox'
    ):
        importlib.reload(shim)

    assert issubclass(shim.DockerSandbox, DockerContainer)
    assert shim.DockerSandboxBackend is DockerContainerBackend


def test_root_old_names_resolve_to_shim() -> None:
    import pydantic_ai_harness

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from pydantic_ai_harness.docker_sandbox import DockerSandbox

        assert pydantic_ai_harness.DockerSandbox is DockerSandbox
        assert pydantic_ai_harness.DockerSandboxBackend is DockerContainerBackend


def test_subclass_of_old_name_keeps_its_own_spec_name() -> None:
    """As before the rename, a subclass's spec block name is its own class name."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from pydantic_ai_harness.docker_sandbox import DockerSandbox

    @dataclass(init=False)
    class MyDocker(DockerSandbox[None]):
        pass

    assert MyDocker.get_serialization_name() == 'MyDocker'
    agent = Agent.from_spec(
        {'model': 'test', 'capabilities': [{'MyDocker': {'image': 'python:3.13-slim'}}]},
        custom_capability_types=[MyDocker],
    )
    assert [type(c) for c in agent.root_capability.capabilities if isinstance(c, DockerContainer)] == [MyDocker]


def test_old_spec_block_name_loads() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from pydantic_ai_harness.docker_sandbox import DockerSandbox

    agent = Agent.from_spec(
        {'model': 'test', 'capabilities': [{'DockerSandbox': {'image': 'python:3.13-slim'}}]},
        custom_capability_types=[DockerSandbox],
    )

    assert [c.image for c in agent.root_capability.capabilities if isinstance(c, DockerContainer)] == [
        'python:3.13-slim'
    ]
