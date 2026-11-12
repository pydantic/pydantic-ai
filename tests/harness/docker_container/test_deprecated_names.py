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


def _import_old_class() -> type[DockerContainer[None]]:
    """The shim warns once on import; these tests assert the class's own warnings instead."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from pydantic_ai_harness.docker_sandbox import DockerSandbox  # pyright: ignore[reportDeprecated]

    return DockerSandbox  # pyright: ignore[reportDeprecated]


_CLASS_WARNING = r'`DockerSandbox` is now `DockerContainer`.*rename the `DockerSandbox` block to `DockerContainer`'


def test_root_old_names_resolve_to_shim() -> None:
    import pydantic_ai_harness

    old = _import_old_class()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert pydantic_ai_harness.DockerSandbox is old  # pyright: ignore[reportDeprecated]
        assert pydantic_ai_harness.DockerSandboxBackend is DockerContainerBackend


def test_constructing_old_class_warns() -> None:
    """Every construction warns, not only the first import, so a long-running process that loads old specs sees it."""
    old = _import_old_class()
    with pytest.warns(HarnessDeprecationWarning, match=_CLASS_WARNING):
        capability = old(image='python:3.13-slim')
    assert isinstance(capability, DockerContainer)


def test_subclass_of_old_name_warns_once_and_keeps_its_own_spec_name() -> None:
    """Defining a subclass warns, since its base class has to migrate; creating one doesn't warn again.

    As before the rename, the subclass's spec block name is its own class name.
    """
    old = _import_old_class()
    with pytest.warns(HarnessDeprecationWarning, match=_CLASS_WARNING):

        @dataclass(init=False)
        class MyDocker(old):
            pass

    assert MyDocker.get_serialization_name() == 'MyDocker'
    agent = Agent.from_spec(
        {'model': 'test', 'capabilities': [{'MyDocker': {'image': 'python:3.13-slim'}}]},
        custom_capability_types=[MyDocker],
    )
    assert [type(c) for c in agent.root_capability.capabilities if isinstance(c, DockerContainer)] == [MyDocker]


def test_old_spec_block_name_loads() -> None:
    old = _import_old_class()
    with pytest.warns(HarnessDeprecationWarning, match=_CLASS_WARNING):
        agent = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'DockerSandbox': {'image': 'python:3.13-slim'}}]},
            custom_capability_types=[old],
        )

    assert [c.image for c in agent.root_capability.capabilities if isinstance(c, DockerContainer)] == [
        'python:3.13-slim'
    ]
