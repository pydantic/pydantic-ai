"""Every workspace-supplying harness capability takes `authority` and `path_authority`, in Python and in specs.

Not a VCR test: it asserts constructor and `from_spec` signatures, and the workspace `get_workspace` builds
without I/O, which no provider request exercises.
"""

from __future__ import annotations

import inspect

import pytest

import pydantic_ai_harness
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import AuthorityWorkspace

CAPABILITIES: dict[str, dict[str, str]] = {
    'BoxdSandbox': {},
    'BoxLiteSandbox': {},
    'DaytonaSandbox': {},
    'DockerContainer': {'image': 'python:3.13-slim'},
    'E2BSandbox': {},
    'KubernetesSandbox': {'warm_pool': 'pool'},
    'ModalSandbox': {},
    'OpenSandbox': {'image': 'python:3.13-slim'},
    'RenderSandbox': {},
    'SmolSandbox': {},
    'SpritesSandbox': {},
    'SSHWorkspace': {'destination': 'user@host', 'working_dir': '/work'},
}
"""Each capability's name, with the spec arguments it requires."""


def _capability(name: str) -> type[AbstractCapability[object]]:
    try:
        return getattr(pydantic_ai_harness, name)
    except ImportError:  # pragma: lax no cover - only without the capability's extra
        pytest.skip(f'{name} needs its extra')


@pytest.mark.parametrize('name', CAPABILITIES)
def test_constructor_and_spec_take_authority(name: str) -> None:
    capability = _capability(name)
    # An inherited `from_spec` passes its arguments through to `__init__`; an explicit one names them.
    explicit_spec = [capability.from_spec] if 'from_spec' in vars(capability) else []
    for function in (capability.__init__, *explicit_spec):
        parameters = inspect.signature(function).parameters
        assert {'authority', 'path_authority'} <= parameters.keys(), function


@pytest.mark.parametrize('name', CAPABILITIES)
def test_authority_wraps_the_runs_workspace(name: str) -> None:
    capability = _capability(name).from_spec(
        **CAPABILITIES[name], authority='none', path_authority={'/work/own': 'operator'}
    )
    workspace = capability.get_workspace(RunContext(deps=None, model=TestModel(), usage=RunUsage()), ref=None)
    assert isinstance(workspace, AuthorityWorkspace)
    assert workspace.durable_policy() == ('none', {'/work/own': 'operator'})
