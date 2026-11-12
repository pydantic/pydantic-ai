"""Every workspace-supplying harness capability takes `authority` and `path_authority`, in Python and in specs."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai_harness import (
    BoxdSandbox,
    BoxLiteSandbox,
    DaytonaSandbox,
    DockerSandbox,
    E2BSandbox,
    ModalSandbox,
    OpenSandbox,
    RenderSandbox,
    SmolSandbox,
    SpritesSandbox,
    SSHWorkspace,
)

CAPABILITIES: list[type[AbstractCapability[Any]]] = [
    BoxdSandbox,
    BoxLiteSandbox,
    DaytonaSandbox,
    DockerSandbox,
    E2BSandbox,
    ModalSandbox,
    OpenSandbox,
    RenderSandbox,
    SmolSandbox,
    SpritesSandbox,
    SSHWorkspace,
]


@pytest.mark.parametrize('capability', CAPABILITIES, ids=lambda capability: capability.__name__)
def test_constructor_and_spec_take_authority(capability: type[AbstractCapability[Any]]) -> None:
    # An inherited `from_spec` passes its arguments through to `__init__`; an explicit one names them.
    explicit_spec = [capability.from_spec] if 'from_spec' in vars(capability) else []
    for function in (capability.__init__, *explicit_spec):
        parameters = inspect.signature(function).parameters
        assert {'authority', 'path_authority'} <= parameters.keys(), function


def test_modal_keeps_authority_through_its_own_init() -> None:
    sandbox = ModalSandbox[Any](authority='none', path_authority={'docs': 'user'})
    assert (sandbox.authority, sandbox.path_authority) == ('none', {'docs': 'user'})
