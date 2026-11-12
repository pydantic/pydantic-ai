"""Every workspace-supplying harness capability takes `authority` and `path_authority`, in Python and in specs.

Not a VCR test: it asserts constructor and `from_spec` signatures, which no provider request exercises.
"""

from __future__ import annotations

import inspect

import pytest

import pydantic_ai_harness
from pydantic_ai.capabilities import AbstractCapability

CAPABILITIES = [
    'BoxdSandbox',
    'BoxLiteSandbox',
    'DaytonaSandbox',
    'DockerSandbox',
    'E2BSandbox',
    'ModalSandbox',
    'OpenSandbox',
    'RenderSandbox',
    'SmolSandbox',
    'SpritesSandbox',
    'SSHWorkspace',
]


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
