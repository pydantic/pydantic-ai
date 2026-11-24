"""What the live `ComputerUse` tests of each provider share. Each test module imports `vcr_config` to use it."""

from __future__ import annotations

from typing import TypeGuard

import pytest

from pydantic_ai.computer_use import Click, TypeText


@pytest.fixture(scope='module')
def vcr_config(vcr_config: dict[str, object], pytestconfig: pytest.Config) -> dict[str, object]:
    """On replay, match each request's body too, so a test fails if the code stops sending the recorded request.

    Recording keeps the default matching. With the body matcher, cassetter writes the interactions out of order.
    """
    if pytestconfig.getoption('--record-mode', default=None) not in (None, 'none'):
        return vcr_config
    return {**vcr_config, 'match_on': ['method', 'uri', 'body']}


TASK = 'Take a screenshot, then click at x=100 y=200, type "hello", and reply "done". Do nothing else.'
DONE = [Click(x=100, y=200), TypeText(text='hello')]


def is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)
