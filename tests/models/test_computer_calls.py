"""Tests for the X11 key names that Claude's and Gemini's computer tools send.

Not VCR tests: `key_from_x11` is a pure mapping that makes no provider request, so a recording would add nothing.
"""

from __future__ import annotations

import pytest

from pydantic_ai.models._computer_calls import key_from_x11


@pytest.mark.parametrize(
    'name, key',
    [('Return', 'enter'), ('Control_L', 'ctrl'), ('KP_5', '5'), ('Page_Down', 'pagedown'), ('Escape', 'escape')],
)
def test_x11_key_names_become_the_computer_tools_names(name: str, key: str) -> None:
    # Claude and Gemini name keys as X11 does. A name the tool spells the same way is only lower-cased.
    assert key_from_x11(name) == key
