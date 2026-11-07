"""Telling a light terminal from a dark one by `COLORFGBG`."""

import io

import pytest

from pydantic_clai2.ui.prompt.terminal_background import detect_background


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize(
    ('colorfgbg', 'background'),
    [
        pytest.param('0;15', '#FFFFFF', id='light'),
        pytest.param('15;0', '#000000', id='dark'),
        pytest.param('7;default;8', '#000000', id='rxvt-dark'),
        pytest.param('default;default', None, id='unknown'),
        pytest.param(None, None, id='unset'),
    ],
)
def test_colorfgbg_tells_light_from_dark(
    monkeypatch: pytest.MonkeyPatch, colorfgbg: str | None, background: str | None
) -> None:
    if colorfgbg is None:
        monkeypatch.delenv('COLORFGBG', raising=False)
    else:
        monkeypatch.setenv('COLORFGBG', colorfgbg)
    assert detect_background(stdout=Terminal()) == background


def test_output_that_is_not_a_terminal_has_no_background(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('COLORFGBG', '15;0')
    assert detect_background(stdout=io.StringIO()) is None
