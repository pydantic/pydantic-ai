"""Decoder attachment and terminal-input handoff, without a renderer."""

from collections.abc import Callable, Generator
from contextlib import contextmanager

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys

from pydantic_clai2.ui.prompt.prompt_keys import PromptKeys


async def test_decoding_meta_paste_arrows_and_lone_escape() -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        keys.dispatch(KeyPress(Keys.ControlLeft, ''))
        keys.dispatch(KeyPress(Keys.BackTab, ''))
        keys.dispatch(KeyPress(Keys.Escape, '\x1b'))
        keys.dispatch(KeyPress('v', 'v'))
        keys.dispatch(KeyPress(Keys.BracketedPaste, 'one\ntwo'))
        keys.dispatch(KeyPress(Keys.Escape, '\x1b'))
        keys.flush()
        keys.flush()
        assert events == [('ctrl-left', ''), ('backtab', ''), ('alt-v', 'v'), ('paste', 'one\ntwo'), ('escape', '')]
        keys.start()
        keys.stop()
        keys.stop()


async def test_attach_failure_unwinds_raw_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    restored: list[bool] = []

    @contextmanager
    def raw_mode() -> Generator[None]:
        try:
            yield
        finally:
            restored.append(True)

    @contextmanager
    def fail(callback: Callable[[], None]) -> Generator[None]:
        raise OSError('attach failed')
        yield  # pragma: no cover -- makes this a context manager that fails on entry.

    with create_pipe_input() as pipe:
        monkeypatch.setattr(pipe, 'raw_mode', raw_mode)
        monkeypatch.setattr(pipe, 'attach', fail)
        keys = PromptKeys(source=pipe, feed=lambda key, data: None, eof=lambda: None)
        with pytest.raises(OSError, match='attach failed'):
            keys.start()
        keys.stop()
    assert restored == [True]


@pytest.mark.parametrize('sequence', ['\x1b[13;2u', '\x1b[27;2;13~'])
async def test_shift_enter_through_actual_decoder_with_split_input(sequence: str) -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        try:
            for char in sequence:
                pipe.send_text(char)
                keys.read()
            assert events == [('shift-enter', sequence)]
        finally:
            keys.stop()


def test_csi_partial_unknown_and_oversized_sequences_do_not_become_draft_text() -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        for sequence in ('\x1b[13;', '\x1b[999u', '\x1b[' + '9' * 31):
            for char in sequence:
                keys.dispatch(KeyPress(Keys.Escape if char == '\x1b' else char, char))
            keys.flush()
        assert events == []
        keys.dispatch(KeyPress(Keys.ControlM, '\x1b[13;2u'))
        assert events == [('shift-enter', '\x1b[13;2u')]
        keys.dispatch(KeyPress(Keys.BracketedPaste, '\x1b[13;2u'))
        assert events[-1] == ('paste', '\x1b[13;2u')
        keys.dispatch(KeyPress(Keys.Escape, '\x1b'))
        keys.dispatch(KeyPress('[', '['))
        keys.dispatch(KeyPress(Keys.Left, '\x1b[D'))
        assert len(events) == 2


@pytest.mark.parametrize(
    'sequence',
    ['\x1b\x7f', '\x1b\x08', '\x1b[27;3;127~', '\x1b[27;3;8~', '\x1b[127;3u', '\x1b[8;3u'],
)
@pytest.mark.parametrize('split', [False, True])
async def test_alt_backspace_through_actual_decoder(sequence: str, split: bool) -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        try:
            for chunk in sequence if split else [sequence]:
                pipe.send_text(chunk)
                keys.read()
            assert [key for key, _ in events] == ['alt-backspace']
        finally:
            keys.stop()


@pytest.mark.parametrize('sequence', ['\x1b[27;3;127~', '\x1b[27;3;8~', '\x1b[127;3u', '\x1b[8;3u'])
def test_modified_alt_backspace_tokens_and_literal_paste(sequence: str) -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        keys.dispatch(KeyPress(Keys.ControlH, sequence))
        keys.dispatch(KeyPress(Keys.BracketedPaste, sequence))
        assert events == [('alt-backspace', sequence), ('paste', sequence)]


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('report', ['\x1b[{code};{modifier}u', '\x1b[27;{modifier};{code}~'])
@pytest.mark.parametrize(
    ('code', 'modifier', 'expected'),
    [
        (13, 1, 'enter'),
        (13, 2, 'shift-enter'),
        (13, 3, 'alt-enter'),
        (13, 5, 'enter'),
        (13, 6, 'enter'),
        (13, 66, 'shift-enter'),
        (13, 130, 'shift-enter'),
        (9, 1, 'tab'),
        (9, 2, 'backtab'),
        (27, 1, 'escape'),
        (127, 1, 'backspace'),
        (127, 2, 'backspace'),
        (127, 5, 'backspace'),
        (32, 2, ' '),
        (91, 5, 'escape'),
        (99, 5, 'ctrl-c'),
        (99, 6, 'ctrl-shift-c'),
        (100, 5, 'ctrl-d'),
        (104, 5, 'backspace'),
        (105, 5, 'tab'),
        (106, 5, 'ctrl-j'),
        (109, 5, 'enter'),
        (114, 5, 'ctrl-r'),
        (120, 5, 'ctrl-x'),
        (115, 5, 'ctrl-s'),
        (118, 3, 'alt-v'),
        (98, 3, 'alt-b'),
        (102, 3, 'alt-f'),
        (57414, 1, 'enter'),
        (57414, 2, 'shift-enter'),
        (57417, 1, 'left'),
        (57418, 1, 'right'),
        (57419, 1, 'up'),
        (57420, 1, 'down'),
        (57421, 1, 'pageup'),
        (57422, 1, 'pagedown'),
        (57423, 1, 'home'),
        (57424, 1, 'end'),
        (57425, 1, 'insert'),
        (57426, 1, 'delete'),
        (57427, 1, 'begin'),
    ],
)
async def test_modified_reporting_preserves_editor_keys(
    code: int, modifier: int, expected: str, report: str, split: bool
) -> None:
    events: list[tuple[str, str]] = []
    sequence = report.format(code=code, modifier=modifier)
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        try:
            for chunk in sequence if split else [sequence]:
                pipe.send_text(chunk)
                keys.read()
            assert events == [(expected, sequence)]
            keys.dispatch(KeyPress(Keys.BracketedPaste, sequence))
            assert events[-1] == ('paste', sequence)
        finally:
            keys.stop()


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('lock', [0, 64, 128])
@pytest.mark.parametrize(
    ('code', 'modifier', 'suffix', 'expected'),
    [
        (1, 1, 'A', 'up'),
        (1, 1, 'B', 'down'),
        (1, 1, 'C', 'right'),
        (1, 1, 'D', 'left'),
        (1, 1, 'H', 'home'),
        (1, 1, 'F', 'end'),
        (2, 1, '~', 'insert'),
        (3, 1, '~', 'delete'),
        (5, 1, '~', 'pageup'),
        (6, 1, '~', 'pagedown'),
        (1, 5, 'D', 'ctrl-left'),
        (1, 3, 'C', 'alt-right'),
        (1, 3, 'D', 'alt-left'),
        (3, 3, '~', 'alt-delete'),
    ],
)
async def test_navigation_reports_with_lock_modifiers(
    code: int, modifier: int, suffix: str, expected: str, lock: int, split: bool
) -> None:
    events: list[str] = []
    sequence = f'\x1b[{code};{modifier + lock}{suffix}'
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append(key), eof=lambda: None)
        try:
            for chunk in sequence if split else [sequence]:
                pipe.send_text(chunk)
                keys.read()
            assert events == [expected]
        finally:
            keys.stop()


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize(
    ('sequence', 'expected'),
    [
        ('\x1b[1089::99;5u', 'ctrl-c'),
        ('\x1b[1089:1057:99;69u', 'ctrl-c'),
        ('\x1b[1089:1057:99;6u', 'ctrl-shift-c'),
        ('\x1b[1074::100;5u', 'ctrl-d'),
        ('\x1b[106::106;5u', 'ctrl-j'),
        ('\x1b[99:67;6u', 'ctrl-shift-c'),
    ],
)
async def test_alternate_key_reports_preserve_control_shortcuts(sequence: str, expected: str, split: bool) -> None:
    events: list[str] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append(key), eof=lambda: None)
        try:
            for chunk in sequence if split else [sequence]:
                pipe.send_text(chunk)
                keys.read()
            assert events == [expected]
        finally:
            keys.stop()


@pytest.mark.parametrize(
    'sequence',
    ['\x1b[13;0u', '\x1b[99;9u', '\x1b[999u', '\x1b[97u', '\x1b[99;5:3u', '\x1b[1;73D', '\x1b[999;65Z'],
)
async def test_unrequested_or_invalid_reports_are_not_editor_keys(sequence: str) -> None:
    events: list[str] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append(key), eof=lambda: None)
        try:
            pipe.send_text(sequence)
            keys.read()
            keys.flush()
            assert events == []
        finally:
            keys.stop()


async def test_enter_linefeed_and_kitty_escape_stay_distinct() -> None:
    events: list[str] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append(key), eof=lambda: None)
        try:
            pipe.send_text('\r\n\x1b[27u')
            keys.read()
            assert events == ['enter', 'ctrl-j', 'escape']
        finally:
            keys.stop()


async def test_cursor_reports_are_not_draft_keys() -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(
            source=pipe,
            feed=lambda key, data: events.append((key, data)),
            eof=lambda: None,
        )
        try:
            pipe.send_text('\x1b[12;34R')
            keys.read()
            assert events == []
        finally:
            keys.stop()


def test_sgr_mouse_reports_are_one_key() -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        keys.dispatch(KeyPress(Keys.Vt100MouseEvent, '\x1b[<64;10;5M'))
    assert events == [('mouse', '\x1b[<64;10;5M')]
