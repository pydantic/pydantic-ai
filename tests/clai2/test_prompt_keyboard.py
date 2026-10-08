"""Modified keyboard protocols must preserve editing, submission, and handoff."""

import io

import pytest

from tests.clai2.test_live_prompt import editor


@pytest.mark.parametrize('enter', ['\r', '\x1b[13u', '\x1b[57414u'])
@pytest.mark.parametrize('newline', ['\x1b[13;2u', '\x1b[27;2;13~', '\x1b[57414;2u', '\n', '\x1b[106;5u'])
async def test_newline_does_not_submit_until_plain_enter(newline: str, enter: str) -> None:
    async with editor() as (live, pipe, _):
        pipe.send_text(f'first{newline}second')
        live.keys.read()
        assert live.buffer.text == 'first\nsecond'
        assert live.queued_messages == ()
        pipe.send_text(enter)
        assert await live.read() == 'first\nsecond'
        assert live.queued_messages == ()
        assert live.buffer.text == ''


@pytest.mark.parametrize('modifier', [65, 129])
async def test_kitty_navigation_with_lock_keys(modifier: int) -> None:
    async with editor() as (live, pipe, _):
        pipe.send_text(f'ab\x1b[1;{modifier}DX')
        live.keys.read()
        assert live.buffer.text == 'aXb'


@pytest.mark.parametrize('modifier', [3, 67, 131])
async def test_alt_delete_does_not_delete_unmodified_text(modifier: int) -> None:
    async with editor() as (live, pipe, _):
        pipe.send_text(f'abc\x1b[D\x1b[D\x1b[3;{modifier}~')
        live.keys.read()
        assert live.buffer.text == 'abc'
        assert live.buffer.cursor == 1


async def test_kitty_control_shortcuts_on_non_latin_layouts() -> None:
    async with editor() as (live, pipe, _):
        pipe.send_text('discard\x1b[1089::99;5u')
        live.keys.read()
        assert live.buffer.text == ''
        with pytest.raises(KeyboardInterrupt):
            await live.read()


async def test_kitty_shortcuts_remain_usable() -> None:
    async with editor() as (live, pipe, _):
        pipe.send_text('discard\x1b[99;5u')
        live.keys.read()
        assert live.buffer.text == ''
        with pytest.raises(KeyboardInterrupt):
            await live.read()
        pipe.send_text('keep\x1b[99;6u')
        live.keys.read()
        assert live.buffer.text == 'keep'
        pipe.send_text('\x1b[99;5u\x1b[100;5u')
        with pytest.raises(KeyboardInterrupt):
            await live.read()
        with pytest.raises(EOFError):
            await live.read()


@pytest.mark.parametrize('fail', [False, True])
async def test_keyboard_protocols_are_released_for_menus_and_on_exit(fail: bool) -> None:
    enable = '\x1b[>4;1m\x1b[>5u'
    disable = '\x1b[<u\x1b[>4;0m'
    output = io.StringIO()
    with pytest.RaisesGroup(RuntimeError):
        async with editor(output=output) as (live, pipe, _):
            assert output.getvalue().count(enable) == 1
            try:
                async with live.suspended():
                    assert output.getvalue().count(disable) == 1
                    if fail:
                        raise ValueError('menu failed')
            except ValueError:
                assert fail
            assert output.getvalue().count(enable) == 2
            pipe.send_text('first\x1b[13;2usecond\r')
            assert await live.read() == 'first\nsecond'
            raise RuntimeError('editor failed')
    assert output.getvalue().count(disable) == 2
