"""Double Escape is an idle-editor gesture, not another cancellation binding."""

import anyio
import pytest
from termflow.tui.completion import Completion

from pydantic_clai2.ui.prompt.live_prompt import PromptRewind
from tests.clai2.test_live_prompt import editor


@pytest.mark.parametrize('split', [False, True])
async def test_double_escape_through_decoder_preserves_draft(split: bool) -> None:
    async with editor() as (live, pipe, _):
        live.buffer.replace('unfinished draft')
        for chunk in ['\x1b', '\x1b'] if split else ['\x1b\x1b']:
            pipe.send_text(chunk)
            live.keys.read()
            if split:
                live.keys.flush()
        live.keys.flush()
        with pytest.raises(PromptRewind):
            await live.read()
        assert live.buffer.text == 'unfinished draft'
        assert live.history.get_strings() == []
        assert not live.interrupts.exit_requested


async def test_escape_timing_intervening_keys_and_repeated_presses() -> None:
    async with editor() as (live, _, _):
        now = 0.0
        live.clock = lambda: now
        live.feed('escape')
        assert 'Esc again' in live.notice
        now = 0.51
        live.feed('escape')
        live.submit('sentinel')
        assert await live.read() == 'sentinel'
        live.feed('left')
        live.feed('escape')
        live.submit('sentinel')
        assert await live.read() == 'sentinel'
        now = 1.01
        live.feed('escape')
        live.feed('escape')
        live.submit('sentinel')
        with pytest.raises(PromptRewind):
            await live.read()
        assert await live.read() == 'sentinel'


@pytest.mark.parametrize('context', ['completion', 'pending', 'search', 'queue', 'suspended'])
async def test_dismissing_other_ui_does_not_arm_rewind(context: str) -> None:
    async with editor() as (live, _, _):
        if context == 'completion':
            live._completions = [Completion('help')]  # pyright: ignore[reportPrivateUsage]
        elif context == 'pending':
            live._completion_pending = True  # pyright: ignore[reportPrivateUsage]
        elif context == 'search':
            live.feed('ctrl-r')
        elif context == 'queue':
            live.submit('queued')
        if context == 'suspended':
            live.feed('escape')
            async with live.suspended():
                pass
        else:
            live.feed('escape')
        if context == 'queue':
            assert await live.read() == 'queued'
        live.feed('escape')
        live.submit('sentinel')
        assert await live.read() == 'sentinel'
        live.feed('escape')
        with pytest.raises(PromptRewind):
            await live.read()


async def test_active_escape_cancels_without_requesting_rewind_or_exit() -> None:
    async with editor() as (live, _, _):
        started, finished = anyio.Event(), anyio.Event()

        async def operation() -> None:
            started.set()
            await anyio.sleep_forever()

        async def run() -> None:
            assert not await live.interrupts.run(operation())
            finished.set()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            await started.wait()
            live.feed('escape')
            live.feed('escape')
            await finished.wait()
        assert not live.interrupts.exit_requested
        live.feed('escape')
        live.submit('sentinel')
        assert await live.read() == 'sentinel'
        live.feed('escape')
        with pytest.raises(PromptRewind):
            await live.read()


@pytest.mark.parametrize('sequence', ['\x1b[D', '\x1b\x7f', '\x1b[13;2u', '\x1b[200~\x1b\x1b\x1b[201~'])
async def test_alt_csi_and_paste_do_not_count_as_escapes(sequence: str) -> None:
    async with editor() as (live, pipe, _):
        pipe.send_text(sequence)
        live.keys.read()
        live.keys.flush()
        live.dismiss_completions()
        live.feed('escape')
        live.submit('sentinel')
        assert await live.read() == 'sentinel'
