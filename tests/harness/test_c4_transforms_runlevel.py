"""The run-level transforms in `c4_transforms_runlevel.py` do what they claim, so the C4 checks over them bite.

A transform that silently changed nothing would pass every check. Each test here runs a probe's native history through
one transform and checks it was transformed: the fallback switched models, compaction replaced history with a summary
or a window, the provider trim cut at the boundary, the processor dropped the model's text. Not a VCR test: every
model is a `FunctionModel`, and the provider trim runs offline.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai.messages import CompactionPart, ModelRequest, ModelResponse, SystemPromptPart, TextPart

from .test_capability_routes import PROBES, _native  # pyright: ignore[reportPrivateUsage]
from .test_history_transformations import TRANSFORMS, NativeRun

_PROBE = next(iter(PROBES))


async def _transformed(transformation: str, directory: Path) -> tuple[NativeRun, list[object]]:
    calls = PROBES[_PROBE]
    run = await _native(calls, directory / 'native')
    native = NativeRun(run.messages, run.ran, _PROBE, calls)
    output = await TRANSFORMS[transformation].apply(native, directory / transformation)  # pyright: ignore[reportArgumentType]
    return native, list(output)


async def test_the_fallback_switch_finishes_on_the_backup(tmp_path: Path) -> None:
    native, output = await _transformed('model_fallback_switch', tmp_path)
    responses = [message for message in output if isinstance(message, ModelResponse)]
    assert len(responses) == sum(isinstance(message, ModelResponse) for message in native.messages)
    assert [response.model_name for response in responses] == [
        'primary',
        *['function:respond:stream'] * (len(responses) - 1),
    ]


async def test_summarizing_replaces_history_with_a_summary(tmp_path: Path) -> None:
    native, output = await _transformed('summarizing', tmp_path)
    assert len(output) < len(native.messages)
    summaries = [
        part
        for message in output
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, CompactionPart)
    ]
    assert [part.content for part in summaries] == ['The user asked for some tool calls, and they ran.']


async def test_the_sliding_window_keeps_the_newest_messages(tmp_path: Path) -> None:
    native, output = await _transformed('sliding_window', tmp_path)
    assert output[-1] == native.messages[-1]
    assert len(output) < len(native.messages)


async def test_the_provider_trim_cuts_at_the_compaction_block(tmp_path: Path) -> None:
    native, output = await _transformed('compaction_part', tmp_path)
    assert output[-1] == native.messages[-1]
    assert isinstance(output[-2], ModelResponse) and isinstance(output[-2].parts[0], CompactionPart)
    before = output[:-2]
    assert all(
        isinstance(part, SystemPromptPart)
        for message in before
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_the_processor_drops_the_models_text(tmp_path: Path) -> None:
    native, output = await _transformed('history_processor', tmp_path)
    assert any(
        isinstance(part, TextPart)
        for message in native.messages
        if isinstance(message, ModelResponse)
        for part in message.parts
    )
    assert not any(
        isinstance(part, TextPart) for message in output if isinstance(message, ModelResponse) for part in message.parts
    )
    assert [message for message in output if isinstance(message, ModelRequest)] == [
        message for message in native.messages if isinstance(message, ModelRequest)
    ]
