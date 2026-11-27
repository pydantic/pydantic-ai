"""The run-level transforms in `c4_transforms_runlevel.py` do what they claim, so the C4 checks over them bite.

A transform that silently changed nothing would pass every check. Each test here runs a probe's native history through
one transform and checks it was transformed: the fallback switched models, compaction replaced history with a summary
or a window, the provider trim cut at the boundary, the processor ran before every request. Not a VCR test: every
model is a `FunctionModel`, and the provider trim runs offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic_ai import RunContext
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelRequest, ModelResponse, SystemPromptPart

from .c4_transforms_runlevel import is_receipt, rebuilt, run_with_processor
from .test_capability_routes import PROBES, _native  # pyright: ignore[reportPrivateUsage]
from .test_history_transformations import TRANSFORMS, NativeRun, Signature, Transformation, signature

_PROBE = next(iter(PROBES))


def _signatures(messages: Sequence[ModelMessage]) -> list[Signature]:
    return [signature(part) for message in messages if isinstance(message, ModelRequest) for part in message.parts]


async def _transformed(transformation: Transformation, directory: Path) -> tuple[NativeRun, list[ModelMessage]]:
    calls = PROBES[_PROBE]
    run = await _native(calls, directory / 'native')
    native = NativeRun(run.messages, run.ran, _PROBE, calls)
    output = await TRANSFORMS[transformation].apply(native, directory / transformation)
    return native, list(output)


async def test_the_fallback_switch_finishes_on_the_backup(tmp_path: Path) -> None:
    native, output = await _transformed('model_fallback_switch', tmp_path)
    responses = [message for message in output if isinstance(message, ModelResponse)]
    assert len(responses) == sum(isinstance(message, ModelResponse) for message in native.messages)
    first, *rest = [response.model_name for response in responses]
    assert first == 'primary'
    assert rest and 'primary' not in rest


async def test_summarizing_replaces_history_with_a_summary(tmp_path: Path) -> None:
    native, output = await _transformed('summarizing', tmp_path)
    assert native.messages[2] not in output  # the tool call's result
    assert any(is_receipt(part) for message in output if isinstance(message, ModelRequest) for part in message.parts)
    summaries = [
        part
        for message in output
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, CompactionPart)
    ]
    assert [part.content for part in summaries] == ['The user asked for some tool calls, and they ran.']


async def test_the_sliding_window_keeps_the_newest_messages(tmp_path: Path) -> None:
    """The tool call and its result make way for a receipt naming what was dropped; the first prompt stays."""
    native, output = await _transformed('sliding_window', tmp_path)
    assert native.messages[0] in output and output[-1] == native.messages[-1]
    assert native.messages[1] not in output and native.messages[2] not in output
    assert any(is_receipt(part) for message in output if isinstance(message, ModelRequest) for part in message.parts)


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


async def test_the_processor_runs_before_every_request(tmp_path: Path) -> None:
    calls = PROBES[_PROBE]
    run = await _native(calls, tmp_path / 'native')
    native = NativeRun(run.messages, run.ran, _PROBE, calls)
    given: list[int] = []

    def counted(ctx: RunContext[None], messages: list[ModelMessage]) -> list[ModelMessage]:
        given.append(len(messages))
        return rebuilt(ctx, messages)

    output = await run_with_processor(native, tmp_path / 'processed', counted)
    assert len(given) == sum(isinstance(message, ModelResponse) for message in native.messages)
    assert _signatures(output) == _signatures(native.messages)
