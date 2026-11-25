"""C4's run-level transformations: a `ModelFallback` switch, compaction and history processors.

`test_history_transformations.py` merges `RUN_LEVEL_TRANSFORMS` into its registry, so each one runs every case and
probe through the same three checks as the cheap transforms there. Each one turns the case's native history into the
history a real run would go on with: the fallback switch reruns the case with the primary model failing from its second
request, and the others apply the real compaction strategy, provider trim or history-processor runner to the native
history, with a `RunContext` of their own. Not VCR tests: every model here is a `FunctionModel`, and the provider trim
runs offline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    TextPart,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls, FunctionModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import SlidingWindowCompaction, SummarizingCompaction
from pydantic_ai_harness.compaction._receipts import is_receipt_part

from ..append_only import drain_events
from .test_capability_routes import RouteRun, _built, _Script  # pyright: ignore[reportPrivateUsage]
from .test_history_transformations import Check, Exempt, NativeRun, Transform, Transformation


def _context() -> RunContext[None]:
    return RunContext(deps=None, model=FunctionModel(_summarize), usage=RunUsage(), run_step=1)


def _standing(part: ModelRequestPart) -> bool:
    """Everything but the standing system prompt, which a compaction keeps ahead of what it replaced."""
    return not isinstance(part, SystemPromptPart)


# --- model_fallback_switch -----------------------------------------------------------------------------------------


async def _model_fallback_switch(native: RouteRun, directory: Path) -> list[ModelMessage]:
    """The case rerun with a `FallbackModel` whose primary fails from its second request, so the backup finishes it."""
    assert isinstance(native, NativeRun)
    script = _Script(native.calls.made)

    def failing(step: int) -> None:
        if step >= 1:
            raise ModelHTTPError(status_code=503, model_name='primary', body='overloaded')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        failing(sum(isinstance(message, ModelResponse) for message in messages))
        return script.respond(messages, info)

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        failing(sum(isinstance(message, ModelResponse) for message in messages))
        async for chunk in script.stream(messages, info):
            yield chunk

    primary = FunctionModel(respond, stream_function=stream, model_name='primary')
    capabilities, observer = await _built(native.calls, directory)
    agent = Agent(FallbackModel(primary, script.model), deps_type=type(None), capabilities=[*capabilities, observer])
    result = await agent.run('go', event_stream_handler=drain_events)
    return result.all_messages()


# --- summarizing and sliding_window --------------------------------------------------------------------------------


def _summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('The user asked for some tool calls, and they ran.')])


async def _summarizing(native: RouteRun, directory: Path) -> list[ModelMessage]:
    """`SummarizingCompaction` keeping only the newest message, with a receipt for what it summarized."""
    compaction = SummarizingCompaction[None](model=FunctionModel(_summarize), keep_messages=1, receipts=True)
    return await compaction.compact(list(native.messages), _context())


async def _sliding_window(native: RouteRun, directory: Path) -> list[ModelMessage]:
    """`SlidingWindowCompaction` keeping only the newest message, with a receipt for what it dropped."""
    return await SlidingWindowCompaction[None](keep_messages=1, receipts=True).compact(
        list(native.messages), _context()
    )


# --- compaction_part -----------------------------------------------------------------------------------------------


async def _compaction_part(native: RouteRun, directory: Path) -> list[ModelMessage]:
    """What Anthropic is sent when its compaction block comes back before the case's last request.

    The block is a `CompactionPart` in a response, so it's no request part to compare: what's checked is the request
    parts the provider trim keeps around it. The trim is called directly: a public run renders it into Anthropic's wire
    format, where the registry's checks, which compare `ModelRequestPart`s, can't read the parts it kept.
    """
    *before, last = native.messages
    block = ModelResponse(
        parts=[CompactionPart(content='Earlier turns, summarized.', provider_name='anthropic')],
        provider_name='anthropic',
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key='test'))
    return model._trim_before_compaction([*before, block, last])  # pyright: ignore[reportPrivateUsage]


# --- history_processor ---------------------------------------------------------------------------------------------


PROCESSED: list[int] = []
"""How many messages each call of the processor was given, so a test can tell it ran."""


def _rebuilt(ctx: RunContext[None], messages: list[ModelMessage]) -> list[ModelMessage]:
    """A user's processor that returns a rebuilt copy, as a filtering one does, with nothing filtered out.

    Sync and taking the context, so the runner takes its executor path. What's checked is that the runner and
    `ProcessHistory` hand every part on as it was.
    """
    PROCESSED.append(len(messages))
    return [replace(message, parts=list(message.parts)) for message in messages]


async def _history_processor(native: RouteRun, directory: Path) -> list[ModelMessage]:
    """The case rerun with a `ProcessHistory` processor, which also rewrites the history the run keeps."""
    assert isinstance(native, NativeRun)
    script = _Script(native.calls.made)
    capabilities, observer = await _built(native.calls, directory)
    processor = ProcessHistory[None](_rebuilt)
    agent = Agent(script.model, deps_type=type(None), capabilities=[*capabilities, observer, processor])
    result = await agent.run('go', event_stream_handler=drain_events)
    return result.all_messages()


_RECEIPT_IN_USER_VOICE: dict[Check, Exempt] = {
    'no_authority_gain': Exempt(
        "Not filed yet: a compaction receipt is a `UserPromptPart`, so the harness's note about the history it "
        "replaced reaches the model in the user's voice with 'user' authority, above the tool output it stands in "
        'for. Being verified by the compaction and voice-authority sessions.'
    )
}
"""The receipt finding, on every case and probe: only the no-authority-gain check fails."""


RUN_LEVEL_TRANSFORMS: dict[Transformation, Transform] = {
    'model_fallback_switch': Transform(
        'model_fallback_switch',
        _model_fallback_switch,
    ),
    'summarizing': Transform(
        'summarizing',
        _summarizing,
        drops=_standing,
        # The summary, and the receipt naming what it replaced: a `UserPromptPart` told apart from kept prompts.
        creates_part=lambda part: isinstance(part, CompactionPart) or is_receipt_part(part),
        known=_RECEIPT_IN_USER_VOICE,
        functions=(
            'pydantic_ai_harness.compaction._summarizing_compaction.SummarizingCompaction.compact',
            'pydantic_ai_harness.compaction._summarizing_compaction.SummarizingCompaction._insert_receipt',
        ),
    ),
    'sliding_window': Transform(
        'sliding_window',
        _sliding_window,
        drops=_standing,
        creates_part=is_receipt_part,
        known=_RECEIPT_IN_USER_VOICE,
        functions=(
            'pydantic_ai_harness.compaction._sliding_window_compaction.SlidingWindowCompaction.compact',
            'pydantic_ai_harness.compaction._sliding_window_compaction.SlidingWindowCompaction._prepend_receipt',
            'pydantic_ai_harness.compaction._sliding_window_compaction.SlidingWindowCompaction._without_receipts',
            'pydantic_ai_harness.compaction._sliding_window_compaction.SlidingWindowCompaction._dropped_messages',
        ),
    ),
    'compaction_part': Transform(
        'compaction_part',
        _compaction_part,
        drops=_standing,
        functions=(
            'pydantic_ai.models.Model._trim_before_compaction',
            'pydantic_ai.models._trim_messages_before_compaction',
        ),
    ),
    'history_processor': Transform(
        'history_processor',
        _history_processor,
        functions=('pydantic_ai.capabilities.process_history._run_history_processor',),
    ),
}
