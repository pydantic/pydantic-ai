"""The `process_history` capability hook: where it runs in a request step, and what it writes back to history.

Not VCR tests: what's under test is core's request step, with `FunctionModel`s standing in for a provider, and no
provider sees anything a recording would pin.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import (
    AbstractCapability,
    CombinedCapability,
    CompactionRequest,
    CompactionResult,
    Hooks,
    ModelFallback,
    ProcessHistory,
    WrapperCapability,
)
from pydantic_ai.exceptions import ModelAPIError, RetryModelRequest
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets._deferred_capability_loader import LOAD_CAPABILITY_TOOL_NAME

from ._inline_snapshot import snapshot


def texts(messages: list[ModelMessage]) -> list[str]:
    """The text of each user prompt and text part, in order."""
    return [
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart | TextPart) and isinstance(part.content, str)
    ]


def recording(seen: list[list[ModelMessage]], reply: str = 'done') -> FunctionModel:
    """A model that records the messages each request sends, and answers with text."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart(reply)])

    return FunctionModel(respond)


def shouted(message: ModelMessage) -> ModelMessage:
    """A copy of a message with its text parts in capitals, with parts of its own."""
    if isinstance(message, ModelRequest):
        return replace(
            message,
            parts=[
                replace(part, content=part.content.upper())
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
                else part
                for part in message.parts
            ],
        )
    return replace(
        message,
        parts=[
            replace(part, content=part.content.upper()) if isinstance(part, TextPart) else part
            for part in message.parts
        ],
    )


HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart('first')]),
    ModelResponse(parts=[TextPart('a')]),
]


@dataclass
class Logged(AbstractCapability[Any]):
    """Logs its `process_history` and `before_model_request` calls, with the messages each one got."""

    name: str
    log: list[str]
    processes: bool = True

    async def process_history(self, ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        self.log.append(f'{self.name}.process_history {texts(messages)}')
        *earlier, last = messages
        assert isinstance(last, ModelRequest)
        return [*earlier, replace(last, parts=[*last.parts, UserPromptPart(self.name)])]

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.log.append(f'{self.name}.before_model_request {texts(request_context.messages)}')
        return request_context


@dataclass
class OnlyBefore(AbstractCapability[Any]):
    """Logs `before_model_request`, and adds a note to what this request alone sends."""

    name: str
    log: list[str]

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.log.append(f'{self.name}.before_model_request {texts(request_context.messages)}')
        request_context.messages = [*request_context.messages, ModelRequest(parts=[UserPromptPart('note')])]
        return request_context


@dataclass
class LoggingWrapper(WrapperCapability[Any]):
    """A wrapper with a `before_model_request` of its own, around its wrapped capabilities' hooks."""

    log: list[str] = field(kw_only=True)

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.log.append('container.before_model_request')
        return await super().before_model_request(ctx, request_context)


def _plain(a: AbstractCapability[Any], b: AbstractCapability[Any]) -> list[AbstractCapability[Any]]:
    return [a, b]


def _each_wrapped(a: AbstractCapability[Any], b: AbstractCapability[Any]) -> list[AbstractCapability[Any]]:
    return [WrapperCapability(a), WrapperCapability(b)]


def _both_wrapped(a: AbstractCapability[Any], b: AbstractCapability[Any]) -> list[AbstractCapability[Any]]:
    return [WrapperCapability(CombinedCapability([a, b]))]


@pytest.mark.parametrize('arrange', [_plain, _each_wrapped, _both_wrapped])
async def test_each_capability_processes_history_right_before_its_before_model_request(arrange: Any):
    """At each capability's position: its `process_history`, then its `before_model_request`, through wrappers too.

    A processor gets what the hooks ahead of it left, including a later capability's request-only note, and its
    result is what's sent and kept.
    """
    log: list[str] = []
    seen: list[list[ModelMessage]] = []
    capabilities = [
        Logged('one', log),
        *arrange(OnlyBefore('two', log), Logged('three', log)),
    ]
    result = await Agent(recording(seen), capabilities=capabilities).run('hi')

    assert log == snapshot(
        [
            "one.process_history ['hi']",
            "one.before_model_request ['hi', 'one']",
            "two.before_model_request ['hi', 'one']",
            "three.process_history ['hi', 'one', 'note']",
            "three.before_model_request ['hi', 'one', 'note', 'three']",
        ]
    )
    assert texts(seen[0]) == ['hi', 'one', 'note', 'three']
    # `three` made the note part of the history it returned.
    assert texts(result.all_messages()) == ['hi', 'one', 'note', 'three', 'done']


@dataclass
class LoggingCombination(CombinedCapability[Any]):
    """A combination with a `before_model_request` of its own, around its children's hooks."""

    log: list[str] = field(default_factory=list[str], kw_only=True)

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.log.append('container.before_model_request')
        return await super().before_model_request(ctx, request_context)


@pytest.mark.parametrize('container', ['wrapper', 'combination'])
async def test_a_container_with_its_own_before_model_request_processes_first(container: str):
    """A wrapper or combination that overrides one of the two hooks runs its children's processors, then its own hook.

    The combination is wrapped, since an agent dispatches to the children of a combination it's given directly.
    """
    log: list[str] = []

    # The deferred child isn't loaded, so neither of its hooks runs.
    async def never(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        raise AssertionError('a deferred capability that was never loaded ran')  # pragma: no cover

    later = Hooks[Any](before_model_request=never, id='later', description='Later.', defer_loading=True)
    children: list[AbstractCapability[Any]] = [Logged('one', log), OnlyBefore('two', log), later]
    capability = (
        LoggingWrapper(CombinedCapability(children), log=log)
        if container == 'wrapper'
        else WrapperCapability(LoggingCombination(children, log=log))
    )
    result = await Agent(recording([]), capabilities=[capability]).run('hi')

    assert log == snapshot(
        [
            "one.process_history ['hi']",
            'container.before_model_request',
            "one.before_model_request ['hi', 'one']",
            "two.before_model_request ['hi', 'one']",
        ]
    )
    assert texts(result.all_messages()) == ['hi', 'one', 'done']


@dataclass
class KeepLast(WrapperCapability[Any]):
    """A wrapper that processes the history itself, keeping only the last message of what `wrapped` returns."""

    async def process_history(self, ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        return (await super().process_history(ctx, messages))[-1:]


async def test_a_wrapper_can_process_the_history_of_a_capability_that_doesnt():
    """The wrapped capability's `process_history` is the default, which returns the history as it was given."""
    log: list[str] = []
    result = await Agent(recording([]), capabilities=[KeepLast(OnlyBefore('one', log))]).run(
        'hi', message_history=HISTORY
    )

    assert log == ["one.before_model_request ['hi']"]
    assert texts(result.all_messages()) == ['hi', 'done']


@dataclass
class Shouting(CombinedCapability[Any]):
    """A combination that processes the history itself, shouting what its children's processors return."""

    async def process_history(self, ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        return [shouted(message) for message in await super().process_history(ctx, messages)]


async def test_a_combination_can_process_the_history_of_children_that_dont():
    """A combination's own `process_history` runs even when none of its children has one."""
    log: list[str] = []
    seen: list[list[ModelMessage]] = []
    capability = WrapperCapability(Shouting([OnlyBefore('one', log)]))
    result = await Agent(recording(seen), capabilities=[capability]).run('hi')

    assert log == ["one.before_model_request ['HI']"]
    assert texts(seen[0]) == ['HI', 'note']
    assert texts(result.all_messages()) == ['HI', 'done']


async def test_processed_history_is_written_back_as_returned():
    """The messages a processor returns are what's sent and what the history holds, new objects and all.

    The next step's processor starts from that history, so it sees its own earlier result.
    """
    seen: list[list[ModelMessage]] = []
    given: list[list[ModelMessage]] = []
    returned: list[list[ModelMessage]] = []

    async def shout(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        given.append(messages)
        returned.append(processed := [shouted(message) for message in messages])
        return processed

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        if len(seen) == 1:
            return ModelResponse(parts=[ToolCallPart('noop', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model), capabilities=[Hooks(process_history=shout)])

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    result = await agent.run('hi', message_history=HISTORY)

    assert texts(given[0]) == ['first', 'a', 'hi']
    assert texts(given[1]) == ['FIRST', 'A', 'HI']
    assert texts(result.all_messages()) == ['FIRST', 'A', 'HI', 'done']
    # The history holds the processor's own objects: the last step's result, and the response after it.
    assert all(kept is new for kept, new in zip(result.all_messages(), returned[1], strict=False))
    assert [texts(messages) for messages in seen] == [['FIRST', 'A', 'HI'], ['FIRST', 'A', 'HI']]


@pytest.mark.parametrize('grouped', [False, True], ids=['deferred', 'deferred_group'])
async def test_a_deferred_capability_processes_history_once_loaded(grouped: bool):
    steps: list[int] = []

    def record(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        steps.append(ctx.run_step)
        return messages

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(LOAD_CAPABILITY_TOOL_NAME, {'id': 'trim'})])
        return ModelResponse(parts=[TextPart('done')])

    trim = (
        CombinedCapability[Any]([Hooks(process_history=record)], id='trim', description='Trims.', defer_loading=True)
        if grouped
        else Hooks(process_history=record, id='trim', description='Trims.', defer_loading=True)
    )
    await Agent(FunctionModel(model), capabilities=[trim]).run('hi')

    assert steps == [2]


async def test_every_fallback_candidate_is_sent_the_history_processed_once():
    calls: list[int] = []
    seen: list[list[ModelMessage]] = []

    def count(messages: list[ModelMessage]) -> list[ModelMessage]:
        calls.append(len(messages))
        return messages[-1:]

    def failure(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        raise ModelAPIError(model_name='primary', message='rate limited')

    agent = Agent(FunctionModel(failure), capabilities=[ProcessHistory(count), ModelFallback(recording(seen))])
    result = await agent.run('hi', message_history=HISTORY)

    assert calls == [3]
    assert [texts(messages) for messages in seen] == [['hi'], ['hi']]
    assert texts(result.all_messages()) == ['hi', 'done']


async def test_every_retried_attempt_is_sent_the_history_processed_once():
    calls: list[int] = []
    seen: list[list[ModelMessage]] = []

    def count(messages: list[ModelMessage]) -> list[ModelMessage]:
        calls.append(len(messages))
        return messages[-1:]

    @dataclass
    class RetryFirstAttempt(AbstractCapability[Any]):
        async def after_model_request(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            if request_context.attempt == 1:
                raise RetryModelRequest()
            return response

    agent = Agent(recording(seen), capabilities=[ProcessHistory(count), RetryFirstAttempt()])
    await agent.run('hi', message_history=HISTORY)

    assert calls == [3]
    assert [texts(messages) for messages in seen] == [['hi'], ['hi']]


@dataclass
class KeepEverything(AbstractCapability[Any]):
    """Compacts by putting a summary ahead of every message it's given, which it keeps as they are."""

    calls: list[CompactionRequest] = field(default_factory=list[CompactionRequest])

    async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
        self.calls.append(request)
        part = CompactionPart(content='summary')
        return CompactionResult(messages=[ModelRequest(parts=[part]), *request.messages], part=part, handler='keep')


async def test_compaction_requested_by_a_processor_compacts_the_processed_history_in_the_same_step():
    """A processor that asks for compaction gets it on the step's first attempt, made from what it returned."""
    seen: list[list[ModelMessage]] = []

    def redact(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        ctx.request_compaction(reason='threshold')
        return [shouted(message) for message in messages]

    compaction = KeepEverything()
    agent = Agent(recording(seen), capabilities=[ProcessHistory(redact), compaction])
    result = await agent.run('hi', message_history=HISTORY)

    [request] = compaction.calls
    assert texts(request.messages) == ['FIRST', 'A', 'HI']
    # The step's only request is the compacted one.
    [sent] = seen
    assert isinstance(sent[0].parts[0], CompactionPart)
    assert texts(sent) == ['FIRST', 'A', 'HI']
    [record] = result.compactions
    assert record.replaced is not None and texts(record.replaced) == ['FIRST', 'A', 'HI']


async def test_messages_a_processor_substitutes_are_kept_through_compaction():
    """New message objects a processor returns are history, so a compaction keeps them rather than the originals.

    A `before_model_request` hook's copies are request-only, and a compaction commits the originals they stand for
    instead; a processor's are written back first, so they are the history's own objects.
    """
    returned: list[ModelMessage] = []

    def shout(messages: list[ModelMessage]) -> list[ModelMessage]:
        returned[:] = [shouted(message) for message in messages]
        return list(returned)

    async def compact(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        ctx.request_compaction()
        return request_context

    agent = Agent(
        recording([]),
        capabilities=[ProcessHistory(shout), Hooks(before_model_request=compact), KeepEverything()],
    )
    result = await agent.run('hi', message_history=HISTORY)

    committed = result.all_messages()
    assert texts(committed) == ['FIRST', 'A', 'HI', 'done']
    assert all(kept is new for kept, new in zip(committed[1:], returned, strict=False))


async def test_hooks_registers_process_history_functions_in_order():
    """Constructor kwarg, bare decorator and parameterized decorator, sync and async, run in registration order."""
    log: list[str] = []

    def first(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        log.append('kwarg')
        return messages

    hooks = Hooks(process_history=first)

    @hooks.on.process_history
    async def second(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        log.append('decorator')
        return messages

    @hooks.on.process_history(timeout=5)
    async def third(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        log.append('decorator with timeout')
        return messages[-1:]

    result = await Agent(recording([]), capabilities=[hooks]).run('hi', message_history=HISTORY)

    assert log == ['kwarg', 'decorator', 'decorator with timeout']
    assert texts(result.all_messages()) == ['hi', 'done']
    assert repr(hooks) == "Hooks({'process_history': 3})"
