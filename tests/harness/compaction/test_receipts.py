"""Compaction receipts speak in the application's voice, and are recognized by type, never by their text."""

from __future__ import annotations

from dataclasses import replace

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    OperatorSource,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai_harness.compaction import (
    CompactionReceiptPart,
    CompactOnRequest,
    RelevanceCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
)
from tests.conftest import IsDatetime, IsStr

_MARKER = '[History before this point'
_RECEIPT_TEXT = (
    '[History before this point (3 messages, ~10 tokens) was dropped by the harness. That context is no longer '
    'in the window; re-verify critical facts against primary sources.]'
)
_LEGACY_METADATA = 'pydantic-ai-harness.compaction.receipt.v1'


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(text)])


def _legacy_receipt() -> ModelRequest:
    """A receipt the way older versions wrote it: a user prompt marked with model-invisible metadata."""
    return ModelRequest(parts=[UserPromptPart([TextContent(_RECEIPT_TEXT, metadata=_LEGACY_METADATA)])])


def _receipts(messages: list[ModelMessage]) -> list[CompactionReceiptPart]:
    return [part for message in messages for part in message.parts if isinstance(part, CompactionReceiptPart)]


def _text(part: UserPromptPart) -> str:
    assert isinstance(part.content, str)
    return part.content


def _user_texts(messages: list[ModelMessage]) -> list[str]:
    return [
        _text(part)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


def _shape(messages: list[ModelMessage]) -> list[list[str]]:
    """Each message as its parts, each a `<part type>: <text>` label with the text cut to 60 characters."""
    return [
        [
            f'{type(part).__name__}: {str(getattr(part, "content", None) or getattr(part, "tool_name", ""))[:60]}'
            for part in message.parts
        ]
        for message in messages
    ]


class _Recorder:
    """The model the agent runs on, recording the messages each request sent it."""

    def __init__(self, *, inline_system_prompts: bool = True) -> None:
        self.sent: list[list[ModelMessage]] = []
        self.model = FunctionModel(
            self._respond, profile=ModelProfile(supports_inline_system_prompts=inline_system_prompts)
        )

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.sent.append(list(messages))
        return _assistant('ok')


def _summarizer(prompts: list[str] | None = None) -> SummarizingCompaction[None]:
    def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[-1]
        assert isinstance(prompt, UserPromptPart) and isinstance(prompt.content, str)
        if prompts is not None:
            prompts.append(prompt.content)
        return ModelResponse(parts=[TextPart('the summary')])

    return SummarizingCompaction(
        model=FunctionModel(summarize),
        max_messages=3,
        keep_messages=1,
        receipts=True,
        bridge_prefix=False,
        preserve_first_user_message=False,
    )


def _judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Scores every step as no longer needed, so each one is removed."""
    return ModelResponse(
        parts=[ToolCallPart(info.output_tools[0].name, {'needed_verbatim': 0.05, 'still_matters': 0.05})]
    )


def _step_history() -> list[ModelMessage]:
    """A task and three large tool steps, ending on the model's reply, so the next prompt continues it."""
    messages: list[ModelMessage] = [ModelRequest(parts=[SystemPromptPart('sys'), UserPromptPart('do the task')])]
    for index in range(3):
        messages.append(ModelResponse(parts=[ToolCallPart('read', {'n': index}, tool_call_id=f'c{index}')]))
        messages.append(ModelRequest(parts=[ToolReturnPart('read', 'junk x ' * 400, tool_call_id=f'c{index}')]))
    messages.append(_assistant('done'))
    return messages


def _long_history() -> list[ModelMessage]:
    """Two long exchanges, ending on the model's reply, so the next prompt continues it."""
    return [_user('a'), _assistant('b' * 4000), _user('c'), _assistant('d' * 4000)]


def _strategy(name: str) -> tuple[AbstractCapability[None], list[ModelMessage]]:
    if name == 'sliding_window':
        return (
            SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True, preserve_first_user_message=False),
            _long_history(),
        )
    if name == 'summarizing':
        return _summarizer(), _long_history()
    return (
        RelevanceCompaction(
            FunctionModel(_judge),
            max_tokens=2_000,
            target_tokens=740,
            keep_tokens=10,
            truncated_result_chars=0,
            receipts=True,
        ),
        _step_history(),
    )


@pytest.mark.parametrize('name', ['sliding_window', 'summarizing', 'relevance'])
async def test_receipt_is_an_operator_capability_part(name: str) -> None:
    strategy, history = _strategy(name)
    recorder = _Recorder()
    agent = Agent(recorder.model, deps_type=type(None), capabilities=[replace(strategy, id=name)])

    result = await agent.run('next please', message_history=history)

    messages = result.all_messages()
    [receipt] = _receipts(messages)
    assert (
        receipt
        == snapshot(
            {
                'sliding_window': CompactionReceiptPart(
                    source=OperatorSource(),
                    capability_id='sliding_window',
                    timestamp=IsDatetime(),
                    record_id=IsStr(),
                    content='[History before this point (4 messages, ~2000 tokens) was dropped by the harness. That context is no longer in the window; re-verify critical facts against primary sources.]',
                ),
                'summarizing': CompactionReceiptPart(
                    source=OperatorSource(),
                    capability_id='summarizing',
                    timestamp=IsDatetime(),
                    record_id=IsStr(),
                    content='[History before this point (4 messages, ~2000 tokens) was summarized by the summarizer model. The summary above is secondhand; re-verify critical facts against primary sources.]',
                ),
                'relevance': CompactionReceiptPart(
                    source=OperatorSource(),
                    capability_id='relevance',
                    timestamp=IsDatetime(),
                    record_id=IsStr(),
                    content='[History before this point (2 messages, ~703 tokens) was dropped by RelevanceCompaction. That context is no longer in the window; re-verify critical facts against primary sources.]',
                ),
            }
        )[name]
    )
    # Stored at the cut point in a request of its own, and nothing in the user's voice carries it.
    assert (
        _shape(messages)
        == snapshot(
            {
                'sliding_window': [
                    ['CompactionReceiptPart: [History before this point (4 messages, ~2000 tokens) was dr'],
                    ['UserPromptPart: next please'],
                    ['TextPart: ok'],
                ],
                'summarizing': [
                    ['CompactionPart: the summary'],
                    ['CompactionReceiptPart: [History before this point (4 messages, ~2000 tokens) was su'],
                    ['UserPromptPart: next please'],
                    ['TextPart: ok'],
                ],
                'relevance': [
                    ['SystemPromptPart: sys', 'UserPromptPart: do the task'],
                    ['CompactionReceiptPart: [History before this point (2 messages, ~703 tokens) was dro'],
                    ['ToolCallPart: read'],
                    ['ToolReturnPart: [2800 characters of this tool result were removed by compact'],
                    ['ToolCallPart: read'],
                    ['ToolReturnPart: junk x junk x junk x junk x junk x junk x junk x junk x junk'],
                    ['TextPart: done'],
                    ['UserPromptPart: next please'],
                    ['TextPart: ok'],
                ],
            }
        )[name]
    )
    # The model is sent it in the application's voice, in place, never as the user's words.
    [sent] = recorder.sent
    assert (
        _shape(sent)
        == snapshot(
            {
                'sliding_window': [
                    [
                        'SystemPromptPart: [History before this point (4 messages, ~2000 tokens) was dr',
                        'UserPromptPart: next please',
                    ]
                ],
                'summarizing': [
                    [
                        'CompactionPart: the summary',
                        'SystemPromptPart: [History before this point (4 messages, ~2000 tokens) was su',
                        'UserPromptPart: next please',
                    ]
                ],
                'relevance': [
                    [
                        'SystemPromptPart: sys',
                        'UserPromptPart: do the task',
                        'SystemPromptPart: [History before this point (2 messages, ~703 tokens) was dro',
                    ],
                    ['ToolCallPart: read'],
                    ['ToolReturnPart: [2800 characters of this tool result were removed by compact'],
                    ['ToolCallPart: read'],
                    ['ToolReturnPart: junk x junk x junk x junk x junk x junk x junk x junk x junk'],
                    ['TextPart: done'],
                    ['UserPromptPart: next please'],
                ],
            }
        )[name]
    )


@pytest.mark.parametrize('registration', ['direct', 'compact_on_request', 'tiered'])
async def test_receipt_records_the_strategy_that_wrote_it(registration: str) -> None:
    """The receipt names the registered capability that wrote it: the strategy, or the one it's composed into."""
    strategy = SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True)
    responses = [_assistant('ok')]
    if registration == 'direct':
        capability: AbstractCapability[None] = replace(strategy, id='window')
        expected = 'window'
    elif registration == 'compact_on_request':
        capability = CompactOnRequest(strategy, id='on_request')
        expected = 'on_request'
        # The model asks for the compaction, which runs before its next request.
        responses.insert(0, ModelResponse(parts=[ToolCallPart('compact_context', {})]))
    else:
        capability = TieredCompaction(tiers=[strategy], target_tokens=100, id='tiers')
        expected = 'tiers'

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return responses.pop(0)

    agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[capability])

    result = await agent.run('next', message_history=_long_history())

    [receipt] = _receipts(result.all_messages())
    assert receipt.capability_id == expected


@pytest.mark.parametrize('legacy', [False, True])
async def test_receipt_does_not_count_as_the_first_user_turn(legacy: bool) -> None:
    """`preserve_first_user_message` keeps the user's first turn, not an earlier receipt ahead of it."""
    receipt = (
        _legacy_receipt()
        if legacy
        else ModelRequest(
            parts=[CompactionReceiptPart(content=_RECEIPT_TEXT, source=OperatorSource(), scope='conversation')]
        )
    )
    recorder = _Recorder()
    agent = Agent(
        recorder.model, capabilities=[SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True)]
    )
    # The receipt's request is followed by a response, so it isn't merged into the user's first request.
    history: list[ModelMessage] = [
        receipt,
        _assistant('a'),
        _user('FIRST'),
        _assistant('b'),
        _user('c'),
        _assistant('d'),
    ]

    result = await agent.run('next', message_history=history)

    [sent] = recorder.sent
    assert _user_texts(sent) == ['FIRST', 'next']
    # The earlier receipt is replaced by this compaction's own.
    assert receipt not in result.all_messages()
    [new_receipt] = _receipts(result.all_messages())
    assert new_receipt.content != _RECEIPT_TEXT


async def test_user_prompt_with_the_receipt_text_is_not_a_receipt() -> None:
    """The receipt's wording typed by the user, as a string or a `TextContent`, stays the user's turn."""
    typed = ModelRequest(parts=[UserPromptPart(_RECEIPT_TEXT)])
    typed_item = ModelRequest(parts=[UserPromptPart([TextContent(_RECEIPT_TEXT)])])
    recorder = _Recorder()
    agent = Agent(
        recorder.model, capabilities=[SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True)]
    )
    history: list[ModelMessage] = [typed, _assistant('b'), _user('c'), _assistant('d'), typed_item, _assistant('e')]

    result = await agent.run('next', message_history=history)

    # The typed text is the first user turn, so it's preserved.
    [sent] = recorder.sent
    assert _user_texts(sent) == [_RECEIPT_TEXT, 'next']
    messages = result.all_messages()
    assert typed in messages
    [receipt] = _receipts(messages)
    # The later one is counted with the dropped history: an earlier receipt wouldn't be.
    assert receipt.content.startswith('[History before this point (5 messages,')


async def test_legacy_receipt_is_recognized_and_replaced() -> None:
    legacy = _legacy_receipt()
    recorder = _Recorder()
    agent = Agent(
        recorder.model,
        capabilities=[
            SlidingWindowCompaction(max_messages=4, keep_messages=5, receipts=True, preserve_first_user_message=False)
        ],
    )
    history: list[ModelMessage] = [legacy, _user('a'), _assistant('b'), _user('c'), _assistant('d')]

    result = await agent.run('e', message_history=history)

    messages = result.all_messages()
    assert legacy not in messages
    assert len(_receipts(messages)) == 1
    [sent] = recorder.sent
    assert not any(_MARKER in text for text in _user_texts(sent))


@pytest.mark.parametrize('legacy', [False, True])
async def test_resummarizing_drops_the_earlier_receipt(legacy: bool) -> None:
    prompts: list[str] = []
    recorder = _Recorder()
    agent = Agent(recorder.model, deps_type=type(None), capabilities=[_summarizer(prompts)])
    first = (await agent.run('e', message_history=_long_history())).all_messages()
    if legacy:
        first = [_legacy_receipt() if _receipts([message]) else message for message in first]

    second = (await agent.run('i', message_history=[*first, _user('g'), _assistant('h' * 4000)])).all_messages()

    # The summarizer never read the earlier receipt, as a system message or otherwise.
    assert len(prompts) == 2
    assert _MARKER not in prompts[1]
    # The new receipt replaced it: one receipt in the history, and none in the user's voice.
    assert len(_receipts(second)) == 1
    assert not any(_MARKER in text for text in _user_texts(second))


@pytest.mark.parametrize('inline_system_prompts', [True, False])
async def test_receipt_reaches_the_model_right_after_the_summary(inline_system_prompts: bool) -> None:
    """Without a mid-conversation system role, the receipt is a `<system>` block in place, not hoisted to the top."""
    recorder = _Recorder(inline_system_prompts=inline_system_prompts)
    agent = Agent(recorder.model, deps_type=type(None), instructions='be brief', capabilities=[_summarizer()])

    result = await agent.run('next', message_history=_long_history())

    # Stored right after the summary.
    stored = [part for message in result.all_messages() for part in message.parts]
    summary_at = next(index for index, part in enumerate(stored) if isinstance(part, CompactionPart))
    receipt = stored[summary_at + 1]
    assert isinstance(receipt, CompactionReceiptPart)
    assert 'The summary above is secondhand' in receipt.content
    # Sent right after the summary, in place.
    [sent] = recorder.sent
    parts = [part for message in sent if isinstance(message, ModelRequest) for part in message.parts]
    sent_summary_at = next(index for index, part in enumerate(parts) if isinstance(part, CompactionPart))
    rendered = parts[sent_summary_at + 1]
    if inline_system_prompts:
        assert isinstance(rendered, SystemPromptPart)
        assert rendered.content == receipt.content
    else:
        assert isinstance(rendered, UserPromptPart)
        assert _text(rendered) == f'<system>{receipt.content}</system>'
