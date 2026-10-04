"""Minimum-token retention through the public compaction API."""

from __future__ import annotations

import inspect

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import (
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    compact_now,
    estimate_token_count,
    pin,
)

StrategyType = type[SlidingWindowCompaction[None]] | type[SummarizingCompaction[None]]


@pytest.fixture(params=[SlidingWindowCompaction, SummarizingCompaction])
def strategy_type(request: pytest.FixtureRequest) -> StrategyType:
    if request.param is SlidingWindowCompaction:
        return SlidingWindowCompaction
    return SummarizingCompaction


def response(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(text)])


def assert_original_suffix(result: list[ModelMessage], suffix: list[ModelMessage], snapshot: bytes) -> None:
    retained = result[-len(suffix) :]
    assert len(retained) == len(suffix)
    assert all(actual is original for actual, original in zip(retained, suffix))
    assert ModelMessagesTypeAdapter.dump_json(retained) == snapshot


class TestMinimumRetention:
    async def test_automatic_agent_compaction_preserves_floor(self, strategy_type: StrategyType) -> None:
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('discard this older request')]),
            response('x' * 192),
            ModelRequest(parts=[UserPromptPart('y' * 12)]),
        ]
        seen: list[ModelMessage] = []

        def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.extend(messages)
            return response('done')

        strategy = strategy_type(max_messages=1, min_keep_tokens=50, preserve_first_user_message=False)
        if isinstance(strategy, SummarizingCompaction):
            strategy.model = TestModel(custom_output_text='summary')
        agent = Agent(FunctionModel(answer), deps_type=type(None), instructions='I' * 100_000, capabilities=[strategy])

        result = await agent.run(message_history=history)

        assert result.output == 'done'
        # Agent adds request metadata/instructions; compaction must retain the original parts.
        retained_parts = [
            part for message in seen for part in message.parts if isinstance(part, (TextPart, UserPromptPart))
        ]
        assert retained_parts == [history[1].parts[0], history[2].parts[0]]
        assert all(message is not history[0] for message in seen)

    @pytest.mark.parametrize('system_count', [1, 2])
    @pytest.mark.parametrize('suffix_kind', ['response', 'system_only', 'retained_pin'])
    async def test_reinjected_pin_stays_before_protected_system_suffix(
        self, strategy_type: StrategyType, system_count: int, suffix_kind: str
    ) -> None:
        missing_pin = pin('older pinned state')
        retained_pin = pin('protected pinned state')
        suffix: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(f'protected system {index}')]) for index in range(system_count)
        ]
        if suffix_kind == 'response':
            suffix.append(response('protected response'))
        elif suffix_kind == 'retained_pin':
            suffix.append(ModelRequest(parts=[retained_pin]))
        messages: list[ModelMessage] = [ModelRequest(parts=[missing_pin]), response('discard'), *suffix]
        snapshot = ModelMessagesTypeAdapter.dump_json(suffix)
        original_snapshot = ModelMessagesTypeAdapter.dump_json(messages)
        strategy = strategy_type(
            max_messages=1,
            min_keep_tokens=estimate_token_count(suffix, tokenizer=len),
            tokenizer=len,
            preserve_first_user_message=False,
        )

        result = await compact_now(strategy, messages, model=TestModel(custom_output_text='summary'))

        assert_original_suffix(result, suffix, snapshot)
        assert ModelMessagesTypeAdapter.dump_json(messages) == original_snapshot
        prefix = result[: -len(suffix)]
        assert any(isinstance(message, ModelRequest) and missing_pin in message.parts for message in prefix)
        parts = [part for message in result if isinstance(message, ModelRequest) for part in message.parts]
        assert sum(part == missing_pin for part in parts) == 1
        assert sum(part == retained_pin for part in parts) == int(suffix_kind == 'retained_pin')
        assert all(message is not messages[1] for message in result)

    @pytest.mark.parametrize('use_tokenizer', [False, True])
    @pytest.mark.parametrize(
        'old_instructions,new_instructions',
        [
            ('I' * 100_000, 'I' * 100_000),
            ('small', 'I' * 100_000),
            ('I' * 100_000, 'small'),
            ('I' * 100_000, None),
        ],
        ids=['unchanged', 'growing', 'shrinking', 'missing-latest'],
    )
    async def test_attached_instructions_do_not_satisfy_floor(
        self,
        strategy_type: StrategyType,
        use_tokenizer: bool,
        old_instructions: str,
        new_instructions: str | None,
    ) -> None:
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('discard')], instructions=old_instructions),
            response('x' * 192),
            ModelRequest(parts=[UserPromptPart('y' * 12)], instructions=new_instructions),
        ]
        before = ModelMessagesTypeAdapter.dump_json(messages)
        suffix = messages[1:]
        strategy = strategy_type(
            max_messages=1,
            min_keep_tokens=200 if use_tokenizer else 50,
            tokenizer=len if use_tokenizer else None,
            preserve_first_user_message=False,
        )

        result = await compact_now(strategy, messages, model=TestModel(custom_output_text='summary'))

        assert_original_suffix(result, suffix, ModelMessagesTypeAdapter.dump_json(suffix))
        assert ModelMessagesTypeAdapter.dump_json(messages) == before
        assert all(message is not messages[0] for message in result)
        assert len(result) == 2 + int(isinstance(strategy, SummarizingCompaction))

    @pytest.mark.parametrize('minimum', [8, 9, 100])
    @pytest.mark.parametrize('instruction_only', [False, True])
    async def test_short_history_with_large_instructions_is_noop(
        self, strategy_type: StrategyType, minimum: int, instruction_only: bool
    ) -> None:
        messages: list[ModelMessage] = [
            ModelRequest(parts=[] if instruction_only else [UserPromptPart('tiny')], instructions='I' * 100_000),
            ModelRequest(parts=[] if instruction_only else [UserPromptPart('tail')], instructions='J' * 200_000),
        ]
        before = ModelMessagesTypeAdapter.dump_json(messages)
        usage = RunUsage()
        strategy = strategy_type(max_messages=1, min_keep_tokens=minimum, tokenizer=len)

        result = await compact_now(strategy, messages, model=TestModel(), usage=usage)

        assert len(result) == len(messages)
        assert all(actual is original for actual, original in zip(result, messages))
        assert ModelMessagesTypeAdapter.dump_json(result) == before
        assert usage.requests == 0

    @pytest.mark.parametrize('scale', [1, 1000])
    async def test_crossing_minimum_retains_whole_message(self, strategy_type: StrategyType, scale: int) -> None:
        # The latest 3k is insufficient; adding the preceding 48k must retain 51k,
        # not discard the 48k message because it overshoots a 50k target.
        messages: list[ModelMessage] = [response('old'), response('x' * (48 * scale)), response('y' * (3 * scale))]
        snapshot = ModelMessagesTypeAdapter.dump_json(messages[1:])
        strategy = strategy_type(
            max_messages=1, min_keep_tokens=50 * scale, tokenizer=len, preserve_first_user_message=False
        )

        result = await compact_now(strategy, messages, model=TestModel(custom_output_text='summary'))

        assert_original_suffix(result, messages[1:], snapshot)
        assert estimate_token_count(result[-2:], tokenizer=len) == 51 * scale
        assert all(message is not messages[0] for message in result)
        assert len(result) == (3 if isinstance(strategy, SummarizingCompaction) else 2)

    @pytest.mark.parametrize('minimum,retained_count', [(3, 1), (4, 2), (51, 2), (52, 3)])
    async def test_exact_threshold_and_one_token_boundary(
        self, strategy_type: StrategyType, minimum: int, retained_count: int
    ) -> None:
        messages: list[ModelMessage] = [response('older'), response('x' * 48), response('yyy')]
        suffix = messages[-retained_count:]
        snapshot = ModelMessagesTypeAdapter.dump_json(suffix)
        strategy = strategy_type(
            max_messages=1, min_keep_tokens=minimum, tokenizer=len, preserve_first_user_message=False
        )

        result = await compact_now(strategy, messages, model=TestModel(custom_output_text='summary'))

        assert_original_suffix(result, suffix, snapshot)
        expected_extra = int(isinstance(strategy, SummarizingCompaction) and retained_count < len(messages))
        assert len(result) == retained_count + expected_extra

    @pytest.mark.parametrize('contents', [[], [''], ['tiny'], ['one', 'two']])
    async def test_insufficient_history_is_unchanged_without_model_call(
        self, strategy_type: StrategyType, contents: list[str]
    ) -> None:
        messages: list[ModelMessage] = [response(text) for text in contents]
        snapshot = ModelMessagesTypeAdapter.dump_json(messages)
        usage = RunUsage()
        strategy = strategy_type(max_messages=1, min_keep_tokens=100, tokenizer=len)

        result = await compact_now(strategy, messages, model=TestModel(), usage=usage)

        assert result == messages
        assert all(actual is original for actual, original in zip(result, messages))
        assert ModelMessagesTypeAdapter.dump_json(result) == snapshot
        assert usage.requests == 0

    @pytest.mark.parametrize('gap', [0, 7])
    @pytest.mark.parametrize('reply_kind', ['return', 'tool_retry', 'generic_retry'])
    async def test_tool_reply_dependencies(self, strategy_type: StrategyType, gap: int, reply_kind: str) -> None:
        call = ModelResponse(parts=[ToolCallPart('lookup', {'key': 'value'}, tool_call_id='lookup-1')])
        reply = (
            ToolReturnPart('lookup', 'result', tool_call_id='lookup-1')
            if reply_kind == 'return'
            else RetryPromptPart(
                'retry', tool_name='lookup' if reply_kind == 'tool_retry' else None, tool_call_id='lookup-1'
            )
        )
        returned = ModelRequest(parts=[reply])
        messages: list[ModelMessage] = [response('discard'), call]
        messages.extend(response(f'intervening {index}') for index in range(gap))
        messages.extend([returned, response('end')])
        suffix = messages[-2:] if reply_kind == 'generic_retry' else messages[1:]
        snapshot = ModelMessagesTypeAdapter.dump_json(suffix)
        minimum = estimate_token_count(messages[-2:], tokenizer=len)
        strategy = strategy_type(
            max_messages=1, min_keep_tokens=minimum, tokenizer=len, preserve_first_user_message=False
        )

        result = await compact_now(strategy, messages, model=TestModel(custom_output_text='summary'))

        assert_original_suffix(result, suffix, snapshot)
        assert all(message is not messages[0] for message in result)
        assert len(result) == len(suffix) + int(isinstance(strategy, SummarizingCompaction))

    @pytest.mark.parametrize('token_budget', [False, True])
    async def test_legacy_retention_does_not_protect_tool_retries(
        self, strategy_type: StrategyType, token_budget: bool
    ) -> None:
        messages: list[ModelMessage] = [
            ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='lookup-1')]),
            ModelRequest(parts=[RetryPromptPart('retry', tool_name='lookup', tool_call_id='lookup-1')]),
            response('end'),
        ]
        suffix = messages[1:]
        strategy = strategy_type(
            max_messages=1,
            keep_messages=2,
            keep_tokens=estimate_token_count(suffix, tokenizer=len) if token_budget else None,
            tokenizer=len,
            preserve_first_user_message=False,
        )

        result = await compact_now(strategy, messages, model=TestModel(custom_output_text='summary'))

        assert_original_suffix(result, suffix, ModelMessagesTypeAdapter.dump_json(suffix))
        assert len(result) == 2 + int(isinstance(strategy, SummarizingCompaction))
        assert all(message is not messages[0] for message in result)

    async def test_custom_tokenizer_controls_boundary(self, strategy_type: StrategyType) -> None:
        seen: list[str] = []

        def word_tokens(text: str) -> int:
            seen.append(text)
            return len(text.split())

        messages: list[ModelMessage] = [
            response('discard me'),
            response('two words'),
            ModelRequest(parts=[UserPromptPart('longsingleword')], instructions='attached instruction ' * 1000),
        ]
        strategy = strategy_type(
            max_messages=1, min_keep_tokens=3, tokenizer=word_tokens, preserve_first_user_message=False
        )
        snapshot = ModelMessagesTypeAdapter.dump_json(messages[1:])

        result = await compact_now(strategy, messages, model=TestModel(custom_output_text='summary'))

        assert_original_suffix(result, messages[1:], snapshot)
        assert 'longsingleword' in seen
        assert 'two words' in seen
        assert all(message is not messages[0] for message in result)

    async def test_default_tokenizer_counts_complete_suffix(self, strategy_type: StrategyType) -> None:
        # The heuristic rounds the combined character count, not each message separately.
        messages: list[ModelMessage] = [response('discard'), response('abc'), response('xyz')]
        suffix = messages[1:]
        snapshot = ModelMessagesTypeAdapter.dump_json(suffix)
        strategy = strategy_type(max_messages=1, min_keep_tokens=1, preserve_first_user_message=False)
        assert estimate_token_count(messages[-1:]) == 0
        assert estimate_token_count(suffix) == 1

        result = await compact_now(strategy, messages, model=TestModel(custom_output_text='summary'))

        assert_original_suffix(result, suffix, snapshot)
        assert len(result) == 2 + int(isinstance(strategy, SummarizingCompaction))

    @pytest.mark.parametrize('minimum', [0, -1, -100])
    def test_non_positive_minimum_is_rejected(self, strategy_type: StrategyType, minimum: int) -> None:
        with pytest.raises(ValueError, match='min_keep_tokens'):
            strategy_type(max_messages=1, min_keep_tokens=minimum)

    @pytest.mark.parametrize('keep_tokens', [0, 1, 100])
    def test_minimum_and_maximum_retention_are_exclusive(self, strategy_type: StrategyType, keep_tokens: int) -> None:
        with pytest.raises(ValueError, match='keep_tokens'):
            strategy_type(max_messages=1, keep_tokens=keep_tokens, min_keep_tokens=1)

    def test_minimum_is_keyword_only_and_disabled_by_default(self, strategy_type: StrategyType) -> None:
        parameter = inspect.signature(strategy_type).parameters['min_keep_tokens']
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None
        assert strategy_type(max_messages=1).min_keep_tokens is None
        assert strategy_type(max_messages=1, min_keep_tokens=1).min_keep_tokens == 1

    @pytest.mark.parametrize('token_budget', [False, True])
    async def test_none_preserves_legacy_retention(self, strategy_type: StrategyType, token_budget: bool) -> None:
        messages: list[ModelMessage] = [response('old'), response('x' * 48), response('yyy')]
        strategy = strategy_type(
            max_messages=1,
            keep_messages=1,
            keep_tokens=50 if token_budget else None,
            min_keep_tokens=None,
            tokenizer=len,
            preserve_first_user_message=False,
        )

        result = await compact_now(strategy, messages, model=TestModel(custom_output_text='summary'))

        assert result[-1] is messages[-1]
        assert all(message is not messages[1] for message in result)
        assert len(result) == 1 + int(isinstance(strategy, SummarizingCompaction))

    async def test_first_user_preservation_is_additive(self, strategy_type: StrategyType) -> None:
        first = ModelRequest.user_text_prompt('original user request')
        messages: list[ModelMessage] = [first, response('discard'), response('x' * 48), response('yyy')]
        snapshot = ModelMessagesTypeAdapter.dump_json(messages[-2:])
        strategy = strategy_type(max_messages=1, min_keep_tokens=50, tokenizer=len)

        result = await compact_now(strategy, messages, model=TestModel(custom_output_text='summary'))

        assert any(message is first for message in result)
        assert_original_suffix(result, messages[-2:], snapshot)
        assert all(message is not messages[1] for message in result)

    async def test_existing_receipt_in_protected_suffix_is_untouched(self, strategy_type: StrategyType) -> None:
        # Obtain a real receipt from the public API rather than depending on its metadata format.
        receipted = await compact_now(
            SlidingWindowCompaction(max_messages=1, keep_messages=2, receipts=True, preserve_first_user_message=False),
            [response('discarded history'), response('kept')],
            model=TestModel(),
        )
        assert len(receipted) == 2
        receipt = receipted[0]
        suffix: list[ModelMessage] = [response('protected before receipt'), receipt, response('tail')]
        messages: list[ModelMessage] = [response('older prefix'), *suffix]
        snapshot = ModelMessagesTypeAdapter.dump_json(suffix)
        strategy = strategy_type(
            max_messages=1,
            min_keep_tokens=estimate_token_count(suffix, tokenizer=len),
            tokenizer=len,
            receipts=True,
            preserve_first_user_message=False,
        )

        result = await compact_now(strategy, messages, model=TestModel(custom_output_text='summary'))

        assert_original_suffix(result, suffix, snapshot)
        assert sum(message is receipt for message in result) == 1
        assert all(message is not messages[0] for message in result)


class TestSummarizingCompactionMinimumRetention:
    @pytest.mark.parametrize('keep_user_messages', [False, True])
    @pytest.mark.parametrize('receipts', [False, True])
    async def test_only_prefix_is_summarized_and_extras_do_not_trim_suffix(
        self, keep_user_messages: bool, receipts: bool
    ) -> None:
        prompts: list[str] = []

        def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            prompts.extend(
                part.content
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            )
            return response('a concise summary')

        older_user = ModelRequest.user_text_prompt('OLDER_USER_SENTINEL ' * 10)
        suffix: list[ModelMessage] = [
            ModelRequest.user_text_prompt('PROTECTED_USER_SENTINEL ' * 10),
            response('PROTECTED_RESPONSE_SENTINEL'),
        ]
        messages: list[ModelMessage] = [older_user, response('OLDER_RESPONSE_SENTINEL'), *suffix]
        before = ModelMessagesTypeAdapter.dump_json(messages)
        snapshot = ModelMessagesTypeAdapter.dump_json(suffix)
        strategy: SummarizingCompaction[None] = SummarizingCompaction(
            max_messages=1,
            keep_messages=1,
            min_keep_tokens=estimate_token_count(suffix, tokenizer=len),
            tokenizer=len,
            keep_user_messages=keep_user_messages,
            keep_user_messages_max_chars=30,
            preserve_first_user_message=False,
            receipts=receipts,
        )

        result = await compact_now(strategy, messages, model=FunctionModel(summarize))

        assert_original_suffix(result, suffix, snapshot)
        assert ModelMessagesTypeAdapter.dump_json(messages) == before
        assert len(prompts) == 1
        assert 'OLDER_USER_SENTINEL' in prompts[0]
        assert 'OLDER_RESPONSE_SENTINEL' in prompts[0]
        assert 'PROTECTED_USER_SENTINEL' not in prompts[0]
        assert 'PROTECTED_RESPONSE_SENTINEL' not in prompts[0]
        retained_users = [
            part.content
            for message in result[: -len(suffix)]
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
            and isinstance(part.content, str)
            and 'OLDER_USER_SENTINEL' in part.content
        ]
        assert bool(retained_users) is keep_user_messages
        assert all(len(text) <= 30 for text in retained_users)

    async def test_model_error_falls_back_to_sliding_window_with_same_minimum(self) -> None:
        calls = 0

        def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            raise ModelAPIError('test', 'summary unavailable')

        messages: list[ModelMessage] = [response('old'), response('x' * 48), response('yyy')]
        snapshot = ModelMessagesTypeAdapter.dump_json(messages)
        chain: FallbackCompaction[None] = FallbackCompaction(
            fallback_chain=[
                SummarizingCompaction(
                    max_messages=1,
                    min_keep_tokens=50,
                    tokenizer=len,
                    preserve_first_user_message=False,
                    model=FunctionModel(fail),
                ),
                SlidingWindowCompaction(
                    max_messages=1, min_keep_tokens=50, tokenizer=len, preserve_first_user_message=False
                ),
            ]
        )

        result = await compact_now(chain, messages, model=TestModel())

        assert calls == 1
        assert len(result) == 2
        assert_original_suffix(result, messages[1:], ModelMessagesTypeAdapter.dump_json(messages[1:]))
        assert ModelMessagesTypeAdapter.dump_json(messages) == snapshot
