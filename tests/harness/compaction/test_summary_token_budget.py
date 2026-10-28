from __future__ import annotations

import copy
import warnings

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, SelectModel
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness.compaction import (
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    compact_now,
)


class TestSummarizingCompaction:
    @pytest.mark.parametrize('custom_prompt', [False, True])
    @pytest.mark.parametrize(
        ('budget', 'settings', 'model_limit', 'expected_limit'),
        [
            (None, None, None, None),
            (120, None, None, 120),
            (120, ModelSettings(max_tokens=60, temperature=0.2), None, 60),
            (120, ModelSettings(max_tokens=300, temperature=0.2), None, 120),
            (None, ModelSettings(max_tokens=60), None, 60),
            (120, None, 60, 60),
            (120, ModelSettings(max_tokens=300), 60, 120),
        ],
    )
    async def test_budget_reaches_summary_prompt_and_model_settings(
        self,
        budget: int | None,
        settings: ModelSettings | None,
        model_limit: int | None,
        expected_limit: int | None,
        custom_prompt: bool,
    ) -> None:
        observed: list[tuple[str, ModelSettings | None]] = []

        def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            prompt = messages[-1]
            assert isinstance(prompt, ModelRequest)
            part = prompt.parts[-1]
            assert isinstance(part, UserPromptPart)
            assert isinstance(part.content, str)
            observed.append((part.content, info.model_settings))
            return ModelResponse(parts=[TextPart('A concise summary.')])

        defaults = ModelSettings(temperature=0.5)
        if model_limit is not None:
            defaults['max_tokens'] = model_limit
        original_defaults = defaults.copy()
        model = FunctionModel(summarize, settings=defaults)
        original_settings = settings.copy() if settings is not None else None
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            strategy = SummarizingCompaction[None](
                max_messages=1,
                keep_messages=1,
                preserve_first_user_message=False,
                max_summary_tokens=budget,
                model_settings=settings,
            )
        assert not captured
        if custom_prompt:
            strategy.summary_prompt = 'Summarize this transcript:\n{messages}'
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Summarize the investigation.')]),
            ModelResponse(parts=[TextPart('The cause is identified.')]),
            ModelRequest(parts=[UserPromptPart('Continue.')]),
        ]

        compacted = await compact_now(strategy, history, model=model)
        compacted.extend(
            [ModelResponse(parts=[TextPart('The fix is ready.')]), ModelRequest(parts=[UserPromptPart('Review it.')])]
        )
        await compact_now(strategy, compacted, model=model)

        expected_settings: ModelSettings = {**defaults, **(settings or {})}
        if expected_limit is not None:
            expected_settings['max_tokens'] = expected_limit
        assert len(observed) == 2
        assert '<previous-summary>' not in observed[0][0]
        assert '<previous-summary>' in observed[1][0]
        for prompt, applied_settings in observed:
            assert applied_settings == expected_settings
            if budget is not None:
                assert f'within {budget} tokens' in prompt
                assert 'rewrite it within this budget rather than appending to it' in prompt
            else:
                assert 'Write the complete summary within' not in prompt
            if not custom_prompt:
                assert 'much shorter than the conversation it replaces' in prompt
        assert settings == original_settings
        assert defaults == original_defaults

    @pytest.mark.parametrize('budget', [0, -1])
    def test_budget_must_be_positive(self, budget: int) -> None:
        with pytest.raises(ValueError, match='max_summary_tokens must be positive'):
            SummarizingCompaction(max_messages=1, max_summary_tokens=budget)

    @pytest.mark.parametrize('capability_limit', [30, 300])
    async def test_budget_applies_after_capability_settings(self, capability_limit: int) -> None:
        class SummarySettings(AbstractCapability[None]):
            def get_model_settings(self) -> ModelSettings:
                return {'max_tokens': capability_limit, 'temperature': 0.2}

        observed: list[ModelSettings | None] = []

        def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            observed.append(info.model_settings)
            return ModelResponse(parts=[TextPart('Summary.')])

        strategy = SummarizingCompaction[None](
            max_messages=1,
            keep_messages=1,
            preserve_first_user_message=False,
            max_summary_tokens=120,
            summarization_capabilities=[SummarySettings()],
        )
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Investigate.')]),
            ModelResponse(parts=[TextPart('Investigation complete.')]),
            ModelRequest(parts=[UserPromptPart('Continue.')]),
        ]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            await compact_now(strategy, history, model=FunctionModel(summarize))
        assert observed == [{'max_tokens': min(120, capability_limit), 'temperature': 0.2}]
        assert not captured

    @pytest.mark.parametrize('model_kind', ['direct', 'fallback', 'wrapper'])
    async def test_budget_respects_capability_selected_model(self, model_kind: str) -> None:
        observed: list[int] = []

        def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.model_settings is not None
            assert 'max_tokens' in info.model_settings
            observed.append(info.model_settings['max_tokens'])
            return ModelResponse(parts=[TextPart('Summary.')])

        selected = FunctionModel(summarize, settings=ModelSettings(max_tokens=40))
        summary_model = (
            FallbackModel(selected)
            if model_kind == 'fallback'
            else WrapperModel(selected)
            if model_kind == 'wrapper'
            else selected
        )
        strategy = SummarizingCompaction[None](
            max_messages=1,
            keep_messages=1,
            preserve_first_user_message=False,
            max_summary_tokens=120,
            summarization_capabilities=[SelectModel(lambda ctx: summary_model)],
        )
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Investigate.')]),
            ModelResponse(parts=[TextPart('Investigation complete.')]),
            ModelRequest(parts=[UserPromptPart('Continue.')]),
        ]
        await compact_now(
            strategy, history, model=FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('Unused.')]))
        )
        assert observed == ([120] if model_kind == 'fallback' else [40])

    @pytest.mark.parametrize('partial_text', ['partial summary', ''])
    @pytest.mark.parametrize('budget', [None, 120])
    @pytest.mark.parametrize('fail_twice', [False, True])
    @pytest.mark.parametrize('model_limit', [None, 80])
    async def test_length_stopped_summary_is_retried_before_commit(
        self, partial_text: str, budget: int | None, fail_twice: bool, model_limit: int | None
    ) -> None:
        calls: list[list[ModelMessage]] = []

        def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(copy.deepcopy(messages))
            if len(calls) == 1 or fail_twice:
                return ModelResponse(parts=[TextPart(partial_text)], finish_reason='length')
            return ModelResponse(parts=[TextPart('Complete summary.')], finish_reason='stop')

        strategy = SummarizingCompaction[None](
            max_messages=1,
            keep_messages=1,
            preserve_first_user_message=False,
            max_summary_tokens=budget,
            model_settings=ModelSettings(max_tokens=model_limit) if model_limit is not None else None,
        )
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Investigate.')]),
            ModelResponse(parts=[TextPart('Investigation complete.')]),
            ModelRequest(parts=[UserPromptPart('Continue.')]),
        ]
        original = copy.deepcopy(history)
        if fail_twice:
            with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries'):
                await compact_now(strategy, history, model=FunctionModel(summarize))
        else:
            compacted = await compact_now(strategy, history, model=FunctionModel(summarize))
            assert [
                part.content
                for msg in compacted
                if isinstance(msg, ModelRequest)
                for part in msg.parts
                if isinstance(part, CompactionPart)
            ] == ['Complete summary.']
        assert history == original
        assert len(calls) == 2
        retry_request = calls[1][-1]
        assert isinstance(retry_request, ModelRequest)
        assert len(retry_request.parts) == 1
        feedback = retry_request.parts[0]
        assert isinstance(feedback, UserPromptPart)
        assert isinstance(feedback.content, str)
        expected_limit = model_limit if model_limit is not None else budget
        limit = f'{expected_limit} tokens' if expected_limit is not None else 'the output token limit'
        assert f'within {limit}' in feedback.content

    async def test_length_stopped_summary_can_fall_back_without_committing(self) -> None:
        summary_calls = 0
        sent: list[list[ModelMessage]] = []

        def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal summary_calls
            summary_calls += 1
            return ModelResponse(parts=[TextPart('partial summary')], finish_reason='length')

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            sent.append(copy.deepcopy(messages))
            return ModelResponse(parts=[TextPart('Done.')])

        strategy = SummarizingCompaction[None](
            FunctionModel(summarize),
            max_messages=1,
            keep_messages=1,
            preserve_first_user_message=False,
            max_summary_tokens=120,
        )
        fallback = FallbackCompaction[None](
            fallback_chain=[
                strategy,
                SlidingWindowCompaction(
                    max_messages=1, keep_messages=1, preserve_first_user_message=False, receipts=False
                ),
            ],
            fallback_on=(UnexpectedModelBehavior,),
            max_tokens=1,
        )
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Investigate.')]),
            ModelResponse(parts=[TextPart('Investigation complete.')]),
        ]
        original = copy.deepcopy(history)
        result = await Agent(FunctionModel(respond), deps_type=type(None), capabilities=[fallback]).run(
            'Continue.', message_history=history
        )
        assert result.output == 'Done.'
        assert result.usage.requests == 3
        assert summary_calls == 2
        assert history == original
        assert len(sent) == 1
        assert [
            part.content
            for msg in sent[0]
            if isinstance(msg, ModelRequest)
            for part in msg.parts
            if isinstance(part, UserPromptPart)
        ] == ['Continue.']
        assert not any(isinstance(part, CompactionPart) for msg in sent[0] for part in msg.parts)
