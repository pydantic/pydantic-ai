from __future__ import annotations

import warnings

import pytest

from pydantic_ai.capabilities import AbstractCapability, SelectModel
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness.compaction import SummarizingCompaction, compact_now


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
        explicit_limit = (settings or {}).get('max_tokens')
        if budget is not None and explicit_limit is not None and explicit_limit > budget:
            assert len(captured) == 1
            assert captured[0].category is UserWarning
            assert f'`max_summary_tokens={budget}` limits `model_settings["max_tokens"]={explicit_limit}`' in str(
                captured[0].message
            )
        else:
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
                assert f'within {expected_limit} tokens' in prompt
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

    @pytest.mark.parametrize('fail_primary', [False, True])
    async def test_budget_respects_fallback_model_defaults(self, fail_primary: bool) -> None:
        observed: list[int] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.model_settings is not None
            assert 'max_tokens' in info.model_settings
            observed.append(info.model_settings['max_tokens'])
            if fail_primary:
                raise ModelAPIError('primary', 'primary unavailable')
            return ModelResponse(parts=[TextPart('Summary.')])

        def fallback(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.model_settings is not None
            assert 'max_tokens' in info.model_settings
            observed.append(info.model_settings['max_tokens'])
            return ModelResponse(parts=[TextPart('Summary.')])

        model = FallbackModel(
            FunctionModel(primary, settings=ModelSettings(max_tokens=60)),
            FunctionModel(fallback, settings=ModelSettings(max_tokens=40)),
        )
        strategy = SummarizingCompaction[None](
            max_messages=1, keep_messages=1, preserve_first_user_message=False, max_summary_tokens=120
        )
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Investigate.')]),
            ModelResponse(parts=[TextPart('Investigation complete.')]),
            ModelRequest(parts=[UserPromptPart('Continue.')]),
        ]
        await compact_now(strategy, history, model=model)
        assert observed == ([40, 40] if fail_primary else [40])

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
        if capability_limit > 120:
            assert len(captured) == 1
            assert 'Resolved summary max_tokens=300' in str(captured[0].message)
        else:
            assert not captured

    @pytest.mark.parametrize('use_fallback', [False, True])
    async def test_budget_respects_capability_selected_model(self, use_fallback: bool) -> None:
        observed: list[int] = []

        def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.model_settings is not None
            assert 'max_tokens' in info.model_settings
            observed.append(info.model_settings['max_tokens'])
            return ModelResponse(parts=[TextPart('Summary.')])

        selected = FunctionModel(summarize, settings=ModelSettings(max_tokens=40))
        summary_model = FallbackModel(selected) if use_fallback else selected
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
        assert observed == [40]
