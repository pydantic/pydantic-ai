from __future__ import annotations

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities import ModelFallback
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai_harness.compaction import SlidingWindowCompaction


async def test_output_fallback_compacts_for_the_next_models_window() -> None:
    old_prompt = 'old ' * 400

    def primary_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert any(
            isinstance(part, UserPromptPart) and part.content == old_prompt
            for message in messages
            for part in message.parts
        )
        return ModelResponse(parts=[TextPart('draft')])

    def backup_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert not any(
            isinstance(part, UserPromptPart) and part.content == old_prompt
            for message in messages
            for part in message.parts
        )
        return ModelResponse(parts=[TextPart('ready')])

    primary = FunctionModel(primary_response, model_name='primary', profile=ModelProfile(context_window=100_000))
    backup = FunctionModel(backup_response, model_name='backup', profile=ModelProfile(context_window=1_000))
    agent = Agent(
        primary,
        capabilities=[
            SlidingWindowCompaction(
                max_fraction=0.5, keep_messages=1, preserve_first_user_message=False, tokenizer=len
            ),
            ModelFallback(backup, output_retry_threshold=1),
        ],
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(old_prompt)]),
        ModelResponse(parts=[TextPart('Previous answer.')]),
    ]
    assert (await agent.run('Answer.', message_history=history)).output == 'ready'
