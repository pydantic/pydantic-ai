"""Pin the summary provenance used by CLAI2's conservative rewind boundary check."""

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.compaction import SummarizingCompaction, compact_now


class TestSummarizingCompaction:
    async def test_summary_is_unowned_context_newer_than_preserved_prompts(self) -> None:
        first_prompt = UserPromptPart('first prompt')
        later_prompt = UserPromptPart('later instructions')
        messages: list[ModelMessage] = [
            ModelRequest(parts=[first_prompt], run_id='first'),
            ModelResponse(parts=[TextPart('first answer')], run_id='first'),
            ModelRequest(parts=[later_prompt], run_id='later'),
            ModelResponse(parts=[TextPart('last answer')], run_id='later'),
        ]
        result = await compact_now(
            SummarizingCompaction(max_messages=1, keep_tokens=0),
            messages,
            model=TestModel(custom_output_text='summary of both prompts'),
        )

        summary, preserved_prompt, last_response = result
        assert isinstance(summary, ModelRequest) and summary.run_id is None
        [part] = summary.parts
        assert isinstance(part, SystemPromptPart)
        assert part.content == 'Summary of previous conversation:\n\nsummary of both prompts'
        assert part.timestamp > max(first_prompt.timestamp, later_prompt.timestamp)
        assert preserved_prompt == messages[0] and last_response == messages[-1]
