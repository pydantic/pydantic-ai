"""Pin the summary provenance used by CLAI2's conservative rewind boundary check."""

from pydantic_ai.messages import CompactionPart, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.compaction import SummarizingCompaction, compact_now
from pydantic_ai_harness.compaction._summary import SUMMARY_METADATA_KEY


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

        # The first prompt precedes the summary of everything after it.
        preserved_prompt, summary, last_response = result
        assert isinstance(summary, ModelRequest) and summary.run_id is None
        assert summary.parts == [
            CompactionPart(content='summary of both prompts', metadata={SUMMARY_METADATA_KEY: True})
        ]
        assert summary.timestamp is not None
        assert summary.timestamp > max(first_prompt.timestamp, later_prompt.timestamp)
        assert preserved_prompt == messages[0] and last_response == messages[-1]
