"""Pin the summary provenance used by CLAI2's conservative rewind boundary check."""

from datetime import UTC, datetime

import pytest

from pydantic_ai.messages import CompactionPart, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.compaction import SummarizingCompaction, compact_now
from pydantic_ai_harness.compaction._summary import SUMMARY_METADATA_KEY, SUMMARY_TIME_METADATA_KEY, summary_time


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
        assert summary.timestamp is not None
        assert summary.parts == [
            CompactionPart(
                content='summary of both prompts',
                metadata={SUMMARY_METADATA_KEY: True, SUMMARY_TIME_METADATA_KEY: summary.timestamp.isoformat()},
            )
        ]
        assert summary.timestamp > max(first_prompt.timestamp, later_prompt.timestamp)
        assert preserved_prompt == messages[0] and last_response == messages[-1]


WRITTEN = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ('part', 'expected'),
    [
        pytest.param(
            CompactionPart(
                content='S', metadata={SUMMARY_METADATA_KEY: True, SUMMARY_TIME_METADATA_KEY: WRITTEN.isoformat()}
            ),
            WRITTEN,
            id='summary',
        ),
        pytest.param(CompactionPart(content='S', metadata={SUMMARY_METADATA_KEY: True}), None, id='saved-before-times'),
        pytest.param(
            CompactionPart(content='S', metadata={SUMMARY_METADATA_KEY: True, SUMMARY_TIME_METADATA_KEY: 'soon'}),
            None,
            id='unreadable',
        ),
        pytest.param(
            CompactionPart(content='S', metadata={SUMMARY_TIME_METADATA_KEY: WRITTEN.isoformat()}),
            None,
            id='not-a-summary',
        ),
    ],
)
def test_summary_time_reads_only_a_summarys_own_time(part: CompactionPart, expected: datetime | None) -> None:
    assert summary_time(part) == expected


class _RecordedBeforeTheTimeWasKept(SummarizingCompaction[None]):
    """Stands in for a durable replay of a summary an earlier version recorded as a bare `str`."""

    async def _summarize(
        self, messages: list[ModelMessage], ctx: RunContext[None], *, previous_summary: str | None = None
    ) -> str:
        return 'recorded summary'


async def test_a_summary_recorded_without_its_time_still_replays() -> None:
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('first prompt')]),
        ModelResponse(parts=[TextPart('first answer')]),
        ModelRequest(parts=[UserPromptPart('later prompt')]),
        ModelResponse(parts=[TextPart('later answer')]),
    ]
    result = await compact_now(
        _RecordedBeforeTheTimeWasKept(max_messages=1, keep_messages=1, preserve_first_user_message=False),
        messages,
        model=TestModel(),
    )

    summary = result[0]
    assert isinstance(summary, ModelRequest)
    assert summary.parts == [CompactionPart(content='recorded summary', metadata={SUMMARY_METADATA_KEY: True})]
