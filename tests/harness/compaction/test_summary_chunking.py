"""`SummarizingCompaction` fitting its summary requests into a summarizer model's smaller context window."""

from __future__ import annotations

import re

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelContextWindowExceededError
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness.compaction import SummarizingCompaction, estimate_token_count

try:
    from logfire.testing import CaptureLogfire

    logfire_installed = True
except ImportError:  # pragma: no cover
    logfire_installed = False

pytestmark = pytest.mark.anyio

_WINDOW = 2_000
"""The summarizer's context window in these tests, far smaller than the histories they summarize."""


class _Summarizer:
    """A summarizer with a *window*-token context window that rejects a request over it, as a provider does.

    Its profile states the window unless *known* is false, so the capability has none to fit. Each request's
    prompt is recorded, and its summary names the request, so a test can see which summary the next request
    was asked to update.
    """

    def __init__(self, *, window: int = _WINDOW, known: bool = True, settings: ModelSettings | None = None) -> None:
        self.window = window
        self.prompts: list[str] = []
        self.request_tokens: list[int] = []
        profile = ModelProfile(context_window=window) if known else None
        self.model = FunctionModel(self._summarize, profile=profile, settings=settings)

    def _summarize(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tokens = estimate_token_count(messages)
        if tokens > self.window:
            raise ModelContextWindowExceededError(model_name='small-summarizer', message='prompt is too long')
        self.request_tokens.append(tokens)
        self.prompts.append(
            '\n'.join(
                part.content
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            )
        )
        return ModelResponse(parts=[TextPart(f'summary {len(self.prompts)}')])


def _reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('done')])


def _agent(compaction: SummarizingCompaction[None]) -> Agent[None, str]:
    return Agent(FunctionModel(_reply), deps_type=type(None), capabilities=[compaction])


def _turns(count: int, size: int = 1_200) -> list[ModelMessage]:
    """*count* question-and-answer turns, each answer *size* characters, each turn numbered."""
    history: list[ModelMessage] = []
    for index in range(count):
        history += [
            ModelRequest(parts=[UserPromptPart(f'question {index}')]),
            ModelResponse(parts=[TextPart(f'answer {index} ' + 'x' * size)]),
        ]
    return history


def _tool_turns(count: int, size: int = 400) -> list[ModelMessage]:
    """*count* tool calls, each to its own tool and answered by a *size*-character return, after one question."""
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('question')])]
    for index in range(count):
        history += [
            ModelResponse(parts=[ToolCallPart(f'tool_{index}', {}, tool_call_id=f'call_{index}')]),
            ModelRequest(parts=[ToolReturnPart(f'tool_{index}', 'y' * size, tool_call_id=f'call_{index}')]),
        ]
    return [*history, ModelResponse(parts=[TextPart('all tools ran')])]


def _compaction(
    summarizer: _Summarizer,
    *,
    tool_return_max_chars: int | None = 500,
    model_settings: ModelSettings | None = None,
    instructions: str = SummarizingCompaction.instructions,
) -> SummarizingCompaction[None]:
    """A compaction that summarizes all of a history with *summarizer* when the next prompt is added."""
    return SummarizingCompaction(
        summarizer.model,
        max_messages=2,
        keep_messages=1,
        preserve_first_user_message=False,
        tool_return_max_chars=tool_return_max_chars,
        model_settings=model_settings,
        instructions=instructions,
    )


def _summary(messages: list[ModelMessage]) -> str:
    [summary] = [part.content for message in messages for part in message.parts if isinstance(part, CompactionPart)]
    assert summary is not None
    return summary


class TestSummarizingCompaction:
    async def test_a_history_over_the_summarizer_window_is_summarized_in_chunks(self) -> None:
        """The #12115 repro: the history doesn't fit the summarizer's window, and the run still gets its summary."""
        summarizer = _Summarizer()
        history = _turns(12)
        assert estimate_token_count(history) > _WINDOW
        agent = _agent(_compaction(summarizer))

        result = await agent.run('next', message_history=history)

        assert len(summarizer.prompts) == snapshot(4)
        assert _summary(result.all_messages()) == f'summary {len(summarizer.prompts)}'
        assert all(tokens <= _WINDOW for tokens in summarizer.request_tokens)

    async def test_one_request_overflows_without_a_known_window(self) -> None:
        """Without a window to fit, the request goes out whole, as before, and the provider rejects it."""
        summarizer = _Summarizer(known=False)
        history = _turns(12)
        agent = _agent(_compaction(summarizer))

        with pytest.raises(ModelContextWindowExceededError):
            await agent.run('next', message_history=history)

    async def test_each_chunk_updates_the_summary_of_the_chunks_before_it(self) -> None:
        summarizer = _Summarizer()
        history = _turns(12)
        agent = _agent(_compaction(summarizer))

        await agent.run('next', message_history=history)

        anchors = [
            re.findall(r'<previous-summary>\n(.*?)\n</previous-summary>', prompt) for prompt in summarizer.prompts
        ]
        assert anchors == snapshot([[], ['summary 1'], ['summary 2'], ['summary 3']])
        # Every turn is summarized exactly once, oldest first.
        questions = [re.findall(r'question (\d+)', prompt) for prompt in summarizer.prompts]
        assert [int(index) for chunk in questions for index in chunk] == list(range(12))

    async def test_the_caller_anchor_starts_the_chain(self) -> None:
        """An earlier compaction's summary is what the first chunk updates."""
        summarizer = _Summarizer()
        history: list[ModelMessage] = [
            ModelRequest(parts=[CompactionPart(content='earlier summary', metadata=_ours_metadata())]),
            *_turns(12),
        ]
        agent = _agent(_compaction(summarizer))

        result = await agent.run('next', message_history=history)

        anchors = [
            re.findall(r'<previous-summary>\n(.*?)\n</previous-summary>', prompt) for prompt in summarizer.prompts
        ]
        assert anchors == snapshot([['earlier summary'], ['summary 1'], ['summary 2'], ['summary 3']])
        assert _summary(result.all_messages()) == f'summary {len(summarizer.prompts)}'

    async def test_a_chunk_keeps_each_tool_call_with_its_return(self) -> None:
        summarizer = _Summarizer()
        history = _tool_turns(40)
        agent = _agent(_compaction(summarizer, tool_return_max_chars=None))

        await agent.run('next', message_history=history)

        assert len(summarizer.prompts) == snapshot(5)
        for index in range(40):
            [prompt] = [prompt for prompt in summarizer.prompts if f'name="tool_{index}"' in prompt]
            assert prompt.count(f'name="tool_{index}"') == 2
            assert f'<message role="tool_call" name="tool_{index}">' in prompt
            assert f'<message role="tool_return" name="tool_{index}">' in prompt

    async def test_the_summary_budget_is_left_free_in_every_request(self) -> None:
        """The summarizer's `max_tokens` is reserved for the summary, so a larger one makes smaller chunks."""
        history = _turns(12)
        default = _Summarizer()
        reserved = _Summarizer(settings=ModelSettings(max_tokens=1_200))

        for summarizer in (default, reserved):
            agent = _agent(_compaction(summarizer))
            await agent.run('next', message_history=history)

        # The default reserve is a quarter of the window, at most 8,192 tokens.
        assert all(tokens <= _WINDOW - _WINDOW // 4 for tokens in default.request_tokens)
        assert all(tokens <= _WINDOW - 1_200 for tokens in reserved.request_tokens)
        assert (len(default.prompts), len(reserved.prompts)) == snapshot((4, 12))

    async def test_a_summarizer_setting_reserve_on_the_capability_counts(self) -> None:
        history = _turns(12)
        summarizer = _Summarizer()
        compaction = _compaction(summarizer, model_settings=ModelSettings(max_tokens=1_200))
        agent = _agent(compaction)

        await agent.run('next', message_history=history)

        assert all(tokens <= _WINDOW - 1_200 for tokens in summarizer.request_tokens)
        assert len(summarizer.prompts) == snapshot(12)

    async def test_a_message_larger_than_a_chunk_has_its_tool_output_cut_to_fit(self) -> None:
        summarizer = _Summarizer()
        history = _tool_turns(3, size=20_000)
        agent = _agent(_compaction(summarizer, tool_return_max_chars=None))

        result = await agent.run('next', message_history=history)

        assert len(summarizer.prompts) == snapshot(5)
        assert all(tokens <= _WINDOW for tokens in summarizer.request_tokens)
        for index in range(3):
            # Each return is summarized, cut with the truncation marker rather than dropped.
            [prompt] = [prompt for prompt in summarizer.prompts if f'name="tool_{index}"' in prompt]
            assert re.search(rf'<message role="tool_return" name="tool_{index}">\ny+\[\.\.\.\]\n</message>', prompt)
        assert _summary(result.all_messages()) == f'summary {len(summarizer.prompts)}'

    async def test_a_history_that_fits_is_summarized_in_one_unchanged_request(self) -> None:
        history = _turns(12)
        small = _Summarizer()
        large = _Summarizer(window=1_000_000)
        unknown = _Summarizer(window=1_000_000, known=False)

        for summarizer in (large, unknown, small):
            agent = _agent(_compaction(summarizer))
            await agent.run('next', message_history=history)

        # A window the request fits sends the one request a summarizer without a known window gets.
        assert len(large.prompts) == 1
        assert large.prompts == unknown.prompts
        assert all(f'question {index}' in large.prompts[0] for index in range(12))
        assert '<previous-summary>' not in large.prompts[0]
        assert len(small.prompts) > 1

    async def test_a_request_whose_fixed_part_alone_overflows_goes_out_whole(self) -> None:
        """With no room for any transcript, the request isn't chunked: the provider decides, as before."""
        summarizer = _Summarizer()
        history = _turns(2, size=10)
        compaction = _compaction(summarizer, instructions='Summarize. ' * 2_000)
        agent = _agent(compaction)

        with pytest.raises(ModelContextWindowExceededError):
            await agent.run('next', message_history=history)

    @pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
    async def test_the_span_records_the_summary_calls(self, capfire: CaptureLogfire) -> None:
        history = _turns(12)
        for summarizer in (_Summarizer(), _Summarizer(window=1_000_000)):
            agent = _agent(_compaction(summarizer))
            agent.instrument = InstrumentationSettings()
            await agent.run('next', message_history=history)

        spans = [span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'compact_messages']
        assert [span['attributes'].get('compaction.summary_calls') for span in spans] == [4, None]


def _ours_metadata() -> dict[str, object]:
    """The metadata marking a `CompactionPart` as a summary `SummarizingCompaction` wrote."""
    return {'pydantic-ai-harness.compaction.summary.v1': True}
