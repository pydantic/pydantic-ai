"""Show a restored conversation the way its turns streamed, without running any tool again."""

from collections.abc import Sequence

from rich.console import Console

from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent, PartEndEvent, PartStartEvent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import StreamRenderer
from pydantic_clai2.ui.rendering.tool_output import terminal_text

RESUMED_TURNS = 10
"""How many of the latest prompts `/resume` shows; the model still receives the whole history."""


def turn_starts(messages: Sequence[ModelMessage]) -> list[int]:
    """Indexes of the requests that start a turn: a user prompt, not a tool result or a steer in a seen run."""
    starts: list[int] = []
    seen_runs: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, ModelRequest) or (message.run_id is not None and message.run_id in seen_runs):
            continue
        if message.run_id is not None:
            seen_runs.add(message.run_id)
        if any(isinstance(part, UserPromptPart) for part in message.parts) and not any(
            isinstance(part, (ToolReturnPart, RetryPromptPart)) for part in message.parts
        ):
            starts.append(index)
    return starts


def prompt_text(part: UserPromptPart) -> str:
    """The prompt as the editor echoed it, with a marker for each attachment."""
    content = [part.content] if isinstance(part.content, str) else part.content
    return '\n'.join(item if isinstance(item, str) else '[attachment]' for item in content)


async def render_history(
    messages: Sequence[ModelMessage], *, console: Console, renderer: StreamRenderer, turns: int = RESUMED_TURNS
) -> None:
    """Print the last `turns` prompts, answers, and tool calls through the renderer a live turn uses.

    Tool output that only streamed (shell previews, diffs) is not saved, so a tool shows its call header.
    """
    starts = turn_starts(messages)
    hidden = max(len(starts) - turns, 0)
    if hidden:
        messages = messages[starts[hidden] :]
        console.print(
            f'{hidden} earlier turn{"s" if hidden > 1 else ""} not shown; the model still has them.',
            style=theme.color(theme.MUTED),
        )
        console.print()
    seen_runs: set[str] = set()
    for message in messages:
        # A prompt in a run already shown was a steer, which the transcript never echoed.
        steered = message.run_id in seen_runs
        if message.run_id is not None:
            seen_runs.add(message.run_id)
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and not steered:
                    await renderer.finish()
                    console.print(f'> {terminal_text(prompt_text(part))}', markup=False, highlight=False)
                    console.print()
                elif isinstance(part, ToolReturnPart) or isinstance(part, RetryPromptPart) and part.tool_name:
                    await renderer.on_stream_event(FunctionToolResultEvent(part))
            continue
        for index, part in enumerate(message.parts):
            if isinstance(part, (TextPart, ThinkingPart)):
                await renderer.on_stream_event(PartStartEvent(index=index, part=part))
                await renderer.on_stream_event(PartEndEvent(index=index, part=part))
            elif isinstance(part, ToolCallPart):
                await renderer.on_stream_event(FunctionToolCallEvent(part))
    await renderer.finish()
