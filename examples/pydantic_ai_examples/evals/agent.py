from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from datetime import datetime

from pydantic_ai import Agent, RunContext

from .models import TimeRangeInputs, TimeRangeResponse


@dataclass
class TimeRangeDeps:
    """Dependencies for the time range inference agent.

    While we could just get the current time using datetime.now() directly in the tools or system prompt, passing it
    via deps makes it easier to use a repeatable value during testing. While there are packages like `time-machine`
    that can do this for you, that kind of monkey-patching approach can become unwieldy as things get more complex.
    """

    now: datetime = field(default_factory=lambda: datetime.now().astimezone())


time_range_agent = Agent[TimeRangeDeps, TimeRangeResponse](
    'gpt-5',
    output_type=TimeRangeResponse,  # type: ignore  # we can't yet annotate something as receiving a TypeForm
    deps_type=TimeRangeDeps,
    system_prompt="Convert the user's request into a structured time range.",
    retries=1,
    instrument=True,
)


@time_range_agent.tool
def get_current_time(ctx: RunContext[TimeRangeDeps]) -> str:
    """Get the user's current time and timezone in the format 'Friday, November 22, 2024 11:15:14 PST'."""
    # (The following comment is not in the docstring because the tool docstring is included in model requests.)
    # In practice, you might unconditionally include this in the system prompt, but using a tool for this helps
    # demonstrate some evaluation capabilities, such as checking whether a specific tool was called (or wasn't).
    now_str = ctx.deps.now.strftime(
        '%A, %B %d, %Y %H:%M:%S %Z'
    )  # Format like: Friday, November 22, 2024 11:15:14 PST
    return f"The user's current time is {now_str}."


async def infer_time_range(inputs: TimeRangeInputs) -> TimeRangeResponse:
    """Infer a time range from a user prompt."""
    deps = TimeRangeDeps(now=inputs['now'])
    return (await time_range_agent.run(inputs['prompt'], deps=deps)).output
