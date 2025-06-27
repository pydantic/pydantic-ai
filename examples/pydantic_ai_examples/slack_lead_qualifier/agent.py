from textwrap import dedent
from types import NoneType

import logfire

### [imports]
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool  ### [/imports]

from .models import Analysis, Profile

### [agent]
agent = Agent(
    'openai:gpt-4o',
    instructions=dedent(
        """
        Welcome team helper! 🎉

        When a new person joins our public Slack, please put together a brief, **public-info-only** snapshot so we can
        greet them in a way that's most useful to them.

        **What to include**

        1. **Who they are:**  Any publicly available details about their professional role or projects (e.g. LinkedIn,
           GitHub, company bio).
        2. **Where they work:**  Name of the organisation and its domain.
        3. **How we can help:**  On a scale of 1–5, estimate how likely they are to benefit from **Pydantic Logfire**
           (our paid observability tool) based on factors such as team size, product maturity, or AI usage.
           *1 = probably not relevant, 5 = very strong fit.*

        **Our products (for context only)**
        • **Pydantic Validation** – Python data-validation (open source)
        • **Pydantic AI** – Python agent framework (open source)
        • **Pydantic Logfire** – Observability for traces, logs & metrics with first-class AI support (commercial)

        **How to research**

        • Use the provided DuckDuckGo search tool for quick, public look-ups.
        • Stick to information people already publish on the open web; never attempt private or sensitive data.
        • If you can't find enough to form a reasonable view, return **None**.

        Respond with a single `Analysis` object (or `None`) as defined in our codebase.
        """
    ),
    tools=[duckduckgo_search_tool()],
    output_type=NativeOutput([Analysis, NoneType]),
)  ### [/agent]


### [analyze_profile]
@logfire.instrument('Analyze profile')
async def analyze_profile(profile: Profile) -> Analysis | None:
    result = await agent.run(profile.as_prompt())
    return result.output  ### [/analyze_profile]
