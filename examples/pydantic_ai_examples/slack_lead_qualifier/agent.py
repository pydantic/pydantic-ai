from textwrap import dedent

import logfire

from pydantic_ai import Agent, NativeOutput
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

from .models import Analysis, Profile, Unknown

### [agent]
agent = Agent(
    'openai:gpt-4o',
    instructions=dedent(
        """
        Your job is to evaluate a user who's joined our public Slack, and provide a summary of how important they might be as a customer to us.

        Our company, Pydantic, offers three products:
        * Pydantic Validation: A powerful library for data validation in Python (free and open source)
        * Pydantic AI: A Python Agent Framework (free and open source)
        * Pydantic Logfire: a general purpose observability framework (Traces, Logs and Metrics) with special support for
        Python, Javascript/TypeScript and Rust. It's particularly useful in AI applications. (commercial paid product)

        We particularly want to find developers working for or running prominent/well funded companies that might pay for Pydantic Logfire.

        Always use your search tool to research the user and the company they work for, based on the email domain or what you find on e.g. LinkedIn and GitHub.
        Note that our products are aimed at software developers, data scientists, and AI engineers, so if the person you find is not in a technical role,
        you're likely looking at the wrong person. In that case, you should search again with additional keywords to narrow it down to developers.

        If you couldn't find anything useful, return Unknown.
        """
    ),
    tools=[duckduckgo_search_tool()],
    output_type=NativeOutput([Analysis, Unknown]),
)  ### [/agent]


### [analyze_profile]
@logfire.instrument('Analyze profile')
async def analyze_profile(profile: Profile) -> Analysis | Unknown:
    result = await agent.run(profile.as_prompt())
    return result.output  ### [/analyze_profile]
