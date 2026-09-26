"""Build a research agent from the blocks packaged as `Researcher`.

Run the packaged equivalent without assembling the blocks:

    uvx --with 'pydantic-ai-harness[researcher]' clai -a pydantic_ai_harness.researcher:researcher_agent -m openai:gpt-5.6-sol
"""

import os

from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai.models import Model

from pydantic_ai_harness import SubAgent, SubAgents, ToolOutputLimits

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'openai:gpt-5.6-sol')

# Keep these written-out blocks in sync with the defaults in `pydantic_ai_harness.researcher`.
INSTRUCTIONS = """\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.
"""

sub_researcher = SubAgent(
    Agent(
        name='researcher',
        description='Research a focused sub-question on the web and report back with findings and source links',
        capabilities=[WebSearch(local=True), WebFetch(local=True), ToolOutputLimits()],
    )
)


def build_agent(model: Model | str = DEFAULT_MODEL) -> Agent:
    """Build a web research agent."""
    return Agent(
        model,
        instructions=INSTRUCTIONS,
        capabilities=[
            WebSearch(local=True),  # Use native search when supported, with DuckDuckGo as the local fallback.
            WebFetch(local=True),  # Use native URL fetching when supported, with a local fallback.
            SubAgents(agents=[sub_researcher], agent_folders=None),
            ToolOutputLimits(),  # Bound large search responses.
        ],
    )


def main() -> None:
    """Start an interactive research session."""
    build_agent().to_cli_sync()


if __name__ == '__main__':
    main()
