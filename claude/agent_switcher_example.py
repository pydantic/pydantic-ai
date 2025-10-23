"""
Example: Dynamic Agent Persona Switching

This demonstrates how to swap an agent's system prompt mid-conversation
using a tool call, without needing a second agent.run() call.

The key is using @agent.system_prompt(dynamic=True) which gets re-evaluated
on each turn, including after tool calls.
"""

from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

# Define the available agent personas
PROMPTS = {
    'alan': 'You are Alan, a die hard Warriors fan. You love talking about the Warriors!',
    'bob': 'You are Bob, a total Lakers homer. The Lakers are the greatest franchise ever!',
}


# Deps stores the current agent persona
@dataclass
class AgentDeps:
    current_agent: str = 'alan'


# Create the agent
agent = Agent('openai:gpt-4o', deps_type=AgentDeps)


@agent.system_prompt(dynamic=True)
def dynamic_persona_prompt(ctx: RunContext[AgentDeps]) -> str:
    """This gets re-evaluated on each turn, so we can change the persona dynamically!"""
    return PROMPTS[ctx.deps.current_agent]


@agent.tool
def change_agent(ctx: RunContext[AgentDeps], agent_name: str) -> str:
    """Change to a different agent persona.

    Args:
        agent_name: Name of the agent to switch to (e.g., 'alan' or 'bob')
    """
    if agent_name not in PROMPTS:
        return f"Unknown agent: {agent_name}. Available: {', '.join(PROMPTS.keys())}"

    # Just modify the deps - the dynamic system prompt will pick it up!
    ctx.deps.current_agent = agent_name
    return f"Switched to {agent_name}"


# Example usage
if __name__ == '__main__':
    import asyncio

    async def main():
        # Start with Alan
        deps = AgentDeps(current_agent='alan')

        # Alan responds
        result = await agent.run(
            'What do you think about basketball?',
            deps=deps
        )
        print(f"Alan: {result.output}\n")

        # Now let's ask Alan to become Bob
        # Important: reuse the same deps object so the state persists!
        result = await agent.run(
            'Hey, change_agent to bob please. Yo Bob, what you think of that bubble chip?',
            deps=deps  # Same deps object!
        )
        print(f"Bob: {result.output}\n")

    asyncio.run(main())
