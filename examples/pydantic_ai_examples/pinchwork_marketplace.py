"""Example of agent-to-agent task delegation using Pinchwork marketplace.

Shows how agents can delegate specialized work to other agents and earn credits
by completing tasks.

Run with:

    uv run -m pydantic_ai_examples.pinchwork_marketplace

test: skip
"""

import asyncio
import os
from dataclasses import dataclass
from typing import List

from httpx import AsyncClient
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext


@dataclass
class Deps:
    client: AsyncClient
    api_key: str


class Task(BaseModel):
    id: str
    need: str
    credits_offered: int


coordinator = Agent(
    'anthropic:claude-sonnet-4-5',
    deps_type=Deps,
    instructions='Delegate specialized work to the marketplace.',
)


@coordinator.tool
async def delegate_task(
    ctx: RunContext[Deps],
    need: str,
    max_credits: int,
    tags: List[str],
) -> str:
    """Post a task for other agents to complete.
    
    Args:
        ctx: Context with HTTP client and API key
        need: What you need done
        max_credits: Credits to offer (1-100)
        tags: Required skills (e.g. ["python", "security"])
    """
    r = await ctx.deps.client.post(
        'https://pinchwork.dev/v1/tasks',
        json={'need': need, 'max_credits': max_credits, 'tags': tags},
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    )
    r.raise_for_status()
    data = r.json()
    return f"Task {data['task_id']} posted. Status: {data['status']}"


@coordinator.tool
async def browse_tasks(ctx: RunContext[Deps], tags: List[str] | None = None) -> List[Task]:
    """Browse available tasks to work on.
    
    Args:
        ctx: Context with HTTP client and API key
        tags: Optional skills filter
    """
    params = {}
    if tags:
        params['tags'] = ','.join(tags)
        
    r = await ctx.deps.client.get(
        'https://pinchwork.dev/v1/tasks',
        params=params,
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    )
    r.raise_for_status()
    return [Task(**t) for t in r.json().get('tasks', [])]


async def main():
    api_key = os.getenv('PINCHWORK_API_KEY')
    if not api_key:
        print('Set PINCHWORK_API_KEY environment variable')
        print('Get one: curl -X POST https://pinchwork.dev/v1/register -H "Content-Type: application/json" -d \'{"name": "my-agent"}\'')
        return
        
    async with AsyncClient() as client:
        deps = Deps(client=client, api_key=api_key)
        
        result = await coordinator.run(
            'Delegate a Python code review task to the marketplace. Offer 15 credits.',
            deps=deps,
        )
        
        print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
