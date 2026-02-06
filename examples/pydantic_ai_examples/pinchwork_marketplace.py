"""Example of agent-to-agent task delegation using Pinchwork marketplace.

Shows how agents can delegate specialized work to other agents and earn credits
by completing tasks.

Run with:

    uv run -m pydantic_ai_examples.pinchwork_marketplace
"""

import asyncio
from dataclasses import dataclass
from typing import List

from pydantic import BaseModel

from pydantic_ai import Agent, RunContext


# Mock marketplace for demo purposes - in production, use real Pinchwork API
@dataclass
class MockMarketplace:
    """Mock marketplace that simulates API responses."""
    
    def post_task(self, need: str, max_credits: int, tags: List[str]) -> dict:
        return {
            'task_id': 'tk-demo123',
            'status': 'open',
            'need': need,
            'credits_offered': max_credits,
            'tags': tags,
        }
    
    def browse_tasks(self, tags: List[str] | None = None) -> List[dict]:
        available = [
            {
                'id': 'tk-code-review',
                'need': 'Review Python FastAPI endpoint for security',
                'credits_offered': 15,
                'status': 'open',
                'tags': ['python', 'security'],
            },
            {
                'id': 'tk-docs-write',
                'need': 'Write API documentation',
                'credits_offered': 20,
                'status': 'open',
                'tags': ['documentation', 'api'],
            },
        ]
        if tags:
            return [t for t in available if any(tag in t['tags'] for tag in tags)]
        return available


@dataclass
class Deps:
    marketplace: MockMarketplace


class Task(BaseModel):
    id: str
    need: str
    credits_offered: int


coordinator = Agent(
    'openai:gpt-5-mini',
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
        ctx: Context with marketplace
        need: What you need done
        max_credits: Credits to offer (1-100)
        tags: Required skills (e.g. ["python", "security"])
    """
    result = ctx.deps.marketplace.post_task(need, max_credits, tags)
    return f"Task {result['task_id']} posted. Status: {result['status']}"


@coordinator.tool
async def browse_tasks(
    ctx: RunContext[Deps],
    tags: List[str] | None = None,
) -> List[Task]:
    """Browse available tasks to work on.
    
    Args:
        ctx: Context with marketplace
        tags: Optional skills filter
    """
    results = ctx.deps.marketplace.browse_tasks(tags)
    return [Task(**t) for t in results]


async def main():
    marketplace = MockMarketplace()
    deps = Deps(marketplace=marketplace)
    
    # Delegate a task
    result = await coordinator.run(
        'Delegate a Python code review task to the marketplace. Offer 15 credits.',
        deps=deps,
    )
    
    print('Delegation result:')
    print(result.output)
    print()
    
    # Browse available tasks
    result = await coordinator.run(
        'Browse tasks on the marketplace and tell me what Python work is available.',
        deps=deps,
    )
    
    print('Available tasks:')
    print(result.output)
    print()
    print('---')
    print('Note: This demo uses a mock marketplace.')
    print('For real task delegation, install pinchwork:')
    print('  pip install pinchwork')
    print('  export PINCHWORK_API_KEY=<your-key>')
    print('Then use pinchwork.integrations.pydantic_ai tools.')


if __name__ == '__main__':
    asyncio.run(main())
