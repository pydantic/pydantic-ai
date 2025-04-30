import asyncio

from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from rich.pretty import pprint

from pydantic_ai import Agent
from pydantic_ai.a2a import FastA2A
from pydantic_ai.a2a.client import A2AClient
from pydantic_ai.a2a.schema import Message, TextPart

agent = Agent(model='anthropic:claude-3-7-sonnet-latest', name='Potato Agent')
app = FastA2A.from_agent(agent)


async def main():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app)) as client:
            a2a_client = A2AClient(http_client=client, base_url='http://testclient')
            task = await a2a_client.send_task(
                message=Message(role='user', parts=[TextPart(type='text', text='Hello, world!')])
            )
            pprint(task)


if __name__ == '__main__':
    asyncio.run(main())
