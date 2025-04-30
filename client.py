import asyncio

from rich.pretty import pprint

from pydantic_ai.a2a.client import A2AClient
from pydantic_ai.a2a.schema import Message, TextPart

client = A2AClient()


async def main():
    message = Message(role='user', parts=[TextPart(type='text', text='Hello, world!')])
    task = await client.send_task(message=message)
    pprint(task)


if __name__ == '__main__':
    asyncio.run(main())
