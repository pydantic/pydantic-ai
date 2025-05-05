from fasta2a.client import A2AClient
from fasta2a.schema import Message, TextPart

client = A2AClient()


async def main():
    a2a_client = A2AClient()

    response = await a2a_client.send_task(
        message=Message(role='user', parts=[TextPart(type='text', text='Hello, world!')]),
    )

    print(response)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
