from pydantic_ai import Agent

agent = Agent('google-vertex:gemini-2.0-flash-exp')


async def main():
    async with agent.run_stream('Hi there!') as result:
        async for message in result.stream_text(delta=True):
            print(message)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
