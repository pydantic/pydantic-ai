import asyncio

from mcp import ClientSession, types
from mcp.client.sse import sse_client


async def main():
    async with sse_client('http://localhost:3101/sse') as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool('run_python_code', {'python_code': "print('hello world')"})
            content = result.content[0]
            if isinstance(content, types.TextContent):
                print('Received:', content.text)
                assert content.text.strip() == 'hello world', 'Unexpected response from server'
            else:
                raise ValueError('Unexpected content type:', content)


if __name__ == '__main__':
    asyncio.run(main())
