import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server('test-server')
log_level = 'unset'


@server.call_tool()
async def query_db(name: str, arguments: dict[str, float]) -> list[TextContent]:
    if name == 'celsius_to_fahrenheit':
        celsius = arguments['celsius']
        return [TextContent(type='text', text=str((celsius * 9 / 5) + 32))]
    elif name == 'get_log_level':
        return [TextContent(type='text', text=log_level)]
    else:
        raise ValueError(f'Unknown tool: {name}')


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name='celsius_to_fahrenheit', description='Convert Celsius to Fahrenheit.', inputSchema={}),
        Tool(name='get_log_level', description='', inputSchema={}),
    ]


@server.set_logging_level()
async def set_logging_level(level: str) -> None:
    global log_level
    log_level = level


async def run_stdio_server():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(run_stdio_server())
