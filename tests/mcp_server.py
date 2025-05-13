import base64
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.types import BlobResourceContents, EmbeddedResource, TextResourceContents

mcp = FastMCP('PydanticAI MCP Server')
log_level = 'unset'


@mcp.tool()
async def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit.

    Args:
        celsius: Temperature in Celsius

    Returns:
        Temperature in Fahrenheit
    """
    return (celsius * 9 / 5) + 32


@mcp.tool()
async def get_weather_forecast(location: str) -> str:
    """Get the weather forecast for a location.

    Args:
        location: The location to get the weather forecast for.

    Returns:
        The weather forecast for the location.
    """
    return f'The weather in {location} is sunny and 26 degrees Celsius.'


@mcp.tool()
async def get_image_resource() -> EmbeddedResource:
    data = Path(__file__).parent.joinpath('assets/kiwi.png').read_bytes()
    return EmbeddedResource(
        type='resource',
        resource=BlobResourceContents(
            uri='resource://kiwi.png',  # type: ignore
            blob=base64.b64encode(data).decode('utf-8'),
            mimeType='image/png',
        ),
    )


@mcp.tool()
async def get_product_name() -> EmbeddedResource:
    return EmbeddedResource(
        type='resource',
        resource=TextResourceContents(
            uri='resource://product_name.txt',  # type: ignore
            text='PydanticAI',
        ),
    )


@mcp.tool()
async def get_image() -> Image:
    data = Path(__file__).parent.joinpath('assets/kiwi.png').read_bytes()
    return Image(data=data, format='png')


@mcp.tool()
async def get_dict() -> dict[str, Any]:
    return {'foo': 'bar', 'baz': 123}


@mcp.tool()
async def get_error(value: bool = False):
    if value:
        return 'This is not an error'

    raise ValueError('This is an error. Call the tool with True instead')


@mcp.tool()
async def get_none():
    return None


@mcp.tool()
async def get_multiple_items():
    return [
        'This is a string',
        'Another string',
        {'foo': 'bar', 'baz': 123},
        await get_image(),
    ]


@mcp.tool()
async def get_log_level(ctx: Context) -> str:  # type: ignore
    """Get the current log level.

    Returns:
        The current log level.
    """
    await ctx.info('this is a log message')
    return log_level


@mcp._mcp_server.set_logging_level()  # pyright: ignore[reportPrivateUsage]
async def set_logging_level(level: str) -> None:
    global log_level
    log_level = level


if __name__ == '__main__':
    mcp.run()
