import base64
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.server.session import ServerSession
from mcp.types import (
    BlobResourceContents,
    CreateMessageResult,
    EmbeddedResource,
    ResourceLink,
    SamplingMessage,
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)
from pydantic import AnyUrl, BaseModel

mcp = FastMCP('Pydantic AI MCP Server')
log_level = 'unset'


@mcp.tool(annotations=ToolAnnotations(title='Celsius to Fahrenheit'))
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
            uri=AnyUrl('resource://kiwi.png'),
            blob=base64.b64encode(data).decode('utf-8'),
            mimeType='image/png',
        ),
    )


@mcp.tool()
async def get_image_resource_link() -> ResourceLink:
    return ResourceLink(
        type='resource_link',
        uri=AnyUrl('resource://kiwi.png'),
        name='kiwi.png',
    )


@mcp.resource('resource://kiwi.png', mime_type='image/png')
async def kiwi_resource() -> bytes:
    return Path(__file__).parent.joinpath('assets/kiwi.png').read_bytes()


@mcp.tool()
async def get_audio_resource() -> EmbeddedResource:
    data = Path(__file__).parent.joinpath('assets/marcelo.mp3').read_bytes()
    return EmbeddedResource(
        type='resource',
        resource=BlobResourceContents(
            uri=AnyUrl('resource://marcelo.mp3'),
            blob=base64.b64encode(data).decode('utf-8'),
            mimeType='audio/mpeg',
        ),
    )


@mcp.tool()
async def get_audio_resource_link() -> ResourceLink:
    return ResourceLink(
        type='resource_link',
        uri=AnyUrl('resource://marcelo.mp3'),
        name='marcelo.mp3',
    )


@mcp.resource('resource://marcelo.mp3', mime_type='audio/mpeg')
async def marcelo_resource() -> bytes:
    return Path(__file__).parent.joinpath('assets/marcelo.mp3').read_bytes()


@mcp.tool()
async def get_product_name() -> EmbeddedResource:
    return EmbeddedResource(
        type='resource',
        resource=TextResourceContents(
            uri=AnyUrl('resource://product_name.txt'),
            text='Pydantic AI',
        ),
    )


@mcp.tool()
async def get_product_name_link() -> ResourceLink:
    return ResourceLink(
        type='resource_link',
        uri=AnyUrl('resource://product_name.txt'),
        name='product_name.txt',
    )


@mcp.resource('resource://product_name.txt', mime_type='text/plain')
async def product_name_resource() -> str:
    return Path(__file__).parent.joinpath('assets/product_name.txt').read_text()


@mcp.tool()
async def get_image() -> Image:
    data = Path(__file__).parent.joinpath('assets/kiwi.png').read_bytes()
    return Image(data=data, format='png')


@mcp.tool()
async def get_dict() -> dict[str, Any]:
    return {'foo': 'bar', 'baz': 123}


@mcp.tool(structured_output=False)
async def get_unstructured_dict() -> dict[str, Any]:
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


@mcp.tool()
async def echo_deps(ctx: Context[ServerSession, None]) -> dict[str, Any]:
    """Echo the run context.

    Args:
        ctx: Context object containing request and session information.

    Returns:
        Dictionary with an echo message and the deps.
    """
    await ctx.info('This is an info message')

    deps: Any = getattr(ctx.request_context.meta, 'deps')
    return {'echo': 'This is an echo message', 'deps': deps}


@mcp.tool()
async def use_sampling(ctx: Context[ServerSession, None], foo: str) -> CreateMessageResult:
    """Use sampling callback."""

    result = await ctx.session.create_message(
        [
            SamplingMessage(role='assistant', content=TextContent(type='text', text='')),
            SamplingMessage(role='user', content=TextContent(type='text', text=foo)),
        ],
        max_tokens=1_024,
        system_prompt='this is a test of MCP sampling',
        temperature=0.5,
        stop_sequences=['potato'],
    )
    return result


class UserResponse(BaseModel):
    response: str


@mcp.tool()
async def use_elicitation(ctx: Context[ServerSession, None], question: str) -> str:
    """Use elicitation callback to ask the user a question."""

    result = await ctx.elicit(message=question, schema=UserResponse)

    if result.action == 'accept' and result.data:
        return f'User responded: {result.data.response}'
    else:
        return f'User {result.action}ed the elicitation'


@mcp._mcp_server.set_logging_level()  # pyright: ignore[reportPrivateUsage]
async def set_logging_level(level: str) -> None:
    global log_level
    log_level = level


if __name__ == '__main__':
    mcp.run()
