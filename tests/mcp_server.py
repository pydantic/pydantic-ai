"""The MCP server the stdio tests spawn as `python -m tests.mcp_server`.

Written against the `fastmcp` server API rather than `mcp.server.fastmcp`, which is the MCP SDK v1
server that SDK v2 removed: `fastmcp.server` is the one surface both FastMCP generations ship, so
this module imports under either. The in-process counterpart is the `fastmcp_server` fixture in
`tests/test_mcp.py`; keep the two in the same idiom.
"""

import base64
from pathlib import Path
from typing import Any, cast

from fastmcp.prompts import Message
from fastmcp.server import FastMCP
from fastmcp.utilities.types import Image
from mcp.types import (
    Annotations,
    AudioContent,
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

mcp: FastMCP[None] = FastMCP('Pydantic AI MCP Server', instructions='Be a helpful assistant.')


@mcp.tool(annotations={'title': 'Celsius to Fahrenheit'})
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


# SDK v2 retypes every `uri` from `AnyUrl` to `str` and rejects an `AnyUrl` instance, so the URIs
# here are plain strings, cast to satisfy the v1 annotation these tests type-check against.
@mcp.tool()
async def get_image_resource() -> EmbeddedResource:
    data = Path(__file__).parent.joinpath('assets/kiwi.jpg').read_bytes()
    return EmbeddedResource(
        type='resource',
        resource=BlobResourceContents(
            uri=cast(AnyUrl, 'resource://kiwi.jpg'),
            blob=base64.b64encode(data).decode('utf-8'),
            mimeType='image/jpeg',
        ),
    )


@mcp.tool()
async def get_image_resource_link() -> ResourceLink:
    return ResourceLink(
        type='resource_link',
        uri=cast(AnyUrl, 'resource://kiwi.jpg'),
        name='kiwi.jpeg',
    )


@mcp.resource('resource://kiwi.jpg', mime_type='image/jpeg')
async def kiwi_resource() -> bytes:
    return Path(__file__).parent.joinpath('assets/kiwi.jpg').read_bytes()


@mcp.tool()
async def get_audio_resource() -> EmbeddedResource:
    data = Path(__file__).parent.joinpath('assets/marcelo.mp3').read_bytes()
    return EmbeddedResource(
        type='resource',
        resource=BlobResourceContents(
            uri=cast(AnyUrl, 'resource://marcelo.mp3'),
            blob=base64.b64encode(data).decode('utf-8'),
            mimeType='audio/mpeg',
        ),
    )


@mcp.tool()
async def get_audio_resource_link() -> ResourceLink:
    return ResourceLink(
        type='resource_link',
        uri=cast(AnyUrl, 'resource://marcelo.mp3'),
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
            uri=cast(AnyUrl, 'resource://product_name.txt'),
            text='Pydantic AI',
        ),
    )


@mcp.tool()
async def get_product_name_link() -> ResourceLink:
    return ResourceLink(
        type='resource_link',
        uri=cast(AnyUrl, 'resource://product_name.txt'),
        name='product_name.txt',
    )


@mcp.resource(
    'resource://product_name.txt',
    mime_type='text/plain',
    annotations=Annotations(audience=['user', 'assistant'], priority=0.5),
)
async def product_name_resource() -> str:
    return Path(__file__).parent.joinpath('assets/product_name.txt').read_text(encoding='utf-8')


@mcp.resource('resource://greeting/{name}', mime_type='text/plain')
async def greeting_resource_template(name: str) -> str:
    """Dynamic greeting resource template."""
    return f'Hello, {name}!'


@mcp.tool()
async def get_image() -> Image:
    data = Path(__file__).parent.joinpath('assets/kiwi.jpg').read_bytes()
    return Image(data=data, format='jpg')


@mcp.tool()
async def get_dict() -> dict[str, Any]:
    return {'foo': 'bar', 'baz': 123}


# `output_schema=None` is the fastmcp spelling of the SDK v1 server's `structured_output=False`.
@mcp.tool(output_schema=None)
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


@mcp.prompt()
def simple_prompt() -> str:
    """A simple prompt template."""
    return 'This is a simple prompt'


@mcp.prompt()
def parameterized_prompt(name: str, topic: str) -> str:
    """A prompt template with parameters."""
    return f"Hello {name}, let's talk about {topic}!"


@mcp.prompt()
def annotated_text_prompt() -> list[Message]:
    """A prompt template with annotated text content."""
    return [
        Message(
            content=TextContent(
                type='text',
                text='annotated text',
                annotations=Annotations(audience=['user'], priority=1.0),
            )
        )
    ]


@mcp.prompt()
def text_meta_prompt() -> list[Message]:
    """A prompt template with `_meta` text metadata."""
    return [Message(content=TextContent(type='text', text='meta text', _meta={'source': 'mcp'}))]


@mcp.prompt()
def image_prompt() -> list[Message]:
    """A prompt template with image content."""
    return [
        Message(
            content=ImageContent(
                type='image',
                data=base64.b64encode(b'image-bytes').decode('utf-8'),
                mimeType='image/jpeg',
                annotations=Annotations(audience=['user'], priority=0.8),
            )
        )
    ]


@mcp.prompt()
def audio_prompt() -> list[Message]:
    """A prompt template with audio content."""
    return [
        Message(
            content=AudioContent(
                type='audio',
                data=base64.b64encode(b'audio-bytes').decode('utf-8'),
                mimeType='audio/mpeg',
                annotations=Annotations(audience=['assistant'], priority=0.3),
            )
        )
    ]


@mcp.prompt()
def embedded_resource_prompt() -> list[Message]:
    """A prompt template with an embedded text resource."""
    return [
        Message(
            content=EmbeddedResource(
                type='resource',
                resource=TextResourceContents(
                    uri=cast(AnyUrl, 'resource://product_name.txt'),
                    text='Pydantic AI',
                    mimeType='text/plain',
                ),
                annotations=Annotations(audience=['user'], priority=0.5),
            )
        )
    ]


@mcp.prompt()
def resource_link_prompt() -> list[Message]:
    """A prompt template with a resource link."""
    return [
        Message(
            content=ResourceLink(
                type='resource_link',
                uri=cast(AnyUrl, 'resource://kiwi.jpg'),
                name='kiwi-image',
                title='Kiwi Image',
                description='A photo of a kiwi fruit',
                mimeType='image/jpeg',
            )
        )
    ]


if __name__ == '__main__':
    mcp.run()
