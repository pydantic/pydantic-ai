import base64
from pathlib import Path
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.utilities.types import Image
from mcp.types import (
    Annotations,
    AudioContent,
    BlobResourceContents,
    CreateMessageResult,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    ResourceLink,
    SamplingMessage,
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)
from pydantic import BaseModel

mcp = FastMCP('Pydantic AI MCP Server', instructions='Be a helpful assistant.')
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
    data = Path(__file__).parent.joinpath('assets/kiwi.jpg').read_bytes()
    return EmbeddedResource(
        type='resource',
        resource=BlobResourceContents.model_validate(
            {'uri': 'resource://kiwi.jpg', 'blob': base64.b64encode(data).decode('utf-8'), 'mimeType': 'image/jpeg'}
        ),
    )


@mcp.tool()
async def get_image_resource_link() -> ResourceLink:
    return ResourceLink(
        type='resource_link',
        uri='resource://kiwi.jpg',
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
        resource=BlobResourceContents.model_validate(
            {'uri': 'resource://marcelo.mp3', 'blob': base64.b64encode(data).decode('utf-8'), 'mimeType': 'audio/mpeg'}
        ),
    )


@mcp.tool()
async def get_audio_resource_link() -> ResourceLink:
    return ResourceLink(
        type='resource_link',
        uri='resource://marcelo.mp3',
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
            uri='resource://product_name.txt',
            text='Pydantic AI',
        ),
    )


@mcp.tool()
async def get_product_name_link() -> ResourceLink:
    return ResourceLink(
        type='resource_link',
        uri='resource://product_name.txt',
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


@mcp.tool()
async def get_log_level(ctx: Context) -> str:
    """Get the current log level.

    Returns:
        The current log level.
    """
    await ctx.info('this is a log message')
    return log_level


@mcp.tool()
async def echo_deps(ctx: Context) -> dict[str, Any]:
    """Echo the run context.

    Args:
        ctx: Context object containing request and session information.

    Returns:
        Dictionary with an echo message and the deps.
    """
    await ctx.info('This is an info message')

    assert ctx.request_context is not None
    deps: Any = getattr(ctx.request_context.meta, 'deps')
    return {'echo': 'This is an echo message', 'deps': deps}


@mcp.tool()
async def use_sampling(ctx: Context, foo: str) -> CreateMessageResult:
    """Use sampling callback."""

    result = await ctx.session.create_message(  # pyright: ignore[reportDeprecated]
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
async def get_client_info(ctx: Context) -> dict[str, Any] | None:
    """Get information about the connected MCP client.

    Returns:
        Dictionary with client info (name, version, etc.) or None if not available.
    """
    client_params = ctx.session.client_params
    if client_params is None:
        return None
    # Dump by alias: the wire spelling is the same under MCP SDK v1 and v2.
    client_info: dict[str, Any] = client_params.model_dump(by_alias=True)['clientInfo']
    return {
        'name': client_info['name'],
        'version': client_info['version'],
        'title': client_info.get('title'),
        'websiteUrl': client_info.get('websiteUrl'),
    }


@mcp.tool()
async def use_elicitation(ctx: Context, question: str) -> str:
    """Use elicitation callback to ask the user a question."""

    result = await ctx.elicit(message=question, response_type=UserResponse)

    if result.action == 'accept' and result.data:
        return f'User responded: {result.data.response}'
    else:
        return f'User {result.action}ed the elicitation'


async def hidden_tool() -> str:
    """A tool that is hidden by default."""
    return 'I was hidden!'


@mcp.tool()
async def enable_hidden_tool(ctx: Context) -> str:
    """Enable the hidden tool, triggering a ToolListChangedNotification."""
    mcp.add_tool(hidden_tool)
    await ctx.session.send_tool_list_changed()
    return 'Hidden tool enabled'


@mcp.prompt()
def simple_prompt() -> str:
    """A simple prompt template."""
    return 'This is a simple prompt'


@mcp.prompt()
def parameterized_prompt(name: str, topic: str) -> str:
    """A prompt template with parameters."""
    return f"Hello {name}, let's talk about {topic}!"


@mcp.prompt()
def annotated_text_prompt() -> list[PromptMessage]:
    """A prompt template with annotated text content."""
    return [
        PromptMessage(
            role='user',
            content=TextContent(
                type='text',
                text='annotated text',
                annotations=Annotations(audience=['user'], priority=1.0),
            ),
        )
    ]


@mcp.prompt()
def text_meta_prompt() -> list[PromptMessage]:
    """A prompt template with `_meta` text metadata."""
    return [PromptMessage(role='user', content=TextContent(type='text', text='meta text', _meta={'source': 'mcp'}))]


@mcp.prompt()
def image_prompt() -> list[PromptMessage]:
    """A prompt template with image content."""
    return [
        PromptMessage(
            role='user',
            content=ImageContent.model_validate(
                {
                    'type': 'image',
                    'data': base64.b64encode(b'image-bytes').decode('utf-8'),
                    'mimeType': 'image/jpeg',
                    'annotations': Annotations(audience=['user'], priority=0.8),
                }
            ),
        )
    ]


@mcp.prompt()
def audio_prompt() -> list[PromptMessage]:
    """A prompt template with audio content."""
    return [
        PromptMessage(
            role='user',
            content=AudioContent.model_validate(
                {
                    'type': 'audio',
                    'data': base64.b64encode(b'audio-bytes').decode('utf-8'),
                    'mimeType': 'audio/mpeg',
                    'annotations': Annotations(audience=['assistant'], priority=0.3),
                }
            ),
        )
    ]


@mcp.prompt()
def embedded_resource_prompt() -> list[PromptMessage]:
    """A prompt template with an embedded text resource."""
    return [
        PromptMessage(
            role='user',
            content=EmbeddedResource(
                type='resource',
                resource=TextResourceContents.model_validate(
                    {
                        'uri': 'resource://product_name.txt',
                        'text': 'Pydantic AI',
                        'mimeType': 'text/plain',
                    }
                ),
                annotations=Annotations(audience=['user'], priority=0.5),
            ),
        )
    ]


@mcp.prompt()
def resource_link_prompt() -> list[PromptMessage]:
    """A prompt template with a resource link."""
    return [
        PromptMessage(
            role='user',
            content=ResourceLink.model_validate(
                {
                    'type': 'resource_link',
                    'uri': 'resource://kiwi.jpg',
                    'name': 'kiwi-image',
                    'title': 'Kiwi Image',
                    'description': 'A photo of a kiwi fruit',
                    'mimeType': 'image/jpeg',
                }
            ),
        )
    ]


if __name__ == '__main__':
    mcp.run()
