from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from pydantic_ai import Agent, BinaryContent, ImageUrl, ModelRetry, VideoUrl
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.messages import (
    FilePart,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.native_tools import CodeExecutionTool, WebFetchTool, WebSearchTool
from pydantic_ai.output import NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.settings import ThinkingEffort

from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from google.genai.types import MediaResolution

    from pydantic_ai.models.google_interactions import GoogleInteractionsModel, GoogleInteractionsModelSettings
    from pydantic_ai.profiles.google import GoogleModelProfile
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'), pytest.mark.vcr]


@pytest.fixture
def interactions_model(gemini_api_key: str, request_capture: RequestCapture) -> GoogleInteractionsModel:
    return GoogleInteractionsModel(
        'gemini-3.8-flash',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
    )


@pytest.mark.parametrize('stream', [False, True])
async def test_stateless_tool_history(
    interactions_model: GoogleInteractionsModel,
    request_capture: RequestCapture,
    stream: bool,
    allow_model_requests: None,
):
    agent = Agent(interactions_model)

    @agent.tool_plain
    def city_temperature(city: str) -> int:
        """Look up the current temperature in the requested city."""
        assert city == 'Paris'
        return 21

    if stream:
        async with agent.run_stream('Use city_temperature to find the temperature in Paris.') as result:
            output = await result.get_output()
            history = result.all_messages()
    else:
        result = await agent.run('Use city_temperature to find the temperature in Paris.')
        output, history = result.output, result.all_messages()
    assert '21' in output
    responses = [message for message in history if isinstance(message, ModelResponse)]
    call = next(part for response in responses for part in response.parts if isinstance(part, ToolCallPart))
    assert call.tool_call_id
    assert any(isinstance(part, ThinkingPart) and part.signature for response in responses for part in response.parts)
    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
    final = await agent.run('Repeat that temperature without calling a tool.', message_history=history)
    assert '21' in final.output
    bodies = [request_capture.body(index=index) for index in range(len(request_capture.paths))]
    assert len(bodies) >= 3
    assert all(body['store'] is False and 'previous_interaction_id' not in body for body in bodies)
    assert all(not response.provider_response_id for response in responses)
    previous_input = bodies[0]['input']
    assert isinstance(previous_input, list)
    second_input = bodies[1]['input']
    assert isinstance(second_input, list)
    assert second_input[: len(previous_input)] == previous_input
    for response in responses:
        assert response.provider_details is not None
        steps = response.provider_details['google_interactions_steps']
        final_input = bodies[-1]['input']
        assert isinstance(final_input, list)
        sent_steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(final_input)
        for step in TypeAdapter(list[dict[str, JsonValue]]).validate_python(steps):
            assert any(
                all(key == 'content' or sent.get(key) == value for key, value in step.items()) for sent in sent_steps
            )
        for part in response.parts:
            if isinstance(part, TextPart):
                assert any(
                    part.content == item.get('text')
                    for sent in sent_steps
                    for item in TypeAdapter(list[dict[str, JsonValue]]).validate_python(sent.get('content', []))
                )
            elif isinstance(part, ToolCallPart):
                assert any(
                    sent.get('id') == part.tool_call_id and sent.get('arguments') == part.args_as_dict()
                    for sent in sent_steps
                )


class City(BaseModel):
    name: str
    country: str


@pytest.mark.parametrize('mode', ['native', 'tool', 'prompted'])
@pytest.mark.parametrize('stream', [False, True])
async def test_structured_output(
    interactions_model: GoogleInteractionsModel, mode: str, stream: bool, allow_model_requests: None
):
    output_type = (
        NativeOutput(City) if mode == 'native' else ToolOutput(City) if mode == 'tool' else PromptedOutput(City)
    )
    agent = Agent(interactions_model, output_type=output_type)
    if stream:
        async with agent.run_stream('Name the capital city of France and its country.') as result:
            output = await result.get_output()
    else:
        output = (await agent.run('Name the capital city of France and its country.')).output
    assert output == City(name='Paris', country='France')


@pytest.mark.parametrize('stream', [False, True])
async def test_retry(interactions_model: GoogleInteractionsModel, stream: bool, allow_model_requests: None):
    agent = Agent(interactions_model)
    calls = 0

    @agent.tool_plain
    def get_password() -> str:
        """Get the password. Retry if the service is temporarily unavailable."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelRetry('Temporarily unavailable. Call get_password again.')
        return 'silver fox'

    if stream:
        async with agent.run_stream('Use get_password to tell me the password.') as result:
            output = await result.get_output()
    else:
        output = (await agent.run('Use get_password to tell me the password.')).output
    assert calls == 2
    assert 'silver fox' in output.lower()


@pytest.mark.parametrize('native_tool', ['code', 'search', 'fetch'])
@pytest.mark.parametrize('stream', [False, True])
async def test_native_tools(
    interactions_model: GoogleInteractionsModel, native_tool: str, stream: bool, allow_model_requests: None
):
    tool = (
        CodeExecutionTool() if native_tool == 'code' else WebSearchTool() if native_tool == 'search' else WebFetchTool()
    )
    prompts = {
        'code': 'Use code_execution to calculate the sum of the first 137 prime numbers. Include the numerical answer.',
        'search': 'Use google_search to find the current stable Python version and cite a source.',
        'fetch': 'Use url_context to read https://ai.google.dev/gemini-api/docs/interactions-overview and summarize it in a sentence.',
    }
    agent = Agent(interactions_model, capabilities=[NativeTool(tool)])
    if stream:
        async with agent.run_stream(prompts[native_tool]) as result:
            assert await result.get_output()
            history = result.all_messages()
    else:
        result = await agent.run(prompts[native_tool])
        assert result.output
        history = result.all_messages()
    responses = [message for message in history if isinstance(message, ModelResponse)]
    assert any(isinstance(part, NativeToolCallPart) for response in responses for part in response.parts)
    assert any(isinstance(part, NativeToolReturnPart) for response in responses for part in response.parts)
    if native_tool in ('search', 'fetch'):
        calls = [part for response in responses for part in response.parts if isinstance(part, NativeToolCallPart)]
        returns = [part for response in responses for part in response.parts if isinstance(part, NativeToolReturnPart)]
        assert any(call.args_as_dict().get('queries' if native_tool == 'search' else 'urls') for call in calls)
        results = [
            item for part in returns for item in TypeAdapter(list[dict[str, JsonValue]]).validate_python(part.content)
        ]
        if native_tool == 'search':
            assert any(item.get('search_suggestions') for item in results)
            assert any(call.provider_details and call.provider_details.get('search_type') for call in calls)
        else:
            assert any(item.get('url') and item.get('status') for item in results)
    final = await agent.run(
        'Summarize the answer briefly using only the preceding information.', message_history=history
    )
    assert final.output


@pytest.mark.parametrize('stream', [False, True])
async def test_image_input(
    interactions_model: GoogleInteractionsModel, image_content: BinaryContent, stream: bool, allow_model_requests: None
):
    agent = Agent(interactions_model)
    prompt = ['Describe this image in a sentence.', image_content]
    if stream:
        async with agent.run_stream(prompt) as result:
            assert await result.get_output()
    else:
        assert (await agent.run(prompt)).output


async def test_deferred_approval(interactions_model: GoogleInteractionsModel, allow_model_requests: None):
    from pydantic_ai import DeferredToolRequests, DeferredToolResults

    agent = Agent(interactions_model, output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def send_invoice(customer: str) -> str:
        """Send the customer's invoice after approval."""
        assert customer == 'Alice'
        return 'Invoice INV-123 sent to Alice.'

    pending = await agent.run('Send the invoice to Alice using send_invoice.')
    assert isinstance(pending.output, DeferredToolRequests)
    call = pending.output.approvals[0]
    history = ModelMessagesTypeAdapter.validate_json(pending.all_messages_json())
    completed = await agent.run(
        message_history=history, deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: True})
    )
    assert isinstance(completed.output, str)
    assert 'INV-123' in completed.output


@pytest.mark.parametrize('stream', [False, True])
async def test_image_output(
    gemini_api_key: str, request_capture: RequestCapture, stream: bool, allow_model_requests: None
):
    from pydantic_ai import BinaryImage

    model = GoogleInteractionsModel(
        'gemini-3.1-flash-image',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
    )
    agent = Agent(model, output_type=BinaryImage)
    if stream:
        async with agent.run_stream('Draw one small red square on a white background.') as result:
            image = await result.get_output()
    else:
        image = (await agent.run('Draw one small red square on a white background.')).output
    assert image.media_type.startswith('image/')
    assert image.data


@pytest.mark.parametrize('stream', [False, True])
async def test_mcp(interactions_model: GoogleInteractionsModel, stream: bool, allow_model_requests: None):
    from pydantic_ai.native_tools import MCPServerTool

    agent = Agent(
        interactions_model,
        capabilities=[
            NativeTool(
                MCPServerTool(id='deepwiki', url='https://mcp.deepwiki.com/mcp', allowed_tools=['read_wiki_structure'])
            )
        ],
    )
    prompt = 'Call read_wiki_structure exactly once for pydantic/pydantic, then name one documentation topic from its result.'
    if stream:
        async with agent.run_stream(prompt) as result:
            assert await result.get_output()
            history = result.all_messages()
    else:
        result = await agent.run(prompt)
        assert result.output
        history = result.all_messages()
    assert any(
        isinstance(message, ModelResponse)
        and any(
            isinstance(part, NativeToolCallPart) and part.tool_name == 'mcp_server:deepwiki' for part in message.parts
        )
        for message in history
    )
    assert (await agent.run('Repeat that documentation topic without calling a tool.', message_history=history)).output


@pytest.mark.parametrize('stream', [False, True])
async def test_file_search(interactions_model: GoogleInteractionsModel, stream: bool, allow_model_requests: None):
    import io

    from pydantic_ai.native_tools import FileSearchTool

    client = interactions_model.client
    store = await client.aio.file_search_stores.create(config={'display_name': 'interactions-test'})
    assert store.name is not None
    try:
        await client.aio.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store.name,
            file=io.BytesIO(b'The secret project code is emerald tiger.'),
            config={'mime_type': 'text/plain', 'display_name': 'secret.txt'},
        )
        agent = Agent(interactions_model, capabilities=[NativeTool(FileSearchTool(file_store_ids=[store.name]))])
        prompt = 'Use file_search to find the secret project code.'
        if stream:
            async with agent.run_stream(prompt) as result:
                assert 'emerald tiger' in (await result.get_output()).lower()
        else:
            assert 'emerald tiger' in (await agent.run(prompt)).output.lower()
    finally:
        await client.aio.file_search_stores.delete(name=store.name, config={'force': True})


@pytest.mark.parametrize('kind', ['document', 'audio', 'video'])
async def test_multimodal_input(
    interactions_model: GoogleInteractionsModel, kind: str, request: pytest.FixtureRequest, allow_model_requests: None
):
    content: BinaryContent = request.getfixturevalue(f'{kind}_content')
    assert (await Agent(interactions_model).run(['Summarize this file in one sentence.', content])).output


@pytest.mark.parametrize('source', ['binary', 'youtube'])
async def test_video_processing_controls(
    interactions_model: GoogleInteractionsModel,
    video_content: BinaryContent,
    request_capture: RequestCapture,
    source: str,
    allow_model_requests: None,
):
    video = VideoUrl('https://www.youtube.com/watch?v=dQw4w9WgXcQ') if source == 'youtube' else video_content
    video.vendor_metadata = {'start_offset': '0s', 'end_offset': '1s', 'fps': 1}
    settings: GoogleInteractionsModelSettings = {'google_video_resolution': MediaResolution.MEDIA_RESOLUTION_LOW}
    result = await Agent(interactions_model, model_settings=settings).run(
        ['Describe this video in one sentence.', video]
    )
    assert result.output
    steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(request_capture.body()['input'])
    content = TypeAdapter(list[dict[str, JsonValue]]).validate_python(steps[0]['content'])
    assert content[1]['processing'] == {'type': 'static', 'start_offset': '0s', 'end_offset': '1s', 'fps': 1}
    assert content[1]['resolution'] == 'low'
    if source == 'youtube':
        assert isinstance(video, VideoUrl)
        assert content[1]['uri'] == video.url
        assert 'data' not in content[1]


@pytest.mark.parametrize('stream', [False, True])
async def test_audio_output(
    gemini_api_key: str, request_capture: RequestCapture, stream: bool, allow_model_requests: None
):

    model = GoogleInteractionsModel(
        'gemini-3.8-flash-tts',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
    )
    settings: GoogleInteractionsModelSettings = {'google_response_format': {'type': 'audio'}}
    messages = [ModelRequest(parts=[UserPromptPart('Say hello in a friendly voice.')])]
    if stream:
        async with model_request_stream(model, messages, model_settings=settings) as result:
            async for _ in result:
                pass
            response = result.get()
    else:
        response = await model_request(model, messages, model_settings=settings)
    assert any(
        isinstance(part, FilePart) and part.content.media_type.startswith('audio/') and part.content.data
        for part in response.parts
    )


async def test_native_voice_configuration(
    gemini_api_key: str, request_capture: RequestCapture, allow_model_requests: None
):

    model = GoogleInteractionsModel(
        'gemini-3.8-flash-tts',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
    )
    settings: GoogleInteractionsModelSettings = {
        'google_response_format': {'type': 'audio'},
        'google_generation_config': {'speech_config': [{'voice': 'Kore', 'language': 'en-US'}]},
    }
    response = await model_request(model, [ModelRequest.user_text_prompt('Say hello.')], model_settings=settings)
    assert any(isinstance(part, FilePart) and part.content.data for part in response.parts)
    config = TypeAdapter(dict[str, JsonValue]).validate_python(request_capture.body()['generation_config'])
    assert config['speech_config'] == [{'voice': 'Kore', 'language': 'en-US'}]


@pytest.mark.parametrize(
    'model_name,kind,native_result',
    [pytest.param('gemini-3.8-flash', kind, True, id=kind) for kind in ('image', 'document', 'audio', 'video')]
    + [
        pytest.param('gemini-4-argon', kind, True, id=f'argon-{kind}')
        for kind in ('image', 'document', 'audio', 'video')
    ]
    + [pytest.param('gemini-3.8-flash', 'document', False, id='fallback-document')],
)
async def test_multimodal_tool_return(
    interactions_model: GoogleInteractionsModel,
    model_name: str,
    kind: str,
    native_result: bool,
    request: pytest.FixtureRequest,
    request_capture: RequestCapture,
    allow_model_requests: None,
):
    content: BinaryContent = request.getfixturevalue(f'{kind}_content')
    provider = interactions_model.provider
    assert provider is not None
    profile: GoogleModelProfile | None = (
        None if native_result else GoogleModelProfile(google_supported_mime_types_in_tool_returns=())
    )
    agent = Agent(GoogleInteractionsModel(model_name, provider=provider, profile=profile))

    @agent.tool_plain
    def get_file() -> BinaryContent:
        """Fetch the file to be summarized."""
        return content

    assert (await agent.run('Use get_file and summarize the file in one sentence.')).output
    steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(request_capture.body(index=-1)['input'])
    result = next(step for step in reversed(steps) if step.get('type') == 'function_result')
    returned = TypeAdapter(list[dict[str, JsonValue]]).validate_python(result['result'])
    if native_result:
        assert any(item.get('type') == kind and item.get('mime_type') == content.media_type for item in returned)
    else:
        assert all(item.get('type') == 'text' for item in returned)
        assert steps[-1]['type'] == 'user_input'
        files = TypeAdapter(list[dict[str, JsonValue]]).validate_python(steps[-1]['content'])
        assert any(item.get('type') == kind and item.get('mime_type') == content.media_type for item in files)
        assert 'get_file' in str(files[0]['text'])


@pytest.mark.parametrize('direction', ['to_interactions', 'to_generate_content'])
async def test_cross_api_history(
    interactions_model: GoogleInteractionsModel, direction: str, allow_model_requests: None
):
    from pydantic_ai.models.google import GoogleModel

    provider = interactions_model.provider
    assert provider is not None
    generate_content = GoogleModel(interactions_model.model_name, provider=provider)
    source, target = (
        (generate_content, interactions_model)
        if direction == 'to_interactions'
        else (interactions_model, generate_content)
    )
    agent = Agent(source)

    @agent.tool_plain
    def get_secret() -> str:
        """Look up the secret phrase."""
        return 'emerald tiger'

    first = await agent.run('Call get_secret to get the secret phrase.')
    history = ModelMessagesTypeAdapter.validate_json(first.all_messages_json())
    assert (
        'emerald tiger'
        in (await agent.run('Repeat the secret phrase.', model=target, message_history=history)).output.lower()
    )


async def test_vertex(vertex_provider: GoogleProvider, allow_model_requests: None):
    agent = Agent(GoogleInteractionsModel('gemini-3.8-flash', provider=vertex_provider))

    @agent.tool_plain
    def get_secret() -> str:
        """Look up the secret phrase."""
        return 'emerald tiger'

    result = await agent.run('Use get_secret and tell me the secret phrase.')
    assert 'emerald tiger' in result.output.lower()
    history = ModelMessagesTypeAdapter.validate_json(result.all_messages_json())
    async with agent.run_stream('Repeat the secret phrase.', message_history=history) as stream:
        assert 'emerald tiger' in (await stream.get_output()).lower()


async def test_image_uri_output(gemini_api_key: str, request_capture: RequestCapture, allow_model_requests: None):
    from pydantic_ai import BinaryImage

    settings: GoogleInteractionsModelSettings = {'google_response_format': {'type': 'image', 'delivery': 'uri'}}
    model = GoogleInteractionsModel(
        'gemini-3.1-flash-image',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
        settings=settings,
    )
    from pydantic_ai.exceptions import ModelHTTPError

    with pytest.raises(ModelHTTPError, match='Image delivery mode is not supported'):
        await Agent(model, output_type=BinaryImage).run('Draw one small red square on a white background.')


async def test_native_url_input(
    interactions_model: GoogleInteractionsModel, request_capture: RequestCapture, allow_model_requests: None
):
    image = ImageUrl('https://storage.googleapis.com/generativeai-downloads/images/scones.jpg')
    result = await Agent(interactions_model).run(['Describe the baked goods in this picture.', image])
    assert any(word in result.output.lower() for word in ('scone', 'pastr', 'biscuit', 'bread'))
    steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(request_capture.body()['input'])
    content = TypeAdapter(list[dict[str, JsonValue]]).validate_python(steps[0]['content'])
    assert content[-1] == {'type': 'image', 'uri': image.url, 'mime_type': 'image/jpeg'}


async def test_video_uri_output_requires_storage(
    gemini_api_key: str, request_capture: RequestCapture, allow_model_requests: None
):
    from pydantic_ai.exceptions import ModelHTTPError

    settings: GoogleInteractionsModelSettings = {'google_response_format': {'type': 'video', 'delivery': 'uri'}}
    model = GoogleInteractionsModel(
        'gemini-omni-1.2-flash',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
        settings=settings,
    )
    with pytest.raises(ModelHTTPError, match='store=true is required'):
        await Agent(model).run('Create a short video of a red ball.')


async def test_native_settings(
    interactions_model: GoogleInteractionsModel, request_capture: RequestCapture, allow_model_requests: None
):

    settings: GoogleInteractionsModelSettings = {
        'thinking': 'low',
        'google_interactions_safety_settings': [{'type': 'harassment', 'threshold': 'block_only_high'}],
        'google_labels': {'purpose': 'test'},
        'service_tier': 'default',
    }
    result = await Agent(interactions_model, model_settings=settings).run('Reply hello.')
    assert result.output
    body = request_capture.body()
    assert body['generation_config'] == {'thinking_level': 'low', 'thinking_summaries': 'auto'}
    assert body['safety_settings'] == settings['google_interactions_safety_settings']
    assert body['labels'] == {'purpose': 'test'}
    assert body['service_tier'] == 'standard'


async def test_google_maps(interactions_model: GoogleInteractionsModel, allow_model_requests: None):

    settings: GoogleInteractionsModelSettings = {'google_native_tools': [{'type': 'google_maps'}]}
    result = await Agent(interactions_model, model_settings=settings).run(
        'Use Google Maps to find the street address of the Eiffel Tower in Paris.'
    )
    assert result.output
    assert any(
        isinstance(message, ModelResponse)
        and any(isinstance(part, NativeToolCallPart) and part.tool_name == 'google_maps' for part in message.parts)
        for message in result.all_messages()
    )
    places = [
        item.get('places')
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, NativeToolReturnPart) and part.tool_name == 'google_maps'
        for item in TypeAdapter(list[dict[str, JsonValue]]).validate_python(part.content)
    ]
    assert any(places)


async def test_image_generation_tool_configuration(
    gemini_api_key: str, request_capture: RequestCapture, allow_model_requests: None
):
    from pydantic_ai.messages import BinaryImage
    from pydantic_ai.native_tools import ImageGenerationTool

    model = GoogleInteractionsModel(
        'gemini-3.1-flash-image',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
    )
    agent = Agent(
        model,
        output_type=BinaryImage,
        capabilities=[NativeTool(ImageGenerationTool(aspect_ratio='16:9', size='1K', output_format='jpeg'))],
    )
    result = await agent.run('Draw a simple red circle on a white background.')
    assert result.output.data and result.output.media_type == 'image/jpeg'
    assert request_capture.body()['response_format'] == {
        'type': 'image',
        'aspect_ratio': '16:9',
        'image_size': '1K',
        'mime_type': 'image/jpeg',
    }


@pytest.fixture
def pptx_content(assets_path: Path) -> BinaryContent:
    return BinaryContent(
        (assets_path / 'dummy.pptx').read_bytes(),
        media_type='application/vnd.openxmlformats-officedocument.presentationml.presentation',
    )


async def test_pptx_input(
    interactions_model: GoogleInteractionsModel, pptx_content: BinaryContent, allow_model_requests: None
):
    result = await Agent(interactions_model).run(['Describe the text on this slide.', pptx_content])
    assert 'dummy' in result.output.lower()


@pytest.mark.parametrize('kind', ['music', 'video'])
async def test_generated_media(
    gemini_api_key: str, request_capture: RequestCapture, kind: str, allow_model_requests: None
):

    model = GoogleInteractionsModel(
        'lyria-3-clip-preview' if kind == 'music' else 'gemini-omni-1.1-flash',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=180)),
    )
    settings: GoogleInteractionsModelSettings = {
        'google_response_format': {'type': 'audio'} if kind == 'music' else {'type': 'video', 'resolution': '360p'}
    }
    prompt = (
        'An instrumental acoustic guitar piece.'
        if kind == 'music'
        else 'A red ball rolling across a plain white floor, fixed camera.'
    )
    response = await model_request(model, [ModelRequest.user_text_prompt(prompt)], model_settings=settings)
    files = [part for part in response.parts if isinstance(part, FilePart)]
    assert files and files[0].content.data
    assert files[0].content.media_type.startswith('audio/' if kind == 'music' else 'video/')
    assert request_capture.body()['store'] is False


@pytest.mark.parametrize('thinking', [False, 'minimal'])
@pytest.mark.parametrize('stream', [False, True])
async def test_minimal_thinking_resolves_to_supported_level(
    interactions_model: GoogleInteractionsModel,
    request_capture: RequestCapture,
    thinking: ThinkingEffort | bool,
    stream: bool,
    allow_model_requests: None,
):
    # Verified against the public Interactions API: this model rejects MINIMAL and accepts LOW.

    settings: GoogleInteractionsModelSettings = {'thinking': thinking}
    agent = Agent(interactions_model, model_settings=settings)
    if stream:
        async with agent.run_stream('Reply exactly OK.') as result:
            assert await result.get_output()
    else:
        assert (await agent.run('Reply exactly OK.')).output
    assert request_capture.body()['generation_config'] == {
        'thinking_level': 'low',
        'thinking_summaries': 'none' if thinking is False else 'auto',
    }
