"""What each provider model sends for a history, captured offline, and which voice each piece of text lands in.

`capture` runs a history through a real provider model the way `Agent` does (`prepare_messages`, then `request`),
answers the HTTP request with a 400 so nothing leaves the process, and returns the request body as the model built it.
`voices` reads that body back as `(voice, text)` pairs in order, so a test can ask which voice a marker string reached
the model in without knowing each provider's wire format.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

import httpx2

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, ModelRequestParameters

from .conftest import try_import

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as mistral_available:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

with try_import() as cohere_available:
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

Voice: TypeAlias = Literal['system', 'developer', 'user', 'data', 'assistant', 'tool']
"""The voice a piece of text reaches the model in, normalized across providers.

The role that carries it, except that text framed in a tag in the user turn is `'data'`, and text in a `<system>` frame
is `'system'` whatever role carries it.
"""

CapturedProvider: TypeAlias = Literal['anthropic', 'openai-chat', 'openai-responses', 'google', 'mistral', 'cohere']

CAPTURED_PROVIDERS: tuple[CapturedProvider, ...] = (
    'anthropic',
    'openai-chat',
    'openai-responses',
    'google',
    'mistral',
    'cohere',
)
"""The providers `capture` can build offline. Bedrock, xAI (gRPC), Groq and Hugging Face aren't wired up yet."""


@dataclass
class _Capture:
    bodies: list[bytes] = field(default_factory=list[bytes])

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(request.content)
        return httpx2.Response(400, json={'error': {'message': 'captured', 'type': 'captured'}}, request=request)

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle))


def _model(provider: CapturedProvider, capture: _Capture) -> Model:
    client = capture.client()
    if provider == 'anthropic':
        return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test', http_client=client))
    if provider == 'openai-chat':
        return OpenAIChatModel('gpt-5', provider=OpenAIProvider(api_key='test', http_client=client))
    if provider == 'openai-responses':
        return OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='test', http_client=client))
    if provider == 'google':
        return GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test', http_client=client))
    if provider == 'mistral':
        return MistralModel('mistral-large-latest', provider=MistralProvider(api_key='test', http_client=client))
    return CohereModel('command-r', provider=CohereProvider(api_key='test', http_client=client))


@dataclass
class CapturedRequest:
    provider: CapturedProvider
    body: dict[str, Any]

    def voices(self) -> list[tuple[Voice, str]]:
        """Every piece of text in the request, in order, with the voice it reaches the model in.

        Tool call arguments aren't text in a voice, so they're left out.

        A model without inline system messages gets the application's voice as a `<system>` frame in the user turn,
        so text in such a frame counts as `'system'`, whatever role carries it. Text in another frame in the user turn,
        sent either in one piece or as an opening tag, the body and a closing tag, is `'data'`.
        """
        voices: list[tuple[Voice, str]] = []
        open_frame: str | None = None
        for voice, text in _VOICE_READERS[self.provider](self.body):
            if _is_inline_system(text):
                voices.append(('system', text))
            elif voice != 'user':
                voices.append((voice, text))
            elif open_frame is not None:
                # Text after the closing tag in the same piece isn't framed, so the whole piece counts as unframed.
                closing = f'</{open_frame}>'
                voices.append(('data' if closing not in text or text.strip().endswith(closing) else voice, text))
                if closing in text:
                    open_frame = None
            elif (opened := _FRAME_OPEN.fullmatch(text.strip())) is not None:
                open_frame = opened.group(1)
                voices.append(('data', text))
            else:
                voices.append(('data' if _is_framed(text.strip()) else voice, text))
        return voices

    def voices_of(self, marker: str) -> list[Voice]:
        """The voices of the pieces of text that contain `marker`."""
        return [voice for voice, text in self.voices() if marker in text]


async def capture(
    provider: CapturedProvider,
    messages: Sequence[ModelMessage],
    params: ModelRequestParameters | None = None,
) -> CapturedRequest:
    """The request body `provider`'s model builds for `messages`, as `Agent` would send it."""
    recorder = _Capture()
    model = _model(provider, recorder)
    params = params or ModelRequestParameters()
    params = model.customize_request_parameters(params)
    try:
        await model.request(model.prepare_messages(list(messages), params), None, params)
    except ModelHTTPError:
        pass
    assert len(recorder.bodies) == 1, f'{provider} sent {len(recorder.bodies)} requests'
    return CapturedRequest(provider, json.loads(recorder.bodies[0]))


def _texts(content: Any) -> Iterator[str]:
    """The text in a message's content, whatever shape the provider gives it."""
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for item in cast(list[Any], content):
            yield from _texts(item)
    elif isinstance(content, dict):
        item = cast(dict[str, Any], content)
        for key in ('text', 'content', 'output'):
            if key in item:
                yield from _texts(item[key])


_FRAME_OPEN = re.compile(r'<([a-z][a-z0-9_]*)(?: [^<>]*)?>')
_FRAMED = re.compile(r'<([a-z][a-z0-9_]*)(?: [^<>]*)?>.*</\1>', re.DOTALL)


def _is_framed(text: str) -> bool:
    """Whether all of `text` is one frame: an opening tag at the start, and its only closing tag at the end."""
    framed = _FRAMED.fullmatch(text)
    return framed is not None and text.count(f'</{framed.group(1)}>') == 1


def _is_inline_system(text: str) -> bool:
    return text.startswith('<system>') and text.endswith('</system>')


def _role_voice(role: str) -> Voice:
    return {'model': 'assistant', 'function': 'tool'}.get(role, role)  # pyright: ignore[reportReturnType]


def _anthropic_voices(body: dict[str, Any]) -> Iterator[tuple[Voice, str]]:
    for text in _texts(body.get('system', [])):
        yield 'system', text
    for message in body['messages']:
        role: Voice = _role_voice(message['role'])
        for block in message['content'] if isinstance(message['content'], list) else [message['content']]:
            if isinstance(block, dict) and block.get('type') == 'tool_result':
                for text in _texts(block.get('content', [])):
                    yield 'tool', text
            else:
                for text in _texts(block):
                    yield role, text


def _chat_voices(body: dict[str, Any]) -> Iterator[tuple[Voice, str]]:
    for message in body['messages']:
        for text in _texts(message.get('content')):
            yield _role_voice(message['role']), text


def _responses_voices(body: dict[str, Any]) -> Iterator[tuple[Voice, str]]:
    for text in _texts(body.get('instructions')):
        yield 'system', text
    for item in body['input']:
        if 'role' in item:
            for text in _texts(item.get('content')):
                yield _role_voice(item['role']), text
        elif item.get('type') == 'function_call_output':
            for text in _texts(item.get('output')):
                yield 'tool', text


def _google_voices(body: dict[str, Any]) -> Iterator[tuple[Voice, str]]:
    system = body.get('systemInstruction') or body.get('config', {}).get('systemInstruction')
    if system:
        for part in system.get('parts', []):
            for text in _texts(part):
                yield 'system', text
    for content in body['contents']:
        role = _role_voice(content.get('role', 'user'))
        for part in content.get('parts', []):
            if 'functionResponse' in part:
                yield 'tool', json.dumps(part['functionResponse'].get('response'))
            else:
                for text in _texts(part):
                    yield role, text


_VOICE_READERS: dict[CapturedProvider, Callable[[dict[str, Any]], Iterator[tuple[Voice, str]]]] = {
    'anthropic': _anthropic_voices,
    'openai-chat': _chat_voices,
    'openai-responses': _responses_voices,
    'google': _google_voices,
    'mistral': _chat_voices,
    'cohere': _chat_voices,
}
