"""What each provider model sends for a history, captured offline, and which voice each piece of text lands in.

`capture` runs a history through a real provider model's `prepare_messages`, then its `request`, answers the HTTP
request with a 400 (or, for Bedrock, stops before botocore builds one) so nothing leaves the process, and returns the
request body as the model built it. It doesn't
normalize the history first the way the agent graph does, so pass it a history an agent run ended with.
`voices_of` reads that body back, so a test can ask which voice a marker string reached the model in without knowing
each provider's wire format.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

import httpx
import httpx2
import pytest

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

with try_import() as bedrock_available:
    import boto3
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import() as openrouter_available:
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

Voice: TypeAlias = Literal['system', 'developer', 'user', 'data', 'assistant', 'tool']
"""The voice a piece of text reaches the model in, normalized across providers.

The role that carries it, except that text framed in a tag in the user turn is `'data'`, and text in a `<system>` frame
is `'system'` whatever role carries it.
"""

CapturedProvider: TypeAlias = Literal[
    'anthropic', 'openai-chat', 'openai-responses', 'google', 'mistral', 'cohere', 'bedrock', 'openrouter'
]

CAPTURED_PROVIDERS: tuple[CapturedProvider, ...] = (
    'anthropic',
    'openai-chat',
    'openai-responses',
    'google',
    'mistral',
    'cohere',
    'bedrock',
    'openrouter',
)
"""The providers `capture` can build offline. xAI (gRPC), Groq and Hugging Face aren't wired up yet."""


_INSTALLED: dict[CapturedProvider, Callable[[], bool]] = {
    'anthropic': anthropic_available,
    'openai-chat': openai_available,
    'openai-responses': openai_available,
    'google': google_available,
    'mistral': mistral_available,
    'cohere': cohere_available,
    'bedrock': bedrock_available,
    'openrouter': openrouter_available,
}


class _Captured(Exception):
    """Raised in place of sending a Bedrock request, once its parameters are recorded."""


@dataclass
class _Capture:
    bodies: list[bytes] = field(default_factory=list[bytes])

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(request.content)
        return httpx2.Response(400, json={'error': {'message': 'captured', 'type': 'captured'}}, request=request)

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle))

    def handle_legacy(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content)
        return httpx.Response(400, json={'error': {'message': 'captured', 'type': 'captured'}}, request=request)

    def legacy_client(self) -> httpx.AsyncClient:
        """A client on the legacy `httpx`, for a provider whose SDK still takes one."""
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle_legacy))

    def handle_bedrock(self, params: dict[str, Any], **_: Any) -> None:
        self.bodies.append(json.dumps(params, default=str).encode())
        raise _Captured()

    def bedrock_client(self) -> BedrockRuntimeClient:
        """A Bedrock Runtime client that records each call's parameters before botocore builds the request."""
        client: BedrockRuntimeClient = boto3.client(  # pyright: ignore[reportUnknownMemberType]
            'bedrock-runtime', region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test'
        )
        client.meta.events.register('before-parameter-build.bedrock-runtime', self.handle_bedrock)
        return client


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
    if provider == 'bedrock':
        return BedrockConverseModel(
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            provider=BedrockProvider(bedrock_client=capture.bedrock_client()),
        )
    if provider == 'openrouter':
        return OpenRouterModel(
            'anthropic/claude-sonnet-4.5', provider=OpenRouterProvider(api_key='test', http_client=client)
        )
    return CohereModel('command-r', provider=CohereProvider(api_key='test', http_client=capture.legacy_client()))


@dataclass
class CapturedRequest:
    provider: CapturedProvider
    body: dict[str, Any]

    def voices_of(self, marker: str, frames: Collection[str] | None = None) -> list[Voice]:
        """The voices of the pieces of text in the request that contain `marker`.

        Tool definitions, their descriptions and parameter schemas, are the application's voice: `'system'`. Tool
        call arguments aren't text in a voice, so they're left out. A model without inline system messages gets
        the application's voice as a `<system>` frame in the user turn, so text in such a frame counts as `'system'`.
        Text in another frame in the user turn is `'data'`. Either kind of frame may come in one piece or as an
        opening tag, the body and a closing tag.

        With `frames`, a piece only counts as `'data'` in one of those frames or a reserved `pyai:` one: text that only
        looks like a frame, such as a page's own `<html>`, is unframed user text.
        """
        return [
            'user' if voice == 'data' and frames is not None and not _expected_frame(frame, frames) else voice
            for voice, text, frame in self._pieces()
            if marker in text
        ]

    def _pieces(self) -> list[tuple[Voice, str, str | None]]:
        """Every piece of text with its voice and, for framed data, the frame's tag."""
        pieces: list[tuple[Voice, str, str | None]] = []
        open_frame: str | None = None
        # The pieces of a frame that hasn't closed yet: framed once it closes, unframed user text if it never does.
        pending: list[str] = []
        pieces.extend(
            ('system', text, None) for text in _strings([self.body.get('tools'), self.body.get('toolConfig')])
        )
        for voice, text in _VOICE_READERS[self.provider](self.body):
            stripped = text.strip()
            if open_frame is not None and voice != 'user':
                pieces.extend(('user', piece, None) for piece in pending)
                open_frame, pending = None, []
            if open_frame is None and _is_framed(stripped, 'system'):
                pieces.append(('system', text, None))
            elif voice != 'user':
                pieces.append((voice, text, None))
            elif open_frame is not None:
                closing = f'</{open_frame}>'
                if (closes := text.count(closing)) == 0:
                    pending.append(text)
                    continue
                pieces.extend((_frame_voice(open_frame), piece, open_frame) for piece in pending)
                # Text after the closing tag in the same piece isn't framed, so the whole piece counts as unframed.
                if closes == 1 and stripped.endswith(closing):
                    pieces.append((_frame_voice(open_frame), text, open_frame))
                else:
                    pieces.append((voice, text, None))
                open_frame, pending = None, []
            elif (opened := _FRAME_OPEN.fullmatch(stripped)) is not None:
                open_frame, pending = opened.group(1), [text]
            elif (tag := _frame_tag(stripped)) is not None:
                pieces.append(('data', text, tag))
            else:
                pieces.append((voice, text, None))
        pieces.extend(('user', piece, None) for piece in pending)
        return pieces


async def capture(
    provider: CapturedProvider, messages: Sequence[ModelMessage], params: ModelRequestParameters | None = None
) -> CapturedRequest:
    """The request body `provider`'s model builds for `messages`, through its `prepare_messages` and `request`.

    `params` are the request parameters a run sent, so the tools it offered are on the wire too.
    """
    if not _INSTALLED[provider]():
        pytest.skip(f'the {provider} extra is not installed')
    recorder = _Capture()
    model = _model(provider, recorder)
    params = model.customize_request_parameters(params or ModelRequestParameters())
    try:
        await model.request(model.prepare_messages(list(messages), params), None, params)
    except (ModelHTTPError, _Captured):
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


_TAG = r'[a-z][a-z0-9_]*(?::[a-z][a-z0-9_]*)?'
_FRAME_OPEN = re.compile(rf'<({_TAG})(?: [^<>]*)?>')
_FRAMED = re.compile(rf'<({_TAG})(?: [^<>]*)?>.*</\1>', re.DOTALL)


def _frame_tag(text: str) -> str | None:
    """The tag of the one frame all of `text` is, an opening tag first and its only closing tag last, or `None`."""
    framed = _FRAMED.fullmatch(text)
    if framed is None or text.count(f'</{framed.group(1)}>') != 1:
        return None
    return framed.group(1)


def _is_framed(text: str, tag: str) -> bool:
    return _frame_tag(text) == tag


def _expected_frame(frame: str | None, frames: Collection[str]) -> bool:
    return frame is not None and (frame in frames or frame.startswith('pyai:'))


def _frame_voice(tag: str) -> Voice:
    return 'system' if tag == 'system' else 'data'


def _strings(value: Any) -> Iterator[str]:
    """Every string nested in a JSON value, as the model reads it."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict | list):
        items = cast(dict[str, Any], value).values() if isinstance(value, dict) else cast(list[Any], value)
        for item in items:
            yield from _strings(item)


def _role_voice(role: str) -> Voice:
    return cast(Voice, {'model': 'assistant', 'function': 'tool'}.get(role, role))


def _anthropic_voices(body: dict[str, Any]) -> Iterator[tuple[Voice, str]]:
    for text in _texts(body.get('system', [])):
        yield 'system', text
    for message in body['messages']:
        role: Voice = _role_voice(message['role'])
        for block in message['content'] if isinstance(message['content'], list) else [message['content']]:
            item = cast(dict[str, Any], block) if isinstance(block, dict) else None
            if item is not None and item.get('type') == 'tool_result':
                for text in _texts(item.get('content', [])):
                    yield 'tool', text
            else:
                for text in _texts(block):
                    yield role, text


def _chat_voices(body: dict[str, Any]) -> Iterator[tuple[Voice, str]]:
    for message in body['messages']:
        for text in _texts(message.get('content')):
            yield _role_voice(message['role']), text


def _responses_voices(body: dict[str, Any]) -> Iterator[tuple[Voice, str]]:
    # `OpenAIResponsesModel` sends system text as `input` items, not `instructions`. If it ever moved there, the text
    # would be found in no voice, which the voice check fails on rather than passing.
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
                for text in _strings(part['functionResponse'].get('response')):
                    yield 'tool', text
            else:
                for text in _texts(part):
                    yield role, text


def _bedrock_voices(body: dict[str, Any]) -> Iterator[tuple[Voice, str]]:
    for text in _texts(body.get('system', [])):
        yield 'system', text
    for message in body['messages']:
        role = _role_voice(message['role'])
        for block in message['content']:
            if 'toolResult' in block:
                for text in _strings(block['toolResult'].get('content')):
                    yield 'tool', text
            else:
                for text in _texts(block):
                    yield role, text


_VOICE_READERS: dict[CapturedProvider, Callable[[dict[str, Any]], Iterator[tuple[Voice, str]]]] = {
    'anthropic': _anthropic_voices,
    'openai-chat': _chat_voices,
    'openai-responses': _responses_voices,
    'google': _google_voices,
    'mistral': _chat_voices,
    'cohere': _chat_voices,
    'bedrock': _bedrock_voices,
    'openrouter': _chat_voices,
}
