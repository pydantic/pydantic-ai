"""Tests for the file-input flags `OpenRouterModel` fills from OpenRouter's model list.

These serve the model list and the chat completions from an `httpx2.MockTransport` rather than a cassette: the real
list holds every model on OpenRouter (over 800 KB), and what is tested is how many times it's fetched and what a
failed fetch leaves behind, which a recording of one successful exchange can't show.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent, BinaryImage
from pydantic_ai.models import override_allow_model_requests

from ..._inline_snapshot import snapshot
from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers import openrouter as openrouter_providers
    from pydantic_ai.providers.openrouter import OpenRouterProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

MODEL_LIST = {
    'data': [
        {'id': 'acme/text-only', 'architecture': {'input_modalities': ['text']}},
        {'id': 'acme/vision', 'architecture': {'input_modalities': ['text', 'image', 'file']}},
        {'id': 'google/gemini-3-flash', 'architecture': {'input_modalities': ['text', 'image']}},
        {'id': 'acme/no-architecture'},
        {'id': 'acme/no-modalities', 'architecture': {'modality': 'text->text'}},
    ]
}


def _completion(message: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': 'gen-1',
        'object': 'chat.completion',
        'created': 0,
        'model': 'acme/text-only',
        'provider': 'Acme',
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', **message}}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }


@dataclass
class FakeOpenRouter:
    model_list_status: int = 200
    model_list_requests: int = 0
    chat_requests: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    replies: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path == '/api/v1/models':
            self.model_list_requests += 1
            return httpx2.Response(self.model_list_status, json=MODEL_LIST)
        self.chat_requests.append(json.loads(request.content))
        return httpx2.Response(200, json=_completion(self.replies.pop(0) if self.replies else {'content': 'ok'}))

    def provider(self) -> OpenRouterProvider:
        return OpenRouterProvider(
            api_key='test', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle))
        )


@pytest.fixture
def openrouter() -> Iterator[FakeOpenRouter]:
    # The suite-wide `no_openrouter_model_list` fixture marks the list as fetched; start from a process that hasn't.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(openrouter_providers, '_input_modalities_by_base_url', {})
        yield FakeOpenRouter()


def _file_input_flags(model: OpenRouterModel) -> dict[str, object]:
    return {
        key: model.profile.get(key)
        for key in (
            'supports_image_input',
            'supports_document_input',
            'supports_audio_file_input',
            'supports_video_input',
        )
    }


class TestOpenRouterInputModalities:
    async def test_run_fills_flags_from_model_list(self, allow_model_requests: None, openrouter: FakeOpenRouter):
        """The list is fetched when a run starts, not when the model is built or its profile read, and only once."""
        provider = openrouter.provider()
        model = OpenRouterModel('acme/text-only', provider=provider)
        assert _file_input_flags(model) == snapshot(
            {
                'supports_image_input': True,
                'supports_document_input': True,
                'supports_audio_file_input': True,
                'supports_video_input': True,
            }
        )
        assert openrouter.model_list_requests == 0

        await Agent(model).run('hello')

        # `file` isn't listed, but OpenRouter parses a PDF to text for a model that doesn't take files.
        assert _file_input_flags(model) == snapshot(
            {
                'supports_image_input': False,
                'supports_document_input': True,
                'supports_audio_file_input': False,
                'supports_video_input': False,
            }
        )

        # Another model on the same base URL, whose profile was read before, is resolved again with the list.
        vision = OpenRouterModel('acme/vision:free', provider=provider)
        assert vision.profile.get('supports_image_input') is True
        await Agent(vision).run('hello')
        await Agent(model).run('hello')
        assert _file_input_flags(vision) == snapshot(
            {
                'supports_image_input': True,
                'supports_document_input': True,
                'supports_audio_file_input': False,
                'supports_video_input': False,
            }
        )
        assert openrouter.model_list_requests == 1

    async def test_tool_image_replaced_on_text_only_model(self, allow_model_requests: None, openrouter: FakeOpenRouter):
        """An image a tool returns to a model OpenRouter lists without `image` input is replaced by a note."""
        openrouter.replies.append(
            {
                'content': None,
                'tool_calls': [
                    {'id': 'call_1', 'type': 'function', 'function': {'name': 'screenshot', 'arguments': '{}'}}
                ],
            }
        )
        agent = Agent(OpenRouterModel('acme/text-only', provider=openrouter.provider()))

        @agent.tool_plain
        def screenshot() -> BinaryImage:
            return BinaryImage(data=b'\x89PNG', media_type='image/png')

        await agent.run('Take a screenshot')

        assert openrouter.chat_requests[-1]['messages'][-1] == snapshot(
            {
                'role': 'tool',
                'tool_call_id': 'call_1',
                'content': 'File 4effda (image/png, 4 bytes) was not sent: this model cannot read images.',
            }
        )

    async def test_hand_written_and_user_profiles_take_precedence(
        self, allow_model_requests: None, openrouter: FakeOpenRouter
    ):
        provider = openrouter.provider()
        # `google_model_profile` sets `supports_video_input` itself, so only the image and audio flags come from the list.
        gemini = OpenRouterModel('google/gemini-3-flash', provider=provider)
        user = OpenRouterModel('acme/text-only', provider=provider, profile={'supports_image_input': True})
        await Agent(gemini).run('hello')
        await Agent(user).run('hello')

        assert _file_input_flags(gemini) == snapshot(
            {
                'supports_image_input': True,
                'supports_document_input': True,
                'supports_audio_file_input': False,
                'supports_video_input': True,
            }
        )
        assert _file_input_flags(user) == snapshot(
            {
                'supports_image_input': True,
                'supports_document_input': True,
                'supports_audio_file_input': False,
                'supports_video_input': False,
            }
        )

    @pytest.mark.parametrize('model_name', ['acme/unlisted', 'acme/no-architecture', 'acme/no-modalities'])
    async def test_unlisted_or_unknown_modalities(
        self, allow_model_requests: None, openrouter: FakeOpenRouter, model_name: str
    ):
        """A model the list doesn't have, or lists without its input modalities, keeps its profile."""
        model = OpenRouterModel(model_name, provider=openrouter.provider())
        await Agent(model).run('hello')

        assert model.profile.get('supports_image_input') is True

    async def test_failed_fetch_keeps_profile(self, allow_model_requests: None, openrouter: FakeOpenRouter):
        """A failed fetch isn't retried in the same process, and leaves the hand-written profile in place."""
        openrouter.model_list_status = 500
        model = OpenRouterModel('acme/text-only', provider=openrouter.provider())

        result = await Agent(model).run('hello')
        await Agent(model).run('hello')

        assert result.output == 'ok'
        assert model.profile.get('supports_image_input') is True
        assert openrouter.model_list_requests == 1

    async def test_no_fetch_without_model_requests(self, openrouter: FakeOpenRouter):
        model = OpenRouterModel('acme/text-only', provider=openrouter.provider())

        with override_allow_model_requests(False), pytest.raises(RuntimeError, match='ALLOW_MODEL_REQUESTS'):
            await Agent(model).run('hello')

        assert openrouter.model_list_requests == 0
