"""The image, audio and video input flags `OpenRouterProvider` takes from its snapshot of OpenRouter's model list.

A weekly workflow refreshes the snapshot (`pydantic_ai/providers/_openrouter_input_modalities.py`), so these tests
replace it with a fixed one rather than depend on what OpenRouter lists for a model in a given week. The end-to-end
tests read the request off a mock client: the claim is what is sent in place of the file, not how OpenRouter answers.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.profiles import ModelProfile

from ...conftest import try_import

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall,
        Function,
    )

    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers import openrouter as openrouter_providers
    from pydantic_ai.providers._openrouter_input_modalities import OPENROUTER_INPUT_MODALITIES
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    from ..mock_openai import MockOpenAI, get_mock_chat_completion_kwargs


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

INPUT_MODALITIES = {
    'acme/text-only': (),
    'acme/vision': ('image',),
    'acme/omni': ('image', 'audio', 'video'),
    'acme/vision:batch': (),
    '~acme/vision-latest': ('image',),
    'google/gemini-3-flash-preview': ('image',),
}

SCREENSHOT = BinaryContent(b'\x89PNG', media_type='image/png', identifier='screenshot')
SCREENSHOT_NOTE = 'File screenshot (image/png, 4 bytes) was not sent: this model cannot read images.'


@pytest.fixture(autouse=True)
def input_modalities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openrouter_providers, 'OPENROUTER_INPUT_MODALITIES', INPUT_MODALITIES)


def file_input_flags(profile: ModelProfile) -> dict[str, bool | None]:
    return {
        'image': profile.get('supports_image_input'),
        'audio': profile.get('supports_audio_file_input'),
        'video': profile.get('supports_video_input'),
        'document': profile.get('supports_document_input'),
    }


def provider_flags(model_name: str) -> dict[str, bool | None]:
    profile = OpenRouterProvider.model_profile(model_name)
    assert profile is not None
    return file_input_flags(profile)


def _completion(message: ChatCompletionMessage) -> ChatCompletion:
    choice = Choice.model_construct(index=0, message=message, finish_reason='stop', native_finish_reason='stop')
    return ChatCompletion.model_construct(
        id='123', choices=[choice], created=1704067200, model='test', object='chat.completion', provider='test'
    )


def _screenshot_call() -> ChatCompletion:
    call = ChatCompletionMessageFunctionToolCall(
        id='call_1', type='function', function=Function(name='screenshot', arguments='{}')
    )
    return _completion(ChatCompletionMessage.model_construct(role='assistant', content=None, tool_calls=[call]))


class TestOpenRouterInputModalities:
    @pytest.mark.parametrize(
        ('model_name', 'expected'),
        [
            pytest.param(
                'acme/text-only', {'image': False, 'audio': False, 'video': False, 'document': None}, id='text-only'
            ),
            pytest.param('acme/vision', {'image': True, 'audio': False, 'video': False, 'document': None}, id='vision'),
            # `video` beats the provider's fallback, which turns video off for a model nothing says reads it.
            pytest.param('acme/omni', {'image': True, 'audio': True, 'video': True, 'document': None}, id='omni'),
            # A variant the snapshot doesn't list takes its base model's kinds.
            pytest.param(
                'acme/vision:free', {'image': True, 'audio': False, 'video': False, 'document': None}, id='tag'
            ),
            # A variant the snapshot lists, because it takes other kinds than its base model, takes its own.
            pytest.param(
                'acme/vision:batch',
                {'image': False, 'audio': False, 'video': False, 'document': None},
                id='listed-tag',
            ),
            pytest.param(
                '~acme/vision-latest', {'image': True, 'audio': False, 'video': False, 'document': None}, id='alias'
            ),
            # A model the snapshot doesn't have keeps the profile it would have without it.
            pytest.param(
                'acme/unlisted', {'image': None, 'audio': None, 'video': False, 'document': None}, id='unlisted'
            ),
            pytest.param(
                'acme/unlisted:free',
                {'image': None, 'audio': None, 'video': False, 'document': None},
                id='unlisted-tag',
            ),
        ],
    )
    def test_flags_from_snapshot(self, model_name: str, expected: dict[str, bool | None]):
        """`document` is never set: OpenRouter parses a PDF to text for a model that doesn't take files."""
        assert provider_flags(model_name) == expected

    def test_hand_written_profile_wins_over_snapshot(self):
        """`google_model_profile` sets `supports_video_input` itself, so only the image and audio flags are listed."""
        assert provider_flags('google/gemini-3-flash-preview') == {
            'image': True,
            'audio': False,
            'video': True,
            'document': None,
        }

    def test_user_profile_wins_over_snapshot(self):
        model = OpenRouterModel(
            'acme/text-only',
            provider=OpenRouterProvider(api_key='test'),
            profile=ModelProfile(supports_image_input=True),
        )
        assert file_input_flags(model.profile) == {'image': True, 'audio': False, 'video': False, 'document': True}

    def test_unlisted_model_keeps_its_profile(self):
        model = OpenRouterModel('acme/unlisted', provider=OpenRouterProvider(api_key='test'))
        assert file_input_flags(model.profile) == {'image': True, 'audio': True, 'video': False, 'document': True}

    @pytest.mark.parametrize(
        ('model_name', 'noted'),
        [pytest.param('acme/text-only', True, id='text-only'), pytest.param('acme/vision', False, id='vision')],
    )
    async def test_tool_image_on_text_only_model_is_sent_as_note(
        self, allow_model_requests: None, model_name: str, noted: bool
    ):
        """OpenRouter rejects an image sent to a text-only model, so a tool's image goes to it as a note instead."""
        client = MockOpenAI.create_mock(
            [_screenshot_call(), _completion(ChatCompletionMessage.model_construct(role='assistant', content='Done.'))]
        )
        agent = Agent(OpenRouterModel(model_name, provider=OpenRouterProvider(openai_client=client)))

        @agent.tool_plain
        def screenshot() -> BinaryContent:
            return SCREENSHOT

        await agent.run('Take a screenshot.')

        sent = repr(get_mock_chat_completion_kwargs(client)[-1]['messages'])
        assert (SCREENSHOT_NOTE in sent) is noted
        assert ('image_url' in sent) is not noted


def test_committed_snapshot_is_sorted_and_holds_only_file_kinds():
    """The snapshot is as `scripts/update_openrouter_input_modalities.py` writes it, so a refresh diffs line by line."""
    assert OPENROUTER_INPUT_MODALITIES
    assert list(OPENROUTER_INPUT_MODALITIES) == sorted(OPENROUTER_INPUT_MODALITIES)
    assert [
        model_id
        for model_id, kinds in OPENROUTER_INPUT_MODALITIES.items()
        if kinds != tuple(kind for kind in ('image', 'audio', 'video') if kind in kinds)
    ] == []
