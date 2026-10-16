from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..realtime.profiles import RealtimeModelProfile


def elevenlabs_realtime_model_profile(model_name: str) -> RealtimeModelProfile:
    """Get the realtime model profile for an ElevenLabs speech-to-speech model."""
    # TODO(elevenlabs-launch): every value below is a placeholder until it is verified against the launch-day
    # API reference and a live recording. Where unverified, the value claims *less* than OpenAI's profile, so
    # an unsupported feature raises a `UserError` up front rather than silently misbehaving on the wire.
    # Branch on `model_name` only once ElevenLabs ships more than one model with different capabilities.
    del model_name
    return {
        # TODO(elevenlabs-launch): `input_audio_buffer.commit` + `response.create` with `turn_detection: null`.
        'supports_manual_turn_control': True,
        # TODO(elevenlabs-launch): `response.cancel`.
        'supports_interruption': True,
        # TODO(elevenlabs-launch): `conversation.item.truncate` (xAI documents it but it doesn't land at barge-in).
        'supports_output_truncation': False,
        # TODO(elevenlabs-launch): `output_modalities: ['text']`; xAI accepts it but always speaks.
        'supports_text_output': False,
        # TODO(elevenlabs-launch): `input_image` content parts.
        'supports_image_input': False,
        # TODO(elevenlabs-launch): `conversation.item.create` for history. Reconnect replays history through
        # this same path, so with `False` a reconnect can't restore the conversation.
        'supports_session_seeding': True,
        'supports_seeding_images': False,
        # TODO(elevenlabs-launch): assistant `output_audio` / user `input_audio` items in seeded history.
        'supports_seeding_audio': False,
        # Not implemented by `ElevenLabsRealtimeModel`, whatever the API offers.
        'supports_webrtc': False,
        # TODO(elevenlabs-launch): a `reasoning` session field.
        'supports_thinking': False,
        # TODO(elevenlabs-launch): does the model keep talking (and answer the user) while a tool call is
        # outstanding? Verify live with a slow tool, as for OpenAI and xAI (`'always'`).
        'async_tool_call_mode': 'always',
        # TODO(elevenlabs-launch): `input_audio_buffer.speech_started` / `.speech_stopped`.
        'emits_input_speech_events': True,
        # TODO(elevenlabs-launch): does a response's `usage.input_tokens` cover the whole conversation?
        'response_usage_covers_context': False,
        # TODO(elevenlabs-launch): PCM rates. ElevenLabs Agents default to 16 kHz; OpenAI Realtime is 24 kHz.
        'audio_input_sample_rate': 24000,
        'audio_output_sample_rate': 24000,
    }
