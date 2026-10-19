from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..realtime.profiles import RealtimeModelProfile


def elevenlabs_agents_realtime_model_profile(model_name: str) -> RealtimeModelProfile:
    """Get the realtime model profile for an ElevenLabs agent.

    The "model name" is an ElevenLabs agent id (e.g. `agent_0101...`), so nothing about the model can
    be inferred from it: every capability fact below describes the Agents platform itself, which is
    the same for every agent. What *does* vary per agent (the audio formats) is validated against
    the profile at connect time (see `ElevenLabsAgentsRealtimeModel.connect`), and the `profile=` argument
    is the escape hatch for agents configured away from ElevenLabs' defaults.
    """
    return {
        # Text mode exists as the per-conversation `conversation.text_only` override, gated by that
        # override's toggle (Security > Overrides in the agent's settings); the connect-time preflight
        # checks the toggle and raises a `UserError` naming it when `output_modality='text'` is requested without it.
        'supports_text_output': True,
        # The agent WebSocket accepts `multimodal_message` file references, but they require a prior
        # REST file upload; discrete image frames are future work.
        'supports_image_input': False,
        # Turn-taking is owned by ElevenLabs' server-side turn model: there is no push-to-talk
        # commit/create surface on the conversation WebSocket.
        'supports_manual_turn_control': False,
        # Barge-in is automatic on user speech. There is no client verb to cancel a response, so
        # `interrupt()` is unsupported even though interruptions are reported (as an `interruption`
        # event followed by an `agent_response_correction` truncating the transcript server-side).
        'supports_interruption': False,
        # The server truncates the stored transcript itself after an interruption
        # (`agent_response_correction`), so there is nothing for `played_ms` to do.
        'supports_output_truncation': False,
        # There is no history-seeding channel on the conversation WebSocket (`contextual_update`
        # injects non-interrupting context, but is not conversation history).
        'supports_session_seeding': False,
        # ElevenLabs' WebRTC transport is LiveKit-based with no server-side control-plane sideband.
        'supports_webrtc': False,
        'supports_thinking': False,
        # A spoken reply ends with its last audio, which the `context_usage` report marks (with it in the
        # agent's `client_events`, as the docs recommend); a text reply ends with `agent_response`. Only an
        # agent without that report falls back to a second without audio.
        'synthesizes_turn_boundary': False,
        'responses_are_requests': True,
        # Usage arrives as `context_usage`, once per reply, with its last audio, and is never attached
        # to a `ModelResponse`: the response usage measures nothing
        # about the context. The live fraction is `ElevenLabsAgentsRealtimeConnection.context_limit_tokens`
        # against the reported `context_tokens` instead.
        'response_usage_covers_context': False,
        # The LLM behind a hosted agent is configured on the agent (and overridable per conversation
        # via `elevenlabs_agents_llm`), so no context window can be inferred from the agent id and the
        # genai-prices lookup is skipped. A pinned value would go unused too, since the session's
        # token-usage fallback is off (`response_usage_covers_context`); the live limit is
        # `ElevenLabsAgentsRealtimeConnection.context_limit_tokens` once the agent reports `context_usage`,
        # and `context_window_used` is the ratio from the same report.
        'context_window': None,
        # `vad_score` events exist but there is no speech start/stop pair.
        'emits_input_speech_events': False,
        # ElevenLabs' defaults: `asr.user_input_audio_format` defaults to `pcm_16000`, and the
        # matching output default. Both are fixed per agent (not per conversation); the connect
        # handshake echoes the actual formats and the adapter raises when they disagree with the
        # profile, pointing at `profile={'audio_..._sample_rate': ...}` for agents configured
        # differently (e.g. `pcm_24000`; telephony `ulaw_8000` agents are rejected outright).
        'audio_input_sample_rate': 16000,
        'audio_output_sample_rate': 16000,
    }


def elevenlabs_realtime_model_profile(model_name: str) -> RealtimeModelProfile:
    """Get the realtime model profile for an ElevenLabs speech-to-speech model."""
    # TODO(elevenlabs-launch): every value below is a placeholder until it is verified against the
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
