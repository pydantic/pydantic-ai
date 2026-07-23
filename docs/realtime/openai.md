# OpenAI Realtime

[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel] connects a Pydantic AI
agent to OpenAI's native speech-to-speech models. See the [realtime overview](index.md) for shared
events, tools, history, frontend transport, and reliability patterns.

## Installation

```bash
pip install "pydantic-ai-slim[realtime,openai]"
```

The provider uses WebSockets plus the `openai` client.

## Configuration

Authentication and base URL come from `provider`, mirroring
[`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]: use `provider='openai'` (the
default, reading `OPENAI_API_KEY`) or an
[`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] for a custom key/base URL. The
WebSocket opens separately, so a custom provider `httpx` client is not used for it.
OpenAI-compatible endpoints exposing a realtime API work too.

[`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings] extends
shared settings with noise reduction, output speed, precise turn detection, and truncation:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime import TurnDetection
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings, SemanticVAD

settings = OpenAIRealtimeModelSettings(
    max_tokens=2_000,
    voice='alloy',
    turn_detection=TurnDetection(sensitivity='high', silence_duration_ms=400),
    openai_input_noise_reduction='near_field',
    openai_output_speed=1.1,
    openai_turn_detection=SemanticVAD(eagerness='high'),
    openai_truncation={'type': 'retention_ratio', 'retention_ratio': 0.8},
)
model = OpenAIRealtimeModel('gpt-realtime', settings=settings)
```

`openai_turn_detection` accepts [`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD] or
[`SemanticVAD`][pydantic_ai.realtime.openai.SemanticVAD] and overrides shared `turn_detection`.
`openai_truncation` also accepts `'auto'` or `'disabled'`; retention ratio keeps a stable,
cheaper prompt-cached audio prefix as a session grows. OpenAI realtime does not expose
`temperature`. Input transcription defaults to `'auto'`; see
[Transcribing user input](index.md#transcribing-user-input).

## Reasoning

The cross-provider [`thinking`][pydantic_ai.realtime.RealtimeModelSettings.thinking] setting mirrors
the unified [`thinking`][pydantic_ai.settings.ModelSettings.thinking] on the request-response models:
`True` enables reasoning at the provider default, and
`'minimal'`/`'low'`/`'medium'`/`'high'`/`'xhigh'` selects an effort level.

It applies only to realtime models that support reasoning — reported by the model's
[`supports_thinking`][pydantic_ai.realtime.RealtimeModelProfile.supports_thinking] profile flag. This
includes OpenAI's `gpt-realtime-2` family (e.g. `gpt-realtime-2.1` and `gpt-realtime-2.1-mini`) and
Gemini's native-audio Live models. The GA `gpt-realtime` is a standard speech-to-speech model without
reasoning, so a `thinking` setting is ignored with a warning rather than sent (the API would otherwise
reject it).

```python {test="skip" lint="skip"}
async with agent.realtime_session(
    model=OpenAIRealtimeModel('gpt-realtime-2.1', settings=OpenAIRealtimeModelSettings(thinking='low')),
) as session:
    ...
```

On OpenAI the effort maps to `reasoning.effort`. OpenAI realtime does not accept a disabled effort,
so `False` omits `reasoning` and leaves the model's default behavior unchanged. On Gemini the setting
maps to a thinking level, and `False` disables thinking. xAI ignores `thinking`. For finer Gemini
control — a token budget, or thought summaries — set
[`google_thinking_config`][pydantic_ai.realtime.google.GoogleRealtimeModelSettings.google_thinking_config],
which takes precedence over `thinking`.

!!! note "Reasoning traces are not surfaced"
    `thinking` tunes the model's reasoning effort (a quality-vs-latency trade-off that improves the
    answer); the session does not surface the reasoning itself as `ThinkingPart`s. OpenAI's realtime
    API never streams reasoning (`reasoning.effort` is input-only), and Gemini's native-audio
    *audio*-output sessions emit no thought summaries. Gemini's *text*-output sessions can emit thought
    summaries, but the realtime session does not surface them.
