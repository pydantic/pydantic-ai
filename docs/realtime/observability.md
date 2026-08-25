# Usage and observability

Realtime audio bills by the second in both directions, so knowing what a session cost — and capping
it — matters even more than for a text run. Realtime sessions accumulate standard
[`RunUsage`][pydantic_ai.usage.RunUsage], enforce standard
[`UsageLimits`][pydantic_ai.usage.UsageLimits], and emit OpenTelemetry spans — viewable in
[Pydantic Logfire](../logfire.md) — through Pydantic AI's normal instrumentation. This lets voice
and follow-up text runs share one usage budget and trace.

## Usage and limits

Read cumulative usage from
[`RealtimeSession.usage`][pydantic_ai.realtime.RealtimeSession.usage]. It includes input/output
tokens, provider audio and cache breakdowns where available, and tool-call counts. Usage updates are
not emitted as session events. As with a standard run's
[usage limits](../agent.md#usage-limits), pass `usage=` to accumulate into a shared object — for
example one carried across a voice call and its follow-up text runs — and `usage_limits=` to cap a
session:

```python
from pydantic_ai import Agent
from pydantic_ai.realtime import RealtimeTurnCompleteEvent
from pydantic_ai.usage import RunUsage, UsageLimits

agent = Agent()
shared = RunUsage()


async def main():
    async with agent.realtime(
        'openai:gpt-realtime',
        usage=shared,
        usage_limits=UsageLimits(total_tokens_limit=100_000),
    ).session() as session:
        await session.send('Say hello.')
        async for event in session:
            if isinstance(event, RealtimeTurnCompleteEvent):
                break
        print(shared)
        #> RunUsage(requests=1)
```

Input-transcription usage is reported separately in `RunUsage.details` under
`input_transcription_*` keys. It is not included in response token totals or attributed to a
`ModelResponse`, because transcription can use a separate model and billing meter.

Token and tool-call limits are checked as usage accrues. Request limits are checked before sending
text, explicitly creating a response, or returning a tool result. With server-side VAD, the provider
can begin a response without a client request; that limit is checked at the first response event.
Breaches raise [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] from iteration.

Provider-specific usage fields belong on the
[OpenAI](openai.md#feature-support-and-limitations),
[Azure OpenAI](azure.md#feature-support-and-limitations),
[Google Gemini](gemini.md#feature-support-and-limitations), and
[xAI](xai.md#feature-support-and-limitations) pages.

## Logfire instrumentation

Call `logfire.instrument_pydantic_ai()` or set `instrument=True` on the agent:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

The session creates an `invoke_agent` span with cumulative usage and conversation content, subject
to the normal content-redaction setting. Nested `chat {model}` spans represent provider responses,
and `execute_tool` spans represent tools and delegated agent runs. `model turn complete` and `interrupt`
spans mark those boundaries. A tool round can produce several response spans within one turn.

You may see runs of `model turn complete (interrupted)` spans with no `chat` span between them.
That's normal on OpenAI server VAD: the provider starts a response for each detected speech segment,
so a user who keeps talking cancels each auto-response before it produces output. Every cancelled or
interrupted response still draws a boundary, displayed as `model turn complete (interrupted)`.

| Attribute | Set on | Meaning |
| --- | --- | --- |
| `pydantic_ai.realtime` | Spans the session emits itself (session, response, boundary, and `user speech` spans) | Always `True`; marks spans that belong to a realtime session. `execute_tool` spans come from the [`Instrumentation`][pydantic_ai.capabilities.Instrumentation] capability and don't carry it. |
| `gen_ai.output.type` | Response spans | `speech` or `text`. |
| `pydantic_ai.response.state` | Interrupted response spans | `'interrupted'`. |
| Response-level usage | OpenAI, Azure OpenAI, and xAI response spans | Tokens attributed to that response. |

Gemini can report usage only on a later completed turn after a function-call response; cumulative
session usage remains authoritative.

When providers report both user speech start and end, Pydantic AI records a `user speech` span.
Providers without both boundaries do not get a guessed duration.

On a [WebRTC sideband](deployment.md#browser-webrtc-server-sideband) a `speak {model}` span additionally covers how
long the model was *audible*, which the response spans can't show: the provider generates audio far
ahead of playing it, so this span routinely outlasts the `model turn complete` that ended the
response.

The session span also reports `pydantic_ai.audio_chunks_dropped` and
`pydantic_ai.transcript_items_dropped`, summed across bounded
[audio and transcript consumers](audio.md#consuming-audio-and-transcripts). These totals are written
when the session closes.

See [Debugging and monitoring](../logfire.md) for Logfire setup and privacy controls.

## Gateway trace propagation

Routing through the [Pydantic AI Gateway](../gateway.md) — e.g.
`agent.realtime('gateway/openai:gpt-realtime')` — is provider configuration, documented on the
[OpenAI](openai.md#gateway) and [Gemini](gemini.md#gateway) pages. When a span is active during the
WebSocket handshake, Pydantic AI propagates
[W3C trace context](https://www.w3.org/TR/trace-context/) so gateway spans can join the trace.

The provider connection is established before the realtime session span starts. Wrap the entire
session context in an outer span when the handshake itself must be included:

```python
import logfire

from pydantic_ai import Agent

agent = Agent()


async def main():
    with logfire.span('voice call'):
        async with agent.realtime('openai:gpt-realtime').session() as session:
            await session.send('Say hello.')
```

## Edge cases

- Usage is cumulative session state, not an event stream. Read it after the relevant responses or
  when the session closes.
- A provider can report response-level usage at a different point from the local tool or turn
  boundary. Use the session total for billing and limits.
- Dropped-stream counters represent each slow consumer independently; two lagging audio iterators
  can both contribute drops for the same produced audio.
