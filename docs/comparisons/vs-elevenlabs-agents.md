# Pydantic AI vs ElevenLabs Agents

[ElevenLabs Agents](https://elevenlabs.io/docs/eleven-agents/overview) is a hosted platform for building, deploying, and operating voice agents. It combines speech recognition, an LLM, text-to-speech, and turn-taking, with a visual builder, telephony, client SDKs, and a direct WebSocket API. Pydantic AI is a general agent framework for text and voice: the same typed [`Agent`][pydantic_ai.Agent] can use any supported model provider, run your Python tools and toolsets, return [typed output](../output.md), connect to [MCP](../mcp/client.md), run on [durable execution engines](../durable_execution/overview.md), and be tested with [Pydantic Evals](../evals.md) and traced with OpenTelemetry.

The two products work together. Pydantic AI ships [`ElevenLabsAgentsRealtimeModel`][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeModel], so a Pydantic AI agent can run its voice conversation on a hosted ElevenLabs agent while keeping its instructions and Python tools in Pydantic AI. Use ElevenLabs Agents on its own when you want a hosted voice stack with native telephony and a no-code builder. Add Pydantic AI when the same agent should work across text and voice, when its logic and tools live in your code, or when you need typed outputs, provider choice, application-level testing, and end-to-end observability.

ElevenLabs' [`elevenlabs` Python package](https://elevenlabs.io/docs/eleven-agents/libraries/python) connects a `Conversation` to a hosted agent over WebSocket. Its [`ClientTools`](https://elevenlabs.io/docs/eleven-agents/customization/tools/client-tools) run registered functions in that client, while `client.conversational_ai` manages hosted agents, conversations, tests, tools, and phone calls through the API.

Pydantic AI is one part of a stack: the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for capabilities and complete agents, [Pydantic Evals](../evals.md) for evaluation, [Pydantic Graph](../graph.md) for graph-based workflows, [Pydantic Logfire](https://pydantic.dev/logfire) for observability, and [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) itself for validation. The tables below cover the whole of it.

## Framework

| | ElevenLabs Agents | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | [Python SDK](https://elevenlabs.io/docs/eleven-agents/libraries/python), plus other client SDKs and a WebSocket API | Python |
| License | Hosted service | MIT |
| Model providers | [ElevenLabs-hosted and third-party LLMs, or a custom LLM](https://elevenlabs.io/docs/eleven-agents/customization/llm) | [Many](../models/overview.md) |
| Extensibility | [Visual workflows](https://elevenlabs.io/docs/eleven-agents/customization/workflows), knowledge bases, and client, webhook, code, MCP, and system tools | [Capabilities and toolsets](../extensibility.md); [80+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Hosted agents configured in the dashboard or by API | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | [Analytics](https://elevenlabs.io/docs/eleven-agents/dashboard) and [OpenTelemetry trace export](https://elevenlabs.io/docs/eleven-agents/customization/opentelemetry-traces) | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Durable execution | None documented | [Eight integrations](../durable_execution/overview.md) |
| Interfaces | [Widget](https://elevenlabs.io/docs/eleven-agents/customization/widget), client SDKs, WebSocket, Twilio, and SIP | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) |
| Realtime voice | Hosted cascaded speech pipeline or [Realtime voice mode](https://elevenlabs.io/docs/eleven-agents/customization/realtime-models) | [Speech-to-speech](../realtime/overview.md), five providers |
| Evals | [Simulation, next-reply, and tool-call tests](https://elevenlabs.io/docs/eleven-agents/customization/agent-testing) | [Pydantic Evals](../evals.md) |
| Image generation | None documented | [Image Generation](../image-generation.md) |

## Voice agents, side by side

ElevenLabs Agents owns the hosted voice stack, from audio transport and turn-taking through deployment and telephony. Pydantic AI treats ElevenLabs Agents as one realtime model provider behind the same agent API used for text and other voice providers. If you are choosing how to build a voice agent, these are the rows that decide it:

| | ElevenLabs Agents | Pydantic AI |
|---|---|---|
| Voice pipeline | Hosted ASR + configurable LLM + ElevenLabs TTS, or an ElevenLabs Realtime model | [Five speech-to-speech providers](../realtime/overview.md#provider-support), including hosted ElevenLabs Agents |
| Audio transport | [Client SDKs](https://elevenlabs.io/docs/eleven-agents/libraries/python), [WebSocket](https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets), and web widget | Yours, except when a hosted provider such as ElevenLabs owns it; [browser WebRTC sideband or WebSocket relay](../realtime/deployment.md) for direct providers |
| Telephony | [Twilio and SIP trunking](https://elevenlabs.io/docs/eleven-agents/phone-numbers/overview), inbound and outbound calls, transfers, and DTMF | [Bridge a provider](../realtime/deployment.md#siptelephony-bridge) such as Twilio |
| Turn-taking | [Configurable turn timeout, turn eagerness, interruption, and soft timeout](https://elevenlabs.io/docs/eleven-agents/customization/turn-taking) | [Provider turn detection, barge-in, push-to-talk](../realtime/turns.md) |
| Noise cancellation | None documented | Provider-side only |
| Hand off to another agent mid-call | [Agent transfer system tool](https://elevenlabs.io/docs/eleven-agents/customization/tools/system-tools/agent-transfer) | No built-in handoff; [delegate from a tool](../realtime/tools.md#delegating-work-during-a-call) instead |
| Structured conversation flows | [Visual workflow builder](https://elevenlabs.io/docs/eleven-agents/customization/workflows) | No voice-specific flow builder; use instructions and tools, or [Pydantic Graph](../graph.md) around the agent |
| Tools mid-call | [Client, webhook, code, MCP, and system tools](https://elevenlabs.io/docs/eleven-agents/customization/tools) | [The same tools, toolsets and dependencies](../realtime/tools.md) as a text agent |
| Hosted platform features | [Knowledge bases, guardrails, dynamic variables, analytics, and experiments](https://elevenlabs.io/docs/eleven-agents/overview) | Keep these on the ElevenLabs agent when using [`ElevenLabsAgentsRealtimeModel`](../realtime/elevenlabs.md#configuration-ownership); use Pydantic AI capabilities around the run |
| After the call | [Conversation history, analysis, and post-call webhooks](https://elevenlabs.io/docs/eleven-agents/workflows/post-call-webhooks) | [`Agent.run()` on the call's history](../realtime/history.md#handing-off-to-a-text-agent) for typed output or follow-up |
| Observability | Analytics plus OpenTelemetry from a webhook, API, or monitoring WebSocket | [OpenTelemetry](../realtime/observability.md): session, turn and tool spans, usage attributed per response |
| Evals | [Dashboard, CLI, or API agent tests](https://elevenlabs.io/docs/eleven-agents/customization/agent-testing) with simulations and LLM judges | [Pydantic Evals](../evals.md) on the text hand-off; nothing realtime-specific yet |
| Deployment | Hosted by ElevenLabs | Your process and backend for direct providers; ElevenLabs hosts its side of an `ElevenLabsAgentsRealtimeModel` session |
| The same agent without voice | [Text and voice through the widget and client SDKs](https://elevenlabs.io/docs/eleven-agents/customization/widget) | [`run()`, CLI, web chat, AG-UI, Vercel AI](../interfaces.md), and realtime voice |

## Running a Pydantic AI agent on ElevenLabs Agents

Create and configure the hosted agent in ElevenLabs, then pass its agent id to [`ElevenLabsAgentsRealtimeModel`][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeModel]. This follows the [ElevenLabs realtime setup](../realtime/elevenlabs.md#before-you-connect):

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.realtime.elevenlabs_agents import (
    ElevenLabsAgentsRealtimeModel,
    ElevenLabsAgentsRealtimeModelSettings,
)

agent = Agent(instructions='You are a friendly concierge. Keep answers short.')


@agent.tool_plain
async def find_restaurant(cuisine: str, party_size: int) -> str:
    """Find a restaurant with a free table.

    Args:
        cuisine: The kind of food the guests want, e.g. 'Italian'.
        party_size: How many guests need a seat.
    """
    return f'Trattoria Roma has a table for {party_size} ({cuisine}).'


model = ElevenLabsAgentsRealtimeModel(
    'agent_0101k2example',
    settings=ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync'),
)
realtime_agent = agent.realtime(model)
```

The model connects directly to the hosted agent over its WebSocket API, so the ElevenLabs Python SDK is not required. Pydantic AI can check or synchronize the hosted agent's client-tool definitions, then execute those tools in your Python process. ElevenLabs remains responsible for the hosted pipeline, turn-taking, audio formats, and platform configuration; the [configuration ownership guide](../realtime/elevenlabs.md#configuration-ownership) explains the boundary.

Telephony agents using `ulaw_8000` are not supported through `ElevenLabsAgentsRealtimeModel` yet because Pydantic AI realtime sessions currently require PCM audio. Support for G.711 and other audio formats is tracked in [issue #9038](https://github.com/pydantic/pydantic-ai/issues/9038). Use ElevenLabs Agents directly for those calls today.
