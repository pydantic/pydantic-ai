# Amazon Nova Sonic

[`BedrockRealtimeModel`][pydantic_ai.realtime.bedrock.BedrockRealtimeModel] brings Amazon Nova Sonic
into the typed, server-side realtime agent loop over Bedrock's HTTP/2 bidirectional stream. See the
[realtime overview](index.md).

!!! warning "Experimental"
    Nova Sonic realtime support is experimental. `BedrockRealtimeModel` is reachable through
    [`infer_realtime_model`][pydantic_ai.realtime.infer_realtime_model] (the `bedrock:` prefix) or by
    importing it directly, but it is not exported from the top-level `pydantic_ai.realtime` namespace
    and its API may change.

## Installation

```bash
pip install "pydantic-ai-slim[realtime-bedrock]"
```

This installs the `aws-sdk-bedrock-runtime` SDK for Bedrock bidirectional streaming, which is
separate from the boto3-based `bedrock` group used by the request-response
[`BedrockConverseModel`][pydantic_ai.models.bedrock.BedrockConverseModel]. The experimental SDK
currently requires Python 3.12 or newer.

## Configuration

Use the `bedrock:` prefix (defaulting to `amazon.nova-2-sonic-v1:0`), reading standard AWS SigV4
environment credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` /
`AWS_DEFAULT_REGION`, defaulting to `us-east-1`):

```python {test="skip"}
from pydantic_ai import Agent

agent = Agent(instructions='You are a helpful voice assistant.')


async def main():
    async with agent.realtime_session(model='bedrock:amazon.nova-2-sonic-v1:0') as session:
        await session.send('Say hello.')
        async for event in session:
            ...
```

You can also configure the model explicitly with
[`BedrockRealtimeModelSettings`][pydantic_ai.realtime.bedrock.BedrockRealtimeModelSettings]:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.bedrock import BedrockRealtimeModel, BedrockRealtimeModelSettings

settings = BedrockRealtimeModelSettings(
    voice='matthew',
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    bedrock_endpointing_sensitivity='HIGH',
)
model = BedrockRealtimeModel('amazon.nova-2-sonic-v1:0', settings=settings)
```

Nova expects mono 16-bit LPCM input at 16 kHz and returns mono 16-bit LPCM at 24 kHz. It supports
text/transcript history seeding (see [Message history](index.md#message-history)) and concurrent
asynchronous tool calls (see [Concurrent tools](index.md#concurrent-tools)). Nova always produces
audio, so `output_modality` is not configurable; interruption is automatic (barge-in), and the API
exposes no client cancel or output-truncate verb — so
[`interrupt`][pydantic_ai.realtime.RealtimeSession.interrupt] is unsupported. Manual turn-taking
(push-to-talk) is supported; image input is not.
