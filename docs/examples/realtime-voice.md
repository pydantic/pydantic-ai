Example of a voice assistant built on a [realtime](../realtime/index.md) speech-to-speech model: it
streams your microphone to OpenAI's `gpt-realtime` model and plays the model's spoken replies back
through your speakers. Talk to it — and try interrupting while it's speaking: the model stops and
listens (barge-in).

Demonstrates:

- [realtime sessions](../realtime/index.md)
- [tools](../tools.md)
- [barge-in](../realtime/index.md#turn-taking-and-barge-in) (interrupting the model mid-sentence)

The agent exposes a single `get_weather` tool the model can call mid-conversation, and the terminal
shows a running transcript of both sides of the conversation plus any tool calls.

## Running the Example

The examples dependencies include
[`sounddevice`](https://python-sounddevice.readthedocs.io) for microphone and speaker access. It
also requires the PortAudio system library; on Linux, install `libportaudio2` if importing
`sounddevice` fails.

The realtime model runs on `gpt-realtime`, so you'll need an OpenAI API key set via
`OPENAI_API_KEY`.

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.realtime_voice
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/realtime_voice.py"}```
