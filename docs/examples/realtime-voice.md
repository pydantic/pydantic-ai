Example of a voice assistant built on a [realtime](../realtime/overview.md) speech-to-speech model: it
streams your microphone to OpenAI's `gpt-realtime` model and plays the model's spoken replies back
through your speakers. Talk to it — and try interrupting while it's speaking: the model stops and
listens (barge-in).

Demonstrates:

- [realtime sessions](../realtime/overview.md)
- [tools](../tools.md)
- [barge-in](../realtime/turns.md#barge-in) (interrupting the model mid-sentence)

The agent exposes a single `get_weather` tool the model can call mid-conversation, and the terminal
shows a running transcript of both sides of the conversation plus any tool calls.

Audio I/O runs on [`listentome`](https://github.com/Kludex/listentome), whose microphone is an
async iterator that [`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio] consumes
directly, and whose speaker `write()` suspends until the device has played each chunk from
[`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio]. Both audio directions stay
bounded rather than growing without limit: the microphone stream and the session's audio buffer
each drop their oldest blocks if their consumer falls behind, so a machine that stutters glitches
instead of ending the call.

Barge-in costs the example no code at all: because playback is a single device-paced
[`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] loop, the session can track
the playback position itself, so [`handle_barge_in=True`][pydantic_ai.realtime.AgentRealtime.session]
does the whole thing — dropping the buffered audio the user will never hear, truncating the
provider's transcript to what was really heard, and staying out of the way on an ordinary turn
where the previous reply was heard in full. Playback loops the session can't follow, and triggers
you'd rather own yourself, take the manual paths in
[the barge-in guide](../realtime/turns.md#barge-in) instead.

## Running the Example

The example's dependencies include
[`listentome`](https://github.com/Kludex/listentome) for microphone and speaker access. It
also requires the PortAudio system library: `brew install portaudio` on macOS,
`apt install libportaudio2` on Debian/Ubuntu.

The realtime model runs on `gpt-realtime`, so you'll need an OpenAI API key set via
`OPENAI_API_KEY`.

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.realtime_voice
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/realtime_voice.py"}```
