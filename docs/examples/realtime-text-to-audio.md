The smallest possible [realtime session](../realtime/index.md): send plain text from Python and hear
the model speak the reply. Sending text into an OpenAI realtime session asks the model to respond
right away, so there's no microphone, voice-activity detection, or manual turn-taking to manage —
just [`send()`][pydantic_ai.realtime.RealtimeSession.send] and iterate the session's events.

Demonstrates:

- [realtime sessions](../realtime/index.md)
- the text-in / audio-out path (no audio hardware required)
- streaming [`SpeechPartDelta`][pydantic_ai.messages.SpeechPartDelta] audio and transcript deltas

The script streams the spoken reply back, prints the transcript as it arrives, and saves the audio
to a `.wav` file you can play afterwards. It's a handy starting point for turning an existing text
chatbot into one that talks, or for generating spoken snippets like a voicemail greeting.

## Running the Example

The realtime model runs on `gpt-realtime`, so you'll need an OpenAI API key set via
`OPENAI_API_KEY`.

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.realtime_text_to_audio "Tell me a fun fact about octopuses."
```

The streamed PCM audio is saved to `realtime-response.wav` so you can listen to the result
afterwards.

## Example Code

```snippet {path="/examples/pydantic_ai_examples/realtime_text_to_audio.py"}```
