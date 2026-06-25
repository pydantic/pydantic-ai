Example of a Pydantic AI agent that understands video using [TwelveLabs](https://twelvelabs.io) Pegasus.

Demonstrates:

- [tools](../tools.md)
- [agent dependencies](../dependencies.md)
- wrapping a third-party multimodal API as a tool

In this case the idea is a "video analyst" agent — the user asks questions about a video
(given its URL), and the agent uses the `analyze_video` tool to call TwelveLabs Pegasus,
a video-understanding model, to answer. The LLM decides *what* to ask about the video, and
Pegasus does the actual video understanding.

## Running the Example

You'll need a TwelveLabs API key set via `TWELVELABS_API_KEY`. You can grab a free key at
[twelvelabs.io](https://twelvelabs.io) — there's a generous free tier.

Optionally set `VIDEO_URL` to point the agent at your own publicly-accessible video;
otherwise a short public sample clip is used.

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.twelvelabs_video_agent
```

## Example Code
```snippet {path="/examples/pydantic_ai_examples/twelvelabs_video_agent.py"}```
