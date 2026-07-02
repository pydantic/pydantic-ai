This example shows how to stream markdown from an agent, using the [`rich`](https://github.com/Textualize/rich) library to highlight the output in the terminal.

It'll run the example with both OpenAI and Google Gemini models if the required environment variables are set.

Demonstrates:

* [streaming text responses](../output.md#streaming-text)

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.stream_markdown
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/stream_markdown.py"}```
