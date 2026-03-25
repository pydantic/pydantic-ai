Information about whales — an example of streamed structured response validation.

Demonstrates:

* [streaming structured output](../output.md#streaming-structured-output)

This script streams structured responses about whales, validates the data
and displays it as a dynamic table using [`rich`](https://github.com/Textualize/rich) as the data is received.

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.stream_whales
```

Should give an output like this:

{{ video('53dd5e7664c20ae90ed90ae42f606bf3', 25) }}

## Example Code

```snippet {path="/examples/pydantic_ai_examples/stream_whales.py"}```
