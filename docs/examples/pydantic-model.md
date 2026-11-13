---
description: "A simple Pydantic AI example that constructs a validated Pydantic model from free-form text input, using structured output with any supported model."
---

# Pydantic Model

Simple example of using Pydantic AI to construct a Pydantic model from a text input.

Demonstrates:

- [structured `output_type`](../output.md#structured-output)

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.pydantic_model
```

This example uses an OpenAI model by default. To run it with Gemini, set `PYDANTIC_AI_MODEL` to a model ID with the `google:` prefix.

```bash
PYDANTIC_AI_MODEL="{{ google:medium }}" python/uv-run -m pydantic_ai_examples.pydantic_model
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/pydantic_model.py"}```
