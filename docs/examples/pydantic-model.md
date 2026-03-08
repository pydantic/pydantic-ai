# Pydantic Model

Simple example of using Pydantic AI to construct a Pydantic model from a text input.

Demonstrates:

- [structured `output_type`](../output.md#structured-output)

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.pydantic_model
```

This examples uses `openai:gpt-5` by default, but it works well with other models, e.g. you can run it
with Gemini using:

```bash
PYDANTIC_AI_MODEL=gemini-3-pro-preview python/uv-run -m pydantic_ai_examples.pydantic_model
```

(or `PYDANTIC_AI_MODEL=gemini-3-flash-preview ...`)

## Example Code

```snippet {path="/examples/pydantic_ai_examples/pydantic_model.py"}```
