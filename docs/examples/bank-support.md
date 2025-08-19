Small but complete example of using Pydantic AI to build a support agent for a bank.

Demonstrates:

- [dynamic system prompt](../agents.md#system-prompts)
- [structured `output_type`](../output.md#structured-output)
- [tools](../tools.md)

## Running the Example

With [dependencies installed and environment variables set](./index.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.bank_support
```

(or `PYDANTIC_AI_MODEL=gemini-1.5-flash ...`)

## Example Code

```snippet {path="/examples/pydantic_ai_examples/bank_support.py"}```
