Customer-support agent that uses [capabilities on demand](../capabilities.md#deferred-capability-loading) to scale one agent across multiple support specialties without loading every specialty into every request.

Demonstrates:

- [capabilities on demand](../capabilities.md#deferred-capability-loading)
- [capabilities](../capabilities.md) bundling instructions and a [toolset](../toolsets.md)
- [agent dependencies](../dependencies.md)

The agent advertises two specialists — `orders` and `returns` — by `id` and `description` only. The model sees the catalog on its first request and calls `load_capability(id)` to unlock the specialist that matches the user's question; that specialist's instructions and tools then appear on the next model request. The specialist not loaded on a given run never enters the context window.

This is the same shape as skills-style progressive disclosure, but each specialist can bring both instructions and tools. The order specialist carries order lookup behavior, the returns specialist carries return-policy guidance and return-opening tools, and the model chooses which one to load at runtime.

Contrast with the [bank support example](./bank-support.md), where a single toolset is always loaded: this example shows the same support-agent pattern scaled to multiple specialist modes without bloating every request. As the number of specialties grows, the first request still only needs the compact catalog.

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.support_specialist
```

With a `LOGFIRE_TOKEN` set you can watch the model call `load_capability` on each turn in the [Logfire](../logfire.md) dashboard; without one the example still runs and prints each answer.

## Example Code

```snippet {path="/examples/pydantic_ai_examples/support_specialist.py"}```
