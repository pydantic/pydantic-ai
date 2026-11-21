# `pydantic-ai-core` experiment

Can the core of `pydantic-ai` live in Rust, so that SDKs in other languages reuse it?

This is a small agent whose run loop, request building and response parsing live in one Rust library. A Python host and a TypeScript host drive the same library:

```python
from pydantic_ai_core import Agent

agent = Agent('gpt-5.4-mini', instructions='Answer in one short sentence.')


@agent.tool_plain
def get_temperature(city: str) -> dict[str, str | int]:
    """Get the current temperature in a city."""
    return {'city': city, 'celsius': 21}


result = await agent.run('What is the temperature in Tokyo?')
print(result.output)
#> The temperature in Tokyo is 21°C.
```

```ts
import { Agent } from "./agent.ts";

const agent = new Agent({
  model: "gpt-5.4-mini",
  instructions: "Answer in one short sentence.",
  tools: [
    {
      name: "get_temperature",
      description: "Get the current temperature in a city.",
      parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
      execute: ({ city }) => ({ city, celsius: 21 }),
    },
  ],
});

const result = await agent.run("What is the temperature in Tokyo?");
console.log(result.output);
// The temperature in Tokyo is 21°C.
```

> [!WARNING]
> **This is an experiment for discussion.** It is not a maintained component. Nothing in the Python packages imports it, and no CI job builds it, tests it or regenerates its fixtures. The fixtures go stale when the code they were dumped from changes.

## The design

The core does no I/O and never calls back into the host. The host feeds it an event, and performs the action it gets back.

```
 Host (Python, TypeScript)                       Rust core
 -------------------------                       ---------
 Agent API, tool functions        events ->      run loop as a state machine
 argument validation              <- actions     OpenAI Chat Completions codec
 HTTP, auth                                      model profiles, schema transformers
```

| Event from the host | Action from the core |
|---|---|
| `start`: model, instructions, prompt, tools, message history | `send_request`: method, path and body |
| `response`: status and body text | `call_tools`, or `done` with output, messages and usage |
| `tool_results`: a value or a retry message per call | `send_request` |

Three choices keep a host small:

- **The core holds nothing between calls.** Each step returns the run's state as an opaque string, and the host passes it back. There are no handles to manage across the language boundary, and a run can be paused and resumed from that string.
- **Bodies cross as text.** The host sends the request body exactly as the core wrote it, and hands the response text back unparsed. No host re-serializes what the provider reads or wrote, so every language sends the same bytes.
- **Messages are `pydantic-ai` messages.** The core reads and writes the JSON format of `ModelMessagesTypeAdapter`, so the Python host returns real `ModelRequest` and `ModelResponse` objects, and accepts a history from a real `Agent`.

Every binding goes through one function, `call(request_json) -> response_json`, exported with a C ABI. The Python host loads the native library with `ctypes`. The TypeScript host loads the WASM build, which imports nothing.

## Run it

You need Rust 1.88 or later with the WASM target (`rustup target add wasm32-unknown-unknown`), and a Node version that runs TypeScript files directly. It was tested with Rust 1.99 and Node 24.

```bash
cd experiments/pydantic-ai-core
cargo test                                             # the core against the Python fixtures
cargo build --release                                  # native library, for the Python host
cargo build --release --target wasm32-unknown-unknown  # 276 KB, 86 KB gzipped, for the TypeScript host
node ts/conformance.ts                                 # the TypeScript host against the same fixtures
cd ../..
uv run python experiments/pydantic-ai-core/python/check.py    # the Python host against the same fixtures
```

With `OPENAI_API_KEY` set, both hosts run the example above against the real API:

```bash
uv run python experiments/pydantic-ai-core/python/example.py
node experiments/pydantic-ai-core/ts/example.ts
```

To regenerate the fixtures, run these from the repository root:

```bash
uv run python experiments/pydantic-ai-core/fixtures/generate.py
uv run python experiments/pydantic-ai-core/fixtures/generate_runs.py
```

## How it is checked

The expected values come from the real `pydantic-ai`, never from a hand-written file.

`fixtures/generate_runs.py` runs a real `Agent` on `OpenAIChatModel` in four scenarios: a tool call, a tool retry, a text answer, and a run that continues a message history. The transport replays responses recorded from the OpenAI API in an existing test cassette, and captures the seven requests `pydantic-ai` sends. `fixtures/runs.json` holds both.

The core, the Python host and the TypeScript host each replay those scenarios. Each has to send the request bodies `pydantic-ai` sent, and end with the same output, usage and new messages. The Rust test also requires the key order of every request body to match.

`fixtures/generate.py` covers the two pieces the codec builds on, in `fixtures/golden.json`:

- 170 model names, each with the profile `openai_model_profile` gives it.
- 48 schemas, each through 3 transformers and 3 `strict` values: 432 cases, 118 of which expect a `UserError`.

## What matching Python required

- **Python `repr` is on the wire.** In strict mode, dropped schema keywords move into `description` formatted with `str()`, such as `patternProperties={'^a': {'type': 'string'}}`. `py_repr` reproduces that.
- **In-place mutation is observable.** `OpenAIJsonSchemaTransformer.walk()` merges the root definition back after the walk edited it in place. `after_in_place_walk` reproduces what Python leaves behind; 12 fixtures depend on it.
- **`serde_json` needs `preserve_order`**, and every removal has to be `shift_remove`, to keep Python's dict order.

## Not covered

The run loop and codec:

- Streaming, structured output, output validators, native tools, multimodal input and model settings.
- Any provider other than OpenAI Chat Completions.
- Response timestamps, cost, and the usage details beyond input and output tokens.
- An empty model response, which `Agent` answers by asking again and the core treats as an error.
- Unknown tool names, which each host turns into a retry itself.
- Configurable tool retries: the limit is fixed at one, the `Agent` default.
- Typed errors. A failed call returns one `error` string, so a host cannot tell a provider error from a malformed request.
- Tool argument validation in the TypeScript host, which passes the parsed arguments straight to the tool.
- The cost of the state string: every step parses and writes the whole message history.

Profiles and schema transformers:

- The other profile families, and the provider layer on top of `openai_model_profile`.
- `_OpenAIJsonSchemaTransformer`, which `OpenAIProvider` selects, and the Anthropic, Google and Bedrock transformers.
- Schemas with two or more recursive definitions under `inline_defs`. Python emits their `$defs` in set order, which changes with `PYTHONHASHSEED`, so there is no single output to match.
- Inputs on which Python raises something other than `UserError`.
- Python `repr` of floats with an exponent, such as `1e-07`, and of non-printable Unicode. Both are marked `shortcut:` in `src/schema.rs`.
- What a JavaScript object does to a value before Rust sees it. Integer-like keys move first, so `{"b": ..., "1": ...}` arrives as `{"1": ..., "b": ...}`, and `2.0` arrives as `2`. No fixture has either shape.
