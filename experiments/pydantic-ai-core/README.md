# `pydantic-ai-core` experiment

Can part of `pydantic-ai` live in Rust and be reused by an SDK in another language?

This Rust library ports two pieces of the Python library and checks them against the Python behaviour, from Rust and from TypeScript:

```ts
modelProfile("openai", "gpt-5.4-nano");
// { ok: { json_schema_transformer: "openai", supports_thinking: true, ... } }

transformSchema("openai", { type: "object", properties: { a: { type: "string", minLength: 2 } } }, true);
// { ok: { schema: { ..., additionalProperties: false, required: ["a"] }, is_strict_compatible: true } }
```

## Run it

```bash
cd experiments/pydantic-ai-core
cargo test                                             # Rust against the Python fixtures
cargo build --release --target wasm32-unknown-unknown  # 226 KB, 72 KB gzipped, no imports
node ts/conformance.ts                                 # TypeScript against the same fixtures, through WASM
```

To regenerate the fixtures, run this from the repository root:

```bash
uv run python experiments/pydantic-ai-core/fixtures/generate.py
```

## What is ported

| Python | Here |
|---|---|
| `openai_model_profile(model_name)` | `model_profile("openai", model)` |
| `InlineDefsJsonSchemaTransformer` | `transform_schema("inline_defs", ...)` |
| `OpenAIJsonSchemaTransformer` | `transform_schema("openai", ...)` |
| `_GptOssJsonSchemaTransformer` | `transform_schema("gpt_oss", ...)` |

The profile holds ids where Python holds classes: `json_schema_transformer` is one of the ids above, and `supported_native_tools` is a list of each tool's `kind`.

## How it is checked

`fixtures/generate.py` runs the real `pydantic-ai` and writes `fixtures/golden.json`:

- 170 model names, each with its profile.
- 45 schemas, each through 3 transformers and 3 `strict` values: 405 cases, 112 of which expect a `UserError`.

Both test runners compare serialized JSON, so key order has to match. Error messages have to match exactly.

## How a binding calls it

Every binding goes through one function, `call(request_json) -> response_json`. The WASM build exports it with `alloc` and `dealloc` and imports nothing, so it needs no `wasm-bindgen` and no glue beyond the `invoke` function in `ts/conformance.ts`.

## What matching Python required

- **Python `repr` is on the wire.** In strict mode, dropped keywords move into `description` formatted with `str()`, such as `patternProperties={'^a': {'type': 'string'}}`. `py_repr` reproduces that.
- **In-place mutation is observable.** `OpenAIJsonSchemaTransformer.walk()` merges the root definition back after the walk edited it in place. `after_in_place_walk` reproduces what Python leaves behind; 12 fixtures depend on it.
- **`serde_json` needs `preserve_order`**, and every removal has to be `shift_remove`, to keep Python's dict order.

## Not covered

- The other profile families, and the provider layer on top of `openai_model_profile`.
- `_OpenAIJsonSchemaTransformer`, which `OpenAIProvider` selects, and the Anthropic, Google and Bedrock transformers.
- A Python binding. The fixtures show the outputs are equal; nothing in `pydantic-ai` calls this library.
- Schemas with two or more recursive definitions under `inline_defs`. Python emits their `$defs` in set order, which changes with `PYTHONHASHSEED`, so there is no single output to match.
- Inputs on which Python raises something other than `UserError`.
