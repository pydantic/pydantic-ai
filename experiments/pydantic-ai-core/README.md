# `pydantic-ai-core` experiment

Can part of `pydantic-ai` live in Rust and be reused by an SDK in another language?

This Rust library ports two pieces of the Python library and checks them against the Python behaviour, from Rust and from TypeScript:

```ts
modelProfile("openai", "gpt-5.4-nano");
// { ok: { json_schema_transformer: "openai", supports_thinking: true, ... } }

transformSchema("openai", { type: "object", properties: { a: { type: "string", minLength: 2 } } }, true);
// { ok: { schema: { ..., additionalProperties: false, required: ["a"] }, is_strict_compatible: true } }
```

> [!WARNING]
> **This is an experiment for discussion.** It is not a maintained component. Nothing in the Python packages imports it, and no CI job builds it, tests it or regenerates its fixtures. `fixtures/golden.json` goes stale when `KnownModelName`, `openai_model_profile` or the ported transformers change.

## Run it

You need Rust 1.88 or later with the WASM target (`rustup target add wasm32-unknown-unknown`), and a Node version that runs TypeScript files directly. It was tested with Rust 1.99 and Node 24.

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
- 47 schemas, each through 3 transformers and 3 `strict` values: 423 cases, 118 of which expect a `UserError`.

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
- Python `repr` of floats with an exponent, such as `1e-07`, and of non-printable Unicode. Both are marked `shortcut:` in `src/schema.rs`.
- Typed errors. A failed call returns one `error` string, so a binding cannot tell a ported `UserError` from a malformed request.
- What a JavaScript object does to a schema before Rust sees it. Integer-like keys move first, so `{"b": ..., "1": ...}` arrives as `{"1": ..., "b": ...}`, and `2.0` arrives as `2`. No fixture has either shape, so the TypeScript runner does not show them.
