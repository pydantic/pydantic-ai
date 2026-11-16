// Replays the Python fixtures through the WASM build: `node ts/conformance.ts`.

import { readFileSync } from "node:fs";

type Exports = {
  memory: WebAssembly.Memory;
  alloc: (len: number) => number;
  dealloc: (ptr: number, len: number) => void;
  call: (ptr: number, len: number) => bigint;
};

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
type Response = { ok: Json } | { error: string };

const wasmPath = "../target/wasm32-unknown-unknown/release/pydantic_ai_core.wasm";
const { instance } = await WebAssembly.instantiate(readFileSync(new URL(wasmPath, import.meta.url)));
const { memory, alloc, dealloc, call } = instance.exports as Exports;

function invoke(request: Json): Response {
  const input = new TextEncoder().encode(JSON.stringify(request));
  const inputPtr = alloc(input.length);
  new Uint8Array(memory.buffer, inputPtr, input.length).set(input);
  const packed = BigInt.asUintN(64, call(inputPtr, input.length));
  dealloc(inputPtr, input.length);
  const outputPtr = Number(packed >> 32n);
  const outputLen = Number(packed & 0xffffffffn);
  const output = new TextDecoder().decode(new Uint8Array(memory.buffer, outputPtr, outputLen));
  dealloc(outputPtr, outputLen);
  return JSON.parse(output);
}

export function modelProfile(family: string, model: string): Response {
  return invoke({ fn: "model_profile", family, model });
}

export function transformSchema(transformer: string, schema: Json, strict: boolean | null): Response {
  return invoke({ fn: "transform_schema", transformer, schema, strict });
}

type Golden = {
  profiles: { model: string; expected: Json }[];
  schemas: {
    name: string;
    transformer: string;
    strict: boolean | null;
    schema: Json;
    expected: { [key: string]: Json };
  }[];
};

const golden: Golden = JSON.parse(readFileSync(new URL("../fixtures/golden.json", import.meta.url), "utf8"));
const failures: string[] = [];

for (const { model, expected } of golden.profiles) {
  const actual = modelProfile("openai", model);
  if (!("ok" in actual) || JSON.stringify(actual.ok) !== JSON.stringify(expected)) {
    failures.push(`profile ${JSON.stringify(model)}`);
  }
}

for (const { name, transformer, strict, schema, expected } of golden.schemas) {
  const actual = transformSchema(transformer, schema, strict);
  const matches =
    "error" in expected
      ? "error" in actual && actual.error === expected.error
      : "ok" in actual && JSON.stringify(actual.ok) === JSON.stringify(expected);
  if (!matches) {
    failures.push(`schema ${name} ${transformer} strict=${strict}`);
  }
}

console.log(`${golden.profiles.length} profiles, ${golden.schemas.length} schema cases, ${failures.length} failures`);
for (const failure of failures) {
  console.log(`  ${failure}`);
}
process.exit(failures.length === 0 ? 0 : 1);
