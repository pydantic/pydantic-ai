// Loads the WASM build of the Rust core and exposes its one entry point.

import { readFileSync } from "node:fs";

type Exports = {
  memory: WebAssembly.Memory;
  pydantic_ai_core_alloc: (len: number) => number;
  pydantic_ai_core_dealloc: (ptr: number, len: number) => void;
  pydantic_ai_core_call: (request: number, requestLen: number, responseLen: number) => number;
};

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

const wasmPath = "../target/wasm32-unknown-unknown/release/pydantic_ai_core.wasm";
const wasm = readFileSync(new URL(wasmPath, import.meta.url));
const { instance } = await WebAssembly.instantiate(wasm);
const core = instance.exports as Exports;

export class CoreError extends Error {}

export function callCore<T = Json>(request: Json): T {
  const input = new TextEncoder().encode(JSON.stringify(request));
  const inputPtr = core.pydantic_ai_core_alloc(input.length);
  const lengthPtr = core.pydantic_ai_core_alloc(4);
  new Uint8Array(core.memory.buffer, inputPtr, input.length).set(input);
  const outputPtr = core.pydantic_ai_core_call(inputPtr, input.length, lengthPtr) >>> 0;
  // The call can grow the memory, so the buffer is read again after it.
  const outputLength = new DataView(core.memory.buffer).getUint32(lengthPtr, true);
  const outputBytes = new Uint8Array(core.memory.buffer, outputPtr, outputLength);
  const output = new TextDecoder().decode(outputBytes);
  core.pydantic_ai_core_dealloc(inputPtr, input.length);
  core.pydantic_ai_core_dealloc(lengthPtr, 4);
  core.pydantic_ai_core_dealloc(outputPtr, outputLength);

  const response = JSON.parse(output) as { ok: T } | { error: string };
  if ("error" in response) {
    throw new CoreError(response.error);
  }
  return response.ok;
}
