// Replays the Python fixtures through the WASM build and the TypeScript host:
// `node ts/conformance.ts`.

import { readFileSync } from "node:fs";

import { Agent, ModelRetry } from "./agent.ts";
import { CoreError, callCore } from "./core.ts";
import type { Json } from "./core.ts";

type JsonObject = { [key: string]: Json };

function fixtures<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(`../fixtures/${name}`, import.meta.url), "utf8"));
}

function sameJson(actual: Json, expected: Json): boolean {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

const failures: string[] = [];

type Golden = {
  profiles: { model: string; expected: Json }[];
  schemas: {
    name: string;
    transformer: string;
    strict: boolean | null;
    schema: Json;
    expected: JsonObject;
  }[];
};
const golden = fixtures<Golden>("golden.json");

for (const { model, expected } of golden.profiles) {
  if (!sameJson(callCore({ fn: "model_profile", family: "openai", model }), expected)) {
    failures.push(`profile ${JSON.stringify(model)}`);
  }
}

for (const { name, transformer, strict, schema, expected } of golden.schemas) {
  let matches: boolean;
  try {
    matches = sameJson(callCore({ fn: "transform_schema", transformer, schema, strict }), expected);
  } catch (error) {
    matches = error instanceof CoreError && error.message === expected.error;
  }
  if (!matches) {
    failures.push(`schema ${name} ${transformer} strict=${strict}`);
  }
}

type RunStep = { request: Json; response: Json; tool_results?: JsonObject[] };
type Scenario = {
  name: string;
  start: {
    model: string;
    instructions: string;
    user_prompt: string;
    tools: JsonObject[];
    message_history?: Json[];
  };
  steps: RunStep[];
  output: string;
  usage: Json;
  new_messages: Json[];
};
const { scenarios } = fixtures<{ scenarios: Scenario[] }>("runs.json");
let requests = 0;

// The host replays each run of the real `pydantic-ai`: `fetch` checks the request and answers
// with the recorded response, and the tool gives the results the Python tool gave.
for (const { name, start, steps, output, usage, new_messages } of scenarios) {
  const remainingSteps = [...steps];
  const toolResults = steps.flatMap((step) => step.tool_results ?? []);
  const agent = new Agent({
    model: start.model,
    instructions: start.instructions,
    apiKey: "-",
    tools: start.tools.map((tool) => ({
      name: tool.name as string,
      parameters: tool.parameters_json_schema,
      execute: () => {
        const result = toolResults.shift()!;
        if ("retry" in result) {
          throw new ModelRetry(result.retry as string);
        }
        return result.content;
      },
    })),
    fetch: async (url, init) => {
      const step = remainingSteps.shift()!;
      requests += 1;
      const sameUrl = url === "https://api.openai.com/v1/chat/completions";
      if (!sameUrl || init?.body !== JSON.stringify(step.request)) {
        failures.push(`run ${name}: request ${steps.length - remainingSteps.length}`);
      }
      return new Response(JSON.stringify(step.response), { status: 200 });
    },
  });

  const result = await agent.run(start.user_prompt, start.message_history);
  const newMessages = result.messages.slice(-new_messages.length);
  const sameResult = result.output === output && sameJson(result.usage, usage);
  if (!sameResult || !sameMessages(newMessages, new_messages)) {
    failures.push(`run ${name}: result`);
  }
}

// The core writes its message fields in its own order, so messages are compared key by key.
function sameMessages(actual: Json, expected: Json): boolean {
  if (Array.isArray(expected)) {
    return (
      Array.isArray(actual) &&
      actual.length === expected.length &&
      expected.every((item, index) => sameMessages(actual[index], item))
    );
  }
  if (expected !== null && typeof expected === "object") {
    const keys = Object.keys(expected);
    return (
      actual !== null &&
      typeof actual === "object" &&
      !Array.isArray(actual) &&
      Object.keys(actual).length === keys.length &&
      keys.every((key) => key in actual && sameMessages(actual[key], expected[key]))
    );
  }
  return actual === expected;
}

console.log(
  `${golden.profiles.length} profiles, ${golden.schemas.length} schema cases, ` +
    `${scenarios.length} runs with ${requests} requests, ${failures.length} failures`,
);
for (const failure of failures) {
  console.log(`  ${failure}`);
}
process.exit(failures.length === 0 ? 0 : 1);
