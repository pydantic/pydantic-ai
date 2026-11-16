// A minimal TypeScript host for the Rust core.
//
// The host does the HTTP and runs the tools. The core decides everything else: what to send,
// when to call a tool, when the run is over.

import { callCore } from "./core.ts";
import type { Json } from "./core.ts";

export class ModelRetry extends Error {}

export type Tool = {
  name: string;
  description?: string;
  /** The JSON schema of the arguments. */
  parameters: Json;
  // shortcut: the arguments are not validated against `parameters`; add a validator such as Zod.
  execute: (args: { [key: string]: Json }) => Json | Promise<Json>;
};

export type AgentOptions = {
  model: string;
  instructions?: string;
  tools?: Tool[];
  baseUrl?: string;
  apiKey?: string;
  fetch?: typeof fetch;
};

export type Usage = { requests: number; input_tokens: number; output_tokens: number };
export type RunResult = { output: string; messages: Json[]; usage: Usage };

type ToolCall = { tool_call_id: string; tool_name: string; args: string };
type Action =
  | { type: "send_request"; request: { method: string; path: string; body: string } }
  | { type: "call_tools"; calls: ToolCall[] }
  | ({ type: "done" } & RunResult);
type Step = { state: string; action: Action };

export class Agent {
  options: AgentOptions;

  constructor(options: AgentOptions) {
    this.options = options;
  }

  async run(userPrompt: string, messageHistory: Json[] = []): Promise<RunResult> {
    const { model, instructions = null, tools = [] } = this.options;
    let event: Json = {
      type: "start",
      model,
      instructions,
      user_prompt: userPrompt,
      tools: tools.map(({ name, description = null, parameters }) => ({
        name,
        description,
        parameters_json_schema: parameters,
      })),
      message_history: messageHistory,
    };
    let state: string | null = null;
    for (;;) {
      const step: Step = callCore<Step>({ fn: "run_step", state, event });
      const action = step.action;
      state = step.state;
      if (action.type === "send_request") {
        event = await this.send(action.request);
      } else if (action.type === "call_tools") {
        const results = await Promise.all(action.calls.map((call) => this.callTool(call)));
        event = { type: "tool_results", results };
      } else {
        return { output: action.output, messages: action.messages, usage: action.usage };
      }
    }
  }

  async send(request: { method: string; path: string; body: string }): Promise<Json> {
    const { baseUrl = "https://api.openai.com/v1", fetch: send = fetch } = this.options;
    const apiKey = this.options.apiKey ?? process.env.OPENAI_API_KEY;
    const response = await send(baseUrl + request.path, {
      method: request.method,
      headers: { authorization: `Bearer ${apiKey}`, "content-type": "application/json" },
      body: request.body,
    });
    return { type: "response", status: response.status, body: await response.text() };
  }

  async callTool(call: ToolCall): Promise<Json> {
    const tool = this.options.tools?.find((tool) => tool.name === call.tool_name);
    if (tool === undefined) {
      return { tool_call_id: call.tool_call_id, retry: `Unknown tool name: '${call.tool_name}'.` };
    }
    try {
      const args = parseArgs(call.args);
      return { tool_call_id: call.tool_call_id, content: await tool.execute(args) };
    } catch (error) {
      if (error instanceof ModelRetry) {
        return { tool_call_id: call.tool_call_id, retry: error.message };
      }
      throw error;
    }
  }
}

function parseArgs(args: string): { [key: string]: Json } {
  try {
    return JSON.parse(args || "{}");
  } catch {
    throw new ModelRetry(`The arguments are not valid JSON: ${args}`);
  }
}
