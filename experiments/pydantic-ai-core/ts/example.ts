// Ask a real model a question that needs a tool: `node ts/example.ts`, with `OPENAI_API_KEY` set.

import { Agent } from "./agent.ts";

const agent = new Agent({
  model: "gpt-5.4-mini",
  instructions: "Answer in one short sentence.",
  tools: [
    {
      name: "get_temperature",
      description: "Get the current temperature in a city.",
      parameters: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
        additionalProperties: false,
      },
      execute: ({ city }) => ({ city, celsius: 21 }),
    },
  ],
});

const result = await agent.run("What is the temperature in Tokyo?");
console.log(result.output);
console.log(result.usage);
console.log(result.messages.length, "messages");
