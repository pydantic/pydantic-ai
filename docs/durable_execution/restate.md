# Durable Execution with Restate

[Restate](https://restate.dev) is a lightweight durable execution runtime with first-class support for AI agents. The Pydantic AI integration is provided via the [Restate Python SDK](https://github.com/restatedev/sdk-python/tree/main/python/restate/ext/pydantic).

Visit the [Restate documentation](https://docs.restate.dev/ai/patterns/durable-agents) for more information.

## Durable Execution

Restate makes your agent **durable** by recording every step of its execution in a journal. If your process crashes mid-execution, Restate replays the journal, skips completed steps, and resumes from exactly where it left off.

Your agent runs in a regular HTTP handler inside a Restate **service**. The Restate Server sits in front of your application and manages orchestration, journaling, and retries. Services run like regular Docker containers or serverless functions.

A durable agent has three building blocks:

1. The **handler**: your agent logic, exposed as an HTTP endpoint in a Restate service.
2. **LLM calls**: persisted so responses are not re-fetched on recovery — saving cost and time.
3. **Tool executions**: wrapped in durable steps so side effects are not duplicated.

```text
                  Clients
              (HTTP, Kafka, etc.)
                     |
                     v
            +---------------------+
            |   Restate Server    |      (Journals execution,
            +---------------------+       retries on failure,
                     ^                    manages state)
                     |
        Journal      |   Replay on
        steps,       |   recovery,
        retries      |   schedule calls
                     v
+------------------------------------------------------+
|               Application Process                    |
|   +----------------------------------------------+   |
|   |         Restate Service Handler              |   |
|   |           (Agent Run Loop)                   |   |
|   |    [ Durable Steps (Tool, MCP, Model) ]      |   |
|   +----------------------------------------------+   |
|         |           |                |               |
+------------------------------------------------------+
          |           |                |
          v           v                v
      [External APIs, services, databases, etc.]
```

See the [Restate documentation](https://docs.restate.dev/ai/patterns/durable-agents) for more information.

## Durable Agent

Any Pydantic AI agent can be made durable by wrapping it with `RestateAgent` from the Restate SDK and running it inside a Restate service handler.

Install the Restate SDK:

```bash
pip/uv-add pydantic-ai restate_sdk[serde]
```

Here is a complete example of a durable Pydantic AI agent with Restate:

```python {title="restate_agent.py" test="skip" lint="skip"}
import restate
from pydantic_ai import Agent, RunContext
from restate.ext.pydantic import RestateAgent, restate_context

weather_agent = Agent(  # (1)!
    'openai:gpt-5.2',
    system_prompt='You are a helpful agent that provides weather updates.',
)


@weather_agent.tool()
async def get_weather(_run_ctx: RunContext[None], city: str) -> dict:
    """Get the current weather for a given city."""

    # Do durable tool steps using the Restate context
    async def call_weather_api(city: str) -> dict:
        return {'temperature': 23, 'description': 'Sunny and warm.'}

    return await restate_context().run_typed(  # (2)!
        f'Get weather {city}', call_weather_api, city=city
    )


restate_agent = RestateAgent(weather_agent)  # (3)!

agent_service = restate.Service('WeatherAgent')


@agent_service.handler()
async def run(_ctx: restate.Context, prompt: str) -> str:  # (4)!
    result = await restate_agent.run(prompt)
    return result.output


app = restate.app(services=[agent_service])  # (5)!

if __name__ == "__main__":  # (6)!
    import hypercorn
    import asyncio
    conf = hypercorn.Config()
    conf.bind = ["0.0.0.0:9080"]
    asyncio.run(hypercorn.asyncio.serve(app, conf))
```

1. Define your agent and tools as you normally would with Pydantic AI.
2. Use `restate_context()` actions inside tools to make their execution durable. The result is persisted and retried until it succeeds. Side effects won't be duplicated on recovery.
3. `RestateAgent` wraps the agent so every LLM response is saved in the Restate Server and replayed during recovery.
4. The Restate service handler gives the agent a durable execution context and exposes it as an HTTP endpoint.
5. `restate.app()` creates the application that can be served.
6. Run the application with an ASGI server like Hypercorn.

See the [Restate agent quickstart](https://docs.restate.dev/ai-quickstart) to learn how to run the agent.
