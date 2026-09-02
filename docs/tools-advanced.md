# Advanced Tool Features

This page covers advanced features for function tools in Pydantic AI. For basic tool usage, see the [Function Tools](tools.md) documentation.

## Tool Output {#function-tool-output}

Tools can return anything that Pydantic can serialize to JSON, as well as audio, video, image or document content depending on the types of [multi-modal input](input.md) the model supports:

```python {title="function_tool_output.py"}
from datetime import datetime

from pydantic import BaseModel

from pydantic_ai import Agent, DocumentUrl, ImageUrl
from pydantic_ai.models.openai import OpenAIResponsesModel


class User(BaseModel):
    name: str
    age: int


agent = Agent(model=OpenAIResponsesModel('gpt-5.2'))


@agent.tool_plain
def get_current_time() -> datetime:
    return datetime.now()


@agent.tool_plain
def get_user() -> User:
    return User(name='John', age=30)


@agent.tool_plain
def get_company_logo() -> ImageUrl:
    return ImageUrl(url='https://iili.io/3Hs4FMg.png')


@agent.tool_plain
def get_document() -> DocumentUrl:
    return DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')


result = agent.run_sync('What time is it?')
print(result.output)
#> The current time is 10:45 PM on April 17, 2025.

result = agent.run_sync('What is the user name?')
print(result.output)
#> The user's name is John.

result = agent.run_sync('What is the company name in the logo?')
print(result.output)
#> The company name in the logo is "Pydantic."

result = agent.run_sync('What is the main content of the document?')
print(result.output)
#> The document contains just the text "Dummy PDF file."
```

_(This example is complete, it can be run "as is")_

Some models (e.g. Gemini) natively support semi-structured return values, while some expect text (OpenAI) but seem to be just as good at extracting meaning from the data. If a Python object is returned and the model expects a string, the value will be serialized to JSON.

### Advanced Tool Returns

For scenarios where you need more control over both the tool's return value and the content sent to the model, you can use [`ToolReturn`][pydantic_ai.messages.ToolReturn]. This is particularly useful when you want to:

- Separate the structured return value from additional content sent to the model
- Explicitly send content as a separate user message (rather than in the tool result)
- Include additional metadata that shouldn't be sent to the LLM
- Reveal deferred tools by name for the next model request

Here's an example of a computer automation tool that captures screenshots and provides visual feedback:

```python {title="advanced_tool_return.py"}
from pydantic_ai import Agent, BinaryContent, ToolReturn
from pydantic_ai.models.test import TestModel

agent = Agent(TestModel())

@agent.tool_plain
def click_and_capture(x: int, y: int) -> ToolReturn:
    """Click at coordinates and show before/after screenshots."""
    before_screenshot = BinaryContent(data=b'\x89PNG', media_type='image/png')
    # perform_click(x, y)
    after_screenshot = BinaryContent(data=b'\x89PNG', media_type='image/png')
    return ToolReturn(
        return_value=f'Successfully clicked at ({x}, {y})',
        content=[
            'Before:',
            before_screenshot,
            'After:',
            after_screenshot,
        ],
        metadata={
            'coordinates': {'x': x, 'y': y},
            'action_type': 'click_and_capture',
        },
    )

# The model receives the rich visual content for analysis
# while your application can access the structured return_value and metadata
result = agent.run_sync('Click on the submit button and tell me what happened')
print(result.output)
#> {"click_and_capture":"Successfully clicked at (0, 0)"}
```

- **`return_value`**: The actual return value used in the tool response. This is what gets serialized and sent back to the model as the tool's result. Can include multimodal content directly (see [Tool Output](#function-tool-output) above).
- **`tools`**: Names of tools marked with `defer_loading=True` that this call made available. Pydantic AI records them in a [`ToolAvailabilityDeltaPart`][pydantic_ai.messages.ToolAvailabilityDeltaPart] immediately after this call's [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart], in the same [`ModelRequest`][pydantic_ai.messages.ModelRequest]. The names remain revealed when history is resumed, while the current tool definitions still come from the agent.
- **`content`**: Content sent as a **separate user message** after the tool result. Use this when you explicitly want content to appear outside the tool result, or when combining structured return values with rich content.
- **`metadata`**: Optional metadata that your application can access but is not sent to the LLM. Useful for logging, debugging, or additional processing. Some other AI frameworks call this feature 'artifacts'.

This separation allows you to provide rich context to the model while maintaining clean, structured return values for your application logic. For multimodal content that should be sent natively in the tool result (when supported by the model), return it directly from the tool function or include it in `return_value` (see [Tool Output](#function-tool-output) above).

## Custom Tool Schema

If you have a function that lacks appropriate documentation (i.e. poorly named, no type information, poor docstring, use of \*args or \*\*kwargs and suchlike) then you can still turn it into a tool that can be effectively used by the agent with the [`Tool.from_schema`][pydantic_ai.tools.Tool.from_schema] function. With this you provide the name, description, JSON schema, and whether the function takes a `RunContext` for the function directly:

```python
from pydantic_ai import Agent, Tool
from pydantic_ai.models.test import TestModel


def foobar(**kwargs) -> str:
    return kwargs['a'] + kwargs['b']

tool = Tool.from_schema(
    function=foobar,
    name='sum',
    description='Sum two numbers.',
    json_schema={
        'additionalProperties': False,
        'properties': {
            'a': {'description': 'the first number', 'type': 'integer'},
            'b': {'description': 'the second number', 'type': 'integer'},
        },
        'required': ['a', 'b'],
        'type': 'object',
    },
    takes_ctx=False,
)

test_model = TestModel()
agent = Agent(test_model, tools=[tool])

result = agent.run_sync('testing...')
print(result.output)
#> {"sum":0}
```

Pydantic AI does not validate tool arguments here; it passes them as keyword arguments.

## Strict Mode {#strict-mode}

Some providers support a *strict* mode for tool calls that constrains the model so its tool-call arguments always conform to the tool's JSON schema. Rather than letting the model generate arguments freely and validating them after the fact, the provider restricts generation so that out-of-schema arguments aren't produced in the first place. This is controlled by the `strict` flag, available on every tool registration mechanism ([`@agent.tool`][pydantic_ai.agent.Agent.tool], [`@agent.tool_plain`][pydantic_ai.agent.Agent.tool_plain], [`Tool`][pydantic_ai.tools.Tool], [`FunctionToolset.add_function`][pydantic_ai.toolsets.function.FunctionToolset.add_function], etc.) and on [`ToolDefinition`][pydantic_ai.tools.ToolDefinition]:

```python
from pydantic import BaseModel

from pydantic_ai import Agent


class Reservation(BaseModel):
    restaurant: str
    party_size: int
    outdoor_seating: bool
    dietary_notes: list[str]


agent = Agent('openai:gpt-5')


@agent.tool_plain(strict=True)
def book_table(reservation: Reservation) -> str:
    return f'Booked a table for {reservation.party_size} at {reservation.restaurant}.'
```

Strict mode earns its keep on structured arguments like this: without it the model might emit `party_size` as a string, omit `outdoor_seating`, or invent an extra property — strict generation rules those out up front rather than relying on a validation retry.

Because strict mode guarantees the arguments match the schema exactly, not every schema can be represented under it: some providers require every property to be listed in `required` and objects to set `additionalProperties: false`. A schema that can't be represented this way may be transformed lossily or have the flag ignored for that tool — which is why `strict=True` is best read as a request to *force* strict mode wherever the provider can honor it.

Pydantic AI translates the `strict` flag into a native schema-enforcement feature for **OpenAI**, **Anthropic**, **Google**, and **Bedrock** models; other providers ignore it. Each provider's underlying feature differs:

| Provider | Behavior |
|---|---|
| OpenAI | Strict tool definitions. Enabled automatically when the tool's schema is strict-compatible; `strict=True` forces it. |
| Anthropic | Strict tool definitions. Off unless you opt in with `strict=True`. |
| Bedrock | Strict tool spec, on supported models. Off unless you opt in with `strict=True`. |
| Google (Gemini) | Gemini's [`VALIDATED` function-calling mode](https://ai.google.dev/gemini-api/docs/function-calling#function_calling_config), which ensures the model adheres to the declared schema. On **Gemini 2.5 and newer it is enabled by default** — `VALIDATED` needs no schema changes, so it's a free improvement — and you can opt out with `strict=False`. Gemini's mode is request-wide: any function or output tool with `strict=False` keeps the whole request on `AUTO`. `VALIDATED` is a preview Gemini feature. |

### Strictness values

The `strict` flag is a `bool | None`:

- `True` — force strict mode wherever the provider supports it for the tool's schema.
- `False` — never use strict mode for the tool. On Google, any tool (function or output) with `strict=False` keeps the whole request on `AUTO` rather than `VALIDATED`.
- `None` (**default**) — decide per provider: OpenAI enables strict when the schema is strict-compatible; Google defaults to `VALIDATED` on supported models; Anthropic and Bedrock leave it off unless you explicitly opt in with `strict=True`.

To turn strict mode on for many tools at once, use [agent-wide dynamic tools](#prepare-tools) to set `strict=True` on each [`ToolDefinition`][pydantic_ai.tools.ToolDefinition].

## Dynamic Tools {#tool-prepare}

Tools can optionally be defined with another function: `prepare`, which is called at each step of a run to
customize the definition of the tool passed to the model, or omit the tool completely from that step.

A `prepare` method can be registered via the `prepare` kwarg to any of the tool registration mechanisms:

- [`@agent.tool`][pydantic_ai.agent.Agent.tool] decorator
- [`@agent.tool_plain`][pydantic_ai.agent.Agent.tool_plain] decorator
- [`Tool`][pydantic_ai.tools.Tool] dataclass

The `prepare` method has type [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc]. It receives [`RunContext`][pydantic_ai.tools.RunContext] and a pre-built [`ToolDefinition`][pydantic_ai.tools.ToolDefinition]. It can return that definition unchanged or modified, return a new definition, or return `None` to omit the tool for that step.

Here's a simple `prepare` method that only includes the tool if the value of the dependency is `42`.

As with the previous example, we use [`TestModel`][pydantic_ai.models.test.TestModel] to demonstrate the behavior without calling a real model.

```python {title="tool_only_if_42.py"}

from pydantic_ai import Agent, RunContext, ToolDefinition

agent = Agent('test')


async def only_if_42(
    ctx: RunContext[int], tool_def: ToolDefinition
) -> ToolDefinition | None:
    if ctx.deps == 42:
        return tool_def


@agent.tool(prepare=only_if_42)
def hitchhiker(ctx: RunContext[int], answer: str) -> str:
    return f'{ctx.deps} {answer}'


result = agent.run_sync('testing...', deps=41)
print(result.output)
#> success (no tool calls)
result = agent.run_sync('testing...', deps=42)
print(result.output)
#> {"hitchhiker":"42 a"}
```

_(This example is complete, it can be run "as is")_

The following example changes the `name` parameter's description based on the value of `deps`.

For the sake of variation, we create this tool using the [`Tool`][pydantic_ai.tools.Tool] dataclass.

```python {title="customize_name.py"}
from __future__ import annotations

from typing import Literal

from pydantic_ai import Agent, RunContext, Tool, ToolDefinition
from pydantic_ai.models.test import TestModel


def greet(name: str) -> str:
    return f'hello {name}'


async def prepare_greet(
    ctx: RunContext[Literal['human', 'machine']], tool_def: ToolDefinition
) -> ToolDefinition | None:
    d = f'Name of the {ctx.deps} to greet.'
    tool_def.parameters_json_schema['properties']['name']['description'] = d
    return tool_def


greet_tool = Tool(greet, prepare=prepare_greet)
test_model = TestModel()
agent = Agent(test_model, tools=[greet_tool], deps_type=Literal['human', 'machine'])

result = agent.run_sync('testing...', deps='human')
print(result.output)
#> {"greet":"hello a"}
print(test_model.last_model_request_parameters.function_tools)
"""
[
    ToolDefinition(
        name='greet',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {
                'name': {'type': 'string', 'description': 'Name of the human to greet.'}
            },
            'required': ['name'],
            'type': 'object',
        },
        toolset_id='<agent>',
    )
]
"""
```

_(This example is complete, it can be run "as is")_

### Agent-wide Dynamic Tools {#prepare-tools}

In addition to per-tool `prepare` methods, you can also define an agent-wide `prepare_tools` function. This function is called at each step of a run and allows you to filter or modify the list of all tool definitions available to the agent for that step. This is especially useful if you want to enable or disable multiple tools at once, or apply global logic based on the current context.

The `prepare_tools` function should be of type [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc], which takes the [`RunContext`][pydantic_ai.tools.RunContext] and a list of [`ToolDefinition`][pydantic_ai.tools.ToolDefinition], and returns the tool definitions to expose for that step. Return the `tool_defs` argument to keep every tool as-is, or `[]` to expose no tools.

!!! note
    The list of tool definitions passed to `prepare_tools` includes both regular function tools and tools from any [toolsets](toolsets.md) registered on the agent, but not [output tools](output.md#tool-output).
To modify output tools, you can set a `prepare_output_tools` function instead.

Here's an example that makes all tools strict if the model is an OpenAI model:

```python {title="agent_prepare_tools_customize.py" noqa="I001"}
from dataclasses import replace

from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.models.test import TestModel


async def turn_on_strict_if_openai(
    ctx: RunContext, tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    if ctx.model.system == 'openai':
        return [replace(tool_def, strict=True) for tool_def in tool_defs]
    return tool_defs


test_model = TestModel()
agent = Agent(test_model, capabilities=[PrepareTools(turn_on_strict_if_openai)])


@agent.tool_plain
def echo(message: str) -> str:
    return message


agent.run_sync('testing...')
assert test_model.last_model_request_parameters.function_tools[0].strict is None

# Set the system attribute of the test_model to 'openai'
test_model._system = 'openai'

agent.run_sync('testing with openai...')
assert test_model.last_model_request_parameters.function_tools[0].strict
```

_(This example is complete, it can be run "as is")_

Here's another example that conditionally filters out the tools by name if the dependency (`ctx.deps`) is `True`:

```python {title="agent_prepare_tools_filter_out.py" noqa="I001"}

from pydantic_ai import Agent, RunContext, Tool, ToolDefinition
from pydantic_ai.capabilities import PrepareTools


def launch_potato(target: str) -> str:
    return f'Potato launched at {target}!'


async def filter_out_tools_by_name(
    ctx: RunContext[bool], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    if ctx.deps:
        return [tool_def for tool_def in tool_defs if tool_def.name != 'launch_potato']
    return tool_defs


agent = Agent(
    'test',
    tools=[Tool(launch_potato)],
    capabilities=[PrepareTools(filter_out_tools_by_name)],
    deps_type=bool,
)

result = agent.run_sync('testing...', deps=False)
print(result.output)
#> {"launch_potato":"Potato launched at a!"}
result = agent.run_sync('testing...', deps=True)
print(result.output)
#> success (no tool calls)
```

_(This example is complete, it can be run "as is")_

You can use `prepare_tools` to:

- Dynamically enable or disable tools based on the current model, dependencies, or other context
- Modify tool definitions globally (e.g., set all tools to strict mode, change descriptions, etc.)

If both per-tool `prepare` and agent-wide `prepare_tools` are used, the per-tool `prepare` is applied first to each tool, and then `prepare_tools` is called with the resulting list of tool definitions.

## Tool Choice {#tool-choice}

The `tool_choice` setting in [`ModelSettings`][pydantic_ai.settings.ModelSettings] controls which tools the model can use during a request. This is useful for disabling tools, forcing tool use, or restricting which tools are available.

Pydantic AI distinguishes between **[function tools](tools.md)** (tools you register via `@agent.tool`, [toolsets](toolsets.md), or [MCP](mcp/client.md)), and **output tools** (internal tools used for [structured output](output.md#tool-output)).

### Options

| Value | Description |
|-------|-------------|
| `'auto'` (default) | Model decides whether to use tools. All tools available. |
| `'none'` | Disable function tools. Model can respond with text or use output tools. |
| `'required'` | Force the model to use a function tool. Excludes output tools, so set dynamically via a [capability](#dynamic-tool-choice-via-capabilities) or use [direct model requests](direct.md); raises an error when set statically in `agent.run()`. |
| `['tool_a', ...]` | Restrict to specific tools by name. Excludes output tools — same dynamic/direct requirement as `'required'`. |
| [`ToolOrOutput`][pydantic_ai.settings.ToolOrOutput]`(function_tools=['...'])` | Restrict function tools while auto-including all output tools. |

Tools hidden by [deferred loading](#tool-search) interact with `tool_choice`: a tool that is still
hidden is ignored when forcing by name, and an explicit choice raises only when every requested
tool is hidden. `'required'` raises when every function
tool is hidden. A tool *declared* with its schema deferred can be forced. On providers that carry
revealed definitions outside the `tools` list (OpenAI Responses `additional_tools`), a revealed
tool can't be forced by name either, since by-name forcing can only target declared tools.

### Example

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ToolOrOutput

agent = Agent(TestModel())


@agent.tool_plain
def get_weather(city: str) -> str:
    return f'Sunny in {city}'


@agent.tool_plain
def get_time(city: str) -> str:
    return f'12:00 in {city}'


# Pass tool_choice via model_settings
result = agent.run_sync('Hello', model_settings={'tool_choice': 'none'})

# Use ToolOrOutput to restrict to specific function tools while allowing output
result = agent.run_sync(
    'Hello', model_settings={'tool_choice': ToolOrOutput(function_tools=['get_weather'])}
)
```

### Dynamic tool choice via capabilities {#dynamic-tool-choice-via-capabilities}

`tool_choice='required'` and `['tool_a', ...]` exclude output tools, so setting either one *statically* would force a tool call on every step and leave the agent unable to produce a final response. `agent.run()` raises a `UserError` when it detects these values on the static baseline (the `model_settings` argument of [`Agent.run`][pydantic_ai.agent.AbstractAgent.run], the agent's own `model_settings`, or the underlying model's defaults).

To vary `tool_choice` *per step* — for example, to force a specific tool on the first step and then let the model decide — return a callable from a capability's [`get_model_settings`][pydantic_ai.capabilities.AbstractCapability.get_model_settings]. The callable receives a [`RunContext`][pydantic_ai.tools.RunContext] with full access to `ctx.messages` and `ctx.run_step`, so it can inspect what has already happened in the run and adapt.

```python {title="force_first_call.py"}
from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, ToolReturnPart


class RequireFirstCall(AbstractCapability):
    """Force `tool_name` to be called successfully before anything else."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name

    def get_model_settings(self):
        def settings(ctx: RunContext) -> ModelSettings:
            called = any(
                isinstance(part, ToolReturnPart) and part.tool_name == self.tool_name
                for message in ctx.messages
                if isinstance(message, ModelRequest)
                for part in message.parts
            )
            if called:
                return ModelSettings()
            return ModelSettings(tool_choice=[self.tool_name])

        return settings


agent = Agent('openai:gpt-5.2', capabilities=[RequireFirstCall('get_weather')])


@agent.tool_plain
def get_weather(city: str) -> str:
    return f'Sunny in {city}'
```

Because capability-supplied settings are resolved per step, the callable's returned `tool_choice` is trusted to change across steps and is not rejected by the baseline validator. For a single model request without an agent loop, use [`pydantic_ai.direct.model_request`][pydantic_ai.direct.model_request] instead.

### Provider Support

All providers support `'auto'` and `'none'`. Key differences for other options:

| Provider | `'required'` | Specific tools | Notes |
|----------|:------------:|:--------------:|-------|
| OpenAI | ✓ | ✓ | Full support |
| Anthropic | ⚠️ | ⚠️ | Not supported with extended thinking; adaptive thinking is compatible |
| Google | ✓ | ✓ | |
| Bedrock | ✓ | Single only | Multiple tools fall back to 'any' mode |
| Groq/HuggingFace | ✓ | Single only | Multiple tools fall back to 'required' mode |
| Mistral | ✓ | ✓ | Maps `'required'` to `'any'` mode |
| Cohere | ✓ | ✓ | Maps `'required'` to `'REQUIRED'`; a named subset is applied by trimming the tools array |
| xAI | ✓ | ✓ | Some models may not support forcing; falls back to 'auto' |

The model classes built on `OpenAIChatModel` — Cerebras, Crusoe, Ollama, OpenRouter, Snowflake, Z.AI and Bedrock Mantle Chat — behave as the OpenAI row describes, with two exceptions. Ollama documents `tool_choice` as unsupported and ignores it. OpenRouter raises a `UserError` for an explicit `'required'` or named subset on models that can't combine forced tool choice with thinking, rather than silently dropping the reasoning; forcing that Pydantic AI merely inferred falls back to `'auto'` instead.

### Prompt caching implications {#tool-choice-caching}

Restricting the available tool set via `tool_choice` can invalidate provider prompt caches because most provider APIs cache on the full tools array. Pydantic AI restricts the tool set in two ways:

- **API-level filtering** (cache-preserving): the full tools array is sent and the provider is told to only allow a subset. Used by OpenAI Responses (`allowed_tools`), Google (`allowed_function_names`), and Bedrock when forcing a single tool.
- **Client-side filtering** (breaks cache): the tools array is trimmed before the request. Used when the provider API has no native filter for the given case.

The table below covers the cases where Pydantic AI must filter client-side and therefore breaks cache:

| Provider | Cache-breaking case |
|----------|---------------------|
| Anthropic | `tool_choice` is a list of multiple tools, OR a single tool with extended thinking or on a model that doesn't support forcing |
| OpenAI Chat | `tool_choice` is a list of multiple tools, OR a single tool on a model that doesn't support forcing |
| Bedrock | `tool_choice` is a list of multiple tools, OR a single tool with thinking enabled or on a model that doesn't support forcing |
| Groq / HuggingFace | `tool_choice` is a list of multiple tools |
| Mistral | `tool_choice` is a list (any size) — the API doesn't accept specific tool names |
| Cohere | `tool_choice` is a list (any size) — the API doesn't accept specific tool names |
| xAI | `tool_choice` is a list of multiple tools, OR a single tool on a model that doesn't support forcing |
| OpenAI Responses | Never — `allowed_tools` handles all cases natively |
| Google | Never — `allowed_function_names` handles all cases natively |

If preserving cache hits matters, prefer providers/cases marked "Never", or use `ToolOrOutput` (which keeps the full set) instead of a restrictive list.

## Tool Execution, Retries, and Failures {#tool-retries}

When a tool is executed, its arguments (provided by the LLM) are first validated against the function's signature using Pydantic (with optional [validation context](output.md#validation-context)). If validation fails (e.g., due to incorrect types or missing required arguments), a `ValidationError` is raised, and the framework automatically generates a [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart] containing the validation details. This prompt is sent back to the LLM, informing it of the error and allowing it to correct the parameters and retry the tool call.

If a tool's own logic cannot produce a normal result, choose the exception based on what you want the model to do next:

- Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] when the model should try the tool call again with corrected arguments or a different approach.
- Raise [`ToolFailed`][pydantic_ai.exceptions.ToolFailed] when the tool call should be reported to the model as a failed result, without consuming the tool's retry budget.

Any other exception propagates out of the agent run and is not sent back to the model.

### Requesting a Tool Retry

Raising `ModelRetry` generates a [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart] containing the exception message. That prompt is sent back to the LLM so it can correct the tool call, choose another tool, or try a different approach.

```python
from pydantic_ai import ModelRetry


def my_flaky_tool(query: str) -> str:
    if query == 'bad':
        # Tell the LLM the query was bad and it should try again
        raise ModelRetry("The query 'bad' is not allowed. Please provide a different query.")
    # ... process query ...
    return 'Success!'
```

Both `ValidationError` and `ModelRetry` respect the configured retry limit — set per-tool via [`Tool(max_retries=N)`][pydantic_ai.tools.Tool] (or `@agent.tool(retries=N)`), per-toolset via [`FunctionToolset(max_retries=N)`][pydantic_ai.toolsets.FunctionToolset], or agent-wide via [`Agent(retries={'tools': N})`][pydantic_ai.agent.Agent.__init__], applied in that order of precedence. The agent-wide default can also be overridden per run via [`agent.run(retries={'tools': N})`][pydantic_ai.agent.Agent.run] (and `run_sync`/`run_stream`/`iter`, or for a block of runs via [`agent.override()`][pydantic_ai.agent.Agent.override]); a per-run value replaces the agent-wide default at the bottom of the precedence chain, so explicit per-tool and per-toolset limits still win. A bare `int` at these run-time call sites overrides both budgets (matching construction) — pass a dict such as `retries={'tools': N}` or `retries={'output': N}` to change just one.

Tool retries are tracked **per tool**: every function tool has its own counter, with no global 'tool call' budget shared across the run. When a tool raises `ModelRetry` or its arguments fail validation, only that tool's counter advances. Inside a tool function, [`ctx.max_retries`][pydantic_ai.tools.RunContext.max_retries] reflects that tool's enforcement limit and [`ctx.retry`][pydantic_ai.tools.RunContext.retry] is that tool's own counter. When a tool exhausts its counter, the run raises [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] with message `'Tool {name!r} exceeded max retries count of {N}. Consider raising the retry limit, or see the docs on tool retries: https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-retries'`. User-provided toolsets inherit the agent-wide tool-retry default — or its per-run override — as their default when no per-toolset value is set.

### Which retry limit wins

Two independent budgets — the **tool** budget (per function/output tool) and the **output** budget (output validation) — each resolve through the same layered precedence. The first layer that sets a value wins; unset layers fall through to the next:

| Precedence (highest first) | How to set it | Budget it sets |
|----------------------------|---------------|----------------|
| 1. Per-tool limit | `@agent.tool(retries=N)` / [`Tool(max_retries=N)`][pydantic_ai.tools.Tool]; [`ToolOutput(max_retries=N)`][pydantic_ai.output.ToolOutput.max_retries] for an output tool | that one tool |
| 2. Per-toolset limit | [`FunctionToolset(max_retries=N)`][pydantic_ai.toolsets.FunctionToolset] or [`MCPToolset(max_retries=N)`][pydantic_ai.mcp.MCPToolset] | tools in that toolset |
| 3. Override block | [`agent.override(retries=...)`][pydantic_ai.agent.Agent.override] | tool and/or output |
| 4. Per-run argument | [`agent.run(retries=...)`][pydantic_ai.agent.Agent.run] (and `run_sync`/`run_stream`/`iter`) | tool and/or output |
| 5. Per-run spec | `agent.run(spec={'retries': ...})` | tool and/or output |
| 6. Agent-wide default | [`Agent(retries=...)`][pydantic_ai.agent.Agent.__init__] | tool and/or output |
| 7. Built-in default | — | `1` |

At layers 3–6, a bare `int` sets **both** budgets to that value, while an [`AgentRetries`][pydantic_ai.agent.AgentRetries] dict sets only the keys it names (`{'tools': N}`, `{'output': N}`, or both). Layers 3–5 override the agent-wide default (layer 6) but never a more specific per-tool (layer 1) or per-toolset (layer 2) limit.

### Reporting a Failed Tool Result {#tool-failed}

Not every tool failure is a correction request. When the call is complete but unsuccessful — the resource doesn't exist, the operation isn't supported, the upstream service returned a definitive error — you usually want the model to *see* the failed result and decide what to do next. Raise `ToolFailed` for this:

```python
from pathlib import Path

from pydantic_ai import ToolFailed


def read_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise ToolFailed(f'File not found: {path}')
    return file_path.read_text()
```

The exception message is recorded in message history as a [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] with `outcome='failed'`. Where the model API has a native error or failed-status field for tool results, Pydantic AI uses it. For APIs without a native error channel, the model-visible content is JSON-framed as `{"error": ...}` so the failure is still explicit. The failed outcome is preserved in Pydantic AI message history; protocol adapters may need their own carrier when that history is round-tripped, as described for [AG-UI](ui/ag-ui.md#preserving-failed-tool-outcomes). The call is traced as an error in telemetry.

Unlike `ModelRetry`, `ToolFailed` does **not** consume the per-tool retry budget; bounding repeated failures is the job of [`UsageLimits`][pydantic_ai.usage.UsageLimits] at the run level — specifically [`request_limit`][pydantic_ai.usage.UsageLimits.request_limit], since [`tool_calls_limit`][pydantic_ai.usage.UsageLimits.tool_calls_limit] only counts successful tool invocations.

Rule of thumb: raise `ModelRetry` when you want the model to try again with corrections; raise `ToolFailed` when the call is done and the result is a failure. For MCP server tool errors, the same choice is available as the [`tool_error_behavior`](mcp/client.md#tool-errors) configuration.

You can also raise `ModelRetry` or `ToolFailed` from tool validation and execution hooks. This is useful for converting third-party exceptions without repeating `try`/`except` in every tool; see [Error hooks](hooks.md#error-hooks) and [Tool execution hooks](hooks.md#tool-execution-hooks).

`ToolFailed` is handled for function tools, their `args_validator`, and tool validation or execution hooks. [Output functions](output.md#output-functions) and [output validators](output.md#output-validator-functions) use `ModelRetry` when the model should try again; there, `ToolFailed` is an ordinary exception that aborts the run unless an output-process error hook recovers from it.

### Tool Timeout

You can set a timeout for tool execution to prevent tools from running indefinitely. If a tool exceeds its timeout, it is treated as a retryable failure and a retry prompt is sent to the model (counting towards the retry limit).

```python
import asyncio

from pydantic_ai import Agent

# Set a default timeout for the agent's own tools
agent = Agent('test', tool_timeout=30)


@agent.tool_plain
async def slow_tool() -> str:
    """This tool will use the agent's default timeout (30 seconds)."""
    await asyncio.sleep(10)
    return 'Done'


@agent.tool_plain(timeout=5)
async def fast_tool() -> str:
    """This tool has its own timeout (5 seconds) that overrides the agent default."""
    await asyncio.sleep(1)
    return 'Done'
```

- **Agent-level timeout**: Set `tool_timeout` on the [`Agent`][pydantic_ai.agent.Agent] to apply a default timeout to the tools registered on it.
- **Per-tool timeout**: Set `timeout` on individual tools via [`@agent.tool`][pydantic_ai.agent.Agent.tool], [`@agent.tool_plain`][pydantic_ai.agent.Agent.tool_plain], or the [`Tool`][pydantic_ai.tools.Tool] dataclass. This overrides the agent-level default.

When a timeout occurs, the tool is treated as a retryable failure and the model receives a retry prompt with the message `"Timed out after {timeout} seconds."`. This counts towards the tool's retry limit just like validation errors or explicit [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] exceptions.

Both settings are enforced by [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset], which is what backs the agent's own tools. Tools served by an [MCP server](mcp/client.md), an [external toolset](deferred-tools.md), or a custom [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] do not read them — bind those deadlines at the server or transport level instead. See [Timeouts](timeouts.md#bounding-how-long-a-step-takes) for how tool timeouts relate to the other deadlines in a run.

### Cancelling the Run from a Tool

A tool can abort the entire run by calling [`RunContext.cancel()`][pydantic_ai.tools.RunContext.cancel] -- e.g. when it discovers that further work is pointless, or a stop signal reaches your application while a tool holds the `RunContext`. From outside the run, pass a [`CancellationToken`][pydantic_ai.CancellationToken] to any run method. The run requests cancellation of whatever is in flight and raises [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] to the caller. Async tool tasks executing [in parallel](#parallel-tool-calls-concurrency) are cancelled and drained, but Python cannot forcibly stop a synchronous tool's worker thread; cancellation may wait for the worker or let it finish in the background. Either way, its eventual result is discarded while any side effects remain. Tool calls that already completed keep their results in the message history, and a tool call that never produced a result is repaired automatically when that history is reused. Both cancellation surfaces require being in the same process as the run, so they do not cross a [durable execution](durable_execution/overview.md) serialization boundary such as a Temporal activity.

See [Cancelling a Run](agent.md#cancelling-a-run) for the full picture, including cancelling from outside the run and accessing the cancelled run's state.

### Custom Args Validator {#args-validator}

The `args_validator` parameter lets you define custom validation that runs after Pydantic schema validation but before the tool executes. This is useful for business logic validation, cross-field validation, or validating arguments before requesting [human approval](deferred-tools.md) for deferred tools.

The validator receives [`RunContext`][pydantic_ai.tools.RunContext] as its first argument, followed by the same parameters as the tool function. Return `None` on success, raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to ask the model to correct the arguments and try again, raise [`ToolFailed`][pydantic_ai.exceptions.ToolFailed] to report a terminal failure the model should adapt to instead of retrying, or raise [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] / [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] to [defer the call](deferred-tools.md) (see below).

```python {title="args_validator_approval.py"}
from pydantic_ai import Agent, DeferredToolRequests, ModelRetry, RunContext

agent = Agent('test', deps_type=int, output_type=[str, DeferredToolRequests])


def validate_sum_limit(ctx: RunContext[int], x: int, y: int) -> None:
    """Validate that the sum doesn't exceed the limit from deps."""
    if x + y > ctx.deps:
        raise ModelRetry(f'Sum of x and y must not exceed {ctx.deps}')


# Validation runs *before* approval is requested, so the model can
# fix bad args without bothering the user.
@agent.tool(requires_approval=True, args_validator=validate_sum_limit)
def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
    """Add two numbers (sum must not exceed the configured limit)."""
    return x + y


result = agent.run_sync('add 5 and 3', deps=100)
assert isinstance(result.output, DeferredToolRequests)
# The validated args are ready for the user to approve
print(result.output.approvals[0].args)
#> {'x': 0, 'y': 0}
```

_(This example is complete, it can be run "as is")_

When schema validation fails, or an `args_validator` raises `ModelRetry`, the error message is sent back to the LLM as a retry prompt (with instructions to try again) and respects the tool's `retries` setting. When an `args_validator` raises `ToolFailed`, the model instead receives a failed tool result it should adapt to rather than retry, and the retry budget is left untouched. For [deferred tools](deferred-tools.md), validation runs at deferral time — only tool calls with valid arguments are deferred.

A validator can defer the call itself, just like the tool function can — and it's the better place to make that decision, since bad arguments are rejected before a human is asked to approve them. A validator that raises `ApprovalRequired` or `CallDeferred` doesn't consume the retry budget either: the arguments were valid, so the deferral is a deliberate decision rather than a failure. The tool function is not executed, and the call joins the run's other [deferred tool calls](deferred-tools.md): it's resolved inline by a [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler if you have one, or surfaced in the run's `DeferredToolRequests` output. Once the call is [approved](deferred-tools.md#human-in-the-loop-tool-approval), the validator runs again — this time with [`RunContext.tool_call_approved`][pydantic_ai.tools.RunContext.tool_call_approved] set to `True` — and the tool executes.

Here, an impossible amount is rejected outright while a large one is put in front of a human:

```python {title="args_validator_conditional_approval.py"}
from pydantic_ai import (
    Agent,
    ApprovalRequired,
    DeferredToolRequests,
    ModelMessage,
    ModelResponse,
    ModelRetry,
    RunContext,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

agent = Agent(deps_type=int, output_type=[str, DeferredToolRequests])


def validate_transfer(ctx: RunContext[int], amount: int) -> None:
    if amount > ctx.deps:
        raise ModelRetry(f'Amount must not exceed {ctx.deps}')  # (1)!
    if amount > 100 and not ctx.tool_call_approved:
        raise ApprovalRequired()  # (2)!


@agent.tool(args_validator=validate_transfer)
def transfer_funds(ctx: RunContext[int], amount: int) -> str:
    return f'Transferred {amount}'


def call_transfer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('transfer_funds', {'amount': 500})])
    return ModelResponse(parts=[TextPart('done')])


result = agent.run_sync('transfer 500', deps=1000, model=FunctionModel(call_transfer))
assert isinstance(result.output, DeferredToolRequests)
print(result.output.approvals[0].args)
#> {'amount': 500}
```

1. The model can fix an impossible amount itself, without a human ever seeing the call.
2. Only calls that made it past validation are put in front of a human.

_(This example is complete, it can be run "as is")_

The `args_validator` parameter is available on [`@agent.tool`][pydantic_ai.agent.Agent.tool], [`@agent.tool_plain`][pydantic_ai.agent.Agent.tool_plain], [`Tool`][pydantic_ai.tools.Tool], [`Tool.from_schema`][pydantic_ai.tools.Tool.from_schema], and [`FunctionToolset`][pydantic_ai.toolsets.function.FunctionToolset]. Validators can be sync or async functions.

Under [durable execution](durable_execution/overview.md), a tool with an `args_validator` gets a dedicated validation activity, step, or task wherever the engine wraps that toolset. The validator may perform I/O and can retry, fail, or defer the call. Tools without an `args_validator` schedule no extra durable unit.

The validation result is exposed via the `args_valid` field on [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent]. This reflects all validation — both schema validation and custom `args_validator` validation (if configured): `True` means all validation passed, `False` means validation failed, and `None` means validation was not performed (e.g. tool calls skipped due to the `'early'` end strategy, or deferred tool calls resolved without execution).

### Parallel tool calls & concurrency {#parallel-tool-calls-concurrency}

When a model returns multiple tool calls in one response, Pydantic AI schedules them concurrently using `asyncio.create_task`, executing them in the order the model emitted them.

To stop a specific tool from overlapping with others, mark it `sequential=True` — it then acts as a barrier: tools the model emitted before it finish first, it runs alone, and tools emitted after it start only once it finishes.

```python {title="sequential_tool.py"}
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

agent = Agent()

calls: list[str] = []


@agent.tool_plain
def fetch_record(record_id: int) -> str:
    calls.append(f'fetch_record({record_id})')
    return f'record-{record_id}'


@agent.tool_plain(sequential=True)
def write_to_database(record: str) -> str:
    calls.append(f'write_to_database({record!r})')
    return 'written'


def call_tools(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:  # first request: ask for both tools at once
        return ModelResponse(
            parts=[
                ToolCallPart('fetch_record', {'record_id': 1}),
                ToolCallPart('write_to_database', {'record': 'data'}),
            ]
        )
    return ModelResponse(parts=[TextPart('done')])


result = agent.run_sync('store the record', model=FunctionModel(call_tools))
print(result.output)
#> done
# `write_to_database` waited for `fetch_record` to finish before running.
print(calls)
#> ['fetch_record(1)', "write_to_database('data')"]
```

You can pass the [`sequential`][pydantic_ai.tools.ToolDefinition.sequential] flag when registering any function tool, and the same barrier is available for [output tools](output.md#tool-output) via [`ToolOutput(sequential=True)`][pydantic_ai.output.ToolOutput] (see [Controlling output tool parallelism](output.md#controlling-output-tool-parallelism)). To run an entire run's tools serially regardless of which tools were called, wrap the run in the [`with agent.parallel_tool_call_execution_mode('sequential')`][pydantic_ai.agent.AbstractAgent.parallel_tool_call_execution_mode] context manager, or set `parallel_tool_calls=False` on the [model settings][pydantic_ai.settings.ModelSettings].

Async functions are run on the event loop, while sync functions are offloaded to threads. To get the best performance, _always_ use an async function _unless_ you're doing blocking I/O (and there's no way to use a non-blocking library instead) or CPU-bound work (like `numpy` or `scikit-learn` operations), so that simple functions are not offloaded to threads unnecessarily.

!!! note "Sync tools run on a separate thread"
    Because a sync function runs on a worker thread, a value it sets on a [`contextvars.ContextVar`][contextvars.ContextVar] is not visible outside the function, and asyncio APIs like `asyncio.get_running_loop()` raise an error, since the worker thread has no event loop. Reading context variables still works, but the write limitation also applies to libraries that use them internally, such as tracing and logging integrations. If your tool needs any of these, make it async. The same applies to any sync function the agent runs for you, including [hooks](hooks.md), system prompt functions, output functions, and history processors.

#### Thread executor for long-running servers

By default, sync functions are offloaded to threads using [`anyio.to_thread.run_sync`][anyio.to_thread.run_sync], which creates ephemeral threads on demand. In long-running servers (e.g. FastAPI), these threads can accumulate under sustained traffic, leading to memory growth.

To control thread lifecycle, provide a bounded [`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor] using the [`UseThreadExecutor`][pydantic_ai.capabilities.UseThreadExecutor] capability (per-agent) or the [`Agent.using_thread_executor()`][pydantic_ai.agent.AbstractAgent.using_thread_executor] context manager (global):

```python {test="skip"}
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from pydantic_ai import Agent
from pydantic_ai.capabilities import UseThreadExecutor

# Per-agent: pass as a capability
executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix='agent-worker')
agent = Agent('openai:gpt-5.2', capabilities=[UseThreadExecutor(executor)])

# Global: wrap your server lifespan
@asynccontextmanager
async def lifespan(app):
    executor = ThreadPoolExecutor(max_workers=16)
    with Agent.using_thread_executor(executor):
        yield
    executor.shutdown(wait=True)
```

!!! note "Limiting tool executions"
    You can cap tool executions within a run using [`UsageLimits(tool_calls_limit=...)`](agent.md#usage-limits). The counter increments only after a successful tool invocation. Output tools (used for [structured output](output.md)) are not counted in the `tool_calls` metric.

#### Output Tool Calls

When a model produces a final result — an [output tool](output.md#tool-output) call, or structured [native](output.md#native-output)/[prompted](output.md#prompted-output) or [image](output.md#image-output) output — in parallel with other tools, the agent's [`end_strategy`][pydantic_ai.agent.Agent.end_strategy] parameter controls how these tool calls are executed.
The default `'graceful'` strategy ensures all function tools are executed even after a final result is found, while skipping remaining output tools. The `'exhaustive'` strategy goes further and also executes all output tools. Both are useful when tools have side effects (like logging, sending notifications, or updating metrics) that should always execute.

For more information on how `end_strategy` works with function tools, output tools, and non-tool output, see [Tool calls alongside a final result](output.md#parallel-output-tool-calls).

## Tool Search

Agents with many tools (e.g. [MCP servers](mcp/client.md) exposing dozens of endpoints) can spend a lot of input tokens on tool definitions before any work happens, and tool selection accuracy noticeably degrades past ~30–50 available tools. Marking tools for deferred loading hides them from the model's initial context; the model discovers hidden tools by keyword when it needs them.

For workflow *bundles* — instructions, tools, model settings, and hooks that travel together — see [on-demand capabilities](capabilities/on-demand.md), which build on the same machinery but disclose at the bundle level rather than the individual-tool level.

Reach for it when:

* the agent exposes ~10+ tools or more than ~10k tokens of tool definitions
* tools cover distinct domains (e.g. multiple MCP servers) and only a subset is relevant per request
* the toolset is growing and you want headroom

Skip it when you have a small, hot toolset where every tool is used most turns — deferring everything would just add a discovery round-trip for no benefit. As a rule of thumb, keep your handful of most-used tools eagerly loaded; defer the long tail.

To opt in, set `defer_loading=True` on individual [`Tool`][pydantic_ai.tools.Tool] / [`@agent.tool`][pydantic_ai.agent.Agent.tool] / [`@agent.tool_plain`][pydantic_ai.agent.Agent.tool_plain] registrations, or use [`.defer_loading()`][pydantic_ai.toolsets.AbstractToolset.defer_loading] on a whole toolset (including [`MCPToolset`][pydantic_ai.mcp.MCPToolset]) — pass a list of tool names to hide specific ones, or `None` to hide all.

Once deferred tools exist, search is handled by the auto-injected [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] capability:

* **Native provider search** on supporting models (Anthropic Sonnet 4.5+, Opus 4.5+, Haiku 4.5+ via [BM25/regex](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool); OpenAI Responses on GPT-5.4+). Deferred tools are sent to the provider with `defer_loading` on the wire and the provider manages their visibility.
* **Custom callable** via [`ToolSearch(strategy=...)`][pydantic_ai.capabilities.ToolSearch] — a user-supplied search function. Executed on our side, but routed through the provider's client-executed native surface (Anthropic `tool_reference` blocks, OpenAI `execution='client'`) where supported so the model sees a tool-search call rather than a regular function tool.
* **Local fallback** on every other model: a `search_tools` function tool matches keywords against tool names and descriptions.

Pydantic AI prefers native search whenever available because the discovery exchange happens append-only (a `tool_search_call` + `tool_search_output` pair) while each tool's authored `defer_loading` value remains stable, so prompt caching is preserved across rounds. On the local fallback, revealed tools are tracked separately from their stable definitions and sent only once discovered.

Every searchable deferred tool remains in the search corpus for the whole run, including tools the model has already discovered. Repeating a search is safe and lets the model recover a tool after its earlier discovery is no longer visible, such as after [compaction](capabilities/compaction.md). With the default keyword strategy, undiscovered matches always rank ahead of already-available ones, so when `max_results` trims the list, an already-available tool never displaces an undiscovered match — it only fills leftover slots.

Toolsets that aggregate or wrap deferred definitions can check visibility with [`ctx.is_tool_available(tool_def)`][pydantic_ai.tools.RunContext.is_tool_available] inside `get_tools`. A definition is available when it is not deferred or its name has been revealed in history. Pass the definition the toolset is holding; the name form applies the same test to the current resolved [`ctx.tools`][pydantic_ai.tools.RunContext.tools] snapshot and is intended for model-request hooks and tool execution.

Tools gated as a bundle are covered by [on-demand capabilities](capabilities/on-demand.md); they are not part of the searchable corpus.

Each recorded reveal also surfaces as a [`ToolAvailabilityDeltaEvent`][pydantic_ai.messages.ToolAvailabilityDeltaEvent] in the agent event stream and as the corresponding persistent Vercel AI data chunk or AG-UI activity snapshot.

Provider adapters project each availability delta onto their supported history-based reveal mechanism. Anthropic declares each revealed definition in `tools` with `defer_loading=True` — up front in capability-only runs (no search surface can expose them), appended at reveal in mixed runs, which withhold capability-owned definitions until then — and references it from a native `tool_addition` block in the same request. OpenAI Responses carries a revealed definition in an appended `additional_tools` input item: a tool that was never declared travels in the item alone, while a tool already declared as a deferred `tools` entry keeps that entry and the item reveals it. Other models announce the newly available tools when their schemas are already visible, or receive a synthesized tool-search exchange when its result must reveal a withheld schema.

For the model to find tools well, give them descriptive names with consistent prefixes (`github_*`, `slack_*`, `mortgage_*`) and put the keywords a user might search for in the tool's description. A search returns a handful of matches at a time, so the model may iterate (search → discover → call → search again) — instructions can nudge it: "Search by topic when you don't see a tool you need."

```python {title="tool_search.py"}
from pydantic_ai import Agent

agent = Agent('anthropic:claude-sonnet-4-6')


@agent.tool_plain(defer_loading=True)
def mortgage_calculator(principal: float, rate: float, years: int) -> str:
    """Calculate monthly mortgage payment for a home loan."""
    monthly_rate = rate / 100 / 12
    n_payments = years * 12
    payment = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)
    return f'${payment:.2f}/month'
```

For MCP servers, use [`.defer_loading()`][pydantic_ai.toolsets.AbstractToolset.defer_loading] to hide all tools behind search:

```python {title="tool_search_mcp.py" lint="skip" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

mcp = MCPToolset('http://localhost:8000/mcp')
agent = Agent('anthropic:claude-sonnet-4-6', toolsets=[mcp.defer_loading()])
```

### Configuring `ToolSearch`

Pass an explicit [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] capability to control the strategy or provide a custom search function:

```python {title="tool_search_custom.py"}
from collections.abc import Sequence

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.tools import ToolDefinition


def fuzzy_search(
    ctx: RunContext, queries: Sequence[str], tools: Sequence[ToolDefinition]
) -> list[str]:
    """Match tools whose name or description contains any query word."""
    needles = [n for q in queries for n in q.lower().split()]
    return [
        t.name
        for t in tools
        if any(n in t.name.lower() or n in (t.description or '').lower() for n in needles)
    ]


agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ToolSearch(strategy=fuzzy_search)])


@agent.tool_plain(defer_loading=True)
def mortgage_calculator(principal: float, rate: float, years: int) -> str:
    """Calculate monthly mortgage payment for a home loan."""
    monthly_rate = rate / 100 / 12
    n_payments = years * 12
    payment = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)
    return f'${payment:.2f}/month'
```

Available strategy values:

| `strategy` | Algorithm | Behavior |
|---|---|---|
| `None` (default) | Provider's native algorithm where available, else local keyword matching | Anthropic native BM25 on Sonnet 4.5+/Opus 4.5+/Haiku 4.5+, OpenAI server-executed `tool_search` on GPT-5.4+, local keyword matching elsewhere. |
| `'keywords'` | Local keyword-overlap | The keyword algorithm runs on our side, but the wire shape adapts: client-executed native (Anthropic, OpenAI) where supported so the prompt cache stays warm, regular `search_tools` function tool elsewhere. |
| `'bm25'` / `'regex'` | Anthropic native | Server-executed by Anthropic. The request fails on other providers (OpenAI, Google, etc.) rather than silently substituting a different algorithm. |
| Callable `(ctx, queries, tools) -> names` | User-defined | Same execution-mode handling as `'keywords'`: client-executed native on supporting providers, local `search_tools` function tool elsewhere. |

The execution mode (server-executed, client-executed-native, or local fallback) is auto-derived from the chosen algorithm and the current provider — users don't pick it directly. Native execution is preferred whenever available because it keeps the model-facing tool list stable across discovery rounds, which preserves Anthropic and OpenAI prompt caching.

To force the local `keywords` algorithm on a provider that natively supports tool search, override [`ModelProfile.supported_native_tools`][pydantic_ai.profiles.ModelProfile.supported_native_tools] to exclude `ToolSearchTool` — the capability then falls through to the local `search_tools` function tool.

!!! note "Cross-provider history replay"
    A turn can run on one provider and the next on another (e.g. via [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] or by switching `model=` between runs). Discovered-tool state is preserved across the switch:

    * Local-shape `search_tools` history rendered onto a native-supporting provider (Anthropic, OpenAI) is promoted to the provider's native tool-search wire so the discovered tools' schemas get unlocked from `defer_loading=True` without forcing the model to re-search.
    * Native-shape `tool_search` history rendered onto another provider — whether or not the target has its own native tool search — is translated to the provider-agnostic `search_tools` exchange first, then rendered in the target's supported shape. A provider's own native history retains its exact replay shape.

#### Tool-availability history portability

Stored history can describe a tool becoming callable as either a model-driven search or an
application-driven availability change. Pydantic AI preserves that distinction when the history is
replayed against a different model:

| Stored representation | Anthropic with `tool_addition_mode='by_reference'` | Anthropic with `tool_addition_mode=None` | OpenAI Responses with native search and `tool_addition_mode='with_definitions'` | First-party OpenAI Responses without native search, `tool_addition_mode='with_definitions'` | OpenAI-compatible Responses with `tool_addition_mode=None` | Gemini (`tool_addition_mode=None`) | OpenAI Chat Completions (`tool_addition_mode=None`) |
|---|---|---|---|---|---|---|---|
| Local `search_tools` call and result | Native search | Native search | Native search | Local search + `additional_tools` | Local search | Local search | Local search |
| Anthropic native search | Native search | Native search | Native search | Local search + `additional_tools` | Local search | Local search | Local search |
| OpenAI native search | Native search | Native search | Native search | Local search + `additional_tools` | Local search | Local search | Local search |
| [`ToolAvailabilityDeltaPart`][pydantic_ai.messages.ToolAvailabilityDeltaPart] | `tool_addition` | Native search | `additional_tools` | `additional_tools` | Announcement or local search | Announcement | Announcement |
| `search_tools` result with `metadata['discovered_tools']` | Native search | Native search | Native search | Local search | Local search | Local search | Local search |

Here, **native search** means a paired provider-native search call and result plus the native search
tool. Searchable deferred tools remain in the deferred corpus. **Local
search** means a paired `search_tools` function call and result plus the local search tool; the
revealed tool is present as an eager function tool — except on a `with_definitions` target, where a
structured search result additionally rides an `additional_tools` item and the revealed definition
travels there instead of occupying a `tools` entry (a plain-text legacy result, the last row, has no
structured discovery to carry, so it stays plain local search). For a capability-only corpus, a provider-native
availability change includes neither a search exchange nor a search tool. In a mixed corpus, the
search tool stays on the wire for the tools that remain searchable; an Anthropic capability tool is
appended as a deferred definition when its `tool_addition` is emitted.

A genuine search is evidence of what the model did: it chose a query and received matches. Rewriting
that exchange as `tool_addition` or `additional_tools` would incorrectly turn model-driven discovery
into application-driven control. Provider-native availability changes are therefore used only for
[`ToolAvailabilityDeltaPart`][pydantic_ai.messages.ToolAvailabilityDeltaPart]. On targets without
that control primitive, a mid-conversation system instruction announces
`The following tool(s) are now available: {names}` when the schemas are already visible. A complete
local search exchange is synthesized only when its result must reveal a schema that is actually
withheld, so the tool does not remain locked behind `defer_loading`.

!!! note "Tool discovery and message history"
    Discovered tools are tracked in typed message parts — search call/result pairs and [`ToolAvailabilityDeltaPart`][pydantic_ai.messages.ToolAvailabilityDeltaPart]s. If a [history processor](message-history.md#processing-message-history) strips that evidence, the tools revert to hidden on the next request; see [Processing Message History](message-history.md#processing-message-history) for the preservation contract.

See [`ToolDefinition.defer_loading`][pydantic_ai.tools.ToolDefinition.defer_loading] and [Deferred Loading](toolsets.md#deferred-loading) for more details.

### Tool search and prompt caching {#tool-search-caching}

Prompt caching keys on a **stable prefix**: providers cache the longest unchanged run of tokens from the start of the request, in roughly the order tool definitions → system/instructions → message history. A change at any layer invalidates the cache for that layer and everything after it — so on most providers **changing, adding, removing, or reordering a tool definition invalidates the cache**, because tool definitions sit at the very front.

Tool search works on **every model**, but it only *preserves* the cache where the model supports native tool search — Anthropic Sonnet 4.5+, Opus 4.5+, and Haiku 4.5+, and OpenAI Responses on GPT-5.4+. There, discovery is [append-only](#tool-search) and the deferred tools never enter the prompt prefix, so an identical prefix is re-read from cache across discovery rounds. On every other model — including Google, and older Anthropic and OpenAI models — the local `search_tools` fallback reveals a discovered tool by adding it to the tools array, which **invalidates the cached prefix from the tool definitions onward on each discovery turn** (on Google, a stable `system_instruction` sits ahead of the tool block and can still be reused — see [Related caching controls](#related-caching-controls) below).

!!! note "Why Gemini tool search never keeps the prefix warm"
    Native tool search preserves the cache by handing discovery off to a provider-side primitive that keeps the discovered tools out of the request prefix. Gemini's API exposes no such primitive — unlike Anthropic's `bm25`/`regex` tool search or OpenAI Responses' `tool_search` — so tool search on Gemini always falls back to the local `search_tools` function tool, which reveals each match by adding it to the tools array. Because Gemini caches on the request prefix and tool definitions sit at its front, every discovery turn that reveals a new tool invalidates the cache from the tool block onward. This is a missing-primitive limitation, not a version gate: with Anthropic and OpenAI a newer model *does* support native tool search, but no Gemini model does.

!!! note "Deferring saves context — it is not dynamic registration"
    With either strategy, every tool the model can ever reach must be **declared on the agent or toolset up front**. Deferring keeps unused definitions out of the model's context (and, natively, out of the cached prefix); it does not let you register a brand-new tool the agent was never configured with. Where a provider can append a tool declaration mid-conversation — OpenAI Responses' `additional_tools` item — a tool that was never in the request's `tools` block can still become callable without touching the prefix, because the declaration travels as an appended input item.

For [on-demand capabilities](capabilities/on-demand.md#on-demand-capabilities), loading a capability that reveals no new tool definitions — instructions or model settings only — preserves the cache on every provider, even without native tool search. Anthropic excludes deferred entries from its cache key: capability-only runs pre-advertise them from turn one, making the one-time deferred preamble part of the initial prefix, while mixed runs pay that preamble through the searchable corpus and append revealed deferred entries outside the cached prefix. No Anthropic capability reveal introduces the deferred preamble midway through a run. First-party OpenAI Responses appends the reveal as an `additional_tools` input item without changing `tools[]`. Elsewhere the revealed tool enters the tool-definitions prefix, as does a native tool, and as does a deferred `prepare_tools`/`prepare_output_tools` hook that rewrites tool definitions on load. See [Cache implications](capabilities/on-demand.md#cache-implications) for the full breakdown.

For a genuinely open-ended tool universe, route everything through a single, stable tool. The harness [`CodeMode`](https://pydantic.dev/docs/ai/harness/code-mode/) capability collapses many tools into one `run_code` tool whose definition stays byte-stable; newly discovered tools are surfaced as callables inside the sandbox rather than as new tool schemas, keeping the tool-definitions prefix — and its cache — intact across discoveries.

#### Seeing it in a trace

With native tool search the deferred catalog never enters the cached prefix, so an identical request prefix is re-read from cache on the next turn — `cache_read_tokens` stays warm even as the model discovers new tools:

/// public-trace | https://logfire-us.pydantic.dev/public-trace/243f9aa5-dd9d-4de4-a36a-5818611892f1?spanId=28a1621e429ce027
    title: 'Anthropic: cached tool + system prefix re-read from cache, with deferred tools declared'
///

Change a single tool definition, and the whole prefix is re-created instead — the [same request with one tool's description edited](https://logfire-us.pydantic.dev/public-trace/c3205dc9-6251-40fa-9d0a-ed647be9ba30?spanId=38ef4635ecaf3e0b) records no cache read.

#### Related caching controls

- Restricting the *active* tools with [`tool_choice`](#tool-choice) can also invalidate the cache when Pydantic AI has to trim the array client-side — see [Prompt caching implications](#tool-choice-caching) for the per-provider breakdown and the cache-preserving alternatives (`allowed_tools`, `allowed_function_names`, `ToolOrOutput`).
- To place explicit cache breakpoints on messages, use [`CachePoint`][pydantic_ai.messages.CachePoint] (honored by Anthropic, Bedrock, and OpenRouter). Anthropic's tool, system, and instruction caching settings are documented under [Anthropic prompt caching](models/anthropic.md#prompt-caching).
- On providers that cache tool definitions at the front of the prefix — Anthropic, OpenAI, and xAI — editing a single tool's description invalidates the cached prefix. Google's *implicit* cache is prefix-based on a different layout (its `system_instruction` is a separate field ahead of the tool block), so a large stable system instruction can keep cache hits even when the tool list changes; an explicit [`CachedContent`](models/google.md) instead fixes the tools as an immutable part of the cache by construction.

## See Also

- [Function Tools](tools.md) - Basic tool concepts and registration
- [Toolsets](toolsets.md) - Managing collections of tools
- [Deferred Tools](deferred-tools.md) - Tools requiring approval or external execution
- [Third-Party Tools](third-party-tools.md) - Integrations with external tool libraries
