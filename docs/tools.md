# Function Tools

Function tools provide a mechanism for models to perform actions and retrieve extra information to help them generate a response.

They're useful when you want to enable the model to take some action and use the result, when it is impractical or impossible to put all the context an agent might need into the instructions, or when you want to make agents' behavior more deterministic or reliable by deferring some of the logic required to generate a response to another (not necessarily AI-powered) tool.

If you want a model to be able to call a function as its final action, without the result being sent back to the model, you can use an [output function](output.md#output-functions) instead.

There are a number of ways to register tools with an agent:

* via the [`@agent.tool`][pydantic_ai.Agent.tool] decorator — for tools that need access to the agent [context][pydantic_ai.tools.RunContext]
* via the [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain] decorator — for tools that do not need access to the agent [context][pydantic_ai.tools.RunContext]
* via the [`tools`][pydantic_ai.Agent.__init__] keyword argument to `Agent` which can take either plain functions, or instances of [`Tool`][pydantic_ai.tools.Tool]
* via the [`toolsets`][pydantic_ai.Agent.__init__] keyword argument to `Agent` which takes a list of [toolsets](#toolsets) that you can build yourself or get from a [third party](#third-party-tools), e.g. an [MCP server](mcp/client.md)

!!! info "Function tools vs. RAG"
    Function tools are basically the "R" of RAG (Retrieval-Augmented Generation) — they augment what the model can do by letting it request extra information.

    The main semantic difference between PydanticAI Tools and RAG is RAG is synonymous with vector search, while PydanticAI tools are more general-purpose. (Note: we may add support for vector search functionality in the future, particularly an API for generating embeddings. See [#58](https://github.com/pydantic/pydantic-ai/issues/58))

!!! info "Function Tools vs. Structured Outputs"
    As the name suggests, function tools use the model's "tools" or "functions" API to let the model know what is available to call. Tools or functions are also used to define the schema(s) for [structured output](output.md) when using the default [tool output mode](output.md#tool-output), thus a model might have access to many tools, some of which call function tools while others end the run and produce a final output.

## Registering via Decorator {#registering-function-tools-via-decorator}

`@agent.tool` is considered the default decorator since in the majority of cases tools will need access to the agent [context][pydantic_ai.tools.RunContext].

Here's an example using both:

```python {title="dice_game.py"}
import random

from pydantic_ai import Agent, RunContext

agent = Agent(
    'google-gla:gemini-1.5-flash',  # (1)!
    deps_type=str,  # (2)!
    system_prompt=(
        "You're a dice game, you should roll the die and see if the number "
        "you get back matches the user's guess. If so, tell them they're a winner. "
        "Use the player's name in the response."
    ),
)


@agent.tool_plain  # (3)!
def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))


@agent.tool  # (4)!
def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps


dice_result = agent.run_sync('My guess is 4', deps='Anne')  # (5)!
print(dice_result.output)
#> Congratulations Anne, you guessed correctly! You're a winner!
```

1. This is a pretty simple task, so we can use the fast and cheap Gemini flash model.
2. We pass the user's name as the dependency, to keep things simple we use just the name as a string as the dependency.
3. This tool doesn't need any context, it just returns a random number. You could probably use dynamic instructions in this case.
4. This tool needs the player's name, so it uses `RunContext` to access dependencies which are just the player's name in this case.
5. Run the agent, passing the player's name as the dependency.

_(This example is complete, it can be run "as is")_

Let's print the messages from that game to see what happened:

```python {title="dice_game_messages.py" requires="dice_game.py"}
from dice_game import dice_result

print(dice_result.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content="You're a dice game, you should roll the die and see if the number you get back matches the user's guess. If so, tell them they're a winner. Use the player's name in the response.",
                timestamp=datetime.datetime(...),
            ),
            UserPromptPart(
                content='My guess is 4',
                timestamp=datetime.datetime(...),
            ),
        ]
    ),
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='roll_dice', args={}, tool_call_id='pyd_ai_tool_call_id'
            )
        ],
        usage=Usage(requests=1, request_tokens=90, response_tokens=2, total_tokens=92),
        model_name='gemini-1.5-flash',
        timestamp=datetime.datetime(...),
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='roll_dice',
                content='4',
                tool_call_id='pyd_ai_tool_call_id',
                timestamp=datetime.datetime(...),
            )
        ]
    ),
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='get_player_name', args={}, tool_call_id='pyd_ai_tool_call_id'
            )
        ],
        usage=Usage(requests=1, request_tokens=91, response_tokens=4, total_tokens=95),
        model_name='gemini-1.5-flash',
        timestamp=datetime.datetime(...),
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='get_player_name',
                content='Anne',
                tool_call_id='pyd_ai_tool_call_id',
                timestamp=datetime.datetime(...),
            )
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content="Congratulations Anne, you guessed correctly! You're a winner!"
            )
        ],
        usage=Usage(
            requests=1, request_tokens=92, response_tokens=12, total_tokens=104
        ),
        model_name='gemini-1.5-flash',
        timestamp=datetime.datetime(...),
    ),
]
"""
```

We can represent this with a diagram:

```mermaid
sequenceDiagram
    participant Agent
    participant LLM

    Note over Agent: Send prompts
    Agent ->> LLM: System: "You're a dice game..."<br>User: "My guess is 4"
    activate LLM
    Note over LLM: LLM decides to use<br>a tool

    LLM ->> Agent: Call tool<br>roll_dice()
    deactivate LLM
    activate Agent
    Note over Agent: Rolls a six-sided die

    Agent -->> LLM: ToolReturn<br>"4"
    deactivate Agent
    activate LLM
    Note over LLM: LLM decides to use<br>another tool

    LLM ->> Agent: Call tool<br>get_player_name()
    deactivate LLM
    activate Agent
    Note over Agent: Retrieves player name
    Agent -->> LLM: ToolReturn<br>"Anne"
    deactivate Agent
    activate LLM
    Note over LLM: LLM constructs final response

    LLM ->> Agent: ModelResponse<br>"Congratulations Anne, ..."
    deactivate LLM
    Note over Agent: Game session complete
```

## Registering via Agent Argument {#registering-function-tools-via-agent-argument}

As well as using the decorators, we can register tools via the `tools` argument to the [`Agent` constructor][pydantic_ai.Agent.__init__]. This is useful when you want to reuse tools, and can also give more fine-grained control over the tools.

```python {title="dice_game_tool_kwarg.py"}
import random

from pydantic_ai import Agent, RunContext, Tool

system_prompt = """\
You're a dice game, you should roll the die and see if the number
you get back matches the user's guess. If so, tell them they're a winner.
Use the player's name in the response.
"""


def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))


def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps


agent_a = Agent(
    'google-gla:gemini-1.5-flash',
    deps_type=str,
    tools=[roll_dice, get_player_name],  # (1)!
    system_prompt=system_prompt,
)
agent_b = Agent(
    'google-gla:gemini-1.5-flash',
    deps_type=str,
    tools=[  # (2)!
        Tool(roll_dice, takes_ctx=False),
        Tool(get_player_name, takes_ctx=True),
    ],
    system_prompt=system_prompt,
)

dice_result = {}
dice_result['a'] = agent_a.run_sync('My guess is 6', deps='Yashar')
dice_result['b'] = agent_b.run_sync('My guess is 4', deps='Anne')
print(dice_result['a'].output)
#> Tough luck, Yashar, you rolled a 4. Better luck next time.
print(dice_result['b'].output)
#> Congratulations Anne, you guessed correctly! You're a winner!
```

1. The simplest way to register tools via the `Agent` constructor is to pass a list of functions, the function signature is inspected to determine if the tool takes [`RunContext`][pydantic_ai.tools.RunContext].
2. `agent_a` and `agent_b` are identical — but we can use [`Tool`][pydantic_ai.tools.Tool] to reuse tool definitions and give more fine-grained control over how tools are defined, e.g. setting their name or description, or using a custom [`prepare`](#tool-prepare) method.

_(This example is complete, it can be run "as is")_

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


agent = Agent(model=OpenAIResponsesModel('gpt-4o'))


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

- Provide rich multi-modal content (images, documents, etc.) to the model as context
- Separate the programmatic return value from the model's context
- Include additional metadata that shouldn't be sent to the LLM

Here's an example of a computer automation tool that captures screenshots and provides visual feedback:

```python {title="advanced_tool_return.py" test="skip" lint="skip"}
import time
from pydantic_ai import Agent
from pydantic_ai.messages import ToolReturn, BinaryContent

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def click_and_capture(x: int, y: int) -> ToolReturn:
    """Click at coordinates and show before/after screenshots."""
    # Take screenshot before action
    before_screenshot = capture_screen()

    # Perform click operation
    perform_click(x, y)
    time.sleep(0.5)  # Wait for UI to update

    # Take screenshot after action
    after_screenshot = capture_screen()

    return ToolReturn(
        return_value=f"Successfully clicked at ({x}, {y})",
        content=[
            f"Clicked at coordinates ({x}, {y}). Here's the comparison:",
            "Before:",
            BinaryContent(data=before_screenshot, media_type="image/png"),
            "After:",
            BinaryContent(data=after_screenshot, media_type="image/png"),
            "Please analyze the changes and suggest next steps."
        ],
        metadata={
            "coordinates": {"x": x, "y": y},
            "action_type": "click_and_capture",
            "timestamp": time.time()
        }
    )

# The model receives the rich visual content for analysis
# while your application can access the structured return_value and metadata
result = agent.run_sync("Click on the submit button and tell me what happened")
print(result.output)
# The model can analyze the screenshots and provide detailed feedback
```

- **`return_value`**: The actual return value used in the tool response. This is what gets serialized and sent back to the model as the tool's result.
- **`content`**: A sequence of content (text, images, documents, etc.) that provides additional context to the model. This appears as a separate user message.
- **`metadata`**: Optional metadata that your application can access but is not sent to the LLM. Useful for logging, debugging, or additional processing. Some other AI frameworks call this feature "artifacts".

This separation allows you to provide rich context to the model while maintaining clean, structured return values for your application logic.

## Tool Schema {#function-tools-and-schema}

Function parameters are extracted from the function signature, and all parameters except `RunContext` are used to build the schema for that tool call.

Even better, PydanticAI extracts the docstring from functions and (thanks to [griffe](https://mkdocstrings.github.io/griffe/)) extracts parameter descriptions from the docstring and adds them to the schema.

[Griffe supports](https://mkdocstrings.github.io/griffe/reference/docstrings/#docstrings) extracting parameter descriptions from `google`, `numpy`, and `sphinx` style docstrings. PydanticAI will infer the format to use based on the docstring, but you can explicitly set it using [`docstring_format`][pydantic_ai.tools.DocstringFormat]. You can also enforce parameter requirements by setting `require_parameter_descriptions=True`. This will raise a [`UserError`][pydantic_ai.exceptions.UserError] if a parameter description is missing.

To demonstrate a tool's schema, here we use [`FunctionModel`][pydantic_ai.models.function.FunctionModel] to print the schema a model would receive:

```python {title="tool_schema.py"}
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

agent = Agent()


@agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
def foobar(a: int, b: str, c: dict[str, list[float]]) -> str:
    """Get me foobar.

    Args:
        a: apple pie
        b: banana cake
        c: carrot smoothie
    """
    return f'{a} {b} {c}'


def print_schema(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    tool = info.function_tools[0]
    print(tool.description)
    #> Get me foobar.
    print(tool.parameters_json_schema)
    """
    {
        'additionalProperties': False,
        'properties': {
            'a': {'description': 'apple pie', 'type': 'integer'},
            'b': {'description': 'banana cake', 'type': 'string'},
            'c': {
                'additionalProperties': {'items': {'type': 'number'}, 'type': 'array'},
                'description': 'carrot smoothie',
                'type': 'object',
            },
        },
        'required': ['a', 'b', 'c'],
        'type': 'object',
    }
    """
    return ModelResponse(parts=[TextPart('foobar')])


agent.run_sync('hello', model=FunctionModel(print_schema))
```

_(This example is complete, it can be run "as is")_

If a tool has a single parameter that can be represented as an object in JSON schema (e.g. dataclass, TypedDict, pydantic model), the schema for the tool is simplified to be just that object.

Here's an example where we use [`TestModel.last_model_request_parameters`][pydantic_ai.models.test.TestModel.last_model_request_parameters] to inspect the tool schema that would be passed to the model.

```python {title="single_parameter_tool.py"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent()


class Foobar(BaseModel):
    """This is a Foobar"""

    x: int
    y: str
    z: float = 3.14


@agent.tool_plain
def foobar(f: Foobar) -> str:
    return str(f)


test_model = TestModel()
result = agent.run_sync('hello', model=test_model)
print(result.output)
#> {"foobar":"x=0 y='a' z=3.14"}
print(test_model.last_model_request_parameters.function_tools)
"""
[
    ToolDefinition(
        name='foobar',
        parameters_json_schema={
            'properties': {
                'x': {'type': 'integer'},
                'y': {'type': 'string'},
                'z': {'default': 3.14, 'type': 'number'},
            },
            'required': ['x', 'y'],
            'title': 'Foobar',
            'type': 'object',
        },
        description='This is a Foobar',
    )
]
"""
```

_(This example is complete, it can be run "as is")_

### Custom Tool Schema

If you have a function that lacks appropriate documentation (i.e. poorly named, no type information, poor docstring, use of *args or **kwargs and suchlike) then you can still turn it into a tool that can be effectively used by the agent with the [`Tool.from_schema`][pydantic_ai.Tool.from_schema] function. With this you provide the name, description and JSON schema for the function directly:

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
    }
)

test_model = TestModel()
agent = Agent(test_model, tools=[tool])

result = agent.run_sync('testing...')
print(result.output)
#> {"sum":0}
```


Please note that validation of the tool arguments will not be performed, and this will pass all arguments as keyword arguments.

## Dynamic Tools {#tool-prepare}

Tools can optionally be defined with another function: `prepare`, which is called at each step of a run to
customize the definition of the tool passed to the model, or omit the tool completely from that step.

A `prepare` method can be registered via the `prepare` kwarg to any of the tool registration mechanisms:

* [`@agent.tool`][pydantic_ai.Agent.tool] decorator
* [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain] decorator
* [`Tool`][pydantic_ai.tools.Tool] dataclass

The `prepare` method, should be of type [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc], a function which takes [`RunContext`][pydantic_ai.tools.RunContext] and a pre-built [`ToolDefinition`][pydantic_ai.tools.ToolDefinition], and should either return that `ToolDefinition` with or without modifying it, return a new `ToolDefinition`, or return `None` to indicate this tools should not be registered for that step.

Here's a simple `prepare` method that only includes the tool if the value of the dependency is `42`.

As with the previous example, we use [`TestModel`][pydantic_ai.models.test.TestModel] to demonstrate the behavior without calling a real model.

```python {title="tool_only_if_42.py"}
from typing import Union

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition

agent = Agent('test')


async def only_if_42(
    ctx: RunContext[int], tool_def: ToolDefinition
) -> Union[ToolDefinition, None]:
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

Here's a more complex example where we change the description of the `name` parameter to based on the value of `deps`

For the sake of variation, we create this tool using the [`Tool`][pydantic_ai.tools.Tool] dataclass.

```python {title="customize_name.py"}
from __future__ import annotations

from typing import Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import Tool, ToolDefinition


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
    )
]
"""
```

_(This example is complete, it can be run "as is")_

### Agent-wide Dynamic Tools {#prepare-tools}

In addition to per-tool `prepare` methods, you can also define an agent-wide `prepare_tools` function. This function is called at each step of a run and allows you to filter or modify the list of all tool definitions available to the agent for that step. This is especially useful if you want to enable or disable multiple tools at once, or apply global logic based on the current context.

The `prepare_tools` function should be of type [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc], which takes the [`RunContext`][pydantic_ai.tools.RunContext] and a list of [`ToolDefinition`][pydantic_ai.tools.ToolDefinition], and returns a new list of tool definitions (or `None` to disable all tools for that step).

!!! note
    The list of tool definitions passed to `prepare_tools` includes both regular function tools and tools from any [toolsets](#toolsets) registered to the agent, but not [output tools](output.md#tool-output).
    To modify output tools, you can set a `prepare_output_tools` function instead.

Here's an example that makes all tools strict if the model is an OpenAI model:

```python {title="agent_prepare_tools_customize.py" noqa="I001"}
from dataclasses import replace
from typing import Union

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.models.test import TestModel


async def turn_on_strict_if_openai(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> Union[list[ToolDefinition], None]:
    if ctx.model.system == 'openai':
        return [replace(tool_def, strict=True) for tool_def in tool_defs]
    return tool_defs


test_model = TestModel()
agent = Agent(test_model, prepare_tools=turn_on_strict_if_openai)


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
from typing import Union

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool, ToolDefinition


def launch_potato(target: str) -> str:
    return f'Potato launched at {target}!'


async def filter_out_tools_by_name(
    ctx: RunContext[bool], tool_defs: list[ToolDefinition]
) -> Union[list[ToolDefinition], None]:
    if ctx.deps:
        return [tool_def for tool_def in tool_defs if tool_def.name != 'launch_potato']
    return tool_defs


agent = Agent(
    'test',
    tools=[Tool(launch_potato)],
    prepare_tools=filter_out_tools_by_name,
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


## Tool Execution and Retries {#tool-retries}

When a tool is executed, its arguments (provided by the LLM) are first validated against the function's signature using Pydantic. If validation fails (e.g., due to incorrect types or missing required arguments), a `ValidationError` is raised, and the framework automatically generates a [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart] containing the validation details. This prompt is sent back to the LLM, informing it of the error and allowing it to correct the parameters and retry the tool call.

Beyond automatic validation errors, the tool's own internal logic can also explicitly request a retry by raising the [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] exception. This is useful for situations where the parameters were technically valid, but an issue occurred during execution (like a transient network error, or the tool determining the initial attempt needs modification).

```python
from pydantic_ai import ModelRetry


def my_flaky_tool(query: str) -> str:
    if query == 'bad':
        # Tell the LLM the query was bad and it should try again
        raise ModelRetry("The query 'bad' is not allowed. Please provide a different query.")
    # ... process query ...
    return 'Success!'
```
Raising `ModelRetry` also generates a `RetryPromptPart` containing the exception message, which is sent back to the LLM to guide its next attempt. Both `ValidationError` and `ModelRetry` respect the `retries` setting configured on the `Tool` or `Agent`.

## Toolsets

A toolset represents a collection of tools that can be registered with an agent in one go. They can be reused by different agents, swapped out at runtime or during testing, and composed in order to dynamically filter which tools are available, modify tool definitions, or change tool execution behavior. A toolset can contain locally defined functions, depend on an external service like an [MCP server](mcp/client.md) to provide them, or implement custom logic to list available tools and handle them being called.

The toolsets that will be available during an agent run can be specified in three different ways:

* at agent construction time, via the [`toolsets`][pydantic_ai.Agent.__init__] keyword argument to `Agent`
* at agent run time, via the `toolsets` keyword argument to [`agent.run()`][pydantic_ai.Agent.run], [`agent.run_sync()`][pydantic_ai.Agent.run_sync], [`agent.run_stream()`][pydantic_ai.Agent.run_stream], or [`agent.iter()`][pydantic_ai.Agent.iter]. These toolsets will be additional to those provided to the `Agent` constructor
* as a contextual override, via the `toolsets` keyword argument to the [`agent.override()`][pydantic_ai.Agent.iter] context manager. These toolsets will replace those provided at agent construction or run time during the life of the context manager

```python {title="toolsets.py"}
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset


def tool():
    pass


def extra_tool():
    pass


def override_tool():
    pass


toolset = FunctionToolset(tools=[tool]) # (1)!
extra_toolset = FunctionToolset(tools=[extra_tool])
override_toolset = FunctionToolset(tools=[override_tool])

test_model = TestModel() # (2)!
agent = Agent(test_model, toolsets=[toolset])

result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['tool']

result = agent.run_sync('What tools are available?', toolsets=[extra_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['tool', 'extra_tool']

with agent.override(toolsets=[override_toolset]):
    result = agent.run_sync('What tools are available?', toolsets=[extra_toolset]) # (3)!
    print([t.name for t in test_model.last_model_request_parameters.function_tools])
    #> ['override_tool']
```

1. The [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] will be explained in detail in the next section.
2. We're using [`TestModel`][pydantic_ai.models.test.TestModel] here because it makes it easy to see which tools were available on each run.
3. This `extra_toolset` will be ignored because we're inside an override context

_(This example is complete, it can be run "as is")_

### Function Toolset

As the name suggests, a [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] makes locally defined functions available as tools.

Functions can be added as tools in three different ways:

* via the [`@toolset.tool`][pydantic_ai.toolsets.FunctionToolset.tool] decorator
* via the [`tools`][pydantic_ai.toolsets.FunctionToolset.__init__] keyword argument to the constructor which can take either plain functions, or instances of [`Tool`][pydantic_ai.tools.Tool]
* via the [`toolset.add_function()`][pydantic_ai.toolsets.FunctionToolset.add_function] and [`toolset.add_tool()`][pydantic_ai.toolsets.FunctionToolset.add_tool] methods which can take a plain function or an instance of [`Tool`][pydantic_ai.tools.Tool] respectively

Functions registered in any of these ways can define an initial `ctx: RunContext` argument in order to receive the agent [context][pydantic_ai.tools.RunContext]. The `add_function()` and `add_tool()` methods can also be used from a tool function to dynamically register new tools during a run to be available in future run steps.

```python {title="function_toolset.py"}
from datetime import datetime

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset


def temperature_celsius(city: str) -> float:
    return 21.0


def temperature_fahrenheit(city: str) -> float:
    return 69.8


weather_toolset = FunctionToolset(tools=[temperature_celsius, temperature_fahrenheit])

@weather_toolset.tool
def conditions(ctx: RunContext, city: str) -> str:
    if ctx.run_step % 2 == 0:
        return "It's sunny"
    else:
        return "It's raining"

datetime_toolset = FunctionToolset()
datetime_toolset.add_function(lambda: datetime.now(), name='now')

print(weather_toolset.tool_names)
#> ['temperature_celsius', 'temperature_fahrenheit', 'conditions']

print(datetime_toolset.tool_names)
#> ['now']
```

_(This example is complete, it can be run "as is")_

### Toolset Composition

Toolsets can be composed to dynamically filter which tools are available, modify tool definitions, or change tool execution behavior. Multiple toolsets can also be combined into one.

#### Combining Toolsets

[`CombinedToolset`][pydantic_ai.toolsets.CombinedToolset] takes a list of toolsets and lets them be used as one.

```python {title="combined_toolset.py" requires="function_toolset.py"}
from function_toolset import weather_toolset, datetime_toolset

from pydantic_ai.toolsets import CombinedToolset


combined_toolset = CombinedToolset([weather_toolset, datetime_toolset])
print(combined_toolset.tool_names)
#> ['temperature_celsius', 'temperature_fahrenheit', 'conditions', 'now']
```

_(This example is complete, it can be run "as is")_

#### Filtering Tools

[`FilteredToolset`][pydantic_ai.toolsets.FilteredToolset] wraps a toolset and filters available tools ahead of each step of the run based on a user-defined function that is passed the agent [run context][pydantic_ai.tools.RunContext] and a each tool's [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] and returns a boolean to indicate whether or not a given tool should be available.

```python {title="filtered_toolset.py" requires="function_toolset.py,combined_toolset.py"}
from combined_toolset import combined_toolset

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FilteredToolset

filtered_toolset = FilteredToolset(combined_toolset, lambda ctx, tool_def: 'fahrenheit' not in tool_def.name)

test_model = TestModel() # (1)!
agent = Agent(test_model, toolsets=[filtered_toolset]) # (2)!
result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['weather_temperature_celsius', 'weather_conditions', 'datetime_now']
```

1. We're using [`TestModel`][pydantic_ai.models.test.TestModel] here because it makes it easy to see which tools were available on each run.
2. Because the filter function takes the agent run context, it is only applied when we're performing (or in this case, simulating) a run, not when we just call `tool_defs()` or `tool_names()` as in the previous examples.

_(This example is complete, it can be run "as is")_

#### Prefixing Tool Names

[`PrefixedToolset`][pydantic_ai.toolsets.PrefixedToolset] wraps a toolset and adds a prefix to each tool name to prevent tool name conflicts between different toolsets.

```python {title="combined_toolset.py" requires="function_toolset.py"}
from function_toolset import weather_toolset, datetime_toolset

from pydantic_ai.toolsets import CombinedToolset
from pydantic_ai.toolsets import PrefixedToolset


prefixed_weather_toolset = PrefixedToolset(weather_toolset, prefix='weather')
prefixed_datetime_toolset = PrefixedToolset(datetime_toolset, prefix='datetime')
combined_toolset = CombinedToolset([prefixed_weather_toolset, prefixed_datetime_toolset])
print(combined_toolset.tool_names)
"""
[
    'weather_temperature_celsius',
    'weather_temperature_fahrenheit',
    'weather_conditions',
    'datetime_now',
]
"""
```

_(This example is complete, it can be run "as is")_

#### Renaming Tools

[`RenamedToolset`][pydantic_ai.toolsets.RenamedToolset] wraps a toolset and lets you rename tools using a dictionary mapping old names to new names. This is useful when the names provided by a toolset are ambiguous or would conflict with tools defined by other toolsets, but [prefixing them](#prefixing-tool-names) creates a name that is unnecessarily long or could be confusing to the model.

```python {title="renamed_toolset.py" requires="function_toolset.py,combined_toolset.py"}
from combined_toolset import combined_toolset

from pydantic_ai.toolsets import RenamedToolset


renamed_toolset = RenamedToolset(
    combined_toolset,
    {
        'current_time': 'datetime_now',
        'temperature_celsius': 'weather_temperature_celsius',
        'temperature_fahrenheit': 'weather_temperature_fahrenheit'
    }
)
print(renamed_toolset.tool_names)
"""
['temperature_celsius', 'temperature_fahrenheit', 'weather_conditions', 'current_time']
"""
```

#### Preparing Tool Definitions

[`PreparedToolset`][pydantic_ai.toolsets.PreparedToolset] lets you modify the entire list of available tools ahead of each step of the agent run using a user-defined function that takes the  agent [run context][pydantic_ai.tools.RunContext] and a list of [`ToolDefinition`s][pydantic_ai.tools.ToolDefinition] and returns a list of modified `ToolDefinition`s.

This is the toolset-specific equivalent of the [`prepare_tools`](#prepare-tools) argument to `Agent` that prepares all tool definitions registered to an agent across toolsets.

Note that it is not possible to add or rename tools using `PreparedToolset`. Instead, you can use [`FunctionToolset.add_function()`](#function-toolset) or [`RenamedToolset`](#renaming-tools).

```python {title="prepared_toolset.py" requires="function_toolset.py,combined_toolset.py,renamed_toolset.py"}
from dataclasses import replace
from typing import Union

from renamed_toolset import renamed_toolset

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import PreparedToolset

descriptions = {
    'temperature_celsius': 'Get the temperature in degrees Celsius',
    'temperature_fahrenheit': 'Get the temperature in degrees Fahrenheit',
    'weather_conditions': 'Get the current weather conditions',
    'current_time': 'Get the current time',
}

async def add_descriptions(ctx: RunContext, tool_defs: list[ToolDefinition]) -> Union[list[ToolDefinition], None]:
    return [
        replace(tool_def, description=description)
        if (description := descriptions.get(tool_def.name, None))
        else tool_def
        for tool_def
        in tool_defs
    ]

prepared_toolset = PreparedToolset(renamed_toolset, add_descriptions)

test_model = TestModel() # (1)!
agent = Agent(test_model, toolsets=[prepared_toolset]) # (2)!
result = agent.run_sync('What tools are available?')
print(test_model.last_model_request_parameters.function_tools)
"""
[
    ToolDefinition(
        name='temperature_celsius',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
            'type': 'object',
        },
        description='Get the temperature in degrees Celsius',
    ),
    ToolDefinition(
        name='temperature_fahrenheit',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
            'type': 'object',
        },
        description='Get the temperature in degrees Fahrenheit',
    ),
    ToolDefinition(
        name='weather_conditions',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
            'type': 'object',
        },
        description='Get the current weather conditions',
    ),
    ToolDefinition(
        name='current_time',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {},
            'type': 'object',
        },
        description='Get the current time',
    ),
]
"""
```

1. We're using [`TestModel`][pydantic_ai.models.test.TestModel] here because it makes it easy to see which tools were available on each run.
2. Because the prepare function takes the agent run context, it is only applied when we're performing (or in this case, simulating) a run, not when we just call `tool_defs()` or `tool_names()` as in the previous examples.

#### Wrapping a Toolset

[`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset] wraps another toolset and delegates all responsibility to it.

This is a no-op by default, but enables some useful abilities:

##### Changing Tool Execution

You can subclass `WrapperToolset` to change the wrapped toolset's tool execution behavior by overriding the [`call_tool()`][pydantic_ai.toolsets.AbstractToolset.call_tool] method.

Note that if your subclass adds new dataclass fields or `__init__` arguments, you must must implement [`rewrap_for_run()`][pydantic_ai.toolsets.WrapperToolset.rewrap_for_run] as described in the API docs.

```python {title="logging_toolset.py" requires="function_toolset.py,combined_toolset.py,renamed_toolset.py,prepared_toolset.py"}
from typing_extensions import Any

from prepared_toolset import prepared_toolset

from pydantic_ai.agent import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import WrapperToolset

LOG = []

class LoggingToolset(WrapperToolset):
    async def call_tool(self, ctx: RunContext, name: str, tool_args: dict[str, Any]) -> Any:
        LOG.append(f'Calling tool {name!r} with args: {tool_args!r}')
        try:
            result = await super().call_tool(ctx, name, tool_args)
            LOG.append(f'Finished calling tool {name!r} with result: {result!r}')
        except Exception as e:
            LOG.append(f'Error calling tool {name!r}: {e}')
            raise e
        else:
            return result


logging_toolset = LoggingToolset(prepared_toolset)

agent = Agent(TestModel(), toolsets=[logging_toolset]) # (1)!
result = agent.run_sync('Call all the tools')
print(LOG)
"""
[
    "Calling tool 'temperature_celsius' with args {'city': 'a'}",
    "Calling tool 'temperature_fahrenheit' with args {'city': 'a'}",
    "Calling tool 'weather_conditions' with args {'city': 'a'}",
    "Calling tool 'current_time' with args {}",
    "Finished calling tool 'temperature_celsius' with result: 21.0",
    "Finished calling tool 'temperature_fahrenheit' with result: 69.8",
    'Finished calling tool \'weather_conditions\' with result: "It\'s raining"',
    "Finished calling tool 'current_time' with result: datetime.datetime(...)",
]
"""
```

1. We use [`TestModel`][pydantic_ai.models.test.TestModel] here as it will automatically call each tool.

_(This example is complete, it can be run "as is")_

##### Modifying Toolsets During a Run

You can change the `WrapperAgent`'s `wrapped` property during an agent run to swap out one toolset for another starting at the next run step.

To add or remove available toolsets, you can wrap a [`CombinedToolset`](#combining-toolsets) and replace it during the run with one that can include fewer, more, or entirely different toolsets.

```python {title="wrapper_toolset.py" requires="function_toolset.py"}
from function_toolset import weather_toolset, datetime_toolset

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import WrapperToolset, FunctionToolset

togglable_toolset = WrapperToolset(weather_toolset)

def toggle(ctx: RunContext[WrapperToolset]):
    if ctx.deps.wrapped == weather_toolset:
        ctx.deps.wrapped = datetime_toolset
    else:
        ctx.deps.wrapped = weather_toolset

test_model = TestModel() # (1)!
agent = Agent(
    test_model,
    deps_type=WrapperToolset, # (2)!
    toolsets=[togglable_toolset, FunctionToolset([toggle])]
)
print([t.name for ts in agent.toolsets for t in ts.tool_defs])
#> ['temperature_celsius', 'temperature_fahrenheit', 'conditions', 'toggle']

result = agent.run_sync('Toggle the toolset', deps=togglable_toolset)
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['now', 'toggle']
```

1. We're using [`TestModel`][pydantic_ai.models.test.TestModel] here because it makes it easy to see which tools were available on each run.
2. We're using the agent's dependencies to give the `toggle` tool access to the `togglable_toolset` via the `RunContext` argument

### Building a Custom Toolset

To define a fully custom toolset with its own logic to list available tools and handle them being called, you can subclass [`BaseToolset`][pydantic_ai.toolsets.BaseToolset] or [`AsyncBaseToolset`][pydantic_ai.toolsets.AsyncBaseToolset], depending on whether listing the available tools can be done synchronously or requires an asynchronous network request. Tool calls themselves are always implemented asynchronously.

These abstract classes require you to implement [`tool_defs()`][pydantic_ai.toolsets.AbstractToolset.tool_defs] or [`async_tool_defs()`][pydantic_ai.toolsets.AsyncBaseToolset.async_tool_defs] respectively, as well as the [`max_retries_for_tool()`][pydantic_ai.toolsets.AbstractToolset.max_retries_for_tool], [`get_tool_args_validator()`][pydantic_ai.toolsets.AbstractToolset.get_tool_args_validator] and [`call_tool()`][pydantic_ai.toolsets.AbstractToolset.call_tool] methods.

If you want to reuse a network connection or session across tool listings and calls during an agent run step, you can implement [`__aenter__()`][pydantic_ai.toolsets.AbstractToolset.__aenter__] and [`__aexit__()`][pydantic_ai.toolsets.AbstractToolset.__aexit__], which will be called when the agent that uses the toolset is itself entered using the [`async with agent`][pydantic_ai.Agent.__aenter__] context manager.

#### Deferred Toolset

A deferred tool is one that will be executed not by Pydantic AI, but by the upstream service that called the agent, such as a web application that supports frontend-defined tools provided to Pydantic AI via a protocol like [AG-UI](https://docs.ag-ui.com/concepts/tools#frontend-defined-tools).

!!! note
    This is typically not something you need to bother with, unless you are implementing support for such a protocol between an upstream tool provider and Pydantic AI.

When the model calls a deferred tool, the agent run ends with a [`DeferredToolCalls`][pydantic_ai.output.DeferredToolCalls] object containing the deferred tool call names and arguments, which is expected to be returned to the upstream tool provider. This upstream service is then expected to generate a response for each tool call and start a new Pydantic AI agent run with the message history and new [`ToolReturnPart`s][pydantic_ai.messages.ToolReturnPart] corresponding to each deferred call, after which the run will continue.

To enable an agent to call deferred tools, you create a [`DeferredToolset`][pydantic_ai.toolsets.DeferredToolset], pass it a list of [`ToolDefinition`s][pydantic_ai.tools.ToolDefinition], and provide it to the agent using one of the methods described above. Additionally, you need to add `DeferredToolCalls` to the `Agent`'s [output types](output.md#structured-output) so that the agent run's output type is correctly inferred. Finally, you should handle the possible `DeferredToolCalls` result by returning it to the upstream tool provider.

If your agent can also be used in a context where no deferred tools are available, you will not want to include `DeferredToolCalls` in the `output_type` passed to the `Agent` constructor as you'd have to deal with that type everywhere you use the agent. Instead, you can pass the `toolsets` and `output_type` keyword arguments when you run the agent using [`agent.run()`][pydantic_ai.Agent.run], [`agent.run_sync()`][pydantic_ai.Agent.run_sync], [`agent.run_stream()`][pydantic_ai.Agent.run_stream], or [`agent.iter()`][pydantic_ai.Agent.iter]. Note that while `toolsets` provided at this stage are additional to the toolsets provided to the constructor, the `output_type` overrides the one specified at construction time (for type inference reasons), so you'll need to include the original output types explicitly.

To demonstrate, let us first define a simple agent _without_ deferred tools:

```python {title="deferred_toolset_agent.py"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.toolsets.function import FunctionToolset

toolset = FunctionToolset()


@toolset.tool
def get_default_language():
    return 'en-US'


@toolset.tool
def get_user_name():
    return 'David'


class PersonalizedGreeting(BaseModel):
    greeting: str
    language_code: str


agent = Agent('openai:gpt-4o', toolsets=[toolset], output_type=PersonalizedGreeting)

result = agent.run_sync('Greet the user in a personalized way')
print(repr(result.output))
#> PersonalizedGreeting(greeting='Hello, David!', language_code='en-US')
```

Next, let's define an function for a hypothetical "run agent" API endpoint that can be called by the frontend and takes a list of messages to send to the model plus a dict of frontend tool names and descriptions. This is where `DeferredToolset` and `DeferredToolCalls` come in:

```python {title="deferred_toolset_api.py" requires="deferred_toolset_agent.py"}
from deferred_toolset_agent import agent

from typing_extensions import Any

from pydantic_ai.output import DeferredToolCalls
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import DeferredToolset
from pydantic_ai.messages import ModelMessage

def run_agent(
    messages: list[ModelMessage] = [], frontend_tools: list[ToolDefinition] = {}
) -> tuple[Any | DeferredToolCalls, list[ModelMessage]]:
    deferred_toolset = DeferredToolset(frontend_tools)
    result = agent.run_sync(
        toolsets=[deferred_toolset], # (1)!
        output_type=[agent.output_type, DeferredToolCalls], # (2)!
        message_history=messages, # (3)!
    )
    return result.output, result.new_messages()
```

1. As mentioned above, these `toolsets` are additional to those provided to the `Agent` constructor
2. As mentioned above, this `output_type` overrides the one provided to the `Agent` constructor, so we have to make sure to not lose it
3. We don't include an `user_prompt` keyword argument as we expect the frontend to provide it via `messages`

Now, imagine that the code below is implemented on the frontend, and `run_agent` stands in for an API call to the backend that runs the agent. This is where we actually execute the deferred tool calls and start a new run with the new result included:

```python {title="deferred_tools.py" requires="deferred_toolset_agent.py,deferred_toolset_api.py"}
from deferred_toolset_api import run_agent

from pydantic_ai.messages import ModelMessage, ModelRequest, RetryPromptPart, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.output import DeferredToolCalls

frontend_tool_definitions = [
    ToolDefinition(
        name='get_preferred_language',
        parameters_json_schema={'type': 'object', 'properties': {'default_language': {'type': 'string'}}},
        description="Get the user's preferred language from their browser",
    )
]
def get_preferred_language(default_language: str) -> str:
    return 'es-MX' # (1)!
frontend_tool_functions = {'get_preferred_language': get_preferred_language}

messages: list[ModelMessage] = [
    ModelRequest(
        parts=[
            UserPromptPart(content='Greet the user in a personalized way')
        ]
    )
]

final_output = None
while final_output is None:
    output, new_messages = run_agent(messages, frontend_tool_definitions)
    messages += new_messages

    if not isinstance(output, DeferredToolCalls):
        final_output = output
        break

    print(output.tool_calls)
    """
    [
        ToolCallPart(
            tool_name='get_preferred_language',
            args={'default_language': 'en-US'},
            tool_call_id='pyd_ai_tool_call_id',
        )
    ]
    """
    for tool_call in output.tool_calls:
        if function := frontend_tool_functions.get(tool_call.tool_name):
            part = ToolReturnPart(
                tool_name=tool_call.tool_name,
                content=function(**tool_call.args_as_dict()),
                tool_call_id=tool_call.tool_call_id,
            )
        else:
            part = RetryPromptPart(
                tool_name=tool_call.tool_name,
                content=f'Unknown tool {tool_call.tool_name!r}',
                tool_call_id=tool_call.tool_call_id,
            )
        messages.append(ModelRequest(parts=[part]))

print(repr(final_output))
"""
PersonalizedGreeting(greeting='Hola, David! Espero que tengas un gran día!', language_code='es-MX')
"""
```

1. Imagine that this returns [`navigator.language`](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/language)

_(This example is complete, it can be run "as is")_

## Third-Party Tools

### MCP Tools {#mcp-tools}

See the [MCP Client](./mcp/client.md) documentation for how to use MCP servers with Pydantic AI as [toolsets](#toolsets).

### LangChain Tools {#langchain-tools}

If you'd like to use a tool from LangChain's [community tool library](https://python.langchain.com/docs/integrations/tools/) with Pydantic AI, you can use the [`tool_from_langchain`][pydantic_ai.ext.langchain.tool_from_langchain] convenience method. Note that Pydantic AI will not validate the arguments in this case -- it's up to the model to provide arguments matching the schema specified by the LangChain tool, and up to the LangChain tool to raise an error if the arguments are invalid.

You will need to install the `langchain-community` package and any others required by the tool in question.

Here is how you can use the LangChain `DuckDuckGoSearchRun` tool, which requires the `ddgs` package:

```python {test="skip"}
from langchain_community.tools import DuckDuckGoSearchRun

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import tool_from_langchain


search = DuckDuckGoSearchRun()
search_tool = tool_from_langchain(search)

agent = Agent(
    'google-gla:gemini-2.0-flash',
    tools=[search_tool],
)

result = agent.run_sync('What is the release date of Elden Ring Nightreign?')  # (1)!
print(result.output)
#> Elden Ring Nightreign is planned to be released on May 30, 2025.
```

1. The release date of this game is the 30th of May 2025, which is after the knowledge cutoff for Gemini 2.0 (August 2024).

_(This example is complete, it can be run "as is")_

If you'd like to use multiple LangChain tools or a LangChain [toolkit](https://python.langchain.com/docs/concepts/tools/#toolkits), you can use the [`LangChainToolset`][pydantic_ai.ext.langchain.LangChainToolset] which takes a list of LangChain tools:

```python {test="skip"}
from langchain_community.agent_toolkits import SlackToolkit

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset


toolkit = SlackToolkit()
toolset = LangChainToolset(toolkit.get_tools())

agent = Agent('openai:gpt-4o', toolsets=[toolset])
# ...
```

### ACI.dev Tools {#aci-tools}

If you'd like to use a tool from the [ACI.dev tool library](https://www.aci.dev/tools) with Pydantic AI, you can use the [`tool_from_aci`][pydantic_ai.ext.aci.tool_from_aci] convenience method. Note that Pydantic AI will not validate the arguments in this case -- it's up to the model to provide arguments matching the schema specified by the ACI tool, and up to the ACI tool to raise an error if the arguments are invalid.

You will need to install the `aci-sdk` package, set your ACI API key in the `ACI_API_KEY` environment variable, and pass your ACI "linked account owner ID" to the function.

Here is how you can use the ACI.dev `TAVILY__SEARCH` tool:

```python {test="skip"}
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import tool_from_aci


tavily_search = tool_from_aci(
    'TAVILY__SEARCH',
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent(
    'google-gla:gemini-2.0-flash',
    tools=[tavily_search],
)

result = agent.run_sync('What is the release date of Elden Ring Nightreign?')  # (1)!
print(result.output)
#> Elden Ring Nightreign is planned to be released on May 30, 2025.
```

1. The release date of this game is the 30th of May 2025, which is after the knowledge cutoff for Gemini 2.0 (August 2024).

If you'd like to use multiple ACI.dev tools, you can use the [`ACIToolset`][pydantic_ai.ext.aci.ACIToolset] which takes a list of ACI tool names as well as the `linked_account_owner_id`:

```python {test="skip"}
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import ACIToolset


toolset = ACIToolset(
    [
        'OPEN_WEATHER_MAP__CURRENT_WEATHER',
        'OPEN_WEATHER_MAP__FORECAST',
    ],
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent('openai:gpt-4o', toolsets=[toolset])
```
