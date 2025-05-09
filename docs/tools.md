# Function Tools

Function tools provide a mechanism for models to retrieve extra information to help them generate a response.

They're useful when it is impractical or impossible to put all the context an agent might need into the system prompt, or when you want to make agents' behavior more deterministic or reliable by deferring some of the logic required to generate a response to another (not necessarily AI-powered) tool.

!!! info "Function tools vs. RAG"
    Function tools are basically the "R" of RAG (Retrieval-Augmented Generation) — they augment what the model can do by letting it request extra information.

    The main semantic difference between PydanticAI Tools and RAG is RAG is synonymous with vector search, while PydanticAI tools are more general-purpose. (Note: we may add support for vector search functionality in the future, particularly an API for generating embeddings. See [#58](https://github.com/pydantic/pydantic-ai/issues/58))

There are a number of ways to register tools with an agent:

* via the [`@agent.tool`][pydantic_ai.Agent.tool] decorator — for tools that need access to the agent [context][pydantic_ai.tools.RunContext]
* via the [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain] decorator — for tools that do not need access to the agent [context][pydantic_ai.tools.RunContext]
* via the [`tools`][pydantic_ai.Agent.__init__] keyword argument to `Agent` which can take either plain functions, or instances of [`Tool`][pydantic_ai.tools.Tool]

## Registering Function Tools via Decorator

`@agent.tool` is considered the default decorator since in the majority of cases tools will need access to the agent context.

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
def roll_die() -> str:
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
3. This tool doesn't need any context, it just returns a random number. You could probably use a dynamic system prompt in this case.
4. This tool needs the player's name, so it uses `RunContext` to access dependencies which are just the player's name in this case.
5. Run the agent, passing the player's name as the dependency.

_(This example is complete, it can be run "as is")_

Let's print the messages from that game to see what happened:

```python {title="dice_game_messages.py"}
from dice_game import dice_result

print(dice_result.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content="You're a dice game, you should roll the die and see if the number you get back matches the user's guess. If so, tell them they're a winner. Use the player's name in the response.",
                timestamp=datetime.datetime(...),
                dynamic_ref=None,
                part_kind='system-prompt',
            ),
            UserPromptPart(
                content='My guess is 4',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            ),
        ],
        instructions=None,
        kind='request',
    ),
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='roll_die',
                args={},
                tool_call_id='pyd_ai_tool_call_id',
                part_kind='tool-call',
            )
        ],
        usage=Usage(
            requests=1,
            request_tokens=90,
            response_tokens=2,
            total_tokens=92,
            details=None,
        ),
        model_name='gemini-1.5-flash',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='roll_die',
                content='4',
                tool_call_id='pyd_ai_tool_call_id',
                timestamp=datetime.datetime(...),
                part_kind='tool-return',
            )
        ],
        instructions=None,
        kind='request',
    ),
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='get_player_name',
                args={},
                tool_call_id='pyd_ai_tool_call_id',
                part_kind='tool-call',
            )
        ],
        usage=Usage(
            requests=1,
            request_tokens=91,
            response_tokens=4,
            total_tokens=95,
            details=None,
        ),
        model_name='gemini-1.5-flash',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='get_player_name',
                content='Anne',
                tool_call_id='pyd_ai_tool_call_id',
                timestamp=datetime.datetime(...),
                part_kind='tool-return',
            )
        ],
        instructions=None,
        kind='request',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content="Congratulations Anne, you guessed correctly! You're a winner!",
                part_kind='text',
            )
        ],
        usage=Usage(
            requests=1,
            request_tokens=92,
            response_tokens=12,
            total_tokens=104,
            details=None,
        ),
        model_name='gemini-1.5-flash',
        timestamp=datetime.datetime(...),
        kind='response',
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

    LLM ->> Agent: Call tool<br>roll_die()
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

## Registering Function Tools via Agent Argument

As well as using the decorators, we can register tools via the `tools` argument to the [`Agent` constructor][pydantic_ai.Agent.__init__]. This is useful when you want to reuse tools, and can also give more fine-grained control over the tools.

```python {title="dice_game_tool_kwarg.py"}
import random

from pydantic_ai import Agent, RunContext, Tool

system_prompt = """\
You're a dice game, you should roll the die and see if the number
you get back matches the user's guess. If so, tell them they're a winner.
Use the player's name in the response.
"""


def roll_die() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))


def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps


agent_a = Agent(
    'google-gla:gemini-1.5-flash',
    deps_type=str,
    tools=[roll_die, get_player_name],  # (1)!
    system_prompt=system_prompt,
)
agent_b = Agent(
    'google-gla:gemini-1.5-flash',
    deps_type=str,
    tools=[  # (2)!
        Tool(roll_die, takes_ctx=False),
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

## Function Tool Output

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

## Function Tools vs. Structured Outputs

As the name suggests, function tools use the model's "tools" or "functions" API to let the model know what is available to call. Tools or functions are also used to define the schema(s) for structured responses, thus a model might have access to many tools, some of which call function tools while others end the run and produce a final output.

## Function tools and schema

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
        description='This is a Foobar',
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
        outer_typed_dict_key=None,
        strict=None,
    )
]
"""
```

_(This example is complete, it can be run "as is")_

## Dynamic Function tools {#tool-prepare}

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
        description='',
        parameters_json_schema={
            'additionalProperties': False,
            'properties': {
                'name': {'type': 'string', 'description': 'Name of the human to greet.'}
            },
            'required': ['name'],
            'type': 'object',
        },
        outer_typed_dict_key=None,
        strict=None,
    )
]
"""
```

_(This example is complete, it can be run "as is")_


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
