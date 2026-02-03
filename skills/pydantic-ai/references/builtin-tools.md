# Built-in Tools Reference

Source: `pydantic_ai_slim/pydantic_ai/builtin_tools.py`

Built-in tools are provider-native tools executed by the model provider, not locally.

## WebSearchTool

Web search via provider's infrastructure:

```py {title="web_search_anthropic.py"}
from pydantic_ai import Agent, WebSearchTool

agent = Agent('anthropic:claude-sonnet-4-0', builtin_tools=[WebSearchTool()])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
#> Scientists have developed a universal AI detector that can identify deepfake videos.
```

With OpenAI (requires Responses API):

```py {title="web_search_openai.py"}
from pydantic_ai import Agent, WebSearchTool

agent = Agent('openai-responses:gpt-5', builtin_tools=[WebSearchTool()])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
#> Scientists have developed a universal AI detector that can identify deepfake videos.
```

Configuration options:

```py {title="web_search_configured.py"}
from pydantic_ai import Agent, WebSearchTool, WebSearchUserLocation

agent = Agent(
    'anthropic:claude-sonnet-4-0',
    builtin_tools=[
        WebSearchTool(
            search_context_size='high',
            user_location=WebSearchUserLocation(
                city='San Francisco',
                country='US',
                region='CA',
                timezone='America/Los_Angeles',
            ),
            blocked_domains=['example.com', 'spam-site.net'],
            allowed_domains=None,  # Cannot use both blocked_domains and allowed_domains with Anthropic
            max_uses=5,  # Anthropic only: limit tool usage
        )
    ],
)

result = agent.run_sync('Use the web to get the current time.')
print(result.output)
#> In San Francisco, it's 8:21:41 pm PDT on Wednesday, August 6, 2025.
```

**Provider support**: OpenAI Responses, Anthropic, Google, xAI, Groq (compound models)

## CodeExecutionTool

Execute code in provider's secure sandbox:

```py {title="code_execution_basic.py"}
from pydantic_ai import Agent, CodeExecutionTool

agent = Agent('anthropic:claude-sonnet-4-0', builtin_tools=[CodeExecutionTool()])

result = agent.run_sync('Calculate the factorial of 15.')
print(result.output)
#> The factorial of 15 is **1,307,674,368,000**.
print(result.response.builtin_tool_calls)
"""
[
    (
        BuiltinToolCallPart(
            tool_name='code_execution',
            args={
                'code': 'import math\n\n# Calculate factorial of 15\nresult = math.factorial(15)\nprint(f"15! = {result}")\n\n# Let\'s also show it in a more readable format with commas\nprint(f"15! = {result:,}")'
            },
            tool_call_id='srvtoolu_017qRH1J3XrhnpjP2XtzPCmJ',
            provider_name='anthropic',
        ),
        BuiltinToolReturnPart(
            tool_name='code_execution',
            content={
                'content': [],
                'return_code': 0,
                'stderr': '',
                'stdout': '15! = 1307674368000\n15! = 1,307,674,368,000',
                'type': 'code_execution_result',
            },
            tool_call_id='srvtoolu_017qRH1J3XrhnpjP2XtzPCmJ',
            timestamp=datetime.datetime(...),
            provider_name='anthropic',
        ),
    )
]
"""
```

OpenAI with image output:

```py {title="code_execution_openai.py"}
from pydantic_ai import Agent, BinaryImage, CodeExecutionTool
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

agent = Agent(
    'openai-responses:gpt-5',
    builtin_tools=[CodeExecutionTool()],
    output_type=BinaryImage,
    model_settings=OpenAIResponsesModelSettings(openai_include_code_execution_outputs=True),
)

result = agent.run_sync('Generate a chart of y=x^2 for x=-5 to 5.')
assert isinstance(result.output, BinaryImage)
```

**Provider support**: OpenAI, Google, Anthropic, xAI, Bedrock (Nova 2.0)

## ImageGenerationTool

Generate images via provider's image models:

```py {title="image_generation_openai.py"}
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool

agent = Agent('openai-responses:gpt-5', builtin_tools=[ImageGenerationTool()])

result = agent.run_sync('Tell me a two-sentence story about an axolotl with an illustration.')
print(result.output)
"""
Once upon a time, in a hidden underwater cave, lived a curious axolotl named Pip who loved to explore. One day, while venturing further than usual, Pip discovered a shimmering, ancient coin that granted wishes!
"""

assert isinstance(result.response.images[0], BinaryImage)
```

Use as output type (auto-enables tool):

```py {title="image_generation_output.py"}
from pydantic_ai import Agent, BinaryImage

agent = Agent('openai-responses:gpt-5', output_type=BinaryImage)

result = agent.run_sync('Generate an image of an axolotl.')
assert isinstance(result.output, BinaryImage)
```

Configuration options:

```py {title="image_generation_configured.py"}
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool

agent = Agent(
    'openai-responses:gpt-5',
    builtin_tools=[
        ImageGenerationTool(
            background='transparent',
            input_fidelity='high',
            moderation='low',
            output_compression=100,
            output_format='png',
            partial_images=3,
            quality='high',
            size='1024x1024',
        )
    ],
    output_type=BinaryImage,
)

result = agent.run_sync('Generate an image of an axolotl.')
assert isinstance(result.output, BinaryImage)
```

**Provider support**: OpenAI Responses (gpt-5+), Google (gemini-*-image models)

## WebFetchTool

Fetch URL contents into context:

```py {title="web_fetch_basic.py"}
from pydantic_ai import Agent, WebFetchTool

agent = Agent('google-gla:gemini-2.5-flash', builtin_tools=[WebFetchTool()])

result = agent.run_sync('What is this? https://ai.pydantic.dev')
print(result.output)
#> A Python agent framework for building Generative AI applications.
```

**Provider support**: Anthropic, Google

## MemoryTool

Enable agent memory (Anthropic only):

```py {title="anthropic_memory.py"}
from typing import Any

from anthropic.lib.tools import BetaAbstractMemoryTool
from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)

from pydantic_ai import Agent, MemoryTool


class FakeMemoryTool(BetaAbstractMemoryTool):
    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        return 'The user lives in Mexico City.'

    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        return f'File created successfully at {command.path}'

    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        return f'File {command.path} has been edited'

    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        return f'Text inserted at line {command.insert_line} in {command.path}'

    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        return f'File deleted: {command.path}'

    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        return f'Renamed {command.old_path} to {command.new_path}'

    def clear_all_memory(self) -> str:
        return 'All memory cleared'

fake_memory = FakeMemoryTool()

agent = Agent('anthropic:claude-sonnet-4-5', builtin_tools=[MemoryTool()])


@agent.tool_plain
def memory(**command: Any) -> Any:
    return fake_memory.call(command)


result = agent.run_sync('Remember that I live in Mexico City')
print(result.output)
"""
Got it! I've recorded that you live in Mexico City. I'll remember this for future reference.
"""

result = agent.run_sync('Where do I live?')
print(result.output)
#> You live in Mexico City.
```

## MCPServerTool

Use remote MCP servers via provider:

```py {title="mcp_server_anthropic.py"}
from pydantic_ai import Agent, MCPServerTool

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    builtin_tools=[
        MCPServerTool(
            id='deepwiki',
            url='https://mcp.deepwiki.com/mcp',  # (1)
        )
    ]
)

result = agent.run_sync('Tell me about the pydantic/pydantic-ai repo.')
print(result.output)
"""
The pydantic/pydantic-ai repo is a Python agent framework for building Generative AI applications.
"""
```

**Provider support**: OpenAI Responses, Anthropic, xAI

## FileSearchTool

Vector search through uploaded files (RAG):

```py {title="file_search_openai_upload.py" test="skip"}
import asyncio

from pydantic_ai import Agent, FileSearchTool
from pydantic_ai.models.openai import OpenAIResponsesModel


async def main():
    model = OpenAIResponsesModel('gpt-5')

    with open('my_document.txt', 'rb') as f:
        file = await model.client.files.create(file=f, purpose='assistants')

    vector_store = await model.client.vector_stores.create(name='my-docs')
    await model.client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file.id
    )

    agent = Agent(
        model,
        builtin_tools=[FileSearchTool(file_store_ids=[vector_store.id])]
    )

    result = await agent.run('What information is in my documents about pydantic?')
    print(result.output)
    #> Based on your documents, Pydantic is a data validation library for Python...

asyncio.run(main())
```

**Provider support**: OpenAI Responses, Google (Gemini Files API)

## Dynamic Configuration

Configure built-in tools based on run context:

```python {title="dynamic_builtin_tool.py"}
from pydantic_ai import Agent, RunContext, WebSearchTool


async def prepared_web_search(ctx: RunContext[dict]) -> WebSearchTool | None:
    if not ctx.deps.get('location'):
        return None

    return WebSearchTool(
        user_location={'city': ctx.deps['location']},
    )

agent = Agent(
    'openai-responses:gpt-5',
    builtin_tools=[prepared_web_search],
    deps_type=dict,
)

# Run with location
result = agent.run_sync(
    'What is the weather like?',
    deps={'location': 'London'},
)
print(result.output)
#> It's currently raining in London.

# Run without location (tool will be omitted)
result = agent.run_sync(
    'What is the capital of France?',
    deps={'location': None},
)
print(result.output)
#> The capital of France is Paris.
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `WebSearchTool` | `pydantic_ai.WebSearchTool` | Web search |
| `WebSearchUserLocation` | `pydantic_ai.WebSearchUserLocation` | User location config |
| `CodeExecutionTool` | `pydantic_ai.CodeExecutionTool` | Code execution |
| `ImageGenerationTool` | `pydantic_ai.ImageGenerationTool` | Image generation |
| `WebFetchTool` | `pydantic_ai.WebFetchTool` | URL fetching |
| `MemoryTool` | `pydantic_ai.MemoryTool` | Agent memory |
| `MCPServerTool` | `pydantic_ai.MCPServerTool` | Remote MCP servers |
| `FileSearchTool` | `pydantic_ai.FileSearchTool` | Vector file search |
