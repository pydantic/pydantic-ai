---
name: pydanticai-docs
description: Use this skill for requests related to Pydantic AI framework - building agents, tools, dependencies, structured outputs, and model integrations.
---

# Pydantic AI Documentation Skill

## Overview

This skill provides guidance for using **Pydantic AI** - a Python agent framework for building production-grade Generative AI applications. Pydantic AI emphasizes type safety, dependency injection, and structured outputs.

## Key Concepts

### Agents

Agents are the primary interface for interacting with LLMs. They contain:

- **Instructions**: System prompts for the LLM
- **Tools**: Functions the LLM can call
- **Output Type**: Structured datatype the LLM must return
- **Dependencies**: Data/services injected into tools and prompts

### Models

Supported models include:

- OpenAI: `openai:gpt-4o`, `openai:gpt-5`
- Anthropic: `anthropic:claude-sonnet-4-5`
- Google: `google:gemini-2.0-flash`
- Groq, Azure, Together AI, DeepSeek, Grok, and more

### Tools

Two types of tools:

- `@agent.tool`: Receives `RunContext` with dependencies
- `@agent.tool_plain`: Plain function without context

### Toolsets

Collections of tools that can be registered with agents:

- `FunctionToolset`: Group multiple tools
- `SkillsToolset`: Progressive skill discovery
- `MCPServerTool`: Model Context Protocol servers
- Third-party toolsets (ACI.dev, etc.)

## Instructions

### 1. Fetch Full Documentation

For the most accurate and up-to-date information, always fetch the full documentation:

```
https://ai.pydantic.dev/llms-full.txt
```

### 2. Quick Examples

**Basic Agent:**

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', instructions='You are a helpful assistant.')
result = agent.run_sync('Hello!')
print(result.output)
```

**With Structured Output:**

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

agent = Agent('openai:gpt-4o', output_type=CityInfo)
result = agent.run_sync('Tell me about Paris')
print(result.output)  # CityInfo(name='Paris', country='France', population=...)
```

**With Tools:**

```python
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-4o')

@agent.tool
async def get_weather(ctx: RunContext[str], city: str) -> str:
    # Your implementation
    return f"Weather in {city}: Sunny, 22°C"

result = await agent.run('What is the weather in London?')
```

**With Dependencies:**

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class AppDeps:
    api_key: str
    user_id: str

agent = Agent('openai:gpt-4o', deps_type=AppDeps)

@agent.tool
async def get_user_data(ctx: RunContext[AppDeps]) -> str:
    return f"User: {ctx.deps.user_id}"

result = await agent.run('Get my data', deps=AppDeps(api_key='...', user_id='123'))
```

## When to Use This Skill

Use this skill when the user asks about:

- How to build agents with Pydantic AI
- Tool definitions and toolsets
- Dependency injection patterns
- Structured outputs with Pydantic models
- Model configuration and providers
- Streaming responses
- Testing agents
