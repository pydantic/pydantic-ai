# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Core Development Tasks

- **Install dependencies**: `make install` (requires uv, pre-commit, and deno)
- **Run all checks**: `pre-commit run --all-files`
- **Run tests**: `make test`
- **Build docs**: `make docs` or `make docs-serve` (local development)

### Single Test Commands

- **Run specific test**: `uv run pytest tests/test_agent.py::test_function_name -v`
- **Run test file**: `uv run pytest tests/test_agent.py -v`
- **Run with debug**: `uv run pytest tests/test_agent.py -v -s`

## Project Architecture

### Core Components

**Agent System (`pydantic_ai_slim/pydantic_ai/agent/`)**
- `Agent[AgentDepsT, OutputDataT]`: Main orchestrator class with generic types for dependency injection and output validation
- Entry points: `run()`, `run_sync()`, `run_stream()` methods
- Handles tool management, system prompts, and model interaction

**Model Integration (`pydantic_ai_slim/pydantic_ai/models/`)**
- Unified interface across providers: OpenAI, Anthropic, Google, Groq, Cohere, Mistral, Bedrock, HuggingFace
- Model strings: `"openai:gpt-5"`, `"anthropic:claude-sonnet-4-5"`, `"google:gemini-2.5-pro"`
- `ModelRequestParameters` for configuration, `StreamedResponse` for streaming

**Graph-based Execution (`pydantic_graph/` + `_agent_graph.py`)**
- State machine execution through: `UserPromptNode` → `ModelRequestNode` → `CallToolsNode`
- `GraphAgentState` maintains message history and usage tracking
- `GraphRunContext` provides execution context

**Tool System (`tools.py`, `toolsets/`)**
- `@agent.tool` decorator for function registration
- `RunContext[AgentDepsT]` provides dependency injection in tools
- Support for sync/async functions with automatic schema generation

**Output Handling**
- `TextOutput`: Plain text responses
- `ToolOutput`: Structured data via tool calls
- `NativeOutput`: Provider-specific structured output
- `PromptedOutput`: Prompt-based structured extraction

### Key Design Patterns

**Dependency Injection**
```python
@dataclass
class MyDeps:
    database: DatabaseConn

agent = Agent('openai:gpt-5', deps_type=MyDeps)

@agent.tool
async def get_data(ctx: RunContext[MyDeps]) -> str:
    return await ctx.deps.database.fetch_data()
```

**Type-Safe Agents**
```python
class OutputModel(BaseModel):
    result: str
    confidence: float

agent: Agent[MyDeps, OutputModel] = Agent(
    'openai:gpt-5',
    deps_type=MyDeps,
    output_type=OutputModel
)
```

## Workspace Structure

This is a uv workspace with multiple packages:
- **`pydantic_ai_slim/`**: Core framework (minimal dependencies)
- **`pydantic_evals/`**: Evaluation system
- **`pydantic_graph/`**: Graph execution engine
- **`examples/`**: Example applications
- **`clai/`**: CLI tool

## Testing Strategy

- **Unit tests**: `tests/` directory with comprehensive model and component coverage
- **VCR cassettes**: `tests/cassettes/` for recorded LLM API interactions
- **Test models**: Use `TestModel` for deterministic testing
- **Examples testing**: `tests/test_examples.py` validates all documentation examples
- **Multi-version testing**: Python 3.10-3.13 support

## Key Configuration Files

- **`pyproject.toml`**: Main workspace configuration with dependency groups
- **`pydantic_ai_slim/pyproject.toml`**: Core package with model optional dependencies
- **`Makefile`**: Development task automation
- **`uv.lock`**: Locked dependencies for reproducible builds

## Important Implementation Notes

- **Model Provider Integration**: Each provider in `models/` directory implements the `Model` abstract base class
- **Message System**: Vendor-agnostic message format in `messages.py` with rich content type support
- **Streaming Architecture**: Real-time response processing with validation during streaming
- **Error Handling**: Specific exception types with retry mechanisms at multiple levels
- **OpenTelemetry Integration**: Built-in observability support

## Documentation Development

- **Local docs**: `make docs-serve` (serves at http://localhost:8000)
- **Docs source**: `docs/` directory (MkDocs with Material theme)
- **API reference**: Auto-generated from docstrings using mkdocstrings

## Dependencies Management

- **Package manager**: uv (fast Python package manager)
- **Lock file**: `uv.lock` (commit this file)
- **Sync command**: `make sync` to update dependencies
- **Optional extras**: Define groups in `pyproject.toml` optional-dependencies

## Best Practices

This is the list of best practices for working with the codebase.

### Rename a class

When asked to rename a class, you need to rename the class in the code and add a deprecation warning to the old class.

```python
from typing_extensions import deprecated

class NewClass: ...  # This class was renamed from OldClass.

@deprecated("Use `NewClass` instead.")
class OldClass(NewClass): ...
```

In the test suite, you MUST use the `NewClass` instead of the `OldClass`, and create a new test to verify the
deprecation warning:

```python
def test_old_class_is_deprecated():
    with pytest.warns(DeprecationWarning, match="Use `NewClass` instead."):
        OldClass()
```

In the documentation, you should not have references to the old class, only the new class.

### Writing documentation

Always reference Python objects with the "`" (backticks) around them, and link to the API reference, for example:

```markdown
The [`Agent`][pydantic_ai.agent.Agent] class is the main entry point for creating and running agents.
```

### Coverage

Every pull request MUST have 100% coverage. You can check the coverage by running `make test`.
