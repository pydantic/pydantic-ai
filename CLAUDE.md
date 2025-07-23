# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Core Development Tasks
- **Install dependencies**: `make install` (requires uv, pre-commit, and deno)
- **Run all checks**: `make` (format, lint, typecheck, test with coverage)
- **Format code**: `make format`
- **Lint code**: `make lint`
- **Type checking**: `make typecheck` (uses pyright) or `make typecheck-both` (pyright + mypy)
- **Run tests**: `make test` (with coverage) or `make test-fast` (parallel, no coverage)
- **Build docs**: `make docs` or `make docs-serve` (local development)

### Single Test Commands
- **Run specific test**: `uv run pytest tests/test_agent.py::test_function_name -v`
- **Run test file**: `uv run pytest tests/test_agent.py -v`
- **Run with debug**: `uv run pytest tests/test_agent.py -v -s`

### Multi-Python Testing
- **Install all Python versions**: `make install-all-python`
- **Test all Python versions**: `make test-all-python`

## Project Architecture

### Core Components

**Agent System (`pydantic_ai_slim/pydantic_ai/agent.py`)**
- `Agent[AgentDepsT, OutputDataT]`: Main orchestrator class with generic types for dependency injection and output validation
- Entry points: `run()`, `run_sync()`, `run_stream()` methods
- Handles tool management, system prompts, and model interaction

**Model Integration (`pydantic_ai_slim/pydantic_ai/models/`)**
- Unified interface across providers: OpenAI, Anthropic, Google, Groq, Cohere, Mistral, Bedrock, HuggingFace
- Model strings: `"openai:gpt-4o"`, `"anthropic:claude-3-5-sonnet"`, `"google:gemini-1.5-pro"`
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

agent = Agent('openai:gpt-4o', deps_type=MyDeps)

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
    'openai:gpt-4o',
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
- **`mcp-run-python/`**: MCP server implementation (Deno/TypeScript)

## Testing Strategy

- **Unit tests**: `tests/` directory with comprehensive model and component coverage
- **VCR cassettes**: `tests/cassettes/` for recorded LLM API interactions
- **Test models**: Use `TestModel` for deterministic testing
- **Examples testing**: `tests/test_examples.py` validates all documentation examples
- **Multi-version testing**: Python 3.9-3.13 support

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
