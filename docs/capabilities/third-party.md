# Third-Party Capabilities

[Capabilities](overview.md) are the recommended way for third-party packages to extend Pydantic AI, since they can bundle tools with hooks, instructions, and model settings. See [Extensibility](../extensibility.md) for the full ecosystem, including [third-party toolsets](../toolsets.md#third-party-toolsets) that can also be wrapped as capabilities.

## Task Management

Capabilities for task planning and progress tracking help agents organize complex work:

* [`pydantic-ai-todo`](https://github.com/vstorm-co/pydantic-ai-todo) - `TodoCapability` with `add_todo`, `read_todos`, `write_todos`, `update_todo_status`, and `remove_todo` tools. Supports subtasks, dependencies, and PostgreSQL persistence. Also available as a lower-level `TodoToolset`.

## Context Management

Capabilities for managing long conversations help agents stay within context limits:

* [`summarization-pydantic-ai`](https://github.com/vstorm-co/summarization-pydantic-ai) - Four capabilities for managing long conversations: `ContextManagerCapability` (real-time token tracking, auto-compression at a configurable threshold, and large tool-output truncation); `SummarizationCapability` (LLM-powered history compression); `SlidingWindowCapability` (zero-cost message trimming); `LimitWarnerCapability` (injects a finish-soon hint before hard context limits). Also available as standalone `history_processors`: `SummarizationProcessor`, `SlidingWindowProcessor`, and `LimitWarnerProcessor`.

## Multi-Agent Orchestration

Capabilities for spawning and delegating to specialized subagents help agents tackle complex, parallelizable work:

* [`subagents-pydantic-ai`](https://github.com/vstorm-co/subagents-pydantic-ai) - `SubAgentCapability` adds tools for multi-agent delegation: `task` (spawn a subagent), `check_task`, `wait_tasks`, `list_active_tasks`, `soft_cancel_task`, `hard_cancel_task`, and `answer_subagent`. Supports sync, async, and auto execution modes, nested subagents, and runtime agent creation. Also available as a lower-level toolset via `create_subagent_toolset`.

## Guardrails & Safety

Capabilities for cost control, input/output filtering, and tool permissions help keep agents safe and within budget:

* [`pydantic-ai-shields`](https://github.com/vstorm-co/pydantic-ai-shields) - Ready-to-use guardrail capabilities: `CostTracking` (tracks token usage and USD cost per run, raises `BudgetExceededError` on budget overrun); `ToolGuard` (block or require approval for specific tools); `InputGuard` and `OutputGuard` (custom sync or async validation functions); `PromptInjection`, `PiiDetector`, `SecretRedaction`, `BlockedKeywords`, and `NoRefusals` content shields.

## File Operations & Sandboxing

Capabilities for filesystem access and sandboxed code execution help agents work with files and run code safely:

* [`pydantic-ai-backend`](https://github.com/vstorm-co/pydantic-ai-backend) - `ConsoleCapability` registers `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, and `execute` tools with a fine-grained permission system. Backends include `StateBackend` (in-memory, for testing), `LocalBackend` (real filesystem), `DockerSandbox` (isolated container execution), and `CompositeBackend` (routing across backends). Also available as a lower-level `ConsoleToolset`.

## Agent Skills

Capabilities that implement [Agent Skills](https://agentskills.io) support help agents efficiently discover and perform specific tasks:

* [`pydantic-ai-skills`](https://github.com/DougTrajano/pydantic-ai-skills) - `SkillsCapability` implements Agent Skills support with progressive disclosure (load skills on-demand to reduce tokens). Supports filesystem and programmatic skills; compatible with [agentskills.io](https://agentskills.io).

## Data & Analytics

Capabilities for querying and analyzing structured data help agents answer questions over files and databases:

* [`pydantic-ai-chdb`](https://github.com/chdb-io/pydantic-ai-chdb) - `ChDBCapability` gives agents analytical SQL over local files (Parquet/CSV/JSON), object storage, and remote databases with [chDB](https://clickhouse.com/docs/en/chdb), the in-process ClickHouse engine — the engine itself needs no server or connection string to run (remote sources are reached via ClickHouse table functions, which take their own credentials). Registers `run_select_query` (read-only ClickHouse SQL with parameter binding), `list_databases`, `list_tables`, `describe_table`, `get_sample_data`, `list_functions`, and `attach_file` (opt-in writable sessions) tools plus schema-first instructions. Sessions default to the engine-level `readonly=2` setting with capped results, and typed engine errors are mapped to [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] so the model can correct its queries. Works with [agent specs](../agent-spec.md) out of the box, so it can be loaded via [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] / [`Agent.from_spec`][pydantic_ai.Agent.from_spec]. Also available as a lower-level [toolset](../toolsets.md) via [`ChDBCapability(...).get_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_toolset].

To add your package to this page, open a pull request.



To publish your own capability package, see [Publishing capabilities](custom.md#publishing-capabilities) and [Extensibility](../extensibility.md).
