# Third-Party Capabilities

[Capabilities](overview.md) are the recommended way for third-party packages to extend Pydantic AI, since they can bundle tools with hooks, instructions, and model settings. See [Extensibility](../extensibility.md) for the full ecosystem, including [third-party toolsets](../toolsets.md#third-party-toolsets) that can also be wrapped as capabilities.

Many of the use cases below are also covered by first-party capabilities in Pydantic AI itself or [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/), the official capability library. Where that's the case, we point to the built-in option first, and list the community packages as alternatives.

## Task Management

For model-owned task planning and progress tracking, [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) ships [`Planning`](https://pydantic.dev/docs/ai/harness/planning/), a cache-friendly self-updating task plan. As a community alternative with subtask, dependency, and PostgreSQL persistence support:

* [`pydantic-ai-todo`](https://github.com/vstorm-co/pydantic-ai-todo) - `TodoCapability` with `add_todo`, `read_todos`, `write_todos`, `update_todo_status`, and `remove_todo` tools. Supports subtasks, dependencies, and PostgreSQL persistence. Also available as a lower-level `TodoToolset`.

## Context Management

Pydantic AI has [built-in compaction](compaction.md) — provider-native APIs and model-agnostic history summarization — and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/compaction/) adds a full menu of compaction strategies. As a community alternative:

* [`summarization-pydantic-ai`](https://github.com/vstorm-co/summarization-pydantic-ai) - Four capabilities for managing long conversations: `ContextManagerCapability` (real-time token tracking, auto-compression at a configurable threshold, and large tool-output truncation); `SummarizationCapability` (LLM-powered history compression); `SlidingWindowCapability` (zero-cost message trimming); `LimitWarnerCapability` (injects a finish-soon hint before hard context limits). Also available as standalone `history_processors`: `SummarizationProcessor`, `SlidingWindowProcessor`, and `LimitWarnerProcessor`.

## Multi-Agent Orchestration

Pydantic AI supports [multi-agent patterns](../multi-agent-applications.md) directly, and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/subagents/) ships [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/) for delegating self-contained tasks to named child agents. As a community alternative:

* [`subagents-pydantic-ai`](https://github.com/vstorm-co/subagents-pydantic-ai) - `SubAgentCapability` adds tools for multi-agent delegation: `task` (spawn a subagent), `check_task`, `wait_tasks`, `list_active_tasks`, `soft_cancel_task`, `hard_cancel_task`, and `answer_subagent`. Supports sync, async, and auto-execution modes, nested subagents, and runtime agent creation. Also available as a lower-level toolset via `create_subagent_toolset`.

## Guardrails & Safety {#guardrails-safety}

[Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/guardrails/) provides input and output guardrails that validate or block requests and responses, and Pydantic AI enforces usage, token, and request limits via [`UsageLimits`](../agent.md#usage-limits). As a community alternative bundling several ready-made shields, including USD cost tracking:

* [`pydantic-ai-shields`](https://github.com/vstorm-co/pydantic-ai-shields) - Ready-to-use guardrail capabilities: `CostTracking` (tracks token usage and USD cost per run, raises `BudgetExceededError` on budget overrun); `ToolGuard` (block or require approval for specific tools); `InputGuard` and `OutputGuard` (custom sync or async validation functions); `PromptInjection`, `PiiDetector`, `SecretRedaction`, `BlockedKeywords`, and `NoRefusals` content shields.

## File Operations & Sandboxing {#file-operations-sandboxing}

[Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) ships sandboxed [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/) and [`Shell`](https://pydantic.dev/docs/ai/harness/shell/) capabilities, plus [`CodeMode`](https://pydantic.dev/docs/ai/harness/code-mode/) for running tool calls as sandboxed Python. As a community alternative:

* [`pydantic-ai-backend`](https://github.com/vstorm-co/pydantic-ai-backend) - `ConsoleCapability` registers `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, and `execute` tools with a fine-grained permission system. Backends include `StateBackend` (in-memory, for testing), `LocalBackend` (real filesystem), `DockerSandbox` (isolated container execution), and `CompositeBackend` (routing across backends). Also available as a lower-level `ConsoleToolset`.

## Agent Skills

Pydantic AI supports [Agent Skills natively](on-demand.md#loading-skills-from-markdown-files) through [on-demand capabilities](on-demand.md), which collapse a skill to a one-line catalog entry until the model loads it. As a community alternative:

* [`pydantic-ai-skills`](https://github.com/DougTrajano/pydantic-ai-skills) - `SkillsCapability` implements Agent Skills support with progressive disclosure (load skills on-demand to reduce tokens). Supports filesystem and programmatic skills; compatible with [agentskills.io](https://agentskills.io).

## Data & Analytics {#data-analytics}

Capabilities for querying and analyzing structured data help agents answer questions over files and databases:

* [`pydantic-ai-chdb`](https://github.com/chdb-io/pydantic-ai-chdb) - `ChDBCapability` gives agents analytical SQL over local files (Parquet/CSV/JSON), object storage, and remote databases with [chDB](https://clickhouse.com/docs/en/chdb), the in-process ClickHouse engine — the engine itself needs no server or connection string to run (remote sources are reached via ClickHouse table functions, which take their own credentials). Registers `run_select_query` (read-only ClickHouse SQL with parameter binding), `list_databases`, `list_tables`, `describe_table`, `get_sample_data`, `list_functions`, and `attach_file` (opt-in writable sessions) tools plus schema-first instructions. Sessions default to the engine-level `readonly=2` setting with capped results, and typed engine errors are mapped to [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] so the model can correct its queries. Works with [agent specs](../agent-spec.md) out of the box, so it can be loaded via [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] / [`Agent.from_spec`][pydantic_ai.agent.Agent.from_spec]. Also available as a lower-level [toolset](../toolsets.md) via [`ChDBCapability(...).get_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_toolset].

## Retrieval & RAG {#retrieval-rag}

Pydantic AI demonstrates the retrieval pattern in the [RAG example](../examples/rag.md), where a search tool is registered against a vector database. The following community-developed packages provide reusable implementations of this pattern as capabilities, bundling retrieval tools with citation handling and instructions:

* [`haiku.rag`](https://github.com/ggozad/haiku.rag) - Local-first document RAG for Pydantic AI agents, backed by embedded [LanceDB](https://lancedb.com/) with no database server required. `RAGCapability` provides hybrid vector and full-text search, optional reranking, structure-aware context expansion, and citations to page numbers and section headings. Each document's [`DoclingDocument`](https://docling-project.github.io/docling/concepts/docling_document/) structure is preserved, which is what the context expansion and the page and section citations resolve against. Retrieved figures are passed to vision-capable models as images, and with a multimodal embedder they share the text vector space, so a text query can retrieve a figure and an image can be the query. For questions requiring computation across a corpus, `AnalysisCapability` combines search and citations with model-written Python executed in a [Monty](https://github.com/pydantic/monty) sandbox over a virtual document filesystem. `EvidenceCompactionCapability` keeps multi-turn requests smaller by retaining previously cited evidence while replacing uncited earlier search results, and `CitationPolicyCapability` requires answers to explicitly declare whether, and by what, they are grounded. The retrieval and analysis capabilities support deferred loading. The package also exposes lower-level [toolsets](../toolsets.md) through `create_search_toolset()` and `create_document_toolset()`.

To add your package to this page, open a pull request.

To publish your own capability package, see [Publishing capabilities](custom.md#publishing-capabilities) and [Extensibility](../extensibility.md).
