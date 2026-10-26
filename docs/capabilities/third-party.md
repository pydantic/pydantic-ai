---
description: "Find community capability packages for Pydantic AI, covering multi-agent orchestration, Agent Skills, SQL analytics and RAG."
---

# Third-Party Capabilities

[Capabilities](overview.md) are the recommended way for third-party packages to extend Pydantic AI, since they can bundle tools with hooks, instructions, and model settings. See [Extensibility](../extensibility.md) for the full ecosystem, including [third-party toolsets](../toolsets.md#third-party-toolsets) that can also be wrapped as capabilities.

Before reaching for a package here, check the [capability index](overview.md#available-capabilities), which covers Pydantic AI and [Pydantic AI Harness](../harness/index.md), the official capability library, including planning, compaction, sub-agents, guardrails, file system and shell access, and sandboxes. This page lists community packages for use cases those don't cover, or that build on them.

## Multi-Agent Orchestration

Pydantic AI Harness's [`SubAgents`](../harness/subagents.md) delegates tasks to a roster of named sub-agents, in the foreground or, with managed delegation sessions, in the background. For an orchestrator the model drives itself:

* [`subagents-pydantic-ai`](https://github.com/vstorm-co/subagents-pydantic-ai) by [Vstorm](https://vstorm.co/) - `SubAgentCapability` lets the model create specialist sub-agents at run time (`create_agent`, or one-shot `delegate`) and manage background tasks through its own tools: `check_task` and `wait_tasks` to collect results, `send_message_to_subagent` to steer a running task, `answer_subagent` to reply when a sub-agent asks for clarification, and `soft_cancel_task` and `hard_cancel_task`. Also available as a lower-level toolset via `create_subagent_toolset`.

## Agent Skills

Pydantic AI Harness's [`Skills`](../harness/skills.md) turns each `SKILL.md` package in a workspace directory into an [on-demand capability](on-demand.md) that the model loads when it needs it. Building on `Skills`:

* [`pydantic-ai-skills`](https://github.com/DougTrajano/pydantic-ai-skills) by [Doug Trajano](https://github.com/DougTrajano) - `SkillsCapability` hands `SKILL.md` discovery, validation, and instruction rendering to `Skills` and adds what it leaves out. `read_skill_resource` and `run_skill_script` tools reach a loaded skill's bundled `references/`, `assets/`, and `scripts/` files, with `${SKILL_DIR}` placeholders substituted for real paths, and scripts run through a pluggable executor with container and virtual-filesystem sandboxes. Remote skill libraries come from `GitSkillsRegistry` and `S3SkillsRegistry`, and skills defined in Python can sit in the same catalog as `SKILL.md` skills.

## Data & Analytics {#data-analytics}

Capabilities for querying and analyzing structured data help agents answer questions over files and databases:

* [`pydantic-ai-chdb`](https://github.com/chdb-io/pydantic-ai-chdb) - `ChDBCapability` gives agents analytical SQL over local files (Parquet/CSV/JSON), object storage, and remote databases with [chDB](https://clickhouse.com/docs/en/chdb), the in-process ClickHouse engine — the engine itself needs no server or connection string to run (remote sources are reached via ClickHouse table functions, which take their own credentials). Registers `run_select_query` (read-only ClickHouse SQL with parameter binding), `list_databases`, `list_tables`, `describe_table`, `get_sample_data`, `list_functions`, and `attach_file` (opt-in writable sessions) tools plus schema-first instructions. Sessions default to the engine-level `readonly=2` setting with capped results, and typed engine errors are mapped to [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] so the model can correct its queries. Works with [agent specs](../agent-spec.md) out of the box, so it can be loaded via [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] / [`Agent.from_spec`][pydantic_ai.agent.Agent.from_spec]. Also available as a lower-level [toolset](../toolsets.md) via [`ChDBCapability(...).get_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_toolset].

## Retrieval & RAG {#retrieval-rag}

Pydantic AI demonstrates the retrieval pattern in the [RAG example](../examples/rag.md), where a search tool is registered against a vector database. The following community-developed packages provide reusable implementations of this pattern as capabilities, bundling retrieval tools with citation handling and instructions:

* [`haiku.rag`](https://github.com/ggozad/haiku.rag) - Local-first document RAG for Pydantic AI agents, backed by embedded [LanceDB](https://lancedb.com/) with no database server required. `RAGCapability` provides hybrid vector and full-text search, optional reranking, structure-aware context expansion, and citations to page numbers and section headings. Each document's [`DoclingDocument`](https://docling-project.github.io/docling/concepts/docling_document/) structure is preserved, which is what the context expansion and the page and section citations resolve against. Retrieved figures are passed to vision-capable models as images, and with a multimodal embedder they share the text vector space, so a text query can retrieve a figure and an image can be the query. For questions requiring computation across a corpus, `AnalysisCapability` combines search and citations with model-written Python executed in a [Monty](https://github.com/pydantic/monty) sandbox over a virtual document filesystem. `EvidenceCompactionCapability` keeps multi-turn requests smaller by retaining previously cited evidence while replacing uncited earlier search results, and `CitationPolicyCapability` requires answers to explicitly declare whether, and by what, they are grounded. The retrieval and analysis capabilities support deferred loading. The package also exposes lower-level [toolsets](../toolsets.md) through `create_search_toolset()` and `create_document_toolset()`.

To add your package to this page, open a pull request.

To publish your own capability package, see [Publishing capabilities](custom.md#publishing-capabilities) and [Extensibility](../extensibility.md).
