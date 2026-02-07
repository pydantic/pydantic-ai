# Pydantic AI v2 Traits API: Research Report & Conceptual Design

> **Prepared for**: DouweM / pydantic-ai core team
> **Date**: February 2026
> **Scope**: Deep research on community projects, external frameworks, and conceptual design for a Traits API

---

## Table of Contents

1. [Framework Reports](#1-framework-reports)
   - [1A. Community Pydantic AI Projects](#1a-community-pydantic-ai-projects)
   - [1B. External Frameworks](#1b-external-frameworks)
2. [Comparison Matrix](#2-comparison-matrix)
3. [Conceptual Traits API Proposal](#3-conceptual-traits-api-proposal)
4. [10-Line Agent Example](#4-10-line-agent-example)
5. [Migration Path](#5-migration-path)

---

## 1. Framework Reports

### 1A. Community Pydantic AI Projects

#### Code Puppy (`mpfaffenberger/code_puppy`)

**Overview**: A CLI-based agentic code generation assistant ("the sassy AI code agent that makes IDEs look outdated") built entirely on pydantic-ai. Terminal-first, supports 65+ model providers, strong personality with strict coding discipline.

**Architecture**:

- **`BaseAgent`**: Abstract base defining `name`, `display_name`, `description`, `get_system_prompt()`, `get_available_tools()`. Handles model loading with fallback chains, MCP integration, token estimation (`len(text)/2.5` heuristic), message compaction (both summarization and truncation modes), and lifecycle callbacks (`on_agent_run_start`, `on_agent_run_end`).
- **`JSONAgent`**: Declarative agent definitions via JSON files. Required fields: `name`, `description`, `system_prompt`, `tools`. Tools resolve against both built-in registries and a Universal Constructor registry (prefixed with `uc:`).
- **Agent Manager**: Registry with `_AGENT_REGISTRY`, `_AGENT_HISTORIES`, `_CURRENT_AGENT`. Discovery in three phases: Python class auto-discovery, sub-package scanning, JSON file discovery.
- **26 specialized agents**: language-specific reviewers, planning agent, QA expert, security auditor, scheduler, pack leader (orchestrator), agent creator.

**Composition Model**: Agents compose via `invoke_agent` tool delegation with per-invocation session isolation via `contextvars`, persistent per-session conversation history (pickle files), automatic DBOS workflow wrapping when enabled, and MCP server inheritance.

**Key Capabilities**:
- **12+ tools**: file ops, shell execution (with safety callbacks, timeout, Ctrl-C/Ctrl-X handling, background process support), agent delegation, human-in-the-loop, skills system, reasoning display
- **DBOS durable execution**: Checkpoints agent interactions in SQLite/cloud DB, automatic workflow recovery, each sub-agent invocation gets a unique workflow ID
- **Skills system**: YAML frontmatter `SKILL.md` files with name, description, tags, version, author; enable/disable; search by name/description/tags
- **MCP integration**: Via `/mcp` command, async loading + caching, special handling when DBOS enabled
- **Round-robin model distribution**: Distributes requests across model instances to overcome rate limits
- **Custom commands**: Markdown files in `.claude/commands/`, `.github/prompts/`, or `.agents/commands/` become slash commands
- **Message compaction**: Verifies all tool calls have matching returns before compacting (critical invariant)

**Patterns relevant to traits**:
1. Agent-as-registry pattern with auto-discovery from directories (maps to discoverable traits)
2. Selective tool access per agent via `get_available_tools()` (scoped capability grants)
3. Message compaction with tool-call integrity checking
4. Plugin callback system (`on_agent_run_start`/`on_agent_run_end`)
5. AGENTS.md rule files appended to system prompts (instruction injection)

**Strengths**: Mature, battle-tested, 65+ provider support, practical DBOS integration, smart compaction.
**Gaps**: Tight CLI coupling, no dependency declaration between agents, no conflict detection, flat JSON schema, basic skills compared to others, local-only execution.

---

#### PAI Agent SDK (`youware-labs/pai-agent-sdk`)

**Overview**: Enterprise-grade application framework layering protocol-based abstractions on pydantic-ai. Focuses on environment abstraction, human-in-the-loop workflows, and hierarchical agent composition. Provides what pydantic-ai intentionally leaves to the user: session management, environment portability, tool approval, and composition.

**Architecture**:

- **`AgentRuntime`**: Dataclass and async context manager bundling `env` (Environment), `ctx` (AgentContext), and `agent` (pydantic-ai Agent). Created via `create_agent()` factory.
- **`BaseTool`**: Abstract base class — the most trait-like tool design of any community project:
  ```python
  class BaseTool(ABC):
      name: str
      description: str
      auto_inherit: bool = False       # Automatic subagent inclusion
      def is_available(self, ctx) -> bool       # Runtime availability
      def get_instruction(self, ctx) -> InstructionResult  # Dynamic guidance
      def get_approval_metadata(self) -> dict | None       # HITL metadata
      async def call(self, ctx, *args, **kwargs) -> Any
      async def process_user_input(self, ctx, user_input) -> UserInputPreprocessResult | None
  ```
- **`BaseToolset`**: Extends `AbstractToolset` with `get_instructions()` returning sync or async strings.
- **`InstructableToolset`** (Protocol): Duck-typing protocol — any object with `get_instructions()` can provide system prompt injections.
- **`Instruction`** model: Has `group` (deduplication key) and `content` fields — prevents duplicate prompt sections when multiple tools/toolsets provide similar guidance.
- **Environment protocol**: `LocalEnvironment`, `LocalFileOperator`, `LocalShell` + `DockerEnvironment`, `DockerShell` (containerized execution).
- **`MessageBus`**: Inter-agent communication with typed `BusMessage`.
- **`TaskManager`**: Background task management with `Task` and `TaskStatus`.

**Composition Model**: Four mechanisms:
1. Toolset layering: core + subagent tools + user toolsets + environment toolsets
2. Subagent delegation: Markdown configs with YAML frontmatter (`name`, `description`, `instruction`, `tools`, `model`)
3. Factory functions: `load_builtin_subagent_tools()` (individual) or `load_builtin_unified_subagent_tool()` (merged "delegate" tool)
4. Stream merging: `AgentStreamer` merging events from main agent + all subagents

**Key Capabilities**:
- **Rich tool abstraction**: `is_available()`, `get_instruction()`, `get_approval_metadata()`, `auto_inherit`, `process_user_input()`
- **Hooks at multiple levels**: Per-tool pre/post hooks (`PreHookFunc`, `PostHookFunc`) + global hooks (`GlobalPreHookFunc`, `GlobalPostHookFunc`) with `CallMetadata` for execution context
- **HITL**: `UserInteraction` for approval workflows, `get_approval_metadata()` on tools, `process_user_input()` for validating/transforming user responses
- **Session management**: `ResumableState` for cross-process persistence
- **Model management**: `ModelCapability` enum, `ModelConfig`, `ModelWrapper`, presets by thinking level
- **Streaming**: Full real-time event streaming with lifecycle events, loop indexing, phase distinction

**Patterns relevant to traits**:
1. `BaseTool` as rich abstraction (conditional availability, dynamic instructions, HITL, auto-inherit) — **most trait-like design**
2. Instruction deduplication via `group` field
3. Hook architecture at multiple levels (per-tool + global)
4. Environment protocol for execution portability ("what to do" vs "where to do it")
5. `InstructableToolset` protocol (duck-typing for instruction contribution)
6. Selective tool inheritance for subagents via `tools` field in configs

**Strengths**: Most architecturally sophisticated, protocol-based, rich tool abstraction, env portability, instruction dedup, comprehensive hooks, strong typing (pyright-validated).
**Gaps**: No conflict detection, no YAML/JSON serialization for agent configs, no declarative composition (still imperative), no formal trait dependency resolution.

---

#### Pydantic Deep Agents (`vstorm-co/pydantic-deepagents`)

**Overview**: Framework for "Claude Code-style" autonomous agents in minimal Python. Factory-pattern approach via `create_deep_agent()` assembling capabilities through feature flags. Philosophy: "10 lines of Python" for a production-grade agent.

**Architecture**:

- **`create_deep_agent()`**: Central factory with feature flags: `include_todo`, `include_filesystem`, `include_subagents`, `include_skills`, `include_execute`, `include_general_purpose_subagent`. Assembles in fixed order: TodoToolset → Console/filesystem → SubAgentToolset → SkillsToolset → user toolsets. Builds pydantic-ai `Agent` with all toolsets, adds dynamic `@agent.instructions` decorator.
- **`DeepAgentDeps`**: Dependency container with `backend`, `files`, `todos`, `subagents`, `uploads`. Provides `clone_for_subagent()` for isolated copy with shared backend.
- **`SkillsToolset`**: `list_skills`, `load_skill`, `read_skill_resource` tools. **Progressive disclosure**: only YAML frontmatter loaded initially, full instructions loaded on-demand.
- **Backend abstraction**: `StateBackend` (in-memory), `LocalBackend` (file-based), `DockerSandbox` (isolated with permissions), `CompositeBackend` (multi-backend).

**Key Capabilities**:
- **Planning**: `TodoToolset` with subtask dependencies, cycle detection, PostgreSQL storage, webhook events. Todo state reflected in dynamic system prompts.
- **Filesystem**: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`. Conditional approval for writes and execution.
- **Subagents**: Sync and async modes with background task management. Each subagent gets isolated dependencies via `clone_for_subagent()`.
- **Skills**: YAML frontmatter `SKILL.md` files, directory-based discovery, progressive disclosure, resource file access, path traversal protection.
- **Summarization**: Two processors — `SummarizationProcessor` (LLM-based, token/message/fraction triggers) and `SlidingWindowProcessor` (zero-cost truncation).
- **HITL**: Declarative `interrupt_on` dict mapping tool names to boolean flags.
- **File uploads**: `run_with_files()` helper with metadata tracking.

**Patterns relevant to traits**:
1. Factory with feature flags — precursor to trait composition, but with a **fixed** capability set
2. Progressive skill disclosure — load metadata first, full content on demand
3. Dynamic instructions decorator — assembling runtime-dependent prompt sections from active capabilities
4. Backend protocol abstraction for execution portability
5. `clone_for_subagent()` dependency isolation
6. Conditional capability inclusion (auto-detect sandbox for execute)
7. Declarative `interrupt_on` for HITL

**Strengths**: Clean "10 lines" API, modular packages, good progressive disclosure, backend abstraction, 100% test coverage.
**Gaps**: Fixed capability set (can't add new types without modifying factory), feature flags don't scale, no dependency resolution, no conflict detection, hardcoded `toolsets_factory` for subagents.

---

#### Mike's Traits SDK Design Doc (`mpfaffenberger/pydantic-ai` PR #1)

**Overview**: Design proposal (February 4, 2026) identifying convergence across the three community projects and proposing a unified `Trait` abstraction. Uses a GPU architecture analogy: current agent capabilities (skills, subagents, compaction) are siloed like 1990s fixed-function graphics pipelines. Traits provide "unified shaders for agents."

**Proposed Core Interface**:

A Trait declares:
- `id` — unique identifier (e.g., `"filesystem"`, `"shell"`)
- `requires` — prerequisite trait IDs
- `conflicts_with` — mutually exclusive trait IDs
- `get_toolset()` — tools provided
- `get_system_prompt()` — dynamic prompt contributions
- `get_history_processor()` — message history manipulation
- `on_agent_start()` / `on_agent_end()` — lifecycle hooks

**Composition Runtime**:
1. Dependency validation (verify all `requires` satisfied)
2. Conflict detection (check no `conflicts_with` pairs both present)
3. Topological sorting (order by dependency graph)
4. Automatic merging (toolsets combined, history processors chained, prompts concatenated)

Traits can modify other traits: e.g., `ApprovalTrait` wraps all other traits' toolsets with approval gates.

**Two Critical Constraints**:
1. **Serialization requirement**: Traits must be expressible as pure data (YAML/JSON) — enables marketplaces, git-tracked configs, UI-based assembly
2. **Runtime dependency injection**: Static configs can't satisfy OAuth tokens, tenant DBs — traits need per-request context injection

**15+ Proposed Built-In Traits**:

| Category | Trait | Description |
|----------|-------|-------------|
| Filesystem | `FileSystemTrait` | File read/write/edit/list with ignore patterns and size limits |
| Filesystem | `GrepTrait` | Code search with ripgrep |
| Execution | `ShellTrait` | Shell commands with timeout and destructive-command confirmation |
| Execution | `PythonExecTrait` | Python code execution in sandbox |
| Composition | `SubAgentTrait` | Subagent delegation with directory-based configs |
| Composition | `SwarmTrait` | Multi-agent swarm coordination |
| Memory | `MemoryTrait` | Cross-session memory persistence |
| Memory | `CompactionTrait` | Conversation compaction (smart/truncation strategies) |
| Memory | `SessionTrait` | Session persistence and resumption |
| Skills | `SkillsTrait` | Modular capability packages from directories |
| Interaction | `UserInteractionTrait` | Ask-user-question tool |
| Interaction | `ReasoningTrait` | Transparent reasoning display |
| Safety | `ApprovalTrait` | Tool approval workflows |
| Safety | `SandboxTrait` | Docker-based isolated execution |
| Integration | `MCPTrait` | MCP server integration |
| Integration | `BrowserTrait` | Browser automation |

**Usage Pattern**:
```python
agent = Agent(
    "anthropic:claude-opus-4-5",
    traits=[
        FileSystemTrait(ignore_patterns=[".git", "node_modules"]),
        ShellTrait(timeout=60, confirm_destructive=True),
        SubAgentTrait(subagent_dir="~/.myagent/subagents"),
        CompactionTrait(strategy="smart", token_threshold=160_000),
        SkillsTrait(directories=["~/.myagent/skills"]),
        ApprovalTrait(mode="writes"),
    ],
)
```

**Strengths**: Addresses real convergence, serialization-first, dependency resolution, conflict detection, declarative, lifecycle hooks, well-grounded conceptual framework.
**Gaps**: No reference implementation, interaction between `ApprovalTrait` wrapping and topological sort needs spec, serialization of runtime-only resources needs detail, no trait versioning for marketplace, no testing strategy for trait interactions.

---

### 1B. External Frameworks

#### Claude Code

**Overview**: Anthropic's production CLI agent. Its architecture is deeply relevant because it represents the "target experience" — traits should make building a Claude Code-style agent trivial.

**Skills System**:
- Markdown files with YAML frontmatter (`name`, `description`, `tags`, `version`, `author`)
- Stored in `.claude/skills/`, `~/.claude/skills/`, or `.agents/skills/`
- **Progressive disclosure**: Only short descriptions shown to model; full instructions loaded on-demand via `activate_skill` tool
- Skills can include resource files accessible via `read_skill_resource`

**Hooks System**:
- Event-driven: `PreToolUse`, `PostToolUse`, `Notification`, `Stop`, `SubagentStop`
- Each hook runs as a shell command, receives JSON context via stdin
- Return codes: 0 = proceed, 2 = block (with optional message), else error
- Hooks can auto-approve or auto-deny tool use — effectively a permission system

**Context Compaction**:
- Auto-triggered at ~140K tokens (of 200K context window)
- LLM-based summarization preserving decisions and code changes
- Subagent transcripts compacted independently
- **Critical insight**: Compaction must respect tool call integrity (all tool calls must have matching returns)

**Subagents**:
- Markdown files with YAML frontmatter in `.claude/agents/` or `~/.claude/agents/`
- Can configure: `tools` (allowlist) or `disallowedTools` (denylist), `permissionMode`, `model`, custom hooks and skills
- Independent execution context and compaction

**Permissions/Guardrails**: Multi-layer: global mode → per-subagent mode → hook-based dynamic control → tool allowlists/denylists.

**Patterns for traits**:
1. Progressive prompt disclosure (trait as lazy-loaded instruction segment)
2. Lifecycle hooks as first-class (pre/post tool use with block/allow control)
3. Compaction as configurable strategy with integrity guarantees
4. Subagent definitions as markdown+YAML configs (serialization reference)

---

#### OpenAI Agents SDK

**Overview**: Lightweight, Python-first multi-agent orchestration. Minimal abstractions with clear composition.

**Guardrails** (signature feature):
- `InputGuardrail`: Receives `RunContextWrapper` + input, returns `GuardrailFunctionOutput` with `tripwire_triggered` boolean. Can run **in parallel** with agent execution (default) or **blocking** (`run_in_parallel=False`).
- `OutputGuardrail`: Same structure, runs on final response. Can replace output or trigger tripwire.
- **Tripwire mechanism**: Raises exception, immediately halts execution. Fail-fast pattern.
- **Parallel execution**: Input guardrails run concurrently with agent generation. If guardrail triggers mid-generation, agent is terminated.

**Handoffs**:
- `Handoff(agent=target_agent)` creates a tool-like `transfer_to_<agent_name>`
- `input_filter`: Filters context passed to next agent (remove old messages, strip tool outputs)
- `on_handoff`: Callback for data prefetching
- **Replaces current agent** (not delegation — handoff)
- Nested handoffs collapse prior transcript into summary

**Patterns for traits**:
1. Guardrails as traits with parallel/blocking execution modes
2. Tripwire pattern (immediate halt on failure) — clean fail-fast
3. Input/output filtering on handoffs for context management
4. `run_in_parallel` flag — configurable per-guardrail latency vs. cost tradeoff

---

#### Google ADK (Agent Development Kit)

**Overview**: Comprehensive framework emphasizing hierarchical agent composition and callback-driven extensibility.

**Agent Type Hierarchy**: `LlmAgent`, `SequentialAgent`, `ParallelAgent`, `LoopAgent`, `CustomAgent` — all inherit from `BaseAgent`, enabling uniform composition at any nesting depth.

**Callbacks** (6 types in 3 pairs):
- `before_agent_callback` / `after_agent_callback`
- `before_model_callback` / `after_model_callback`
- `before_tool_callback` / `after_tool_callback`

**Key design pattern**: **Returning `None` from a `before_*` callback means "proceed normally"; returning an object means "skip the step and use this as the result."** This enables short-circuiting (cached responses), input validation, output transformation, and guardrails — all through one mechanism.

**Session/State/Memory Separation**:
- `Session`: Single conversation with history (`Events`) and working memory (`State`)
- `State`: Key-value store with scope prefixes (`app:`, `user:`, `session:`, `temp:`)
- `Memory`: Cross-session recall via `MemoryService`

**Artifacts**: Named, versioned binary/text objects. Stored as `types.Part` with `inline_data` (bytes + MIME type). Auto-versioned. Scoped by app/user/session/filename. Designed for large data that doesn't belong in session state.

**Patterns for traits**:
1. `before_*/after_*` callback pairs — clean, composable lifecycle hooks
2. "Return `None` to proceed, return value to short-circuit" — elegant pattern for all hooks
3. Session state vs. memory separation
4. Artifact management (versioned binary outputs)

---

#### Vercel AI SDK

**Overview**: TypeScript-first toolkit with middleware-driven extensibility and lightweight agent loops.

**Language Model Middleware** (decorator pattern):
- `transformParams`: Modify request before model (RAG injection, input guardrails)
- `wrapGenerate`: Wrap non-streaming call (caching, logging, post-processing)
- `wrapStream`: Wrap streaming call
- Built-in: `extractReasoningMiddleware`, `simulateStreamingMiddleware`, `defaultSettingsMiddleware`
- Middleware stacks — each independent of model provider

**Agent Loop Control**:
- `stopWhen`: Stop conditions (step count, no more tool calls, custom predicates)
- `prepareStep`: Modify settings between steps — change models, tools, messages per-step

**Tool Composition**: `toModelOutput` separates what the tool returns to the application from what goes back to the model (information asymmetry).

**Patterns for traits**:
1. Model middleware as trait pattern (`transformParams`, `wrapGenerate`, `wrapStream`)
2. Step-level configuration via `prepareStep` (dynamic per-step model/tool switching)
3. Stop conditions as pluggable predicates
4. Tool output separation (app-facing vs. model-facing)

---

#### CrewAI

**Overview**: Role-based multi-agent framework with tasks and crews.

**Key Concepts**: Agents defined via role/goal/backstory. Tasks with expected output descriptions. Crews as teams with delegation. Process types: sequential, hierarchical.

**Patterns for traits**:
1. Role/goal/backstory as structured instruction generation
2. Task delegation with expected output contracts
3. Hierarchical process as orchestration pattern

---

#### LangChain/LangGraph

**Overview**: State machine execution with checkpointing.

**Key Concepts**: Nodes (functions), edges (transitions), state (shared TypedDict). Checkpointing for human-in-the-loop. Subgraphs for modular composition. Memory via separate persistence layer.

**Patterns for traits**:
1. Checkpointing as durable execution primitive
2. Subgraphs as composable agent modules
3. Typed state for cross-step communication

---

#### Agent Capability Standard (`synaptiai/agent-capability-standard`)

**Overview**: Open specification defining 36 atomic capabilities across 9 cognitive layers. Philosophy: "Grounded Agency" — agent reliability should be structural, not optional.

**9 Cognitive Layers**:

| Layer | Description | Example Capabilities |
|-------|-------------|---------------------|
| PERCEIVE | Data acquisition | `retrieve`, `search`, `observe` |
| UNDERSTAND | Interpretation | `detect`, `classify`, `extract`, `parse` |
| REASON | Logical processing | `plan`, `decide`, `infer`, `evaluate` |
| MODEL | World representation | `state`, `relate`, `abstract` |
| SYNTHESIZE | Content creation | `generate`, `compose`, `transform`, `summarize` |
| EXECUTE | Action | `mutate`, `invoke`, `deploy`, `configure` |
| VERIFY | Validation | `verify`, `test`, `audit`, `validate` |
| REMEMBER | Persistence | `checkpoint`, `persist`, `index`, `cache` |
| COORDINATE | Multi-agent | `delegate`, `negotiate`, `synchronize`, `broadcast` |

**Core principles**: Grounded claims (evidence-backed), auditable transforms, safety by construction (mutations require checkpoints), composable atoms (one trait one concern), explicit I/O contracts.

**Patterns for traits**:
1. Cognitive layers as trait taxonomy/organization principle
2. Atomicity principle — one trait, one concern
3. Explicit I/O contracts for traits
4. Safety by construction (mutations require checkpoints)

---

#### Durable Execution as Traits

**Current pydantic-ai integrations**:
- `TemporalAgent` extends `WrapperAgent` — freezes model+toolsets at construction, creates `TemporalModel` routing API calls through activities, wraps each toolset via `temporalize_toolset()`, uses `_temporal_overrides()` context manager
- `DBOSAgent` extends `WrapperAgent` — creates `DBOSModel` to checkpoint model requests as DBOS steps, replaces `MCPServer` instances with `DBOSMCPServer` wrappers, wraps `run()`/`run_sync()` as `@DBOS.workflow`

**Both follow the same pattern**: transparently modify agent behavior by (1) wrapping the model to route through durable infrastructure, (2) wrapping toolsets to checkpoint tool execution, (3) constraining certain operations.

**Assessment: Natural fit for traits.** The existing implementations already follow the trait pattern. A `TemporalTrait()` would provide:
- Activity wrapping for model requests and tool calls
- Workflow definition wrapping `agent.run()`
- State persistence via Temporal's replay mechanism
- Configuration (activity timeouts, retry policies, per-tool activity config)

The mapping is natural because the behavior is **additive** (adds durability without changing core logic), **composable** (could combine with guardrail/hook traits), and has a **clean boundary** (only touches I/O paths).

The `WrapperAgent` approach has one key limitation: combining `TemporalAgent` with other wrapper-based behaviors requires nesting wrappers. A traits system solves this:

```python
# Current (nesting required)
agent = Agent('openai:gpt-5')
temporal_agent = TemporalAgent(agent)
# How to also add guardrails?

# With traits (flat composition)
agent = Agent('openai:gpt-5', traits=[
    TemporalTrait(activity_config=...),
    GuardrailTrait(input_guardrails=[...]),
])
```

---

## 2. Comparison Matrix

### Capability Coverage

| Capability | Code Puppy | PAI Agent SDK | Deep Agents | Mike's Design | Claude Code | OpenAI Agents | Google ADK | Vercel AI |
|-----------|-----------|--------------|------------|--------------|------------|--------------|-----------|----------|
| **File operations** | Yes | Yes (via Env) | Yes | FileSystemTrait | Yes | - | - | - |
| **Shell execution** | Yes (signals) | Yes (Docker) | Yes (sandbox) | ShellTrait | Yes | - | - | - |
| **Subagent delegation** | Yes (invoke) | Yes (unified) | Yes (sync/async) | SubAgentTrait | Yes (md) | Handoffs | Agent tree | - |
| **Skills/lazy prompts** | Yes (YAML) | Partial | Yes (YAML) | SkillsTrait | Yes (YAML) | - | - | - |
| **Conversation compaction** | Yes (2 modes) | Via processors | Yes (2 modes) | CompactionTrait | Yes (LLM) | - | - | - |
| **HITL/Approval** | Yes (callbacks) | Yes (rich) | Yes (interrupt_on) | ApprovalTrait | Yes (hooks) | - | - | - |
| **Session persistence** | Yes (pickle) | Yes (resumable) | Via backend | SessionTrait | Yes | - | Session svc | - |
| **MCP integration** | Yes | Yes | - | MCPTrait | Yes | - | - | - |
| **Durable execution** | DBOS | - | - | - | - | - | - | - |
| **Input guardrails** | - | Via hooks | - | - | Via hooks | **Yes** (parallel) | Callbacks | Middleware |
| **Output guardrails** | - | Via hooks | - | - | Via hooks | **Yes** (tripwire) | Callbacks | Middleware |
| **Lifecycle hooks** | Start/end | Pre/post tool | - | Start/end | Pre/post tool | - | 6 types (3 pairs) | Middleware |
| **Model middleware** | - | - | - | - | - | - | before/after_model | **Yes** (core pattern) |
| **Dynamic per-step config** | - | - | - | - | - | - | - | prepareStep |
| **Environment abstraction** | - | **Yes** (local/Docker) | **Yes** (backends) | SandboxTrait | - | - | - | - |
| **Artifacts/file output** | - | - | - | - | - | - | **Yes** (versioned) | - |
| **Instruction dedup** | - | **Yes** (group) | - | - | - | - | - | - |
| **Conflict detection** | - | - | - | **Yes** | - | - | - | - |
| **Dependency resolution** | - | - | - | **Yes** | - | - | - | - |
| **Serializable config** | JSON agents | - | - | **Yes** (YAML) | YAML agents | - | - | - |
| **Tool output separation** | - | - | - | - | - | - | - | **Yes** (toModelOutput) |
| **Memory (cross-session)** | - | - | - | MemoryTrait | MEMORY.md | - | **Yes** (svc) | - |
| **Stop conditions** | - | - | - | - | - | - | - | **Yes** (pluggable) |

### Composition Style

| Framework | Style | Extension Point | Scaling |
|-----------|-------|----------------|---------|
| Code Puppy | Registry + JSON | New agent class/JSON file | Per-agent |
| PAI Agent SDK | Protocol + factory | Implement protocol | Per-tool/toolset |
| Deep Agents | Factory + feature flags | Modify `create_deep_agent()` | Fixed set |
| Mike's Design | Declarative + dependency graph | New Trait class + YAML | Open-ended |
| Claude Code | Filesystem conventions | New md file in directory | Open-ended |
| OpenAI Agents SDK | Constructor composition | Add to agent constructor | Per-agent |
| Google ADK | Hierarchical + callbacks | Callbacks on any agent | Unlimited nesting |
| Vercel AI SDK | Middleware stacking | Wrap model with middleware | Stackable |

---

## 3. Conceptual Traits API Proposal

### 3.1 What Is a Trait?

A **trait** is a composable, declarative agent capability that provides some combination of:
- **Tools** (via a toolset)
- **Instructions** (dynamic system prompt contributions)
- **History processors** (message history manipulation)
- **Lifecycle hooks** (before/after agent run, model request, tool call)
- **Guardrails** (input/output validation with halt or transform semantics)
- **Model middleware** (request/response transformation)
- **Configuration** (model settings, stop conditions)

A trait is a **superset of a toolset**. Every toolset is trivially a trait (one that only provides tools), but a trait can provide much more.

### 3.2 Proposed `Trait` Interface

```python
from abc import ABC, abstractmethod
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai._agent_graph import HistoryProcessor

class Trait(ABC, Generic[AgentDepsT]):
    """A composable agent capability."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this trait (e.g., 'filesystem', 'shell')."""
        ...

    @property
    def requires(self) -> Sequence[str]:
        """Trait IDs that must also be present. Default: none."""
        return ()

    @property
    def conflicts_with(self) -> Sequence[str]:
        """Trait IDs that are mutually exclusive with this trait. Default: none."""
        return ()

    def get_toolset(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Return a toolset providing this trait's tools, or None."""
        return None

    def get_instructions(self, ctx: RunContext[AgentDepsT]) -> str | None:
        """Return dynamic instructions to add to the system prompt, or None.

        Can return different instructions based on runtime state (e.g., current
        working directory, active todos, available skills).

        If multiple traits return instructions with the same `instruction_group`,
        only the first is kept (deduplication).
        """
        return None

    @property
    def instruction_group(self) -> str | None:
        """Deduplication key for instructions. Traits with the same non-None
        group will only contribute instructions once. Default: None (no dedup)."""
        return None

    def get_history_processor(self) -> HistoryProcessor[AgentDepsT] | None:
        """Return a history processor, or None."""
        return None

    # --- Lifecycle hooks ---

    async def on_agent_start(self, ctx: RunContext[AgentDepsT]) -> None:
        """Called before the agent run begins."""
        pass

    async def on_agent_end(self, ctx: RunContext[AgentDepsT]) -> None:
        """Called after the agent run completes."""
        pass

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], messages: list[ModelMessage]
    ) -> list[ModelMessage] | None:
        """Called before each model request. Return None to proceed normally,
        or return modified messages. (Inspired by Google ADK pattern.)"""
        return None

    async def after_model_response(
        self, ctx: RunContext[AgentDepsT], response: ModelResponse
    ) -> ModelResponse | None:
        """Called after each model response. Return None to proceed normally,
        or return a modified response."""
        return None

    async def before_tool_call(
        self, ctx: RunContext[AgentDepsT], tool_name: str, tool_args: dict[str, Any]
    ) -> dict[str, Any] | bool | None:
        """Called before each tool call.
        - Return None: proceed normally
        - Return dict: proceed with modified args
        - Return False: block the tool call (with optional message via exception)
        (Inspired by Claude Code hooks + Google ADK callbacks.)
        """
        return None

    async def after_tool_call(
        self, ctx: RunContext[AgentDepsT], tool_name: str, tool_args: dict[str, Any], result: Any
    ) -> Any | None:
        """Called after each tool call. Return None to use original result,
        or return a replacement result."""
        return None

    # --- Guardrails ---

    async def check_input(
        self, ctx: RunContext[AgentDepsT], user_input: str | Sequence[UserContent]
    ) -> GuardrailResult | None:
        """Input guardrail. Return None to pass, or a GuardrailResult to
        halt (tripwire) or transform the input."""
        return None

    async def check_output(
        self, ctx: RunContext[AgentDepsT], output: OutputDataT
    ) -> GuardrailResult | None:
        """Output guardrail. Return None to pass, or a GuardrailResult to
        halt (tripwire) or transform the output."""
        return None

    @property
    def run_guardrails_in_parallel(self) -> bool:
        """Whether to run this trait's guardrails in parallel with generation.
        Default: True (for latency). Set False for cost savings.
        (Inspired by OpenAI Agents SDK.)"""
        return True

    # --- Context management ---

    async def __aenter__(self) -> Self:
        """Async context manager entry (for resource setup)."""
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """Async context manager exit (for resource cleanup)."""
        return None
```

### 3.3 `GuardrailResult` Type

```python
@dataclass
class GuardrailResult:
    """Result from a guardrail check."""

    action: Literal['halt', 'transform', 'warn']
    """What to do:
    - 'halt': Stop execution immediately (tripwire, inspired by OpenAI SDK)
    - 'transform': Replace the input/output with `replacement`
    - 'warn': Log warning but proceed
    """
    message: str = ''
    """Explanation for the guardrail result."""
    replacement: Any = None
    """Replacement value when action is 'transform'."""
```

### 3.4 Composition Runtime

When an agent is constructed with `traits=[...]`, the runtime:

1. **Validates dependencies**: For each trait, verify all `requires` IDs are present in the trait list
2. **Detects conflicts**: Check no `conflicts_with` pairs are both present
3. **Topologically sorts**: Order traits by dependency graph (traits with no deps first)
4. **Merges toolsets**: `CombinedToolset` of all non-None `get_toolset()` results
5. **Chains instructions**: Concatenate all `get_instructions()` results with deduplication by `instruction_group`
6. **Chains history processors**: Compose all `get_history_processor()` results in order
7. **Registers hooks**: Collect all lifecycle hook implementations for dispatch
8. **Registers guardrails**: Collect `check_input`/`check_output` implementations with parallel/blocking config

**Hook dispatch order**: Traits are called in topological order. For "before" hooks, first trait in order is called first. For "after" hooks, reverse order (like middleware unwinding).

**Guardrail dispatch**: Input guardrails run before (or in parallel with) generation. If any returns `halt`, execution stops immediately. Output guardrails run after generation. Halts take priority over transforms.

### 3.5 Built-In Trait Catalog

Organized by category (inspired by Agent Capability Standard's cognitive layers adapted for practical use):

#### Execution Traits

| Trait | ID | Provides | Config |
|-------|----|----------|--------|
| `FileSystemTrait` | `filesystem` | Tools: `read_file`, `write_file`, `edit_file`, `list_files`, `glob` + Instructions: working directory, ignore patterns | `root_dir`, `ignore_patterns`, `max_file_size`, `require_approval_for_writes` |
| `GrepTrait` | `grep` | Tools: `grep` (ripgrep-based search) + Instructions: search guidance | `max_results` |
| `ShellTrait` | `shell` | Tools: `run_command` + Hooks: `before_tool_call` blocks destructive commands unless confirmed | `timeout`, `confirm_destructive`, `allowed_commands`, `blocked_commands` |
| `PythonExecTrait` | `python_exec` | Tools: `execute_python` | `sandbox` (None/Docker/E2B), `timeout` |
| `BrowserTrait` | `browser` | Tools: `navigate`, `click`, `extract`, `screenshot` | `headless`, `timeout` |

#### Composition Traits

| Trait | ID | Provides | Config |
|-------|----|----------|--------|
| `SubAgentTrait` | `subagent` | Tools: `delegate_to_agent`, `list_agents` + Instructions: available subagents | `subagent_dir`, `subagent_configs`, `unified_tool` (bool) |
| `HandoffTrait` | `handoff` | Tools: `transfer_to_<name>` per target + Hooks: input filtering on handoff | `targets`, `input_filter` |

#### Memory & Context Traits

| Trait | ID | Provides | Config |
|-------|----|----------|--------|
| `CompactionTrait` | `compaction` | History processor: auto-summarization or sliding window | `strategy` ("smart"/"truncate"/"sliding"), `token_threshold`, `preserve_recent_n` |
| `MemoryTrait` | `memory` | Tools: `save_memory`, `search_memories` + Instructions: memory usage guidance | `memory_dir`, `backend` |
| `SessionTrait` | `session` | Hooks: auto-save/restore session state | `session_store`, `auto_save` |
| `ArtifactTrait` | `artifact` | Tools: `save_artifact`, `list_artifacts` | `storage_backend`, `max_size` |

#### Knowledge Traits

| Trait | ID | Provides | Config |
|-------|----|----------|--------|
| `SkillsTrait` | `skills` | Tools: `list_skills`, `load_skill`, `read_skill_resource` + Instructions: available skill catalog (progressive disclosure) | `skill_dirs`, `auto_load` |
| `KnowsCurrentTime` | `knows_time` | Instructions: current date/time injected into system prompt | `timezone`, `format` |
| `KnowsUserDetails` | `knows_user` | Instructions: user name, preferences, role | `user_provider` (callable returning user info) |
| `KnowsProjectContext` | `knows_project` | Instructions: project description, tech stack, conventions | `context_file` (e.g., `CLAUDE.md` path) |

#### Safety Traits

| Trait | ID | Provides | Config |
|-------|----|----------|--------|
| `ApprovalTrait` | `approval` | Hooks: `before_tool_call` wraps other traits' tools with approval | `mode` ("writes"/"destructive"/"all"), `auto_approve_patterns` |
| `InputGuardrailTrait` | `input_guardrail` | Guardrail: `check_input` with custom validation | `validator_fn`, `parallel` (bool) |
| `OutputGuardrailTrait` | `output_guardrail` | Guardrail: `check_output` with custom validation | `validator_fn` |
| `SandboxTrait` | `sandbox` | Hooks: routes execution traits through sandbox + conflicts_with non-sandboxed execution | `sandbox_type` ("docker"/"e2b") |

#### Durability Traits

| Trait | ID | Provides | Config |
|-------|----|----------|--------|
| `TemporalTrait` | `temporal` | Model middleware: routes model requests through Temporal activities. Toolset wrapping: checkpoints tool execution. Workflow definition. | `activity_timeout`, `retry_policy`, `task_queue`, `per_tool_config` |
| `DBOSTrait` | `dbos` | Model middleware: checkpoints model requests as DBOS steps. MCP wrapping. | `conductor_key`, `app_version` |
| `PrefectTrait` | `prefect` | Similar to Temporal/DBOS pattern | `flow_config` |

#### Interaction Traits

| Trait | ID | Provides | Config |
|-------|----|----------|--------|
| `UserInteractionTrait` | `user_interaction` | Tools: `ask_user_question` | `timeout`, `default_response` |
| `ReasoningTrait` | `reasoning` | Tools: `share_reasoning` (transparency tool) | - |
| `TodoTrait` | `todo` | Tools: `read_todos`, `write_todos` + Instructions: current todo state | `storage_backend`, `enable_dependencies` |

#### Model Configuration Traits

| Trait | ID | Provides | Config |
|-------|----|----------|--------|
| `StopConditionTrait` | `stop_condition` | Configuration: pluggable stop predicates | `max_steps`, `custom_predicate` |
| `DynamicModelTrait` | `dynamic_model` | Hooks: `before_model_request` switches model based on step/context | `model_selector` (callable) |

### 3.6 Design Questions Answered

#### Should instructions be traits?

**Yes.** `KnowsCurrentTime`, `KnowsUserDetails`, `KnowsProjectContext` are traits that provide _only_ instructions (no tools, no hooks). This is valid because:
- They compose naturally with other traits
- They can have dependencies (e.g., `KnowsUserDetails` might require a user provider)
- They benefit from deduplication (`instruction_group`)
- They're serializable (can be expressed in YAML)
- They keep the Agent constructor clean — no need for separate `instructions` param when traits cover it

However, the `instructions` param on `Agent` should remain for simple cases where a full trait is overkill.

#### Should `CanOutput(T)` be a trait?

**Probably not.** Output type is a fundamental Agent type parameter (`Agent[AgentDepsT, OutputDataT]`) that affects type checking and validation at a level deeper than traits operate. However, a related concept could work:

- `OutputGuardrailTrait` — validates output _after_ generation
- `OutputTransformTrait` — post-processes output (e.g., format conversion)

The output _type_ itself should stay on the Agent constructor. What traits can do is _validate_ and _transform_ output through guardrails.

#### How do traits relate to existing `AbstractToolset`?

`Trait` is a superset of `AbstractToolset`. Every toolset can be used as-is within a trait via `get_toolset()`. The relationship:

```
Trait
├── get_toolset() → AbstractToolset (tools)
├── get_instructions() → str (system prompts)
├── get_history_processor() → HistoryProcessor (message manipulation)
├── lifecycle hooks (before/after agent/model/tool)
├── guardrails (input/output validation)
└── configuration (model settings, stop conditions)
```

Existing `AbstractToolset` subclasses, `FunctionToolset`, `MCPServer`, etc. all continue to work. You can pass them directly in the `toolsets=` param for simple cases, or wrap them in a trait for richer behavior.

#### How do traits relate to built-in tools?

Built-in tools (`WebSearchTool`, `CodeExecutionTool`, etc.) are a natural fit for traits:

```python
# Current
agent = Agent('openai:gpt-5', builtin_tools=[WebSearchTool()])

# With traits (these could be the same objects, implementing both interfaces)
agent = Agent('openai:gpt-5', traits=[WebSearchTrait(search_context_size='medium')])
```

The existing `AbstractBuiltinTool` already has `__init_subclass__` auto-registration. Built-in tools could implement `Trait` as well, giving them the ability to contribute instructions (search result formatting guidance) and hooks (rate limiting).

#### How do traits relate to MCP servers?

MCP servers are toolsets. They continue to work in `toolsets=`. For richer MCP behavior (instructions about available tools, authentication hooks), an `MCPTrait` wraps an MCP server with additional capabilities:

```python
agent = Agent('openai:gpt-5', traits=[
    MCPTrait(server=MCPServerHTTP('http://localhost:8080'), instructions="Database query tool...")
])
```

#### Serialization format?

**YAML as primary, JSON as alternative.** Traits serializable as data:

```yaml
# agent.yaml
model: "anthropic:claude-sonnet-4-5"
traits:
  - id: filesystem
    config:
      root_dir: "."
      ignore_patterns: [".git", "node_modules", "__pycache__"]
      max_file_size: 1_000_000
  - id: shell
    config:
      timeout: 60
      confirm_destructive: true
  - id: skills
    config:
      skill_dirs: ["~/.myagent/skills", "./.agent/skills"]
  - id: compaction
    config:
      strategy: smart
      token_threshold: 160_000
  - id: approval
    config:
      mode: writes
  - id: knows_time
  - id: knows_project
    config:
      context_file: "./CLAUDE.md"
```

Loading:
```python
agent = Agent.from_config("agent.yaml")
# or
agent = Agent.from_config({"model": "...", "traits": [...]})
```

For traits with runtime-only config (callables, connections), the YAML references a trait ID and the runtime provides the implementation:

```python
# Register custom trait implementations
trait_registry.register("my_custom_auth", MyAuthTrait)

# YAML references the registered ID
# - id: my_custom_auth
#   config:
#     provider: "oauth2"
```

#### How do guardrails (#1197), middleware (#2885), skills (#3365, #4144), fallbacks (#3212) map to traits?

| Issue | Trait Mapping |
|-------|--------------|
| #1197 Guardrails | `InputGuardrailTrait`, `OutputGuardrailTrait` — with parallel/blocking execution, tripwire/transform/warn semantics |
| #2885 Middleware/hooks | Lifecycle hooks on `Trait` base class (`before_model_request`, `after_model_response`, `before_tool_call`, `after_tool_call`) |
| #3365 Anthropic Skills | `SkillsTrait` — YAML frontmatter markdown files with progressive disclosure |
| #4144 Official skills | Built-in trait catalog with official trait implementations |
| #3212 Builtin tool fallback | `before_tool_call` hook can intercept failed built-in tool calls and route to custom implementations |
| #4159 Extension philosophy | Traits _are_ the extension philosophy — open, composable, discoverable |
| #530 Persist messages | `SessionTrait` for auto-save/restore of message history |
| #3056/#3180 Durable execution | `TemporalTrait`, `DBOSTrait`, `PrefectTrait` |

#### How does durable execution fit?

Durable execution traits (`TemporalTrait`, `DBOSTrait`) work at the **model middleware** and **toolset wrapping** layers:

1. `before_model_request` hook routes the model API call through a durable activity/step
2. Toolsets returned by other traits get wrapped via `visit_and_replace()` to checkpoint tool execution
3. The agent's `run()` method gets wrapped as a workflow/flow

This is exactly what `TemporalAgent` and `DBOSAgent` already do, but as composable traits rather than exclusive wrapper agents. The trait system's `visit_and_replace()` on `AbstractToolset` is the key mechanism — durability traits can wrap _any_ toolset provided by _any_ other trait.

### 3.7 Interaction Between Traits

**Ordering matters.** Consider:

```python
traits=[
    ApprovalTrait(mode="writes"),      # Wraps write tools with approval
    FileSystemTrait(root_dir="."),     # Provides write tools
    TemporalTrait(task_queue="main"),  # Wraps everything for durability
]
```

The runtime applies these in dependency order:
1. `FileSystemTrait` provides its toolset (tools exist)
2. `ApprovalTrait` wraps those tools via `before_tool_call` (approval gates added)
3. `TemporalTrait` wraps the final toolset via `visit_and_replace()` (durability added)

Result: Tool call → Temporal activity → Approval check → File operation → Temporal checkpoint.

---

## 4. 10-Line Agent Example

### The Target Developer Experience

```python
from pydantic_ai import Agent
from pydantic_ai.traits import (
    FileSystemTrait, ShellTrait, GrepTrait, SubAgentTrait,
    SkillsTrait, CompactionTrait, ApprovalTrait, TodoTrait,
)

agent = Agent(
    "anthropic:claude-sonnet-4-5",
    traits=[
        FileSystemTrait(ignore_patterns=[".git", "node_modules"]),
        GrepTrait(),
        ShellTrait(timeout=60, confirm_destructive=True),
        SubAgentTrait(subagent_dir="~/.myagent/agents"),
        SkillsTrait(skill_dirs=["~/.myagent/skills"]),
        CompactionTrait(strategy="smart", token_threshold=160_000),
        ApprovalTrait(mode="writes"),
        TodoTrait(),
    ],
)
```

This creates a Claude Code-style coding agent with:
- File operations with `.git`/`node_modules` exclusion
- Code search via ripgrep
- Shell execution with timeout and destructive-command confirmation
- Subagent delegation from markdown definitions
- Skills system with progressive disclosure
- Auto-compaction at 160K tokens
- Write approval for safety
- Todo tracking for planning

**Compare with current pydantic-ai** (requires ~100+ lines of custom toolsets, manual instruction assembly, custom history processors, and hand-rolled approval logic) or with the community projects (which each provide this but with different APIs and fixed capability sets).

### Variants

**YAML-first** (for deployment/marketplace):
```python
agent = Agent.from_config("coding-agent.yaml")
```

**With durable execution**:
```python
agent = Agent(
    "anthropic:claude-sonnet-4-5",
    traits=[
        FileSystemTrait(), ShellTrait(), GrepTrait(),
        CompactionTrait(strategy="smart"),
        TemporalTrait(task_queue="agents"),
    ],
)
```

**With guardrails** (enterprise):
```python
agent = Agent(
    "anthropic:claude-sonnet-4-5",
    traits=[
        FileSystemTrait(), ShellTrait(),
        InputGuardrailTrait(validator_fn=check_prompt_injection, parallel=True),
        OutputGuardrailTrait(validator_fn=check_pii_leakage),
        ApprovalTrait(mode="all"),
    ],
)
```

**Preset bundle**:
```python
from pydantic_ai.traits.presets import coding_agent_traits

agent = Agent("anthropic:claude-sonnet-4-5", traits=coding_agent_traits(yolo_mode=True))
```

---

## 5. Migration Path

### Principle: Fully Additive, Zero Breaking Changes

Traits are a new layer _on top of_ the existing API. Nothing is removed or deprecated.

### Phase 1: Trait Interface + First-Party Traits

1. **Introduce `Trait` base class** in `pydantic_ai.traits`
2. **Add `traits` parameter to `Agent.__init__`**: sits alongside existing `tools`, `toolsets`, `instructions`, `history_processors`
3. **Trait resolution**: At agent construction, traits are resolved into their component parts and merged with existing params
4. **Ship first-party traits**: `FileSystemTrait`, `ShellTrait`, `GrepTrait`, `CompactionTrait`, `ApprovalTrait` — covering the most common coding agent needs
5. **Existing code continues to work unchanged**: `toolsets=`, `tools=`, `instructions=`, `history_processors=` all work as before

```python
# Still works (no change)
agent = Agent('openai:gpt-5', tools=[my_tool], instructions="Be helpful")

# New option
agent = Agent('openai:gpt-5', traits=[FileSystemTrait(), ShellTrait()])

# Mix and match
agent = Agent(
    'openai:gpt-5',
    traits=[FileSystemTrait(), CompactionTrait()],
    tools=[my_custom_tool],             # Still works
    instructions="Additional guidance",  # Still works
    toolsets=[my_mcp_server],           # Still works
)
```

### Phase 2: Lifecycle Hooks + Guardrails

1. **Add hook dispatch** to the agent graph execution (before/after model, before/after tool)
2. **Add guardrail dispatch** with parallel/blocking execution support
3. **Ship safety traits**: `InputGuardrailTrait`, `OutputGuardrailTrait`, `SandboxTrait`
4. This addresses issues #1197 (Guardrails) and #2885 (Middleware/hooks)

### Phase 3: Durability + Knowledge Traits

1. **Migrate `TemporalAgent`/`DBOSAgent`** to also be available as `TemporalTrait`/`DBOSTrait` (existing wrapper agents continue to work)
2. **Ship knowledge traits**: `KnowsCurrentTime`, `KnowsProjectContext`, `SkillsTrait`
3. **Ship composition traits**: `SubAgentTrait`, `TodoTrait`

### Phase 4: Serialization + Marketplace

1. **Add `Agent.from_config()` / `Agent.to_config()`** for YAML/JSON serialization
2. **Trait registry** for custom trait discovery
3. **Trait versioning** for marketplace distribution
4. **Community trait packages** via PyPI (`pip install pydantic-ai-trait-github`)

### What Doesn't Change

- `Agent` constructor signature — `traits` is additive
- `AbstractToolset` and all subclasses — fully compatible
- `@agent.tool` and `@agent.tool_plain` decorators
- `builtin_tools` parameter
- `history_processors` parameter
- `instructions` and `system_prompt` parameters
- `RunContext` and dependency injection
- `WrapperAgent` pattern (still works, traits are an alternative)
- All model integrations
- All existing tests

### For Community Project Maintainers

The migration is straightforward:

**Code Puppy**: Each `BaseAgent` subclass maps to a trait. `JSONAgent` maps to YAML-serialized trait configs. The agent manager's registry maps to the trait registry. DBOS integration maps to `DBOSTrait`.

**PAI Agent SDK**: `BaseTool` already has `get_instruction()`, `is_available()`, `auto_inherit`, `get_approval_metadata()` — these map directly to trait methods. `InstructableToolset` maps to traits with `get_instructions()`. `PreHookFunc`/`PostHookFunc` map to `before_tool_call`/`after_tool_call`. Environment protocol stays separate (orthogonal to traits).

**Pydantic Deep Agents**: `create_deep_agent(include_todo=True, include_filesystem=True, ...)` becomes `Agent(traits=[TodoTrait(), FileSystemTrait(), ...])`. Feature flags become trait selection. The factory function can remain as a convenience wrapper.

---

## Appendix: Sources

### Community Projects
- Code Puppy: `github.com/mpfaffenberger/code_puppy`
- PAI Agent SDK: `github.com/youware-labs/pai-agent-sdk`
- Pydantic Deep Agents: `github.com/vstorm-co/pydantic-deepagents`
- Pydantic AI Deep Agent (Wh1isper): `github.com/ai-zerolab/pydantic-ai-deepagent`
- Mike's Traits SDK Design Doc: `github.com/mpfaffenberger/pydantic-ai/pull/1`

### External Frameworks
- Claude Code: `claude.ai/code` (skills, hooks, subagents, compaction)
- OpenAI Agents SDK: `github.com/openai/openai-agents-python`
- Google ADK: `github.com/google/adk-python`
- Vercel AI SDK: `ai-sdk.dev`
- CrewAI: `github.com/crewAIInc/crewAI`
- LangChain/LangGraph: `github.com/langchain-ai/langgraph`
- Agent Capability Standard: `github.com/synaptiai/agent-capability-standard`

### Pydantic AI Codebase
- Agent constructor: `pydantic_ai_slim/pydantic_ai/agent/__init__.py:229`
- AbstractToolset: `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py:62`
- WrapperAgent: `pydantic_ai_slim/pydantic_ai/agent/wrapper.py:29`
- HistoryProcessor: `pydantic_ai_slim/pydantic_ai/_agent_graph.py:73`
- Built-in tools: `pydantic_ai_slim/pydantic_ai/builtin_tools.py`
- Durable execution: `pydantic_ai_slim/pydantic_ai/durable_exec/`

### Relevant Issues
- #1197: Guardrails
- #2885: Middleware/hooks
- #3212: Built-in tool fallback
- #3365: Anthropic Skills
- #4144: Official skills
- #4159: Extension philosophy
- #530: Persist messages
- #3056/#3180: Durable execution
