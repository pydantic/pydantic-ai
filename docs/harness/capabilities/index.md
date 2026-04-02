# Capabilities

Each capability is an `AbstractCapability` subclass that can be attached to any
Pydantic AI agent via the `capabilities` parameter.

| Capability | Description |
|---|---|
| AdaptiveReasoning | Dynamically adjust reasoning effort based on task complexity |
| Approval | Require human approval before executing sensitive operations |
| Compaction | Compress conversation history to stay within context limits |
| FileSystem | Read, write, and navigate the local filesystem |
| Guardrails | Validate inputs/outputs and enforce cost and tool constraints |
| KnowsCurrentTime | Inject the current date and time into the system prompt |
| Memory | Persistent key-value memory across agent sessions |
| Planning | Break complex tasks into plans before execution |
| RepoContextInjection | Inject repository structure and context into the system prompt |
| SecretMasking | Detect and redact secrets in agent inputs and outputs |
| SessionPersistence | Save and restore full conversation sessions |
| Shell | Execute shell commands with safety controls |
| Skills | Progressive tool loading via search and activate |
| SlidingWindow | Keep conversation history within a sliding token window |
| StuckLoopDetection | Detect and break out of repetitive agent loops |
| SubAgent | Delegate subtasks to specialised child agents |
| SystemReminders | Inject periodic reminders into the conversation |
| ToolErrorRecovery | Automatically retry or recover from tool execution errors |
| ToolOrphanRepair | Repair orphaned tool calls in conversation history |
| ToolOutputManagement | Control and format tool output for the model |

Detailed documentation for each capability will be added as they are merged.
