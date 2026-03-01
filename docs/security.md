# Security considerations

Pydantic AI gives you flexible building blocks for agent systems. That flexibility also means security depends on how your agent is configured and which tools it can call.

This page covers common risk patterns and practical mitigations using existing Pydantic AI APIs.

## 1) Prompt-injection and tainted input

Untrusted user content can influence model behavior, tool arguments, and generated SQL or shell-like commands.

Recommended mitigations:

- Validate and constrain tool inputs with typed parameters and explicit checks in tool implementations.
- Keep system prompts explicit about what the model must never do.
- Treat model output as untrusted until it passes your own validation.
- Prefer narrow, purpose-built tools over broad "do anything" tools.

Related docs:

- [Tools](tools.md)
- [Dependencies](dependencies.md)
- [Output](output.md)

## 2) Unbounded loops and retry behavior

Agents can run longer than intended if retries and limits are too permissive for a task.

Recommended mitigations:

- Set bounded retries with `retries` / `max_retries` patterns where applicable.
- Add application-level timeouts and cancellation around agent runs.
- Monitor retry-heavy paths in production and tighten limits when needed.

Related docs:

- [Retries](retries.md)
- [Agents](agent.md)

## 3) Risky tool execution

Tools that touch the filesystem, databases, external APIs, or subprocesses can have high impact when fed model-influenced inputs.

Recommended mitigations:

- Gate high-impact actions behind explicit approval flows.
- Enforce allowlists for destinations, commands, schemas, or file paths.
- Run tools with least privilege credentials and scoped network/data access.
- Make side-effectful tools idempotent where possible.

Related docs:

- [Tools](tools.md)
- [Common tools](common-tools.md)

## 4) Human oversight for high-stakes actions

For actions with financial, legal, privacy, or production impact, include a human-in-the-loop checkpoint.

Recommended mitigations:

- Require explicit user confirmation before irreversible actions.
- Log proposed actions and rationale before execution.
- Separate "plan" steps from "execute" steps in your workflow.

## 5) MCP server trust boundaries

When connecting to external MCP servers, treat them as third-party systems with their own trust and data handling boundaries.

Recommended mitigations:

- Only connect to MCP servers you trust and understand.
- Scope what data is sent to each server.
- Require user approval for sensitive MCP operations.
- Review MCP security guidance for client implementations.

Related docs:

- [MCP overview](mcp/overview.md)
- [MCP client](mcp/client.md)

## Security tooling (optional)

You can layer standard Python security tooling on top of Pydantic AI projects:

- static analysis and linting for insecure patterns
- dependency and supply-chain scanning
- runtime policy/guardrail checks
- observability with redaction for sensitive fields

Pydantic AI integrates with standard Python tooling; pick the controls that match your risk profile and deployment environment.
