# Hosts and Protocols

Read only when the migration includes a Deep Agents host, client, or deployment surface. Migrate the underlying agent separately, then preserve the external contract with an adapter.

| Source surface | Target direction | Required check |
| --- | --- | --- |
| Deep Agents Code or another CLI host | Compose `Coder` or narrower Harness capabilities behind the existing command boundary | Inventory config and environment precedence, hooks and plugins, skills and memory roots, MCP servers, sandbox selection, approval modes, sessions, interactive and headless output, exit status, cancellation, and updates. Do not treat CLI behavior as `create_deep_agent` behavior. |
| frontend or AG-UI client | an application event/state adapter | Preserve coordinator messages, subagent projections, todo and custom state, tool-call lifecycle, interrupts, thread identity, reconnect behavior, files, sandbox artifacts, and final-result timing. |
| Agent Client Protocol | the Harness ACP adapter when its session contract matches, otherwise an application adapter | Verify protocol negotiation, stdio lifecycle, sessions, content blocks, tool presentation, filesystem and terminal ownership, approvals, cancellation, history, errors, and client capability fallbacks. |
| A2A endpoint or client | an application server/client adapter around the migrated agent | Preserve the protocol version, agent card, methods, task and context identity, history, streaming, file parts, authentication, cancellation, tracing, and conformance behavior. |
| managed deployment | the selected host and application services | Preserve API and stream schemas, thread/run/store semantics, queues, schedules, tenancy, authentication, secrets, regional and data-retention rules, observability, retries, rollout, and recovery. |

If the target lacks a protocol feature, retain the existing adapter or propose a focused adapter built on public Pydantic AI run, message, event, and deferred-tool primitives. Do not bury a wire-level behavior change inside the agent migration.

Primary sources: [Deep Agents Code](https://docs.langchain.com/oss/deepagents/code/overview), [frontend](https://docs.langchain.com/oss/python/deepagents/frontend/overview), [ACP](https://docs.langchain.com/oss/python/deepagents/acp), [A2A](https://docs.langchain.com/oss/python/deepagents/a2a), and [going to production](https://docs.langchain.com/oss/python/deepagents/going-to-production).
