---
title: Permissions
description: Ask a person before the agent writes files or runs commands, by permission mode.
---

# Permissions

Ask a person before the agent writes files or runs commands, by permission mode.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/permissions/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

A coding agent's [`FileSystem`](filesystem.md) and [`Shell`](shell.md) tools run with your permissions as soon as the model calls them. Sometimes that is what you want; sometimes you want to watch what the agent does and stop a `rm -rf` or a `git push` before it happens. Marking each tool `requires_approval=True` asks about every call, including the harmless `ls`, so the person stops reading the requests.

## The solution

`Permissions` holds the calls a permission mode asks about and lets the rest run:

| Mode | File writes | Commands |
|---|---|---|
| `'full'` (default) | run | run |
| `'auto'` | run | run, unless they match `ask_commands` |
| `'read-only'` | ask | ask |

A call that asks raises [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired], so it uses the same [deferred approval flow](../deferred-tools.md#human-in-the-loop-tool-approval) as `requires_approval=True`. Without a handler the run ends with the call in its [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output. To ask a person inline instead, add a [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler that calls `ask_permissions` with an approver:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls, LocalWorkspace
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.permissions import (
    PermissionDecision,
    PermissionRequest,
    Permissions,
    ask_permissions,
)
from pydantic_ai_harness.shell import Shell


async def approve(request: PermissionRequest) -> PermissionDecision:
    answer = input(f'{request.reason}\n  {request.subject}\nRun it? [y/N/always] ')
    return PermissionDecision(approved=answer in ('y', 'always'), remember=answer == 'always')


agent = Agent(
    '{{ anthropic:large }}',
    capabilities=[
        LocalWorkspace('.'),
        FileSystem(),
        Shell(),
        Permissions('auto'),
        HandleDeferredToolCalls(handler=lambda ctx, requests: ask_permissions(requests, approve)),
    ],
)
```

`ask_permissions` asks about each held call in turn. A denied call does not run, and the model gets a tool result saying the user denied it, with the person's `reason` when they gave one. A `PermissionDecision` with `remember=True` also lets later calls of the same tool on the same command or path run without asking, for the rest of the conversation (by the run's `conversation_id`). That memory is the `remembered` set, which you can pass in to own: share it, clear it, or drop it with the conversation. It lives in this process, so inside a durable execution workflow it is neither read nor written, and every call asks.

Approvals that other capabilities ask for, such as a tool registered with `requires_approval=True`, are left for the next handler.

## Which calls are checked

- **File writes** are the `FileSystem` tools `write_file`, `edit_file`, and `create_directory`, recognized by their tool kind, so a renamed or prefixed tool is still recognized. They only reach files inside the file system's root, so `'auto'` lets them run.
- **Commands** are the tools whose metadata declares `code_arg_language='shell'` and a `code_arg_name`, the argument holding the command, as the `Shell` tools that take a command (`run_command`, `start_command`, and `shell`) do.
- **Sub-agents** run only the capabilities that reach them. A delegate does not see a `Permissions` passed to the parent's run, so pass it in [`SubAgents(shared_capabilities=[permissions, handler])`](subagents.md) to check delegates too.
- Every other tool runs in every mode. Gate a tool of your own with `requires_approval=True`, a [`ToolGuardrail`](guardrails.md), or a [`ToolCallJudge`](tool-call-judge.md).

`ask_commands` is a list of regular expressions, each searched for anywhere in the command. `DEFAULT_ASK_COMMANDS` asks about forced or recursive deletes (`rm -rf`), `git push`, `git reset --hard` and `git clean`, `sudo`, network clients (`curl`, `wget`, `ssh`, and others), package installs, `dd`, and `mkfs`. It is a convenience, not a sandbox: a command can do the same thing in a form no pattern matches, such as a script it writes first. Use a [sandbox](../workspace.md) when the agent must not reach something.

## Changing the mode during a run

`mode` can be a function of the run context. It is read on every call, so an application can switch modes while a run is going, and the run's next call follows the new mode:

```python
from pydantic_ai_harness.permissions import PermissionMode, Permissions

current: PermissionMode = 'read-only'
permissions = Permissions(lambda ctx: current)
```

## Telemetry

`Permissions` emits no spans of its own. A held call shows up in core's tool spans and in the run's deferred tool requests and results, which carry the request's `subject` and `reason` as metadata.

## Related

- [Approval Reviewer](approval-reviewer.md): let a second model approve the low-risk calls that need approval.
- [Tool Call Judge](tool-call-judge.md): let a second model block calls.
- [Ask User](ask-user.md): let the model ask a person multiple-choice questions.
