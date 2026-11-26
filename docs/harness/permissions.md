---
title: Permissions
description: Ask a person before the agent writes files or runs commands, by permission mode and rules.
---

# Permissions

Ask a person before the agent writes files or runs commands, by permission mode, and allow, ask about, or deny tool calls and shell commands by rule.

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

A call that asks raises [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired], so it uses the same [deferred approval flow](../deferred-tools.md#human-in-the-loop-tool-approval) as `requires_approval=True`. Without a handler the run ends with the call in its [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output. To ask a person inside the run instead, add a [`HandleDeferredToolCalls`](../capabilities/handle-deferred-tool-calls.md) whose handler is `approval_ask` with an approver:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls, LocalWorkspace
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.permissions import (
    PermissionDecision,
    PermissionRequest,
    Permissions,
    approval_ask,
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
        HandleDeferredToolCalls(approval_ask(approve)),
    ],
)
```

`approval_ask` asks the approver about each held call in turn, and returns its answers in place. A denied call does not run, and the model gets a tool result saying the user denied it, with the person's `reason` when they gave one. A `PermissionDecision` with `remember=True` also lets later calls of the same tool on the same command or path run without asking, for the rest of the conversation (by the run's `conversation_id`). That memory is the `remembered` set, which you can pass in to own: share it, clear it, or drop it with the conversation. It lives in this process, so inside a durable execution workflow the approval is kept in the message history instead, in the metadata of the approved call's `ToolReturnPart`: it holds when the workflow replays, and in a later run given that history. Since a marker in the history skips asking, only pass in history you trust; compaction that drops the part makes the call ask again.

Approvals that other capabilities ask for, such as a tool registered with `requires_approval=True`, are left for the next handler.

In a [durable](../durable_execution/overview.md) run, `approval_ask` takes a resume token, so the engine records its answer. Only AWS Lambda durable functions with [`AWSLambdaDurability`](aws-lambda.md#approvals) support that so far; other engines refuse a handler that takes a token with a `UserError` (see [Engines](../capabilities/handle-deferred-tool-calls.md#engines)), so there pass the two-parameter handler `HandleDeferredToolCalls(lambda ctx, requests: ask_permissions(requests, approve))` instead. On Lambda, `approval_ask` runs in the step that records its answer, so its approver must answer quickly, and the callback created for the wait stays open: set `HandleDeferredToolCalls(..., timeout=...)` so it expires. To wait for a person there, write a handler that takes the token and hands it on, and use `permission_requests` to find the held calls in the requests it gets. The answer it resolves the token with can approve or deny them, but not remember an approval for the conversation: only `ask_permissions` and `approval_ask` record `remember=True`.

## Which calls are checked

- **File writes** are the `FileSystem` tools `write_file`, `edit_file`, and `create_directory`, recognized by their tool kind, so a renamed or prefixed tool is still recognized. They only reach files inside the file system's root, so `'auto'` lets them run.
- **Commands** are the tools whose metadata declares `code_arg_language='shell'` and a `code_arg_name`, the argument holding the command, as the `Shell` tools that take a command (`run_command`, `start_command`, and `shell`) do.
- **Sub-agents** run only the capabilities that reach them. A delegate does not see a `Permissions` passed to the parent's run, so pass it in [`SubAgents(shared_capabilities=[permissions, handler])`](subagents.md) to check delegates too.
- Every other tool runs in every mode, unless a [rule](#rules) matches it. Gate a tool of your own with `requires_approval=True`, a [`ToolGuardrail`](guardrails.md), or a [`ToolCallJudge`](tool-call-judge.md).

`ask_commands` is a list of regular expressions, each searched for anywhere in the command. `DEFAULT_ASK_COMMANDS` asks about forced or recursive deletes (`rm -rf`), `git push`, `git reset --hard` and `git clean`, `sudo`, network clients (`curl`, `wget`, `ssh`, and others), package installs, `dd`, and `mkfs`. It is a convenience, not a sandbox: a command can do the same thing in a form no pattern matches, such as a script it writes first. Use a [sandbox](../workspace.md) when the agent must not reach something.

## Rules

`rules` allow, ask about, or deny calls by tool, arguments, and shell command, before the mode decides. A call no rule matches follows the mode:

```python
from pydantic_ai_harness.permissions import PermissionRule, Permissions

permissions = Permissions(
    'read-only',
    rules=[
        PermissionRule('allow', command='git status'),
        PermissionRule('allow', command='rg'),
        PermissionRule('ask', command='git push'),
        PermissionRule('deny', command='rm -rf', reason='Delete files one at a time.'),
        PermissionRule('allow', tools=['write_file'], args={'path': 'docs/*'}),
        PermissionRule('ask', tools=['deploy'], args={'env': 'prod*'}),
    ],
)
```

A `PermissionRule` matches a call when every field it sets matches:

- `tools` is a [`ToolSelector`][pydantic_ai.tools.ToolSelector], as for other tool-scoped capabilities: `'all'` (the default), a list of names, a metadata match, or a predicate.
- `args` maps argument names to globs for their string values. Each value is matched as given and as a normalized path, so `notes/../secrets/key` is also matched as `secrets/key`. On a `'windows'` workspace both forms are compared case-insensitively, with `/` for separators. Symlinks are not resolved. An allow rule needs both forms to match. An ask or deny rule needs either, and also matches the end of a longer path, so `secrets/*` covers `/project/secrets/key` and `../secrets/key`. Values over 4096 characters aren't globbed: ask and deny rules match them, and allow rules don't.
- `command` is the words of a shell command, described below. An allow rule that globs a shell tool's command argument through `args` never matches, since a glob would let `git status; python evil.py` through `git *`: use `command` to allow commands.

When several rules match, `'deny'` beats `'ask'`, which beats `'allow'`, whatever their order, so a broad allow rule cannot override a deny rule.

- **`'allow'`** runs the call without the `'read-only'` mode asking. A command that matches `ask_commands` still asks in `'auto'` and `'read-only'`: remove a pattern from `ask_commands` to let it run.
- **`'ask'`** holds the call like the mode does. A file write or command goes to the approver of `approval_ask` or `ask_permissions` with the rule's `reason`. A call to any other tool is held with a plain [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired], like `requires_approval=True`, with the rule's `reason` in its metadata. The approver is not asked about it, which is left for another handler, so add one, as shown below.
- **`'deny'`** refuses the call without running it, even when it was approved, for example by another capability's approval. The model gets a tool result with `outcome='denied'` saying a permission rule refused it, with the rule's `reason`. A call that needs approval for another reason, such as `requires_approval=True`, is put to the person before the rule refuses it.

A second handler answers the asks `ask_permissions` leaves. A two-parameter handler like this one runs before a handler that takes a resume token, such as `approval_ask`, whatever their order in the capabilities list, so it skips the calls `permission_requests` returns and leaves them to `approval_ask`:

```python
from pydantic_ai import DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai_harness.permissions import permission_requests


async def approve_other_calls(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
    held = {request.call.tool_call_id for request in permission_requests(requests)}
    return DeferredToolResults(
        approvals={
            call.tool_call_id: input(f'Run {call.tool_name}({call.args_as_dict()})? [y/N] ') == 'y'
            for call in requests.approvals
            if call.tool_call_id not in held
        }
    )


other_calls = HandleDeferredToolCalls(handler=approve_other_calls)
```

A `command` rule only matches tools that declare a shell command argument (see [Which calls are checked](#which-calls-are-checked)), and judges a compound command by each of its simple commands, split at `;`, `&&`, `||`, `|`, and `&`:

- A command runs by rule only when an `'allow'` rule matches every simple command in it. An allow rule matches a simple command that starts with its words and has no redirection other than to `/dev/null` or a file descriptor. So `PermissionRule('allow', command='git status')` lets `ls && git status --short` run when `ls` is allowed too, but not `git status > ~/.bashrc` or `FOO=1 git status`.
- An `'ask'` or `'deny'` rule matches a simple command that contains its first word followed by its other words in order, other words allowed in between. A short-option cluster such as `-rf` matches the same options in any grouping or case (`-fr`, `-Rf`, `-r -f`). So `sudo rm -rf /` and `ls; rm -fr /` match `'rm -rf'`, `git -C repo push` matches `'git push'`, its first word also matches a path ending in it, such as `/bin/rm`, and a pattern with a literal character matches the words it could expand to, so `sudo /bin/r[m] -rf x` matches too, while a bare `*` matches no rule word.
- A rule also matches the text inside one quoted word, as in `sh -c "rm -rf /"`. That text may be data, such as a commit message or a search pattern, so a deny rule matched only there asks instead, and `git commit -m "push the fix"` doesn't match `'git push'`, whose words come from two places.
- Some commands are too complex to split: those with `$` (expansions, substitutions, and `$'...'` quoting), backticks, quoted or escaped operators (`echo '|'`), brace expansion (`{rm,-rf,x}`), a pattern in a command name (`/bin/r[m]`), process substitution, subshells, here-documents, line breaks or carriage returns, or unbalanced quotes. Any command on a `'windows'` workspace is one too, since PowerShell quotes differently, and its words are compared ignoring case. No `command` rule allows such a command. Its text is checked against the ask and deny rules with quotes removed, so `python -c 'a(); rm -rf x'` still matches `'rm -rf'`, as an ask since the match is in quoted text. When part of it is expanded by the shell (`$` or a backtick outside single quotes, brace expansion, a line continuation, or a pattern in a command name), a rule's words could be hidden, so it asks whenever ask or deny command rules apply to the tool. Otherwise, with no rule matching, the mode decides, so `grep -E 'a|b' x.py` or a multi-line script runs in `'full'` with only a deny rule.

An allow rule trusts everything after its words, so allowing `find` also allows `find . -delete`: allow the narrowest commands you can. Deny rules match words, not behavior, so `rm --recursive --force` does not match `'rm -rf'`. Rules are checked when a call is about to run, so calls to [external tools](../deferred-tools.md#external-tool-execution), which your application runs, are not checked. Like `ask_commands`, rules are a convenience, not a sandbox.

In an agent spec, `rules` is a list of mappings of these fields (`PermissionRuleSpec`), with `tools` limited to the serializable forms.

## Protected paths

`protected_paths` names paths only a person may approve a change to, such as the agent's own settings, as globs. `protected_commands` names commands that change them, such as the agent's own settings command, as words like an ask rule's `command`. Outside `'full'`, a call that changes one always asks:

```python
from pydantic_ai_harness.permissions import Permissions

permissions = Permissions(
    'auto',
    protected_paths=['/home/me/.config/my-agent', '/home/me/.config/my-agent/*', '.git/hooks/*'],
    protected_commands=['my-agent config set'],
)
```

- A **file write** to a matching path asks. The path is matched as resolved against the working directory and, where the workspace can follow symlinks, as where it leads. A relative glob also matches the end of a path, so `.git/hooks/*` covers every repository's hooks.
- A **command** asks when one of its words names a matching path, a redirection target or an `--option=value` value included, with quotes removed and `~` and environment variables expanded as in this process. A command made only of programs that read the files they name (`cat`, `grep`, `head`, `ls`, and a few more), without redirection, still runs. A command matching one of `protected_commands`, by the same word matching as an ask rule's `command`, asks too.
- A rule's allow does not let such a call run, and a rule's deny still refuses it. The held request's `PermissionRequest.person_only` is `True`, and an [`ApprovalReviewer`](approval-reviewer.md) never reviews it. A `PermissionDecision(remember=True)` doesn't cover later calls either, so a person approves each one.

The check is best effort, by the command's text: a path a command builds at run time, as `cd .git && ...` or a script the agent wrote does, is not seen. It keeps an agent from changing these paths without a person noticing, and is not a sandbox: use [one](../workspace.md) when the agent must not reach a path.

## Limits from outside the run

When one party configures an agent and another runs it, such as an organization rolling a coding agent out to its developers, `max_mode` and `managed_rules` set limits that the person running it can't loosen with `mode` or `rules`:

```python
from pydantic_ai_harness.permissions import PermissionRule, Permissions

permissions = Permissions(
    'full',
    max_mode='auto',
    managed_rules=[
        PermissionRule('deny', command='curl', reason='Use the corporate proxy.'),
        PermissionRule('ask', command='git push'),
        PermissionRule('allow', command='git status'),
    ],
)
```

- **`max_mode`** is the widest mode `mode` may take, or a function returning it for a run. `mode_for()` returns the stricter of the two, so with the example above the agent runs in `'auto'`. A wider `mode` acts as `max_mode`.
- **`managed_rules`** are checked before `rules`. A managed `'deny'` refuses a call even when it was approved, and a managed `'ask'` holds a call even when a rule in `rules` allows it.
- Every `'deny'`, managed or not, is checked before any `'ask'`, so approving a call never runs one that a rule in `rules` denies: the person running the agent can still make it stricter.
- `max_mode` is checked on its own, after the person's `mode`. Only an `'allow'` in `managed_rules` skips the blanket ask of a `'read-only'` `max_mode`, and it doesn't skip a `'deny'` or `'ask'` in `rules`, a stricter `mode`, or `protected_paths`.
- A call held by a managed rule or by `max_mode` is `person_only`, like a change to a protected path: an [`ApprovalReviewer`](https://pydantic.dev/docs/ai/harness/approval-reviewer/) never reviews it, and `remember=True` doesn't cover later calls. A managed `'ask'` on a tool that is neither a command nor a file write takes the plain approval flow, with `{'permissions': {'person_only': True}}` in its deferred request's metadata.
- Several `Permissions` on one agent each check every call, so a second `Permissions('full')` can't loosen the first.

A sub-agent is held to these limits when the parent's `Permissions` reaches it, through `SubAgents(shared_capabilities=[...])`. Like the rest of `Permissions`, they are checked in the agent's process, on a machine the person running it may control: they keep an agent within an organization's settings, and are not a sandbox.

## Changing the mode during a run

`mode` can be a function of the run context. It is read on every call, so an application can switch modes while a run is going, and the run's next call follows the new mode:

```python
from pydantic_ai_harness.permissions import PermissionMode, Permissions

current: PermissionMode = 'read-only'
permissions = Permissions(lambda ctx: current)
```

## Telemetry

`Permissions` emits no spans of its own. A held call shows up in core's tool spans and in the run's deferred tool requests and results, which carry the request's `subject` and `reason` as metadata. A call a rule denies shows up in core's tool span with the denial as its result, and in the history as a `ToolReturnPart` with `outcome='denied'`.

## Related

- [Approval Reviewer](approval-reviewer.md): let a second model approve the low-risk calls that need approval.
- [Tool Call Judge](tool-call-judge.md): let a second model block calls.
- [Ask User](ask-user.md): let the model ask a person multiple-choice questions.
