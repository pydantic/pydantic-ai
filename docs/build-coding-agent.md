---
title: Build your own coding agent in Python
description: "Build your own coding agent in Python with Pydantic AI: start from Coder in a few lines, take it apart into file, shell, planning and sub-agent capabilities, and run it in your terminal, your editor, a web UI, or CI."
---

# Build your own coding agent in Python

Build your own coding agent in Python with [Pydantic AI](index.md) and [Pydantic AI Harness](harness/index.md): an agent that reads your code, edits files, runs commands and tests, works with any model that can call tools, and is yours to change down to the last tool. Pydantic AI is the agent framework; Pydantic AI Harness is its library of [capabilities](capabilities/overview.md), plus complete agents built from them, such as [`Coder`](harness/coder.md). A *harness* is everything around the model that turns it into an agent: the loop, the tools, the workspace, and the context management.

This guide starts from `Coder` in a few lines, takes it apart into the capabilities it is made of, swaps and adds pieces, and then runs the result where you use coding agents: the terminal, your editor, a web UI, and CI.

## Start from `Coder`

Install the Harness with the `coder` extra, a model provider, and the CLI:

```bash
pip/uv-add "pydantic-ai-harness[coder,anthropic,cli]"
```

Then write `coding_agent.py`, and set `ANTHROPIC_API_KEY`:

```python {title="coding_agent.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', name='coder', capabilities=[LocalWorkspace('.'), Coder()])

if __name__ == '__main__':
    agent.to_cli_sync()
```

Run it in a project and talk to it:

```bash
python coding_agent.py
```

Give it a real task, such as "Find out why `tests/test_parser.py` fails, fix the cause without weakening the test, and rerun it." The agent explores the repository, reads and edits files, and runs the test to check its fix; review the diff before you keep it. To run one task from code instead, call `agent.run_sync('Fix the failing test in tests/test_parser.py.')` and print `result.output`.

[`LocalWorkspace('.')`][pydantic_ai.capabilities.LocalWorkspace] is where the agent works ([workspaces](workspace.md)): its file tools and commands run on your machine, in this directory, as you. It is not a sandbox; [Run it in a sandbox](#run-it-in-a-sandbox) shows how to isolate it.

`Coder` gives the model these tools:

- `read_file`, `write_file`, `edit_file`, `list_files` and `grep` for the code.
- `shell` for commands. A command still running after 270 seconds keeps running in the background, so the agent can start a server or a long test run and check on it.
- `delegate_task` to hand a self-contained sub-task to a fresh run of the same agent.

It also loads the repository's `AGENTS.md` and `CLAUDE.md` into the instructions, clears old tool results as the context fills up, caps oversized tool output, repairs malformed tool arguments, and turns on prompt caching. The [Coder page](harness/coder.md) documents every tool and default.

## Take it apart

`Coder` is not a black box. It is a regular combined [capability](capabilities/overview.md): a list of other capabilities, each of which you can use on its own. This agent is built from the same pieces:

```python {title="coding_agent.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching, LocalWorkspace
from pydantic_ai_harness import ClearToolResults, FileSystem, RepoContext, Shell, SubAgents, ToolOutputLimits
from pydantic_ai_harness.repair_tool_arguments import RepairToolArguments
from pydantic_ai_harness.tool_output_limits import Band, Truncate

agent = Agent(
    '{{ anthropic:large }}',
    name='coder',
    instructions='You are a coding agent. Investigate, make focused changes, and verify them before you finish.',
    capabilities=[
        LocalWorkspace('.'),  # where files and commands live
        FileSystem(  # read_file, write_file, edit_file, list_files, grep
            content_hashes=False,
            max_read_chars=50_000,
            tools=['read_file', 'write_file', 'edit_file', 'list_files', 'grep'],
            max_retries=5,
        ),
        Shell(denied_commands=[], allow_interactive=True, default_timeout=270, tools=['shell']),  # shell
        RepoContext(expose_inventory_tool=False),  # AGENTS.md and CLAUDE.md
        SubAgents(include_self=True),  # delegate_task to a fresh run of this agent
        ClearToolResults(max_fraction=0.7),  # clear old tool results as the context fills
        ToolOutputLimits(bands=[Band(over=64_000, action=Truncate(max_chars=64_000))]),  # cap tool output
        RepairToolArguments(),  # fix malformed JSON tool arguments
        Caching(),  # prompt caching
    ],
)
```

This is close to `Coder()`, not identical: `Coder` has longer default instructions and turns off strict tool schemas for its own tools. The [Coder composition](harness/coder.md#composition) lists the exact configuration. From here every line is yours to change: allowlist commands with `Shell(allowed_commands=[...])`, make the file tools [read-only](harness/filesystem.md), or drop delegation.

## Change the pieces

Each change below works on the `Coder` agent and on the taken-apart one alike, because both are lists of capabilities.

### Add planning, skills, and approvals

Capabilities compose, so adding behavior means adding entries to the list:

```python {test="skip" lint="skip"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import GuardrailResult, Planning, Skills, ToolGuardrail
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.guardrails import ToolCallInfo


async def confirm_command(call: ToolCallInfo) -> GuardrailResult:
    answer = await asyncio.to_thread(input, f'Run `{call.args["command"]}`? [y/N] ')
    return GuardrailResult.allow() if answer == 'y' else GuardrailResult.block('The user declined this command.')


agent = Agent(
    '{{ anthropic:large }}',
    name='coder',
    capabilities=[
        LocalWorkspace('.'),
        Coder(),
        Planning(),  # a task plan the model keeps current
        Skills('.agents/skills', missing_directories='skip'),  # SKILL.md procedures, loaded when relevant
        ToolGuardrail(guard=confirm_command, tools=['shell']),  # ask before every command
    ],
)
```

- [Planning](harness/planning.md) gives the model a plan it writes and updates as it works, and reminds it of the plan as the conversation grows.
- [Skills](harness/skills.md) reads [Agent Skills](capabilities/on-demand.md) (`SKILL.md` files) from a directory in the workspace, here `.agents/skills`, and loads each one only when the model asks for it; `missing_directories='skip'` lets the agent start before that directory exists. Install the `skills` extra.
- [Guardrails](harness/guardrails.md) check tool calls before they run. This one asks you in the terminal before each call to the `shell` tool, including calls from the agent's delegates, which run with the same capabilities. File edits and other tools, like the `run_tests` tool below, need guardrails of their own if you want to approve them too. To decide on a server or in a queue instead, return `GuardrailResult.approve()` and the run ends with [deferred tool requests](deferred-tools.md#human-in-the-loop-tool-approval) you answer later. To let a second model approve the harmless calls first, add [Approval Reviewer](harness/approval-reviewer.md).

Other capabilities slot in the same way: [Memory](harness/memory.md) across sessions, [Web Search](capabilities/web-search.md) and [Web Fetch](capabilities/web-fetch.md) for documentation, [summarizing compaction](harness/compaction.md) for very long conversations, [Detect Loops](harness/detect-loops.md), and [Spend Limits](harness/spend.md). The [capability index](harness/index.md#capabilities) lists all of them.

### Run it in a sandbox

`LocalWorkspace` runs the model's commands on your machine. To isolate them, swap it for a sandbox capability; the tools stay the same and now run in the sandbox. A local [Docker](harness/docker-sandbox.md) container with your project mounted at `/workspace`, where commands start, and no network:

```python {test="skip" lint="skip"}
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.docker_sandbox import DockerSandbox

sandbox = DockerSandbox(
    image='python:3.13-slim',
    docker_args=['--volume', f'{Path(".").resolve()}:/workspace'],
    allow_internet_access=False,
)
agent = Agent('{{ anthropic:large }}', name='coder', capabilities=[sandbox, Coder()])
```

The mount is writable, so the agent's edits land in your project while the rest of your files stay out of reach. With no network, the image needs your test dependencies already installed: build one from your project instead of using `python:3.13-slim` as is.

Or a cloud sandbox, such as [Modal](harness/modal-sandbox.md), [E2B](harness/e2b-sandbox.md), [Daytona](harness/daytona-sandbox.md), [Sprites](harness/sprites-sandbox.md), or [boxd](harness/boxd-sandbox.md):

```python {test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox

agent = Agent('{{ anthropic:large }}', name='coder', capabilities=[ModalSandbox(), Coder()])
```

[Workspaces](workspace.md) explains how a sandbox is created, reused across runs, shared with another agent, and cleaned up.

### Use a different model

The model is a string. Swap it for a model from [any provider](models/overview.md) Pydantic AI supports, install that provider's extra, and set its API key:

```python {test="skip" lint="skip"}
agent = Agent('{{ openai:large }}', name='coder', capabilities=[LocalWorkspace('.'), Coder()])
```

Models differ in how well they edit code and check their own work, so try a few on your own tasks. To pick the model per run, leave it off the agent and pass `model=` to `run()`, or let users choose one in the [CLI](cli.md) or [web UI](web.md). [Model Fallback](capabilities/model-fallback.md) moves to another model when a request fails.

### Add your own tool

`Coder` can already run tests through `shell`, but a dedicated tool pins down your project's exact test command. A tool is a Python function, and it gets the run's workspace through [`ctx.workspace`][pydantic_ai.tools.RunContext.workspace], so this one runs the tests on your machine or in the sandbox, whichever the agent has:

```python {test="skip" lint="skip"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness.coder import Coder

agent = Agent('{{ anthropic:large }}', name='coder', capabilities=[LocalWorkspace('.'), Coder()])


@agent.tool
async def run_tests(ctx: RunContext, path: str = 'tests') -> str:
    """Run the test suite, or the tests under one path, and return the report."""
    result = await ctx.workspace.run(['uv', 'run', 'pytest', '-x', '-q', path], timeout=600)
    return f'[exit {result.exit_code}]\n{result.stdout}{result.stderr}'
```

[Function tools](tools.md) covers arguments, retries and return types; [toolsets](toolsets.md) and [MCP servers](capabilities/mcp.md) add tools in bulk.

### Package it as a capability

When a tool comes with instructions, bundle them into a [`Capability`][pydantic_ai.capabilities.Capability] ([custom capabilities](capabilities/custom.md)) so any agent can reuse the pair:

```python {test="skip" lint="skip"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability, LocalWorkspace
from pydantic_ai_harness.coder import Coder


async def changed_files(ctx: RunContext) -> str:
    """List the tracked files changed since this branch left main, including uncommitted changes."""
    result = await ctx.workspace.run(['git', 'diff', '--name-only', '--merge-base', 'origin/main'], timeout=30)
    if result.exit_code != 0:
        return f'Could not list changed files: {result.stderr}'
    return result.stdout or 'No changes.'


our_conventions = Capability(
    id='our_conventions',
    instructions='Run Python through `uv run`. Before finishing, review every file `changed_files` lists.',
    tools=[changed_files],
)

agent = Agent(
    '{{ anthropic:large }}',
    name='coder',
    capabilities=[LocalWorkspace('.'), Coder(), our_conventions],
)
```

A capability subclass can also hook into the run: inspect each model request, rewrite tool results, or stop the run. [Building custom capabilities](capabilities/custom.md) covers the hooks, and every Harness capability's [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness) is a worked example.

## Run it where you code

The agent is one `Agent` object. The same module serves the terminal, an editor, a browser, and CI.

### In the terminal

[`agent.to_cli_sync()`][pydantic_ai.agent.AbstractAgent.to_cli_sync] above opens Pydantic AI's [CLI](cli.md): an interactive chat in your terminal, with commands such as `/multiline` and `/usage`. The [`clai`](cli.md#custom-agents) command runs the same agent with `clai --agent coding_agent:agent`.

### In your editor

Editors like [Zed](https://zed.dev/docs/ai/external-agents) run external coding agents over the [Agent Client Protocol](harness/acp.md) (ACP). Serve the agent over stdio, working in the directory the editor opened:

```python {title="coding_agent_acp.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.experimental.acp import AcpSession, AcpSessionConfig, run_acp_stdio_sync

agent = Agent('{{ anthropic:large }}', name='coder', capabilities=[Coder()])


def session_config(session: AcpSession) -> AcpSessionConfig[None]:
    return AcpSessionConfig(deps=None, workspace=LocalWorkspaceBackend(session.cwd))


if __name__ == '__main__':
    run_acp_stdio_sync(agent, session_config=session_config)
```

Install the `acp` extra and register `python /absolute/path/to/coding_agent_acp.py` as an external agent in your editor. File reads and edits show up as file links and diffs, and tools that need approval prompt you in the editor. ACP support is experimental; the [ACP page](harness/acp.md) covers the editor configuration, session persistence, and model selection.

### In a browser or your own app

[`agent.to_web()`][pydantic_ai.agent.Agent.to_web] serves a chat UI for the agent as an ASGI app:

```python {test="skip" lint="skip"}
app = agent.to_web()
```

Install `pydantic-ai-slim[web]` and run it with `uvicorn coding_agent:app`. With `LocalWorkspace`, every message works in the same directory; see the [web chat UI](web.md). To put the agent behind your own frontend, the [AG-UI](ui/ag-ui.md) and [Vercel AI](ui/vercel-ai.md) adapters stream its text, tool calls and approvals to the browser.

### In CI

[GitHub Agentic Workflows](harness/gh-aw.md) runs a Pydantic AI agent on issues, pull requests, or a schedule, in a container behind an egress firewall. It runs `Coder` by default, or your own agent when the workflow names its module. The [gh-aw guide](harness/gh-aw.md) walks through a complete repository.

## Keep long runs going

A coding run can take an hour. To survive a crashed worker or a deploy mid-run, run the agent under a [durable execution](durable_execution/overview.md) engine. With Temporal, add its [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] capability next to the workspace:

```python {test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.temporal import TemporalDurability
from pydantic_ai_harness.coder import Coder

agent = Agent(
    '{{ anthropic:large }}',
    name='coder',  # names the agent's Temporal activities
    capabilities=[LocalWorkspace('.'), Coder(), TemporalDurability()],
)
```

The capability alone does not make a run durable: the run has to execute inside a Temporal workflow your worker runs, as the [Temporal guide](durable_execution/temporal.md) shows. A resumed run picks up after its last completed step, so the workspace has to survive too, and commands that already ran are not undone. `Coder` also works under DBOS and Prefect; the [workspace guide](workspace.md#harness-workspace-capabilities) lists what each engine supports. Without a workflow engine, [Step Persistence](harness/step-persistence.md) saves a snapshot after every step, so you can resume or fork a run later.

## See what it did

A coding run makes dozens of model requests and tool calls. To see each one with its arguments, output, tokens and cost, set up [Logfire](logfire.md) with one command:

```bash
uvx logfire-cli wizard --task pydantic-ai
```

It signs you in, picks a project, and has your coding agent add the instrumentation. To add it by hand, call `logfire.configure()` and `logfire.instrument_pydantic_ai()` before the agent runs; see [Using Logfire](logfire.md#using-logfire). The traces are standard OpenTelemetry, so [any OTel backend](logfire.md#using-opentelemetry) works too.

To tell whether a new model, instruction, or tool makes the agent better, keep a few representative tasks with checks of their own and run them against each version, by hand or with [Pydantic Evals](evals.md).

## Next steps

- [Coder](harness/coder.md): every tool, default, and setting of the agent you started from.
- [Pydantic AI Harness](harness/index.md): the capabilities you can add, by what they give the agent.
- [Workspaces](workspace.md): local directories, sandboxes, and sharing one between agents.
- [Building custom capabilities](capabilities/custom.md): the hooks behind every piece on this page.
- [You've built this agent before](https://pydantic.dev/articles/harness-week): the Harness Week post, which builds one agent from these parts over five days.
- [What makes a good agent harness](https://pydantic.dev/articles/what-makes-a-good-harness): why timing matters, getting the right context and corrections to the model at the right moment.
