# CLAI plugins

A plugin is a Python file that teaches CLAI new tricks: react when something
happens, add a `/command`, give the agent a tool, or change how a tool's output
is shown on screen.

Everything a plugin can do goes through one object, the `PluginHost`. There is no
global registry to import and no magic file to name. You get a `host`, you tell it
what you want, you're done.

## On-demand authoring help

The default CLAI agent exposes `read_clai_customization_guide`. When you ask for
customization, its instructions tell it to read this guide first. Only the short
hint and tool description are present initially; the bundled text is read on tool
invocation. That hint is ordered after the guidance the plugins contribute and
before the repository instruction file, so it never leads the system prompt. Reading it needs neither a checkout nor a network connection.
This tool needs no arguments and ignores extra arguments supplied by a model.
Other tools keep their existing validation.

The [bundled guide](src/pydantic_clai2/customization.md) includes examples for
commands, hooks, settings, renderers, custom Termflow menus, and custom model
launchers. It also names current limits: PluginHost does not register providers,
replace the prompt editor, or alter the built-in model catalog. Those need a
custom agent launcher or a source change, as explained in the guide.

Custom agents can opt in with `customization_guide()` from
`pydantic_clai2.customization`. The tool only returns documentation; it does not
write files, activate plugins, or grant permission to execute generated code.
Keep the bundled guide aligned with this contract when changing plugin APIs.

## Default code rendering

The default stream renderer buffers fenced code until the fence closes or the
text part ends, then highlights the whole block. This preserves multiline lexer
context. Unlabelled and Markdown fences stay literal; unknown languages use
plain text. Long code lines wrap to the terminal width. Prose still streams line
by line. A plugin renderer that handles a text event replaces this default
rendering for that event.

## Tool retries

CLAI defaults to three retries per tool call. `/set run.tool_retries N` changes
that default for subsequent turns; `N` must be a non-negative integer, with `0`
disabling retries. Explicit retry limits on a tool or toolset take precedence.
Output-validation and HTTP transport retry budgets are unchanged.

## Credentials

CLAI's `/login openai-codex` and the vllm and openrouter connections store tokens
in the configured keyring backend, not plugin settings. Large token bundles use
multiple entries to fit Windows Credential Manager's size limit. When no keyring
backend exists, credentials go to a per-account `0600` file under the user's CLAI config
directory instead. None of this changes plugin APIs. See
[Codex authentication](README.md#codex-authentication) for storage and security
details.

## Logfire: default agent tracing

The built-in `logfire` plugin (`pydantic_clai2.logfire`) is enabled by default in
the stock CLI. It registers Pydantic AI's `Instrumentation` capability with an
isolated Logfire instance, not process-wide instrumentation or custom tracing
hooks. Agent/model/tool spans include timing, token usage, failures, text content,
and binary image attachments by default, including retained history used by
later turns. This may export source code, file contents, and screenshots; verify
the configured telemetry destination first.

Credentials are read from `LOGFIRE_TOKEN` or the SDK's `logfire_credentials.json`
in `$XDG_CONFIG_HOME/pydantic-clai2/logfire/`, defaulting to
`~/.config/pydantic-clai2/logfire/`. SDK configuration is read only from that user
directory too. Repository-local configuration/credentials and the SDK's
`LOGFIRE_CONFIG_DIR`/`LOGFIRE_CREDENTIALS_DIR` overrides are ignored. Relative
`XDG_CONFIG_HOME` values fall back to `~/.config`. A checkout cannot select the
telemetry destination through its own files. Without credentials the default
`if-token-present` mode does not export to Logfire or start interactive setup. Console logging is disabled. Other SDK configuration,
such as explicit OTLP exporters, still applies.

Manage it with `/plugins disable logfire`, `/plugins enable logfire`, or
`/plugins reload logfire`. To change its defaults:

```text
/plugins add logfire pydantic_clai2.logfire '{"include_content": false, "include_binary_content": false}'
```

Options are `service_name` (default `pydantic-clai2`), `include_content` and
`include_binary_content` (both default `true`), and `send_to_logfire` (default
`"if-token-present"`, or `false`). The explicit plugin option takes precedence
over `LOGFIRE_SEND_TO_LOGFIRE`. Tokens are not accepted in plugin settings.
Content flags do not suppress all metadata: tool names and definitions may still
be recorded. Logfire's usual scrubbing is enabled.

Unload flushes and shuts down only this plugin's providers. Reload creates a new
instance. The supplied agent and global providers are unchanged, and the existing
global propagator is preserved. The SDK may install shared executor propagation
helpers; those hooks are not removed on unload. Core's normal
instrumentation precedence applies: the plugin's explicit per-run capability
wins while enabled; disabling it restores the supplied agent's own tracing
behavior. Custom launchers must pass `builtin_plugins=DEFAULT_PLUGINS` to opt in
to stock built-ins. See [telemetry](README.md#telemetry-and-references).

## Where plugins live

Plugins are trusted Python code. Drop-in files execute automatically at startup;
this directory is an executable startup configuration, not a sandbox. The default
Coder runs as your OS user and can modify it, just as it can modify your shell
startup files. Use a separate OS identity or sandbox for untrusted agent work.

Two ways to install one:

1. Drop a `.py` file (or a package folder) into
   `$XDG_CONFIG_HOME/pydantic-clai2/plugins/` (default `~/.config/pydantic-clai2/plugins/`).
   Its name is the file name without `.py`.
2. Point CLAI at anything importable, from the shell or from inside CLAI:

   ```sh
   clai2 plugins add notify my_package.notify
   /plugins add coder pydantic_ai_harness.coder:Coder '{"unrestricted_filesystem": true}'
   ```

No restart needed when you do it from inside CLAI. A plugin you add or enable is
active for the next prompt; one you disable or remove is gone for the next
prompt. From the shell, `clai2 plugins ...` only saves; it loads on the next
start. Plugins in the folder and the ones you added by name are managed the
same way. Explicit module declarations take precedence over a same-named drop-in
file. Reloading a disabled plugin is rejected; enable it first. Drop-in entry
modules are compiled from current source. Installed modules use `importlib.reload`,
which can retain globals removed from source; initialize plugin state explicitly.

Plugins are trusted code running as you. Only install what you trust.

## Worktree startup

```bash
clai2 --worktree my-task
```

`--worktree` (or `-w`) creates `<repository-root>/.worktrees/NAME` and changes to
that directory before reading project settings or activating plugins. Relative paths in your plugin, the coding tools,
and `repo_context` therefore refer to that checkout. User plugins and settings
still load from the same database directory, even with a relative `--database`
path. Only committed project files reach the new checkout. Worktrees and their
`clai/NAME` branches stay on disk after the session ends; plugins do not own their
cleanup. See [Git worktrees](README.md#git-worktrees) for naming and cleanup.

## The built-in plugins

The coding tools are a plugin too, and so are asking you multiple-choice
questions mid-run, reading the repository's instruction file, and keeping the
conversation inside the context window. These five plugins are marked
`(built-in)` and enabled unless you say otherwise. Built-ins activate first, in
the order listed below, so `coder`'s guidance leads the system prompt; saved,
drop-in, and project plugins follow in name order. Registration order is also the
order plugin instructions, renderers, and status segments are consulted in.
`/plugins` and `/plugins list` stay alphabetical for scanning:

| Id | Backed by | Settings | Does |
|---|---|---|---|
| `coder` | `pydantic_ai_harness.coder:Coder` | `{"unrestricted_filesystem": true, "repo_context": false}` | the file and shell tools |
| `ask_user` | `pydantic_clai2.ask_user_menu:activate` | `{}` | the `ask_user_question` tool: multiple-choice questions answered from the terminal |
| `repo_context` | `pydantic_clai2.repo_context` | `{}` | reads `CLAUDE.md` or `AGENTS.md` from the launch directory into the instructions |
| `persistence` | `pydantic_clai2.sessions` | `{}` | Harness step checkpoints for interrupted session recovery |
| `compaction` | `pydantic_clai2.compaction` | `{}` | automatic summarisation with a truncation fallback, `/compact`, and the context warning |

### Optional harness capabilities

`/plugins` and `/plugins list` also include every other public harness capability,
including each compaction strategy and guardrail. These entries start disabled.
`Coder`, `AskUser`, and `RepoContext` use the integrated entries above instead of
appearing twice. Deprecated aliases, toolsets, stores, and the ACP server adapter
are not separate capabilities.

Press Space to enable an entry. Its preview shows the import path and any load
error. Listing disabled entries does not import their modules or require their
optional packages. Some capabilities need an extra installed in CLAI's Python
environment, credentials, or constructor settings before they can load. Supply
JSON constructor settings by replacing the declaration under the same id:

```text
/plugins add sliding_window_compaction pydantic_ai_harness.compaction:SlidingWindowCompaction '{"max_messages": 40}'
```

For callbacks, stores, or other Python objects, use a plugin module that builds
the capability and calls `host.add(...)`, registered under that id. The menu does
not construct these objects or install dependencies. Avoid enabling overlapping
tool providers together, such as `filesystem` or `shell` alongside `coder`.
Removing an optional built-in restores its disabled declaration; enable and
disable choices persist between launches. Project, drop-in, and saved declarations
retain their usual precedence over built-ins.

`/plugins disable coder` gives you a chat-only CLAI (a writing or research setup
with `ExaSearch` instead, say); `/plugins enable coder` brings the tools back;
`/plugins remove coder` cannot forget a built-in, so it resets it to its
defaults. `/plugins disable repo_context` stops the instruction file from being
read. To run a built-in with different options, add your own declaration under
the same name and it takes the built-in's place:

```text
/plugins add coder pydantic_ai_harness.coder:Coder '{"unrestricted_filesystem": false, "repo_context": false}'
/plugins add repo_context pydantic_clai2.repo_context '{"walk_up": true}'
```

Keep `"repo_context": false` on a replacement `coder`: `Coder` bundles its own
`RepoContext`, and with the `repo_context` plugin also on, the instruction file
would reach the model twice.

`repo_context` wraps harness `RepoContext` with the launch directory as the
workspace and its default filenames. Its settings:

| Key | Default | Does |
|---|---|---|
| `walk_up` | `false` | also load instruction files from every directory between the workspace and your home directory |
| `inventory_tool` | `false` | give the agent `inventory_agent_context`, which maps the repo's `.claude`, `.agents`, `.codex`, and `.grok` assets |
| `nested_traversal` | `false` | when the agent reads or lists a directory, tell it about that directory's instruction file |
| `nested_inject` | `"pointer"` | what nested traversal adds: `"pointer"` (one line naming the file) or `"contents"` |

A repository can declare plugins too, in `.clai/settings.json`; they show as
`(project)` and rank just above the built-ins. They start off, because a
repository must not run code as you just because you opened it: CLAI names the
ones waiting at startup, and `/plugins enable NAME` approves one. See
[Project settings](README.md#project-settings).

`compaction` directly registers harness `FallbackCompaction` with
`max_fraction=threshold`; harness owns the automatic trigger. `/compact` runs the
same chain unconditionally. Only `ModelAPIError`, `FallbackExceptionGroup`, and
`UsageLimitExceeded` cause summarisation to fall back to truncation; other exceptions
propagate. `/plugins disable compaction` turns automatic compaction,
`/compact`, and its context warning off; a declaration under the same name
changes its settings (`strategy`, `threshold`, `protected_tokens`,
`context_window`, `summarization_model`; see the README):

```text
/plugins add compaction pydantic_clai2.compaction '{"threshold": 0.7, "context_window": 200000}'
```

### `ask_user`: questions answered from the terminal

The second built-in, `ask_user` (`pydantic_clai2.ask_user_menu:activate`), gives
the model the harness's `AskUser` capability: one tool, `ask_user_question`, for
asking you one to ten multiple-choice questions when the task is ambiguous. Each
question appears inline above a compact numbered picker, keeping the conversation
visible. Up/Down moves the highlight; Enter or an option's number selects it.
For multi-select questions, Enter or a number toggles that choice; select `Done`
to submit at least one choice. The title says `question 2 of 3` when there are
several. Esc or Ctrl-C declines the whole request and lets the model continue.
The picker uses `host.full_screen()` only to flush streaming output and suspend
the editor's input reader. It does not switch to the alternate screen. The draft
is restored on exit, and your picks are printed to the transcript afterwards.
`/plugins disable ask_user` takes the tool away.

The inline `ask_user_question` picker also offers `Other (type answer)`.
Choose it to type your own answer instead of the suggested options, including for
multi-select questions. Enter submits nonblank text. Esc returns to the choices
and keeps your draft; Ctrl-C declines the whole request. Backspace and arrow keys
edit the text. Multiline paste is inserted as text and waits for Enter; it does
not submit an answer or select choices. The conversation stays visible while you type. Custom answers
appear in the transcript and reach the model as a one-item list under the question's header.

The capability does not know it is in a terminal. It hands an `AskUserRequest`
to an `Answerer` (one async callable returning an `AskUserResponse`) and waits.
To answer questions somewhere else, a web page or a chat bridge, say, replace
the built-in with your own plugin under the same name that constructs `AskUser`
with a different answerer:

```python
from pydantic_ai_harness.ask_user import AskUser, AskUserAnswer, AskUserRequest, AskUserResponse

from pydantic_clai2.plugins import PluginHost


async def ask_over_http(request: AskUserRequest) -> AskUserResponse:
    # POST request.questions to your front end, keyed by request.id, and wait
    # for the reply; return AskUserResponse(cancelled=True) if the user dismisses it.
    picks = [AskUserAnswer(header=q.header, selected=(q.options[0].label,)) for q in request.questions]
    return AskUserResponse(answers=tuple(picks))


def activate(host: PluginHost[None]) -> None:
    host.add(AskUser(answerer=ask_over_http))
```

```text
/plugins add ask_user my_ask_user
```

The request and its response are also emitted as `AskUserRequestedEvent` and
`AskUserAnsweredEvent`, so a plugin that only wants to watch (log the question,
show a "waiting for you" state) registers `@host.on(EventClass)` or
`@host.render(EventClass)` without being the answerer.

## Managing plugins

`/plugins` on its own opens a full-screen menu, the same kind Code Puppy uses
for `/agent` and `/mcp`:

```text
 Plugins

 [x] coder         pydantic_ai_harness.coder:Coder   | coder
 [x] notify        ~/.config/pydantic-clai2/plugins  | source  ~/.config/.../notify.py
 [ ] audit         my_package.audit                  | state   enabled, loaded
                                                     | adds    2 commands, 1 hook, 0 tools
                                                     | error   none

 Up/Down move - Space enable/disable - R reload - D remove - Enter/Q close
```

The left side lists every plugin with `[x]` for on and `[ ]` for off. The right
side shows details for the highlighted one: where it came from, whether it
loaded, what it registered, and the last error if loading failed. Every key
acts immediately; there is no save step, so Enter, Q, Esc, and Ctrl-C all just close.
Closing returns to the prompt without printing the plugin list. Use `/plugins list`
to print it.
Adding a plugin needs a name and a module, so that stays a typed command.

With arguments `/plugins` is a plain command, and `clai2 plugins ...` outside
CLAI does the same thing:

| Command | Does |
|---|---|
| `/plugins list` | show every plugin and whether it is on |
| `/plugins add NAME module[:attr] [JSON]` | save it and load it now |
| `/plugins remove NAME` | forget an installed declaration; persistently disable a drop-in (delete its file yourself to remove it); reset a built-in or project-declared plugin to its declaration |
| `/plugins enable NAME` / `disable NAME` | load or unload, remembered across restarts |
| `/plugins reload NAME` | re-import the file and load it again (for editing a plugin while CLAI runs) |
| `/reload` | reload CLAI's own Python modules for development and rebuild the shell without restarting the process |

`/reload` takes no arguments. It uses `importlib.reload`, preserves the conversation,
agent, selected model, and active settings, and reactivates enabled plugins against
the refreshed shell types. Each loaded plugin receives `session_end` before reload
and `session_start` when loaded again. Plugin hosts and their registrations are
recreated, but installed module globals not overwritten by the new source can
survive. Initialize mutable state in `activate`. Disabled and unapproved project
plugins stay off. Failed imports or shell rebuilds restore the
previous module bindings and report the error, but cannot undo import-time side
effects. Reload ordering is planned from current module-scope source imports,
including newly referenced local modules and package initializers. Lazy imports
inside functions and `TYPE_CHECKING` guards do not create eager dependencies;
new modules are imported only when reached by the updated code. Literal guards
and direct platform/version comparisons select their active branch without
executing guard expressions. Other conditions are analyzed conservatively and
may require a restart if their alternatives form a cycle. Invalid source
or a detected import cycle fails before module reloads begin. Restart for changes
to startup code, dynamically loaded dependencies, or agent construction.
Third-party dependencies are not recursively reloaded.

If loading fails or is cancelled, registered `session_end` handlers receive
`reason='error'` under cancellation shielding before the partial host is dropped.
Each handler has a five-second cooperative timeout. Errors and timeouts are
reported separately, and remaining handlers are still attempted. Register cleanup
once a resource is owned; cleanup may run before `session_start` finishes.
Handlers must cooperate with cancellation: blocking code and additional shields
can exceed that timeout. Caller cancellation still propagates after cleanup;
cleanup errors do not replace the original load error.

What "load" and "unload" mean for your plugin:

- Load runs `activate(host)` and then fires `session_start` for that plugin, so a
  plugin loaded mid-session sees the same first event as one loaded at start.
- Unload fires `session_end` for that plugin, then drops everything it
  registered: handlers, commands, tools, renderers, status fragments. Nothing else is touched.
- Both only happen between prompts, never while the agent is running.
- Drop-in entry modules load from current source. Installed entry modules use
  `importlib.reload`, which retains globals absent from the new source. Initialize
  plugin state explicitly on activation.

## The first plugin: give the agent web search

Pydantic AI Harness ships capabilities that are plugins as they are. `ExaSearch`
adds `web_search` and `get_page` tools backed by [Exa](https://exa.ai). Install
the extra and set the key, then add the class by name:

```sh
pip install 'pydantic-ai-harness[exa]'
export EXA_API_KEY=...
```

```text
/plugins add exa pydantic_ai_harness.exa:ExaSearch '{"num_results": 8}'
```

The JSON is passed to the constructor, so any keyword `ExaSearch` accepts that
JSON can express works here; an option that takes a Python object, like
`client`, needs the file form below. Ask the agent something that needs the web on the next prompt and it has
the tools. `/plugins disable exa` takes them away again.

The same thing as a plugin file, `~/.config/pydantic-clai2/plugins/search.py`,
which is the shape to start from when you want more than one capability, or a
`/command`, or a hook alongside it:

```python
from pydantic_ai_harness.exa import ExaSearch

from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost) -> None:
    host.add(ExaSearch(num_results=8, text_summary=True))
```

A plugin module exposes one function, `activate(host)`. CLAI calls it once when
the plugin loads. Everything you register inside it stays until the plugin is
unloaded or CLAI quits. Any Pydantic AI capability goes through `host.add`, so
`YouSearch` from `pydantic_ai_harness.youdotcom`, core's `WebSearch`, or one you
wrote yourself all work the same way.

`module:attr` may also name a function that takes a host, if you prefer a name
other than `activate`.

## Reacting to a moment

Plugins are not only for tools. This one rings the terminal bell when a turn
finishes, so you can tab away during a long run:

```python
from pydantic_clai2.plugins import PluginHost, TurnEnd


def activate(host: PluginHost) -> None:
    @host.on('turn_end')
    async def ping(event: TurnEnd) -> None:
        host.console.bell()
```

## What you can register

### React to a moment: `@host.on(name)`

`name` is a string. Your editor autocompletes it, and the type checker knows what
`event` your handler receives for each name. A typo is an error, not silence.
`host.on` is always used as a decorator.

Four names belong to CLAI itself. They fire outside the agent run, in the shell:

| Name | When | Event fields | Can change things? |
|---|---|---|---|
| `session_start` | CLAI has started, before the first prompt | `agent`, `settings` | no |
| `session_end` | CLAI is quitting | `reason`: `exit`, `eof`, or `error` | no |
| `turn_start` | you pressed Enter on a prompt | `text` | yes: edit `event.text`, or `event.cancel()` |
| `turn_end` | the turn finished, failed, or was interrupted | `text`, `outcome`, `result`, `error` | no |

Ctrl-C during an agent run keeps the prompt and captured partial messages in
conversation history for the next turn. Cancellation still reaches the running
tools for cleanup; it does not undo completed side effects or retry the run.
Retained failed turns and restored interrupted sessions are marked interrupted so
core can close unanswered tool calls on the next prompt without replaying them.
A prompt cancelled by `turn_start` never starts an agent run and is not retained.

Every other name is a Pydantic AI lifecycle hook, spelled exactly as on core's
`Hooks().on`, with the same handler signature. The ones people reach for:

| Name | When |
|---|---|
| `before_run` / `after_run` | an agent run starts / finishes |
| `before_model_request` | just before the model is called; you can edit the request |
| `before_tool_execute` | a tool is about to run; raise to stop it |
| `after_tool_execute` | a tool has returned |
| `tool_execute_error` | a tool raised |
| `event` | every stream event; prefer `@host.on(EventClass)` below |

The full list and every signature are in the
[hooks reference](https://pydantic.dev/docs/ai/core-concepts/hooks/). Core's
`wrap_*` and `on_*_error` methods drop their prefix here (`run`, `tool_execute`,
`run_error`), matching `Hooks().on`.

### React to a typed event: `@host.on(EventClass)`

Tools and capabilities emit typed events (a shell started, a file was written).
Pass the class instead of a string:

```python
from pydantic_ai import RunContext
from pydantic_ai_harness.filesystem import FileWrittenEvent
from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost[None]) -> None:
    @host.on(FileWrittenEvent)
    async def log_write(ctx: RunContext[None], event: FileWrittenEvent) -> None:
        host.console.print(f'wrote {event.path}')
```

Some events let you say no. If the event has a `cancel()` method, calling it stops
the action before it happens (for example `FileChangeRequestEvent`).

### Add a `/command`: `host.commands.register(...)`

```python
from pydantic_clai2.commands import Command

host.commands.register(
    Command(
        name='greet',
        description='Say hello',
        handler=lambda args: 'Hello ' + (' '.join(args) or 'there'),
    )
)
```

The handler gets the arguments as a list of strings and returns the text to show.
It may be `async`. Add `complete=` to offer Tab suggestions. The registry filters
command names and returned candidates by case-sensitive substring, replacing the
whole typed fragment when selected. Return full candidates, not just suffixes.
Names must be unique;
clashing with a built-in is an error at startup, not a silent override.

Path-like input is not dispatched to commands. A slash, dot, or backslash in the
first token after the leading `/` makes it a prompt instead, so screenshot paths
such as `/Users/me/Desktop/Screen Shot.png` appear as follow-ups in the queue.
For path text not converted to an image attachment by the editor, surrounding
whitespace is trimmed and the remaining text reaches the agent without further
rewriting. Quoted paths are prompts too. Unknown command-shaped names such as
`/missing` still report an error. Routing itself does not read files; the editor's
separate image-paste handling can attach existing images before routing. See
[Image input](#image-input) for the resulting hook payloads.

### Give the agent tools or instructions: `host.add(capability)`

```python
from pydantic_ai.capabilities import Capability

tools = Capability(instructions='Prefer British spelling.')


@tools.tool_plain
def word_count(text: str) -> int:
    return len(text.split())


host.add(tools)
```

`host.add` also accepts a function that takes a `RunContext` and returns a
capability (or `None`), for tools that should only exist in some runs.

Markdown in streamed answers and thinking uses OSC 8 hyperlinks for link labels
when writing to a terminal. The URL is also shown as text. Transcript replay keeps
hyperlinks after resize, but does not replay clipboard, title, or palette commands.
Redirected Markdown output does not emit hyperlinks. Destinations longer than
2,048 characters stay visible but do not get clickable metadata.

### Draw an event yourself: `@host.render(EventClass)`

Built-in tool rendering shows one summary line per call by default, clipped to
the terminal width and followed by a blank line. Tool names are pink; arguments
and bullet markers are muted grey. Shell output and completion details, grep results, and file
diffs are hidden from the terminal, not from the model. Set
`/set display.tool_output true` to restore detailed output; `display.shell_lines`
and `display.grep_lines` then control preview lengths (20 lines each by default).
This setting does not suppress plugin renderers or interactive questions.

CLAI shows unknown tool calls as `● tool_name`, with the name in pink. To show something
better, return a Rich renderable (a `str` is fine). Return `None` to say "not mine,
use the default".

```python
from pydantic_ai_harness.filesystem import FileWrittenEvent
from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost[None]) -> None:
    @host.render(FileWrittenEvent)
    def show_write(event: FileWrittenEvent) -> str:
        return f'wrote {event.path}'
```

CLAI flushes any streaming text before it prints what you return, so your output
never lands in the middle of a paragraph.

### Take the whole screen mid-run: `async with host.full_screen()`

A widget opened from inside a tool call, including the inline `ask_user` picker,
has to wait for streamed text to finish and the editor and status row to
get out of the way. `host.full_screen()` flushes pending output, suspends the
editor's input reader, and restores the editor and its draft when the block exits.
The editor remains active during agent turns. Enter steers the active run through
core's `RunContext.enqueue(priority='asap')`, without starting another turn or
cancelling tools. Alt+Enter queues a separate turn with its own `turn_start` and
`turn_end` hooks. Slash commands always wait until the current turn ends.
When idle, either submit key starts a turn. Shift-Enter inserts a newline.
Modified-key reporting is enabled only while the editor owns input.
Option+Backspace (Alt+Backspace) deletes the word before the cursor, like Ctrl-W,
including trailing whitespace. Spaces, tabs, and newlines separate words. Text
after the cursor is preserved. Your terminal must send Option as Alt/Meta for
this shortcut; legacy and modified-key encodings are supported.

Completion rows remain visible while a replacement lookup runs, but stale results
cannot be selected. Popup height changes reuse available space without adding
blank transcript lines on each key. Completion providers should be read-only.
Their errors are shown in the footer rather than ending the session; on menu
handoff or shutdown, a blocked synchronous lookup may finish in the background
and its result is ignored. A single daemon worker and a latest-only queue bound
this work; a stuck provider delays further lookups, not input or process exit.
While work or turn lifecycle
hooks are active, a `Working` label and spinner appear in the editor's top border.
The spinner uses the same pink `ACCENT` as tool names, while the label and border
stay muted. The indicator appears
without adding an input row or changing the draft. The indicator uses the editor's refresh cycle, adds no
background task, and is hidden while a full-screen interface owns the terminal.
The editor reserves bottom rows with terminal scrolling margins. Both partial
and complete output go straight to the transcript region, without suspending or
repainting the input box. The shell paints changed editor rows itself, using
Termflow layout helpers; it does not run a prompt-toolkit renderer. Its cursor
is a nonblinking highlighted cell, separate from the transcript cursor.
Resize blanks the visible viewport and buffers transcript writes until the size
has been stable for 250 ms. It then replays a bounded recent transcript tail and
restores the draft; it does not erase terminal scrollback or conversation history.
The buffer includes startup and plugin lifecycle output. It retains ANSI styling,
not arbitrary terminal-control operations.
Use `host.full_screen()` for widgets instead of printing cursor-control sequences
into the transcript. Large output bursts during resize spill to a private temporary
file and are flushed in order after the viewport is rebuilt.
The Termflow smoothing defaults match Code Puppy: responses use 12 ms ticks, a 0.5-second
catch-up window, and at least one character per tick; thinking uses 20 ms ticks,
a 0.4-second window, and at least two characters per tick, and renders as dimmed
Markdown after the `Thinking` heading on the same row. Plugins do not need their own redraw logic. Pending message previews appear
above the editor in execution order (`Follow-up:` for messages, `Command:` for
slash commands), and disappear when consumed. The preview is read-only; clipping
and flattening multiline text for display do not change the submitted text.
Control bytes are escaped in completion labels, queued previews, and prompt echoes
rather than being executed as terminal commands.
Esc in the live editor cancels active work, including turn lifecycle hooks,
without clearing the draft or requesting exit. While a plugin owns the screen,
its menu retains control of Esc.
Slash-command handlers already
run with the editor suspended; tool-driven widgets must take the screen explicitly:

```python
from pydantic_clai2.plugins import PluginHost


async def choose(host: PluginHost[None]) -> str:
    async with host.full_screen():
        return await show_my_menu()
```

When no editor or stream is active, taking the screen is a no-op. It only settles
the screen; drawing, and restoring the terminal afterwards, is the widget's job. One
widget owns the screen at a time: a second `full_screen()` (from a parallel tool
call, say) waits for the first block to exit. Do not nest it inside itself.

### Read your settings: `host.settings(Model)`

The JSON passed to `plugins add` is validated against a model you define:

```python
from pydantic import BaseModel


class NotifySettings(BaseModel):
    sound: bool = True


settings = host.settings(NotifySettings)
```

Bad or missing values fail at startup with a message naming your plugin.
CLAI ignores unknown names in its own saved settings and preserves their values for
other versions or branches. This does not relax validation of plugin declarations
or `host.settings(Model)`.

### Reach the conversation and the status row: `host.conversation`, `host.status`

`host.conversation` is the retained history: `messages` is a snapshot,
`await commit_messages(...)` persists and swaps it between turns, and `resolved_model()` is the
model the next prompt will use. `host.status` is the footer's state; set
`context_alert` to paint the context figure in the warning colour. The built-in
`compaction` plugin uses both. A host built outside the shell gets an in-memory
`Transcript` and a detached `Status`, so tests need no special case. The status
row itself is CLAI's; a plugin adds to it with the next registration.

### Add to the status row: `host.status_segment(fn)`

`fn` takes no arguments and returns a short string. It is appended after the
built-in figures, painted muted, and dropped when the plugin unloads.

```python
import os

from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost[None]) -> None:
    @host.status_segment
    def where() -> str:
        return os.getcwd()
```

The row is repainted about ten times a second, so `fn` runs that often: keep it
cheap, synchronous, and free of blocking IO. Return `''` to contribute nothing
for a frame. Fragments are truncated from the right on a narrow terminal and are
readable text only; the colours in the row belong to CLAI. A fragment that raises
shows its error name in the row instead, so one broken plugin cannot take the
footer down.

## Rules that keep plugins predictable

- Handlers are `async`. There is no sync variant of anything.
- CLAI host observers return `None`. Core hooks keep their exact core return
  contracts: for example `before_model_request` must return its `ModelRequestContext`.
  For cancelable host events, edit the event or call `event.cancel()`.
- Raising in `turn_start` prevents the turn. Raising in core `before_tool_execute`
  fails the agent run, not only that tool. Use core's documented tool-denial
  mechanisms when the model should recover instead of ending the run, such as
  `pydantic_ai.exceptions.SkipToolExecution` for skipping an individual tool.
- Startup plugins load in alphabetical ID order. CLAI host handlers run in
  activation order, while `/plugins list` remains alphabetical. Core hook ordering
  follows core composition, including reverse order for `after_*` hooks. The first renderer that returns something wins. A
  plugin loaded later goes to the end of the line.
- Anything you print, print through `host.console`, so it stays in step with
  streaming output.

## Testing a plugin

`PluginHost` is an ordinary object. Build one with a `Console` writing to a
`StringIO`, call your `activate`, then call the handlers you registered with
hand-made events. No terminal, no model, no network.

```python
from io import StringIO

from rich.console import Console

from pydantic_clai2.plugins import PluginHost, TurnEnd

host = PluginHost(name='notify', console=Console(file=StringIO()), settings={})
activate(host)
for handler in host.handlers:
    await handler(TurnEnd(text='hi', outcome='completed'))
```

A plugin that reads the history gets a `Transcript` by default; pass
`conversation=Transcript(messages=[...], model=TestModel())` to seed it.

## vllm connection

Open `/add_model`, choose `vllm`, then enter a trusted HTTP(S) server root or `/v1` URL, and optionally a token. CLAI queries `/v1/models` and opens a searchable model picker. HTTP sends tokens unencrypted; use HTTPS outside trusted local networks.

## openrouter connection

Open `/add_model`, choose `openrouter`, then choose **Sign in with browser** or **Enter API key**. Browser sign-in opens OpenRouter's [PKCE authorization flow](https://openrouter.ai/docs/use-cases/oauth-pkce) and receives an authorization code on a temporary loopback listener. CLAI exchanges the code for a user-controlled API key over HTTPS. If the browser cannot reach CLAI (for example over SSH), paste the final callback URL or authorization code into the terminal. If no browser opens, open the printed authorization URL manually. Login times out after five minutes; Ctrl-C cancels it. You can revoke the generated key on OpenRouter.

Manual entry still accepts a key from https://openrouter.ai/keys in a masked prompt. After either method, select a model from the live catalog. CLAI validates the key with `/api/v1/key` before fetching `/api/v1/models`. Cancelling before model selection leaves the saved connection unchanged.

The connection is saved in the configured Python keyring backend after selection; backend security depends on your keyring configuration. Tokens are not stored in SQLite or command history. The selected model persists across restarts. Select the provider again to browse its live models or reconfigure the saved connection. Discovery is explicit and has a 20-second network timeout; redirects are not followed. Agent inference uses Pydantic AI core.


## Saved API keys

Open `/keys` to browse and manage saved API keys in a full-screen menu.
Use A to add, Enter to replace a value, R to rename, and D to delete with
confirmation. Values are masked and are not shown in previews. Changes save
immediately; Esc or Ctrl-C closes the menu. Errors appear inside the menu.

The compatibility command `/set api_key` prompts for a name and masked API key. Names are trimmed
and uppercased automatically, so `my_vllm_key` becomes `MY_VLLM_KEY`. Use letters,
numbers, and underscores, starting with a letter or underscore. Saving an existing
name asks before replacing it. Ctrl-C or Ctrl-D cancels without saving. Do not put
the secret on the command line.

When saved keys exist, vLLM's token prompt and OpenRouter's **Enter API key** flow
show a searchable list of names. Choose one, enter a different key privately, or
choose **No API key** for vLLM. Esc closes the picker without connecting. Browser
login flows are unchanged. Select keys only for endpoints you trust.

Named keys use the existing credential backend, separate from provider logins and
SQLite settings. If no OS keyring exists, CLAI warns that it saved them in the
per-user `0600` plaintext file `credentials-api-keys.json` in its config directory.
Key values never appear in the picker or confirmation. Names are labels, not
exported environment variables. Selecting a saved key stores a reference, not a copy. Discovery and each new
turn resolve its current value. Replacing a key updates connections that reference
it. Deleting it makes those connections fail until you restore the same name or
reconfigure them. Keys referenced by saved connections cannot be renamed. A cross-process lock
serializes key changes and connection saves so concurrent CLAI sessions do not
overwrite each other's key edits. The lock file contains no credentials.

Existing connections with inline credentials, manually entered connection keys,
and browser logins remain unchanged. To switch an existing connection to a
reference, reconfigure it through `/add_model` and select a saved key. Changes do not
alter an already running request or revoke credentials at the provider.

### Adding and selecting models

`/add_model` opens the provider catalog, connection setup, and per-model settings.
`/add_model PROVIDER:NAME` adds a model directly. Adding a model also selects it
for the next prompt. `/model` is a flat picker of added models, and `/model NAME`
selects one without the menu. Its Tab suggestions contain only added models.
The list persists across sessions. The currently configured model is retained
when upgrading; `/set model NAME` also saves the model in this list.

### Terminal themes

Theme selection and cancellation do not print status messages. Terminal colour
controls are never replayed as conversation text.

```text
/theme tokyo_night
/set display.theme github_light
```

`/theme` without arguments opens a searchable picker with a sample conversation,
including Markdown, thinking, tool output, code, warnings, errors, and the input
area. Browsing previews colours without applying them; Enter confirms, and Esc
or Ctrl-C cancels. Choices are `default` and `termflow.themes.PALETTES`.
`default` preserves CLAI's existing appearance without changing terminal colours
on startup or exit. `/theme default` restores it after a palette selection.
The picker, `/set`, project settings, and persisted `display.theme` values share
validation. A project override takes precedence again on the next startup.

```python
from rich.console import Console
from pydantic_clai2 import theme

Console().print('Ready for your next prompt.', style=theme.color(theme.INFO))
```

Resolve the `ACCENT`, `INFO`, `WARNING`, `ERROR`, `MUTED`, and `THINKING` roles
through `theme.color(role)` at render time. The constants retain their brand
values; the resolver reads the active palette. `theme.sgr(role)` resolves raw
ANSI surfaces itself. `theme.current()` returns the selected Termflow
`TerminalPalette`, or `None` for the existing default appearance.

Selecting a bundled palette changes terminal foreground, background, and ANSI
slots via Termflow's OSC sequences. CLAI resets them to terminal defaults when
you return to `default` or exit a selected palette, including errors and
cancellation. Redirected output receives no palette-changing sequences.
Unsupported terminals may ignore changes; supported ones may recolour ANSI
scrollback. The early splash retains brand colours. Code uses the terminal
foreground and ANSI syntax colours.
Diff colours stay unchanged in `default`; bundled palettes use Termflow's diff
defaults. Plugins cannot register custom palettes. Theme selection adds no model
requests, hooks, or telemetry.
### Model settings and custom parameters

`/model_settings` opens a searchable list of added models. Enter configures a
model without changing the active model. Esc returns from settings to this list;
Esc again closes it. `/model_settings PROVIDER:NAME` opens that model directly.
Tab completes added models.
`Ctrl+S` in `/add_model` opens the same editor. Edits save immediately and apply
on the next prompt. `r` resets a field; Esc or Ctrl-C goes back. Fixed choices
open a picker; numeric fields accept typed values, and empty input resets.

Model preferences are shared across checkouts. Reading saved preferences ignores
unknown fields, so newer settings do not break an older reader with this
compatibility fix. Editing or resetting a known field preserves unknown fields
in the store. New edits still reject unknown keys and invalid values.
An invalid value in a known field stops that turn with a repair message, not the
shell; use `/model_settings` to fix or reset it and try again. CLAI does not silently
run with different settings or delete saved preferences.

Older branches must receive this fix too. The minimum read-side backport is
`ModelSettingsForm.model_validate(values, extra='ignore')` in
`model_settings_from_json`; keep the form itself strict. Also backport preservation
of unknown keys on save and the shell's model-settings validation error handler.

The editor offers OpenAI reasoning effort, Responses reasoning context, mode,
summary, and verbosity, and Claude classic/adaptive thinking and effort.
The editor hides generic request fields such as timeouts and penalties.
Reasoning GPT models do not show sampling controls. Previously saved overrides
remain visible so they can be reset. Choices depend on the model and API: Chat Completions does not get Responses
controls. OpenRouter and vLLM GPT routes expose Chat Completions reasoning effort
and service tier, not Responses-only controls. `all_turns` appears only on compatible models, and adaptive Claude
models do not get a token budget. Classic thinking budgets must be at least
1024 and below an explicit `max_tokens`. If classic thinking has no output cap,
CLAI reserves the thinking budget plus 4096 output tokens. Other unset fields
use the provider default.
Explicit native thinking settings take precedence over generic `thinking`.
GPT-6 and GPT-5.6 families, including provider-qualified and namespaced names,
default to `thinking=true`, `service_tier=default`, reasoning effort `medium`,
context `all_turns`, mode `standard`, summary `detailed`, and verbosity `low`.
Explicit per-model values win; reset restores the family default without saving
it as an override. Other models keep their existing defaults. Provider-specific
fields are consumed only by APIs that support them; this does not add Responses
controls to Chat Completions or other protocols.

Code Puppy runtime parity is not complete. In particular, its progress-aware
main/sub-agent streaming retries require core recovery support before CLAI can
expose working retry controls. See [the parity audit](MODEL_SETTINGS_AUDIT.md).

Pydantic AI owns adaptive-thinking translation and preserved-thinking replay,
including Fable 5.1's recovery when a changed conversation prefix invalidates a
thinking block. CLAI does not strip thinking or implement a second recovery loop.
See [core's thinking block binding documentation](https://pydantic.dev/docs/ai/models/anthropic/#thinking-block-binding).

For native Anthropic models, **Preserved Thinking** controls
`thinking.block_binding.prefix_mismatch_behavior`: `error` rejects a mismatched
prefix; `drop_block` continues without the mismatched reasoning block. It appears
on adaptive-capable Claude models. Fable 5.1 also exposes **Thinking Display**:
`updates` or `summarized`. Setting either control without a thinking mode selects
adaptive thinking; neither can be combined with disabled thinking. Resetting the
mode clears its budget, display, and binding overrides. **Interleaved Thinking**
adds the beta header on classic Claude 4 models. CLAI adds the display beta when
requesting updates; core adds the block-binding beta. These native controls do
not add Anthropic protocol support to third-party Chat Completions endpoints.

GLM-4.5 and newer expose **Thinking (GLM)** and **Clear Thinking (GLM)**;
GLM-5.2 and newer also expose **Reasoning Effort (GLM)**. Clear Thinking set to
true clears earlier reasoning; false preserves it. These controls send GLM's
native `thinking.type`, `thinking.clear_thinking`, and `reasoning_effort` body
fields. Only explicit overrides are sent. Disabled thinking cannot be combined
with an effort override. A proxy that needs `chat_template_kwargs` instead of
this native shape still needs custom parameters. Custom parameters win over the
generated body on conflict.

Open `custom_params` for Code Puppy-style **Custom Params**. Enter adds or edits
`key = value`; editing the key renames it. `d` deletes a pair and Esc goes back.
Dotted keys nest in the request body's `extra_body`, for example:

```text
chat_template_kwargs.thinking = medium
reasoning.effort = max
```

Values accept JSON booleans, numbers, null, arrays, and objects, or unquoted
text. Quote numeric-looking strings to keep them strings. Parameters are saved
per model and applied last, overriding built-in request fields on conflict.
An extra-body object replaces the corresponding generated object, rather than
deep-merging it. For example, overriding `reasoning.effort` replaces the generated
`reasoning` object; include custom `reasoning.context` too if you need both.
They deliberately bypass the model compatibility checks: the endpoint must
support what you send. This is also the escape hatch for custom endpoints and
provider options not listed in the form. Reset `custom_params` to remove all
pairs. Do not put credentials here: values are stored as plaintext in SQLite.

### Persisting conversation changes

Use `await host.conversation.commit_messages(messages)` for between-turn history
changes. It commits to storage before replacing the live history, and rejects
changes while an operation is running. `replace_messages(...)` remains an
in-memory compatibility API; it does not save by itself. `Transcript` implements
`commit_messages` without disk IO for headless plugin tests.

`host.conversation.step_store` is the configured Harness `StepStore`, or `None`
for an in-memory host. The built-in `persistence` plugin binds
`StepPersistence(capture_frontier=True)` to it. Do not register a second recorder
for the same store and run. Removing the plugin removes step capture on subsequent
turns; conversation-head saving is owned by `Session` and continues independently.

Harness exports `SnapshotSaved` from `pydantic_ai_harness.step_persistence`.
Subscribe through `@host.on(SnapshotSaved)` to observe committed checkpoints.
It carries `persistence_run_id`, `conversation_id`, `step_index`, and `state`.
This is a notification, not the durable source of truth or permission to replay a
tool. A durable replay may notify again. An observer failure cannot roll back the
already committed snapshot. No new CLAI lifecycle hooks are introduced.

Session naming is a shell-owned background service over Harness's `SessionNamer`.
It never writes into the agent transcript or loads plugin code. `/resume` does
not fire plugin load/unload hooks or restore previous plugin approvals. Cross-project
resume keeps the current working directory and the saved conversation's original
project grouping. The
project/session browser is a dedicated Termflow widget: unlike a single-pane
`MenuBuilder`, it has two independently navigable panes and two-line cards. Its
pure frame and scripted-key tests follow the same headless menu conventions.
The selected project stays highlighted while browsing sessions. The focused pane
is labeled **SELECT PROJECT** or **SELECT SESSION**, with matching key hints.

The resume transcript preview displays at most 24,000 characters of the newest-first
text, with a truncation notice for longer histories. Search is Unicode
case-insensitive and includes text instructions in multimodal prompts.

## Image input

Clipboard and image-path paste are part of the shell's prompt editor, not a plugin
API. Ctrl-V or Alt-V attaches clipboard images; bracketed paste of existing image
paths attaches local files. See [Pasting images](README.md#pasting-images) for
platform requirements and limits.

`turn_start.text` and `turn_end.text` contain the text caption with attachment
markers removed, possibly an empty string for an image-only turn. A `turn_start`
handler may rewrite the caption or cancel the entire turn. Rewriting the text
does not remove the images. Core hooks receive the native multimodal request with
`BinaryContent` image parts. Plugins that inspect or transform image content
should use core hooks rather than parsing terminal markers. There are no new
host lifecycle hooks. Images are persisted with the conversation, including the
accepted request when a turn fails or is cancelled.

## Headless CLI runs

`clai2 -p "PROMPT" [-m PROVIDER:NAME]` runs one saved turn without an editor.
Session and turn hooks still run; stream renderers do not. Host console output
is suppressed, and stdout contains only the final answer. Plugin load failures
abort the run. The `ask_user` plugin is skipped even if saved settings enable or
replace it; this does not change those settings. `host.full_screen()` raises in
headless mode. Plugins must not bypass the host by reading terminal input or
printing directly to stdout. `--resume SESSION-ID` restores history without a
browser or tool replay.
