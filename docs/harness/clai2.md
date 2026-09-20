# CLAI 2.0

A separately installable terminal client for Pydantic AI, using `Coder(unrestricted_filesystem=True)` by default.
Python 3.11+ is required by Termflow. Tracking issue: https://github.com/pydantic/pydantic-ai-harness/issues/875.

CLAI file tools can access paths outside the workspace, including `/tmp`, and do
not protect secret files or repository metadata. OS permissions still apply.
Relative paths use the launch workspace. Use a custom agent with `Coder()` to
retain workspace-scoped file tools. Shell output is displayed dimly.

## Word deletion

Option+Backspace (Alt+Backspace) deletes the word before the cursor, like Ctrl-W,
including trailing whitespace. Spaces, tabs, and newlines separate words. Text
after the cursor is preserved. Your terminal must send Option as Alt/Meta for
this shortcut; legacy and modified-key encodings are supported.

## Interrupting a turn

Press Ctrl-C once to cancel the active agent turn and return to input. Tool cleanup
and terminal restoration finish before the next prompt. Press Ctrl-C again within
two seconds to exit, including across the transition back to input. At the prompt,
the first press clears input and the second exits. Ctrl-D and `/exit` also quit.
External application cancellation still propagates; cancelled turns are not added
to conversation history, but completed tool side effects cannot be undone.

## Input history

Submitted prompts and slash commands persist across restarts for Up/Down recall,
including multiline input. They are stored as plaintext in `input-history` next
to `config.db`: `$XDG_CONFIG_HOME/pydantic-clai2/input-history`, or
`~/.config/pydantic-clai2/input-history` by default. On POSIX the file is restricted
to its owner (mode 0600). Avoid entering secrets in the prompt: input history is
not encrypted. Delete this file while CLAI is closed to clear saved input.
`/new` clears model conversation history, not input recall. Model responses and
tool results are not saved to this file.

## CI coverage

The `CLAI coverage` check combines branch coverage from Python 3.11 and 3.14
and requires 100% for `src/pydantic_clai2`. It is separate from Harness coverage;
passing CLAI test jobs alone does not mean either coverage gate has passed.
Tracked under [#875](https://github.com/pydantic/pydantic-ai-harness/issues/875).

## Start chatting

Launch `clai2`. The default model is `openai-codex:gpt-6-astra`.
Run `/login openai-codex` to connect your ChatGPT/Codex subscription.
Type `/set model ` and press Tab to pick another provider-qualified model name.
The choice is saved in SQLite and used for the next prompt without restarting.

From a source checkout, launch with `uv run --project pydantic-clai2 clai2`.

For API-key providers, set the provider's API key environment variable before starting.
Codex uses subscription OAuth instead, not `OPENAI_API_KEY`. The default Coder
can read and modify files and execute commands with your user permissions. Run it
in a workspace you trust. CLAI does not add a sandbox or approval layer.

The startup splash adapts Code Puppy's stdlib-only, alternate-screen Pydantic
pyramid, with CLAI lettering. The persistent `CLAI 2.0` banner uses `ansi_shadow`.
The splash is disabled for redirected output, CLI arguments, small terminals,
Windows, `NO_COLOR`, or `CLAI_NO_SPLASH=1`.

## Git worktrees

```bash
py-cli clai2 --worktree my-task
```

```bash
py-cli clai2 -w
```

A Git worktree is another checkout of the same repository with its own branch
and working files. Run these commands inside a repository with at least one
commit. `--worktree NAME` creates a `clai/NAME` branch from the current `HEAD`
and starts CLAI at `<repository-root>/.worktrees/NAME`.
`-w` is the short form; omit the name to generate one. Names start with a letter
or digit and contain only ASCII letters, digits, hyphens, and underscores.

After checkout succeeds, CLAI adds `/.worktrees/` to Git's local `info/exclude`
file to keep generated checkouts out of `git status`, without changing your
tracked `.gitignore`.
Uncommitted changes, ignored files, and untracked files are not copied. Project settings, repository instructions, and coding
tools use the new worktree root. Your user settings and plugins stay available;
a relative `--database` path still refers to the directory you launched from.

CLAI prints the new path and branch. Existing branches and non-empty directories
are rejected. If checkout fails, CLAI tries to remove only the branch it just
created, without forcing deletion. If cleanup or the ignore edit fails, the error
names the retained branch or checkout for recovery. The worktree and branch
remain after exit, including startup
errors after creation, so CLAI does not delete your work. Enter that directory
and run `clai2 --resume` to continue a saved session. `--worktree` cannot be
combined with `--resume`, `config`, or `plugins`.

When you no longer need the checkout, use Git's own cleanup commands from your
original repository root. Without `--force`, Git refuses to remove a dirty worktree:

```bash
git worktree remove .worktrees/my-task
git branch -d clai/my-task
```

!!! warning "Worktrees are not sandboxes"
    A worktree separates working files, not permissions. CLAI's default tools can
    still access files outside it. Creation runs before the agent starts and emits
    no agent telemetry spans.

## Codex authentication

`/login openai-codex` opens the browser and uses core's `OpenAICodexOAuthFlow`:
authorization code with PKCE, state validation, and a callback at
`http://localhost:1455/auth/callback`. It times out after five minutes. The browser
must be able to reach that callback on the machine running CLAI.

Tokens live in the configured Python `keyring` backend under service `pydantic-clai2`,
not in SQLite or `~/.codex/auth.json`. Choose an OS-backed credential store: CLAI
uses the configured backend and does not enforce its encryption or storage policy.
Installing or selecting a plaintext backend can store tokens in plaintext. Core owns
token refresh through CLAI's `OpenAICodexCredentialSource`. Tests mock keyring,
the browser, and OAuth exchange and do not access real credentials.

The default Coder shell runs under your OS identity, without a sandbox. Commands
can read files and access credential backends available to that identity, including
CLAI's tokens. Keyring is storage, not isolation from model-controlled commands.
Use a separate OS account or isolated environment for untrusted repositories.

The requested default does not guarantee model availability for a subscription.
Custom agents supplied to `chat` retain their model unless settings explicitly
select an override. `/login` is async, and plugin command handlers may also return
an awaitable string.

## Settings and commands

Preferences live in `$XDG_CONFIG_HOME/pydantic-clai2/config.db`, falling back to
`~/.config/pydantic-clai2/config.db`. Use `--database PATH` to select another database.
A repository can add its own layer with a `.clai/settings.json` file, found by
walking up from the launch directory to the git root. Conversation messages and CLAI's
Codex tokens are not written to the settings database. Plugin settings are arbitrary
JSON stored in plaintext in this database, including secrets if you put them there.
Pass secret references or use plugin-owned credential storage instead of embedding keys.

```text
/set
/set model <Tab>
/set display.thinking false
/set run.request_limit 10000
```

`/set` on its own opens a full-screen menu, the same kind Code Puppy uses: the
settings on the left, details for the highlighted one on the right (current
value, default, what it does). Type to filter. Enter edits: booleans and the
model get a picker (the model list is searchable, with "Type a value..." for
anything not listed), everything else a typed input that validates as you go.
An empty value resets. `R` resets the highlighted setting. Esc closes. Every
edit saves and applies immediately, the same as `/set KEY VALUE`.

## Models and their settings

`/model` opens a searchable provider list, then a model picker for that provider.
Esc from the model list returns to providers. Providers are unique prefixes from
the merged catalog, including `openai-codex`. Its suggestions include
`gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol`, and `gpt-6-astra`; availability
depends on your account. Unknown prices and context limits are not inferred.

The model catalog combines genai-prices' catalog
filtered to providers Pydantic AI can run, plus core's own model list, plus
whatever you have set now. The left side shows model names and marks the current
model; token counts stay in the details. The right side shows the provider, context window,
prices, and any settings you have saved for that model. Type to filter. Enter
makes it the model for the next prompt. `Ctrl+S` opens that model's settings:
`max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `timeout`, the two
penalties, `parallel_tool_calls`, `thinking`, and `service_tier`. They are
saved per model and passed to every run with that model. Unsupported settings
may be ignored or rejected by the provider; select only settings your provider supports. `/model NAME` sets the model without the menu.

Tab completes setting names, boolean values, and model names from Pydantic AI's
built-in catalog without network access. Provider prefixes include `openai-codex:`,
which core supports but does not currently include in that model catalog. Complete
the provider prefix, then enter the model identifier; suggestions do not establish
subscription availability. Custom model identifiers are accepted too.

The command registry uses Termflow's `Completer`, `Document`, and `Completion`
types. The current input widget and popup still use prompt-toolkit through a small
adapter; replacing that editor with a Termflow-based editor is separate work.
`/set SETTING` shows its current value. `/set` changes apply to subsequent prompts
and preserve conversation history; splash changes apply at next startup.

Precedence is defaults, SQLite overrides, `CLAI_MODEL`, then explicit CLI flags.
Settings are validated before writes. `/set` updates the active settings snapshot;
legacy `/config` writes apply on restart; plugin changes apply on the next prompt.
`--request-limit` controls the full prompt's model-request budget.

Interactive commands: `/login`, `/set`, `/theme`, `/model`, `/help`, `/new`, `/exit`, `/config`, `/plugins`, and `/reload`.
Tab completion suggests commands, settings, boolean values, plugin identifiers,
and paths after `@`. Path completion inserts a path; it does not attach file contents.
Unknown slash commands are not sent to the model. Up/down recall prompt history
within this process. Ctrl-D exits. Ctrl-C at input clears the line; during a run it
exits and unwinds the agent. No cancelled run is automatically retried.

## Reload CLAI during development

```text
/reload
```

After editing `pydantic_clai2` source, run `/reload` without arguments. It uses
`importlib.reload` on loaded CLAI modules and rebuilds the prompt loop, commands,
and session with the updated code. The Python process, agent, dependencies passed
to `chat`, conversation messages, selected model, and active settings are kept.
Enabled plugins unload and activate again so their handlers use the refreshed
shell types. Disabled and unapproved project plugins stay off.

If an import or shell rebuild fails, CLAI reports the error and restores the
previous module bindings. Correct the source and retry `/reload`. Plugin hosts
and their registrations are recreated. Installed module globals not overwritten
by the new source can survive; initialize mutable state in `activate`.
Import-time side effects cannot be undone.

Reload ordering follows the modules' existing imports. Restart after changing
import dependencies, startup code, or the custom agent's construction. `/reload`
does not rerun the CLI or recursively reload third-party packages. Use
`/plugins reload NAME` when you only want to reload one plugin.

## Bring an agent

```python
import asyncio
from pydantic_ai import Agent
from pydantic_clai2 import chat

agent = Agent('test')  # No capabilities required.
asyncio.run(chat(agent, deps=None))
```

`Session(agent, deps=..., plugins=..., on_stream_event=...)` is the noninteractive
API. Call `await session.prompt(text)` for each turn. Native `agent.run` drives the
loop through tools to completion. Successful turns retain `result.all_messages()`;
failed or cancelled turns leave the previous history intact, though external tool
side effects may already have occurred. History is in memory only. Structured
outputs are supported and displayed after completion.

### Themes

```text
/theme
/theme tokyo_night
/set display.theme github_light
/theme default
```

`/theme` opens a searchable picker. Its preview shows a sample conversation with
Markdown, thinking, a tool call, syntax highlighting, warnings, errors, and the
input/status area. Each bundled palette paints the sample's foreground and
background. Browsing does not apply a palette or save a setting. Enter confirms;
Esc or Ctrl-C keeps your current choice. Narrow terminals show the list alone.

`default` preserves CLAI's existing brand colours, including Markdown, menus,
status, and diff highlighting. Starting and exiting with this choice leaves your
terminal palette untouched. The default preview has no forced background.
`/theme default` restores this appearance after trying another palette.

All other choices come from Termflow's bundled registry, including
`catppuccin_mocha`, `catppuccin_latte`, `tokyo_night`, and `github_light`. CLAI adds
no new palettes. `/theme NAME` and `/set display.theme NAME` apply the same
validated preference immediately and save it for future sessions. Tab completes
these names. The `/set` theme row also lets you reset immediately;
`/config reset display.theme` removes the saved override for the next startup.

For a bundled palette, Markdown and newly opened menus use Termflow's
`to_render_style()`, and shell roles use its colours. Termflow changes the terminal
foreground, background, and 16 ANSI slots via OSC escape sequences. Supported
terminals may also recolour existing ANSI-styled scrollback. CLAI resets terminal
colours when you return to `default` or exit a selected palette, including failure
and cancellation. Unsupported terminals may ignore these changes. Redirected
output receives no palette-changing sequences. Your terminal configuration file
is not modified.

The early splash retains its brand colours. Syntax highlighting keeps Monokai;
bundled palettes use Termflow's default diff colours. Theme selection adds no
model requests or telemetry.

### Streaming

Streaming matches Code Puppy's separate output and thinking paths:

- Markdown uses Termflow `SmoothWriter`: 12 ms ticks, 0.5-second catch-up,
  minimum one visible character per tick. Prose is parsed line-by-line. Fenced
  code is buffered until the closing fence or text part end, then highlighted
  as a whole block so multiline strings and comments keep their context.
  Unlabelled and Markdown fences stay literal, including indentation and blank
  lines. Unknown languages use plain text. Long code lines wrap to the terminal width.
- Thinking deltas feed `StreamSmoother` immediately: 20 ms ticks, 0.4-second
  catch-up, minimum two characters per tick. They display as dim literal text,
  without waiting for newlines or interpreting Markdown.
- Smoothing applies only to interactive terminal output. Redirected output is
  written directly. Parts drain before the next heading, tool status, or prompt.

`/set display.smooth_seconds 0.5` restores the Code Puppy response catch-up
window if you previously saved a slower preference. This response-only setting
accepts 0.1 to 5 seconds; thinking retains its separate 0.4-second window.
Empty thinking parts show no heading. The CLI disables core's first-run
observability banner.
Cancellation discards queued output. Incomplete Markdown lines are still buffered
until a newline or part end; smoothing does not remove that parsing delay.
Thinking signatures without text cannot be shown. A supplied agent's existing
stream handler is preserved. Custom renderer integrations must await `finish()`
and use `await abort()` on cancellation.

Response and thinking parts end with a blank separator line; responses have no
repeated CLAI heading. Intermediate text is flushed when a tool-call part begins,
before the tool's arguments finish streaming. Incomplete lines within a text part
still wait for a newline or part boundary, as in Code Puppy's Markdown path.

Tool calls print once with a filled-circle marker and the tool name, followed by one blank line. Long names
are truncated to one terminal row. Completion activity remains in the footer
rather than adding a separate `Finished:` line to the transcript.

## Grep previews

Grep calls display the expression and path. Results show the first 20 logical
lines by default; `/set display.grep_lines 10` changes the next turn's preview
(0 to 1000). `Truncated N result lines` counts returned lines hidden by the UI,
including context lines. If the tool itself capped the search, a separate notice
states that the additional result count is unknown. No matches is shown explicitly.
The model still receives the original tool result.

Shell output is rendered one completed line at a time. Carriage-return progress
updates replace the buffered line rather than printing control-code text; the last
update appears at newline or tool completion. CRLF works across chunk boundaries.
Long display lines are ellipsized to terminal width. Multiline commands show their
first line and the number of additional command lines rather than dumping scripts.
Full output remains in the log; display formatting does not alter model results.

## Shell preview limit

Shell output defaults to the first 20 logical lines per command. Change it with
`/set display.shell_lines 50` (0 to 1000; zero hides output). The setting applies
to the next prompt. After the command returns, `Truncated N lines` reports omitted
lines from the log snapshot at that time, including an unterminated final line.
The capability's 16 KB event preview cap can shorten the preview further. Full
output remains in the displayed log path. Background commands can keep writing
after the snapshot; those future lines are not included in its count.

Read headers show the path, zero-based offset, and effective line limit (Coder
default and maximum: 2000). Listing headers show the directory, recursive mode,
result limit (default 200), and optional glob. Coder listings recurse using
ripgrep and honor ignore rules. These displayed defaults describe Coder tools.

File-write/edit headers include the path on the same line as the tool name.
Shell headers include the command on that line. Arguments use cyan, with no
repeated completion heading before the diff or output.

## Tool details

Native capability events drive specialized output: `FileEditedEvent` renders its
bounded unified diff using Termflow `DiffRenderer`, the same renderer Code Puppy
uses. The default appearance keeps CLAI's existing addition and deletion
backgrounds; bundled palettes use Termflow's defaults. Both use brighter markers.
Code syntax colours retain the Monokai default. Successful file writes also show the proposed diff from their matching
`FileChangeRequestEvent`: new files show additions, overwrites show before/after
changes. Without a matching request event, only the written path is shown. Failed
or cancelled writes do not display a success diff. Large diffs retain the
filesystem's truncation notice. Coder shell events show the command, attached
combined output, exit status (or background state), and durable log paths. Output
is capped by the capability and truncation is marked. The standalone `Shell`
capability does not yet emit these Coder-specific shell events.

Terminal control characters in model text and diffs are escaped before rendering.
Shell output permits ANSI SGR color/style sequences, decoded into Rich text rather
than passed directly to the terminal. Styles persist across chunks and lines per
command; cursor movement, clipboard commands, and other controls remain escaped.
Plain shell output stays dim. ANSI generated by Termflow itself is retained.

## Status line

The terminal footer shows the selected model, activity, latest reported
context tokens, and streamed output estimate, including text, thinking, and
string tool-argument deltas. The estimate is characters divided by four, not a
provider tokenizer count. On completion it is replaced by reported run output
usage. Context is the most recent response's reported input plus output tokens,
not cumulative conversation billing or a context-window percentage; `?` means
unavailable. During a request it may reflect the previous response.

While running, the footer reserves the terminal's bottom row using ANSI scrolling
regions. Its text shimmers with a moving highlight at ten frames per second, with no spinner and a
a 16-colour fallback when truecolour is unavailable. Prompt-toolkit owns the footer while accepting input. The run footer is
disabled for redirected output and restores normal scrolling on cancellation or
failure. The cursor is hidden during runs and restored on completion, failure,
or cancellation. No model requests or telemetry are added for status reporting.

## Plugins

Everything beyond the prompt loop is a plugin, including the default coding
tools. A plugin is a Python file with an `activate(host)` function. Through
`host` it can react to lifecycle moments and typed events, add `/commands`, give
the agent tools, draw its own output, and read validated settings.

```python
from pydantic_clai2.plugins import PluginHost, TurnEnd


def activate(host: PluginHost) -> None:
    @host.on('turn_end')
    async def ping(event: TurnEnd) -> None:
        host.console.bell()
```

Drop the file in `~/.config/pydantic-clai2/plugins/`, or register anything
importable with `/plugins add NAME module[:attr] [JSON]`. It is live for the
next prompt; no restart. `/plugins` alone opens a full-screen menu to enable, disable,
reload, and remove. Plugins are trusted code running as you.

[PLUGINS.md](PLUGINS.md) has the full list of hooks, events, and rules.

## Telemetry and references

CLAI emits no additional telemetry. Pydantic AI's own instrumentation covers model
requests, tools, and capability hooks when configured on the supplied agent.

- [Pydantic AI agent execution and events](https://pydantic.dev/docs/ai/core-concepts/agent/)
- [Capability events](https://pydantic.dev/docs/ai/capabilities/overview/)
- [Code Puppy splash](https://github.com/code-puppy/code_puppy/blob/main/code_puppy/splash.py)
- [Code Puppy streaming](https://github.com/code-puppy/code_puppy/blob/main/code_puppy/agents/event_stream_handler.py)
- [Code Puppy command registry](https://github.com/code-puppy/code_puppy/blob/main/code_puppy/command_line/command_registry.py)

See `THIRD_PARTY_NOTICES.md` for attribution.

## vllm connection

Open `/model`, choose `vllm`, then enter a trusted HTTP(S) server root or `/v1` URL, and optionally a token. CLAI queries `/v1/models` and opens a searchable model picker. HTTP sends tokens unencrypted; use HTTPS outside trusted local networks.

## openrouter connection

Open `/model`, choose `openrouter`, then paste an API key from https://openrouter.ai/keys in the masked prompt, then select a model from the live catalog. CLAI validates the key with `/api/v1/key` before fetching `/api/v1/models`. This flow uses API-key authentication, not browser OAuth.

The connection is saved in the configured Python keyring backend after selection; backend security depends on your keyring configuration. Tokens are not stored in SQLite or command history. The selected model persists across restarts. Select the provider again to browse its live models or reconfigure the saved connection. Discovery is explicit and has a 20-second network timeout; redirects are not followed. Agent inference uses Pydantic AI core.
