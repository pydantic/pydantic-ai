# CLAI 2.0

A separately installable terminal client for Pydantic AI, using `Coder(unrestricted_filesystem=True)` by default.
Python 3.11+ is required by Termflow. Tracking issue: https://github.com/pydantic/pydantic-ai-harness/issues/875.

CLAI file tools can access paths outside the workspace, including `/tmp`, and do
not protect secret files or repository metadata. OS permissions still apply.
CLAI attaches the launch directory as the agent's workspace, so relative paths
and commands start there. Commands get CLAI's environment minus LLM provider API
keys. Use a custom agent with `Coder()` to retain workspace-scoped file tools. Shell output is displayed dimly.

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
`/clear` (alias `/new`), or bare `clear`, clears model conversation history, not
input recall, and clears the screen back to the startup banner.
Model responses and tool results are not saved to this file.

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

For API-key providers, export the provider's API key before starting, or put it in
a `.env` in the launch directory. See [Environment variables](#environment-variables)
for discovery, precedence, and trust requirements.
Codex uses subscription OAuth instead, not `OPENAI_API_KEY`. The default Coder
can read and modify files and execute commands with your user permissions. Run it
in a workspace you trust. CLAI does not add a sandbox or approval layer.

The startup splash adapts Code Puppy's stdlib-only, alternate-screen Pydantic
pyramid, with CLAI lettering. The persistent `CLAI 2.0` banner uses `ansi_shadow`.
The splash is disabled for redirected output, CLI arguments, small terminals,
Windows, `NO_COLOR`, or `CLAI_NO_SPLASH=1`.

## Environment variables

CLAI trusts the launch directory and its parents for automatic dotenv loading.
A `.env` can change provider endpoints while reusing your exported API keys, and
alter how tool subprocesses execute. Set `PYTHON_DOTENV_DISABLED=1` in your
environment before launching CLAI in an untrusted directory.

CLAI loads the nearest `.env` file at startup using `python-dotenv`, before reading
settings or importing agents and plugins. It searches the launch directory first,
then its parents, and loads only the first file found. The search is not limited
to Git repository boundaries. With `--worktree`, loading happens before switching
directories. Missing files and named pipes are ignored. Unreadable or non-UTF-8
files are skipped with a diagnostic on stderr.

Use `.env` for provider API keys and settings such as `CLAI_MODEL` or
`CLAI_NO_SPLASH`. Existing environment variables take precedence, including empty
values. Keep `.env` files containing credentials out of version control.

## Git worktrees

```bash
py-cli clai2 --worktree my-task
```

```bash
py-cli clai2 -w
```

A Git worktree is another checkout of the same repository with its own branch
and working files. Run these commands inside a repository with at least one
commit. `--worktree NAME` creates a `clai-NAME` branch from the current `HEAD`
and starts CLAI at `<repository-root>/.worktrees/NAME`. If that worktree already
exists, CLAI reopens it; if only the `clai-NAME` branch exists, CLAI checks it out.
`-w` is the short form; omit the name to generate one. Names start with a letter
or digit and contain only ASCII letters, digits, hyphens, and underscores.

After checkout succeeds, CLAI adds `/.worktrees/` to Git's local `info/exclude`
file to keep generated checkouts out of `git status`, without changing your
tracked `.gitignore`.
Uncommitted changes, ignored files, and untracked files are not copied. Project settings, repository instructions, and coding
tools use the new worktree root. Your user settings and plugins stay available;
a relative `--database` path still refers to the directory you launched from.

CLAI prints the path and branch. A directory at that path that is not a Git
worktree is rejected. If checkout fails, CLAI tries to remove only the branch it just
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
git branch -d clai-my-task
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

Tokens are encrypted into `0600` files in `$XDG_CONFIG_HOME/pydantic-clai2/`
(`credentials-ACCOUNT.enc`), not stored in SQLite or `~/.codex/auth.json`. The key
that decrypts them is the only entry CLAI keeps in the configured Python `keyring`
backend (service `pydantic-clai2`, account `encryption-key`). CLAI reads that entry
at most once per session, so macOS asks for keychain access at most once, instead of
once per saved credential. Logins that older versions saved as keyring entries are
moved into encrypted files the first time they are read. Choose an OS-backed credential store: CLAI
uses the configured backend and does not enforce its encryption or storage policy.
Installing or selecting a plaintext backend can store the key in plaintext. Core owns
token refresh through CLAI's `OpenAICodexCredentialSource`. Tests mock keyring,
the browser, and OAuth exchange and do not access real credentials.

If Codex cannot refresh your login, CLAI tells you to run `/login openai-codex`
in an interactive session, then retry your message. This replaces the generic
connection error that can hide an expired login. Headless runs show the same
advice on stderr and exit with code 1. CLAI does not retry the turn automatically.

The default Coder shell runs under your OS identity, without a sandbox. Commands
can read files and access credential backends available to that identity, including
CLAI's tokens. Keyring is storage, not isolation from model-controlled commands.
Use a separate OS account or isolated environment for untrusted repositories.

The requested default does not guarantee model availability for a subscription.
Custom agents supplied to `chat` retain their model unless settings explicitly
select an override. `/login` is async, and plugin command handlers may also return
an awaitable string.

## GitHub Copilot subscriptions

From a source checkout:

```bash
uv run --project pydantic-clai2 clai2
```

Run `/login github-copilot`, then open `/model add` and choose `github-copilot`.
The provider menu also starts login when no credentials exist. No application
registration or client ID configuration is required. CLAI supplies the same
[public Copilot OAuth client ID as Pi](https://github.com/earendil-works/pi/blob/fde38ed7c2f64434beffc6c0ec3b9994cb89ae23/packages/ai/src/auth/oauth/github-copilot.ts#L10-L11)
and requests `read:user` access to your GitHub profile. This identifies the existing
Copilot OAuth application, not a separately registered CLAI application.
`GITHUB_COPILOT_CLIENT_ID` is an optional override for your own device-enabled OAuth
application; unset or blank uses the bundled default. The workspace pins the merged
Pydantic AI device-flow implementation until its release.

Login prints a code and `https://github.com/login/device`, then starts polling.
Open the link on this or another device and approve only your own session's code.
CLAI does not launch a browser, so a text browser cannot block login or take over
an SSH terminal. Ctrl-C stops polling; GitHub controls expiry. There is no localhost callback.

GitHub authorization does not establish Copilot access. The model menu queries
your account's catalog and keeps only picker-enabled `/chat/completions` models.
It includes current-model details and `Ctrl+S` settings. Subscription and
organization policy still control inference access. A known ID also works with
`/model github-copilot:claude-haiku-4.5`.

The `github-copilot` keyring account is separate from Codex and named API keys.
Without a keyring, CLAI reports the plaintext `credentials-github-copilot.json`
fallback under the user's CLAI config directory, created with mode `0600`.
Tokens and issuance time stay out of settings, history, and login output.
Expiring tokens require another `/login github-copilot`; there is no automatic
refresh. Failed or cancelled authorization preserves the previous login.

Saved login takes precedence over `GITHUB_COPILOT_API_KEY`,
`GITHUB_COPILOT_API_TOKEN`, and `COPILOT_GITHUB_TOKEN`, checked in that order when
no login is saved. CLAI does not read `GH_TOKEN`, `GITHUB_TOKEN`, or another
application's token files. Core owns inference and its telemetry; CLAI adds no
login-specific spans. Bare `/login` asks which sign-in to run.

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
edit saves and applies immediately, the same as `/set KEY VALUE`. `/settings` is
an alias of `/set` and accepts the same arguments.

While a turn is running, `/set`, `/settings`, `/system_prompt`, `/model`, `/model add`, `/model settings`,
`/theme`, and `/spinner` typed without further arguments open their menu right
away instead of queueing.
The turn keeps running: its output is held while the menu is open
and printed in order when the menu closes. A question from the agent waits for
the menu to close. Model and run settings saved in the menu apply once the
running turn ends. `/model settings` edits to the running model apply to its
next model request in the same turn. With arguments, these commands queue like
any other.

`/plugins` runs right away during a turn too, with or without arguments, so
`/plugins disable NAME` does not wait behind the turn or your queued messages.
Commands, status segments, and model providers change at once. The running turn
keeps the tools and hooks it started with, because the agent binds them when a
run begins; the change reaches the agent on your next prompt, and CLAI says so.
A plugin turned off mid-turn finishes its cleanup when the turn ends. While a
delegated task is running, `/plugins` still refuses changes, as between turns.

`/system_prompt` shows the full system prompt sent with the latest request and
edits your own instructions, the `run.instructions` setting. CLAI sends them after
its built-in instructions, `AGENTS.md`, and plugin instructions, which stay
read-only; yours replace none of them, and resetting removes them. Editing opens
`$VISUAL` or `$EDITOR`; on Windows, when neither is set, or when the editor cannot
start, a built-in editor opens instead. Changes apply from your next prompt.

## Models and their settings

`/model` selects from models you have already added. Choose **Add a model...**
to browse providers. `/model PROVIDER:NAME` switches directly to any model. A model
not yet in your list is added and selected; CLAI does not check that it exists, so a
mistyped name fails on the next prompt with the provider's error. Model names normally
start with a provider (`openai:gpt-5`), so no real model is called `add` or `settings`.

| Command | What it does |
| --- | --- |
| `/model` | Pick a saved model, add one, or delete one |
| `/model NAME` | Select `NAME`, adding it first if needed |
| `/model add` | Browse providers and their models |
| `/model add NAME` | The same as `/model NAME` |
| `/model settings` | Choose a saved model to configure |
| `/model settings NAME` | Configure `NAME` |

`/add_model` and `/model_settings` still work as deprecated spellings of
`/model add` and `/model settings`.

`/model add` opens a searchable provider list, then a model picker for that provider.
Esc from the model list returns to providers. Providers are unique prefixes from
the merged catalog, including `openai-codex`. Its suggestions include
`gpt-6.1-sol`, `gpt-6-astra`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6-luna`,
`gpt-5.6-terra`, and `gpt-5.6-sol`; availability depends on your account.
Unknown prices and context limits are not inferred.

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

CLAI installs the SDKs for OpenAI and Anthropic. Selecting a model whose provider SDK is
missing from the Python CLAI runs on fails right away, naming the install command, instead
of on the next prompt. TypeSafe's Jev needs the `typesafe` extra:

```bash
pip/uv-add "pydantic-clai2[typesafe]"
```

From a pydantic-ai checkout, run CLAI with the extra instead:

```bash
uv run --package pydantic-clai2 --extra typesafe clai2
```

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

Interactive commands: `/login`, `/set` (alias `/settings`), `/system_prompt`, `/theme`, `/model`, `/help`, `/clear` (alias `/new`), `/exit`, `/config`, `/plugins`, and `/reload`.
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
Enabled plugins unload and load again so their handlers use the refreshed
shell types. Disabled and unapproved project plugins stay off.

If an import or shell rebuild fails, CLAI reports the error and restores the
previous module bindings. Correct the source and retry `/reload`. Plugin
instances and their contributions are recreated. Installed module globals not
overwritten by the new source can survive; initialize mutable state in `__init__`.
Import-time side effects cannot be undone.

Reload ordering follows the modules' existing imports. Restart after changing
import dependencies, startup code, or the custom agent's construction. `/reload`
does not rerun the CLI or recursively reload third-party packages. Use
`/plugins reload NAME` when you only want to reload one plugin.

## The stock agent in your own code

`open_stock_agent` opens CLAI's stock coding agent without the terminal. Use it with
`agent.run`, `run_stream_events`, `iter`, or anything else that takes an agent:

```python
import asyncio
from pathlib import Path

from pydantic_clai2 import open_stock_agent


async def main() -> None:
    async with open_stock_agent(workspace=Path.cwd(), model='anthropic:claude-opus-5-5') as agent:
        result = await agent.run('Summarize this repository in one paragraph.')
        print(result.output)


asyncio.run(main())
```

The agent has the `coder`, `repo_context`, and `compaction` built-ins, configured as
in `clai2`. The file and shell tools work in `workspace`, unless a capability you pass
supplies a workspace, and `AGENTS.md` or `CLAUDE.md` is read from it. Model names
resolve as in `clai2`, with CLAI's per-model defaults. Nothing else you saved for
`clai2` applies: no saved, drop-in, or project plugins, no `.clai/settings.json`, no
`/model settings`, and no `chain:` fallback chains. `plugin_settings` changes a built-in's
settings, merged over the stock ones, such as `{'coder': {'agent_folders': []}}`.
`capabilities` adds your own capabilities, bound beside the built-ins so delegated
tasks carry them too. Plugins close when the `async with` block exits.

To serve it to an editor, pass the agent to [`run_acp_stdio`](acp.md). Every session
then works in `workspace`. To follow the folder each client session opens instead,
return a `workspace` from a `session_config`, as
[Rooting tools at the workspace](acp.md#rooting-tools-at-the-workspace) shows. With a
capability that requires approval, such as a `ToolGuardrail` returning
`GuardrailResult.approve()`, pass `plugin_settings={'coder': {'sub_agents': False}}`:
a delegated task cannot yet pass an approval up to the client
([#4302](https://github.com/pydantic/pydantic-ai/issues/4302)).

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
and status. Its diffs use green additions and red deletions. Starting and exiting with this choice leaves your
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

The early splash and the `CLAI 2.0` banner keep Pydantic's brand colours under
every palette, except on 16-colour terminals, where the palette owns the ANSI
slots. Syntax highlighting keeps Monokai;
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

Tool calls print once with a filled-circle marker and the tool name, followed by one blank line. Tools
without a specialized summary list their arguments after the name as `name=value` pairs, with pink names and
muted compact-JSON values. Each value shows at most 40 characters by default; `/set display.tool_arg_chars 80`
changes the next turn's limit (0 to 1000; zero hides arguments). The whole line is truncated to one terminal row. Completion activity remains in the footer
rather than adding a separate `Finished:` line to the transcript.

With `/set display.tool_calls grouped`, each call adds to a streak of the same tool on one line instead
of printing its own: `● shell 4, grep 2, shell 3`. On a terminal the line is redrawn from column zero
as each call arrives, so the last count is final only once another tool, visible text or thinking, a diff, or a widget follows.
In the interactive prompt the line keeps counting above anything printed meanwhile, such as a command typed
mid-turn. A tool that no longer fits the row starts the next line. Elsewhere the line prints once, when it ends.
The `grouped` style ignores `display.tool_output` and `display.tool_arg_chars`, and hides shell command output.
`edit_file` and `write_file` calls are not counted: they print their summary and diff as in the `detailed` style.
`/set` previews both styles when you pick one. The setting applies to the next turn; the default is `detailed`.

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
uses. The default appearance uses Claude Code's green addition and red deletion
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
tools. A plugin is a `Plugin` subclass that declares what it contributes the way
a [capability](../capabilities/overview.md) does: it overrides methods to react to
lifecycle moments, add `/commands`, give the agent capabilities, draw its own
output, and take validated settings.

```python
from pydantic_clai2.plugins import Plugin, TurnEnd


class Bell(Plugin):
    async def on_turn_end(self, event: TurnEnd) -> None:
        self.host.console.bell()
```

Drop the file in `~/.config/pydantic-clai2/plugins/`, register anything
importable with `/plugins add NAME module[:Class] [JSON]`, or clone a trusted
repository with `/plugins add https://github.com/your-org/my-plugin.git` inside
CLAI. Git repositories need an `__init__.py` or `plugin.py` at their root; install
any dependencies in CLAI's Python environment first. See
[where plugins live](https://github.com/pydantic/pydantic-ai/blob/main/src/pydantic_clai2/PLUGINS.md#where-plugins-live)
for supported URLs and checkout management.
The plugin is live for the next prompt; no restart. `/plugins` alone opens a full-screen menu to enable, disable,
reload, and remove. Plugins are trusted code running as you.

[PLUGINS.md](https://github.com/pydantic/pydantic-ai/blob/main/src/pydantic_clai2/PLUGINS.md) has every method, event, and rule.

## Telemetry and references

CLAI emits no additional telemetry. Pydantic AI's own instrumentation covers model
requests, tools, and capability hooks when configured on the supplied agent.

- [Pydantic AI agent execution and events](https://pydantic.dev/docs/ai/core-concepts/agent/)
- [Capability events](https://pydantic.dev/docs/ai/capabilities/overview/)
- [Code Puppy splash](https://github.com/mpfaffenberger/code_puppy/blob/main/code_puppy/splash.py)
- [Code Puppy streaming](https://github.com/mpfaffenberger/code_puppy/blob/main/code_puppy/agents/event_stream_handler.py)
- [Code Puppy command registry](https://github.com/mpfaffenberger/code_puppy/blob/main/code_puppy/command_line/command_registry.py)

See `THIRD_PARTY_NOTICES.md` for attribution.

## vllm connection

Open `/model add`, choose `vllm`, then enter a trusted HTTP(S) server root or `/v1` URL, and optionally a token. CLAI queries `/v1/models` and opens a searchable model picker. HTTP sends tokens unencrypted; use HTTPS outside trusted local networks.

## openrouter connection

Open `/model add`, choose `openrouter`, then paste an API key from https://openrouter.ai/keys in the masked prompt, then select a model from the live catalog. CLAI validates the key with `/api/v1/key` before fetching `/api/v1/models`. This flow uses API-key authentication, not browser OAuth.

The connection is saved in the configured Python keyring backend after selection; backend security depends on your keyring configuration. Tokens are not stored in SQLite or command history. The selected model persists across restarts. Select the provider again to browse its live models or reconfigure the saved connection. Discovery is explicit and has a 20-second network timeout; redirects are not followed. Agent inference uses Pydantic AI core.
