# CLAI 2.0

A separately installable terminal client for Pydantic AI. The coding tools,
`Coder(unrestricted_filesystem=True)`, are the built-in `coder` plugin: on by
default, `/plugins disable coder` for a chat-only shell. The model can also ask
you multiple-choice questions mid-run through the built-in `ask_user` plugin;
see [Questions from the model](#questions-from-the-model). The built-in
`repo_context` plugin reads `AGENTS.md` or `CLAUDE.md` from the launch directory
into the agent's instructions; `/plugins disable repo_context` turns that off.
On Windows, CLAI does not provide an agent workspace or repository context yet.
`coder` clears old tool results as the context fills. The built-in `compaction`
plugin, [described below](#compacting-the-conversation), is on by default beside it:
it summarises older messages before the context runs out and adds `/compact`.
Other harness capabilities are not listed in `/plugins`; add one on purpose with
`/plugins add`, see [other harness capabilities](PLUGINS.md#other-harness-capabilities).
The disabled built-in `google_workspace` connects Gmail, Calendar, and Drive with a
token kept in `/keys`. `/google_workspace` opens its settings menu: the `/keys`
entry to use (`GOOGLE_ACCESS_TOKEN` by default), products, and read-only tools; see
[its settings](PLUGINS.md#google_workspace-gmail-calendar-and-drive-tools).
`/plugins enable logfire_mcp` lets the agent query your Logfire telemetry and opens
a settings menu (region, tools, and a key picked from `/keys`, never stored in plugin
settings; otherwise browser sign-in, which also signs new users up and works over SSH:
`/logfire_mcp login`). Reopen it with `/plugins configure logfire_mcp`; see
[Logfire MCP](PLUGINS.md#logfire-mcp-query-your-telemetry).
`/mcp` manages MCP servers the way Code Puppy's `/mcp` does. Bare `/mcp` shows a
status dashboard. `/mcp install` opens a form where you name the server, pick
`stdio`, `http`, or `sse`, type its URL or command, edit the rest of its JSON
configuration in your editor, and switch on OAuth sign-in for remote servers.
`/mcp edit NAME` reopens the same form. `/mcp start NAME`, `stop NAME`,
`restart NAME`, `status NAME`, `logs NAME [LINES]`, `auth NAME [logout]`,
`remove NAME`, and `trust [status|accept|revoke]` manage servers, and `/mcp help`
lists them all. Saved servers are available to the agent from the next prompt,
with tool names prefixed by the server name. A server connects on the first
prompt that needs it (or `/mcp start`) and stays connected; one that cannot
connect is marked `error` and left out rather than failing the prompt. A
repository's Claude Code-style `.mcp.json` (or `.clai/mcp_servers.json`) loads
once you review it and run `/mcp trust accept`. See
[Connect MCP servers](PLUGINS.md#connect-mcp-servers) for storage, secrets, OAuth,
and project trust. `/plugins disable mcp` removes the command and tools.

Python 3.11+ is required.

CLAI file tools can access paths outside the workspace, including `/tmp`, and do
not protect secret files or repository metadata. OS permissions still apply.
CLAI attaches the launch directory as the agent's workspace, so relative paths
and commands start there. A plugin or agent capability that supplies its own
workspace, such as a sandbox, takes the launch directory's place; a capability
function that supplies none leaves the launch directory in use. Commands get
CLAI's environment minus LLM provider API keys. To keep the file tools inside the
launch directory, set **Unrestricted filesystem** to `false` in
`/plugins configure coder`. Shell commands are not restricted either way.

Tool calls show a single-line summary followed by a blank line by default.
Tool and argument names are pink; argument values and bullet markers are muted grey.
Successful file writes and edits also show their diffs, even with `display.tool_output` off.
Shell output, exit details and log paths, and grep results stay out of the terminal;
the model still receives full tool results. Long summaries are clipped to the terminal width.
Use `/set display.tool_output true` to show shell and grep details, or
`/set display.tool_output false` to hide those details without hiding file diffs. With `display.tool_output` on,
`display.shell_lines` and `display.grep_lines` limit previews to 20 lines by
default. Plugin-provided rendering, including interactive questions, is unchanged.

To count calls instead of listing them, run `/set display.tool_calls grouped`. Consecutive calls share
one live line, such as `● shell 4, grep 2, shell 3`. A count grows in place and is final once a different
tool or other output follows. The `grouped` style shows only names and counts, so it ignores
`display.tool_output` and `display.tool_arg_chars`. File edits and writes still print their summary and diff,
which ends the line. `/set` previews both styles when you pick one. The default is `detailed`, one line per call.

## Source layout

The shell entry point lives in `_app.py`. Related implementations live in
`cli/`, `config/`, `runtime/`, `models/`, `plugins/`, and `ui/` (`prompt/`,
`menus/`, and `rendering/`). Built-in plugins live in `builtin_plugins/`; MCP has its own
`mcp/` package. Plugin authors can still import `pydantic_clai2.plugins` and
`pydantic_clai2.commands` directly. See [the plugin guide](PLUGINS.md) for
examples.

## Startup

`clai2 --help` parses arguments without loading the agent or plugins. Interactive
startup defers model menus and provider integrations until you open those menus,
log in, or run a prompt. Once the prompt is ready, a background thread imports
them, so the first prompt usually finds them loaded; if it arrives sooner, it waits
for the rest of those imports. Enabled plugins still load before the first prompt;
their initialization contributes to startup time.
An enabled plugin whose module is not installed, such as a built-in saved by another
CLAI version, is skipped without a message; `/plugins list` shows why. Library
warnings are hidden so they do not break up the display; pass `-W default` to
Python or set `PYTHONWARNINGS=default` to see them. Log messages from Logfire and
OpenTelemetry, such as export retries and timeouts, never print to the terminal
either; logging handlers that a plugin configures still receive them.
`/login` offers both Codex and GitHub Copilot without loading their integrations for
completion. Copilot requests use your saved login through the lazy provider resolver.

## Herdr integration

Enable `/plugins enable herdr` inside a [herdr](https://herdr.dev) pane to report
CLAI2's state, session reference, model/token metadata, and conversation title.
It starts disabled and does nothing outside herdr. See
[the plugin guide](PLUGINS.md#herdr-integration) for details and limitations.

## Desktop notifications

The built-in `notifications` plugin is enabled by default. Interactive sessions
send a desktop notification when a turn finishes or fails, and when the model
asks a question through `ask_user`. Cancelled turns do not notify. Messages use
the title `CLAI2` and generic status text, not prompts, answers, paths, or errors.

macOS uses the system `osascript` notification service. Allow notifications for
Script Editor in System Settings > Notifications; Focus modes can suppress them.
Linux uses `/usr/bin/notify-send` when installed and a desktop notification service is
available. Windows, redirected output, headless mode, and SSH sessions do not
send notifications. These are local OS notifications, not terminal escape
sequences, so local tmux sessions need no passthrough configuration. CLAI does
not detect terminal focus; notifications are submitted even while you are
looking at the terminal. Delivery and presentation depend on OS settings.

Use `/plugins disable notifications` to persistently turn them off and
`/plugins enable notifications` to restore them. `/plugins remove notifications`
resets the built-in default. Missing services, nonzero exits, and a two-second
submission timeout do not fail the turn. No notification-specific telemetry is
emitted.

## Code highlighting

Fenced code uses the fence's language for syntax highlighting. CLAI renders a
block when its closing fence arrives, or when the text part ends if the fence
is unfinished. This keeps multiline strings and comments correctly colored.
Prose outside fences still streams line by line. Unlabelled and Markdown fences
stay literal, including indentation and blank lines; unknown languages use plain
text. Long code lines wrap to the terminal width. Response and reasoning code
blocks use the terminal foreground and ANSI syntax colours, rather than pale
text intended for a dark background. Bundled themes supply their own ANSI colours.

## Word deletion

Option+Backspace (Alt+Backspace) deletes the word before the cursor, like Ctrl-W,
including trailing whitespace. Spaces, tabs, and newlines separate words. Text
after the cursor is preserved. Your terminal must send Option as Alt/Meta for
this shortcut; legacy and modified-key encodings are supported.

## Option keys on macOS

CLAI asks the terminal to report modified keys, so Option+Enter (steer) works in
Herdr, and in iTerm2 even with its default Option key setting. If Option+Enter
still queues the message like Enter, the terminal is sending a plain Enter:

- Terminal.app: turn on Settings > Profiles > Keyboard > Use Option as Meta key.
- tmux: tmux forwards Alt+Enter, but does not pass CLAI's request on to the
  outer terminal. Set that terminal to send Option as Alt, for example iTerm2's
  Settings > Profiles > Keys > Left Option key: Esc+. Herdr needs no setup.
- For Shift-Enter (newline) inside tmux, add `set -g extended-keys on` to
  `~/.tmux.conf`. Without it, tmux sends Shift-Enter as Enter.

## Interrupting a turn

Press Esc or Ctrl-C to cancel the active agent turn without discarding your draft.
Tool cleanup finishes before the next queued message starts. Esc does not request
exit, even when pressed repeatedly. Arrow keys and Alt-key shortcuts keep their
editing behavior. The active full-screen interface retains control of Esc rather
than passing it to the agent turn; its behavior depends on that interface.
After cancelling with Ctrl-C, press Ctrl-C again within two seconds to exit,
including across the transition back to input. Esc does not arm this exit shortcut.
At the prompt,
the first press clears input and the second exits. Ctrl-D on an empty input and
`/exit` also quit after earlier queued messages finish.
The interrupted prompt and captured partial responses and tool results stay in
conversation history, so you can follow up with a clarification. No interrupted
run is automatically retried. Unanswered tool calls in retained failed or
interrupted turns are closed out by core on the next prompt, without replaying
those tools. External application cancellation still propagates, and completed
tool side effects cannot be undone.

## Rewinding a conversation

Between turns, press Esc twice within half a second to open **Rewind conversation**.
Use Up/Down to choose an earlier prompt, then Enter to rewind. The newest prompt
is selected first. Esc or Ctrl-C closes the menu without changing your draft or
conversation. An empty conversation shows a disabled placeholder.

Rewinding removes the selected prompt and all later messages from the saved
conversation, then restores that prompt and its image attachments to the editor.
This replaces any unsent draft and its attachments; cancel the menu to keep them.
Edit the restored prompt and press Enter when ready. It does **not** undo file changes or other
tool side effects, replay tools, or erase the terminal scrollback. Only prompts
still present in the retained history are available. Prompts predating retained
rewritten context, such as a compaction summary, are disabled even if compaction
kept a copy of the prompt. That summary can contain later facts, so the original
rewind boundary no longer exists. Prompts with attachment types the editor
cannot restore are also disabled.

A press that cancels a running turn, closes completions, or leaves history search
does not count toward the shortcut. Wait for queued prompts to finish first.
Typing another key or opening another menu resets the double-Esc sequence.

## Shell commands with `!`

A line that starts with `!`, after trimming surrounding whitespace, runs in the
system shell instead of starting an agent turn:

```text
> !git status
$ git status
Shell command and output saved for the next prompt
...
Done (0.1s)
```

The command runs through the system shell (`/bin/sh -c` on POSIX, `cmd.exe` on
Windows), not your login shell, so zsh or fish syntax and shell aliases are not
available. It runs in CLAI's working directory with the terminal's input. Stdout
and stderr are captured separately and displayed as they arrive. Because output
is piped, programs that require a terminal for their output, such as full-screen
editors, may not work. CLAI reports `Done` or the exit code with the elapsed time.
Ctrl-C interrupts the
command and returns to the prompt. On POSIX the command runs in its own session,
so CLAI forwards the Ctrl-C to its process group, and 0.25 seconds later (as
`subprocess.run` waits) kills whatever is still running there, including
background jobs and programs that ignore Ctrl-C. Only a process that detaches
on purpose with `setsid()`, as daemons do, outlives the command. Detached jobs
should redirect stdout and stderr: CLAI stops collecting their output after the
shell exits. Without a controlling terminal, programs that prompt through `/dev/tty`, such as `sudo`
or `ssh` password prompts, cannot read your input. On Windows the console
delivers the Ctrl-C, and `taskkill` then ends the command's process tree. As at other times, a second Ctrl-C within two seconds exits
CLAI. The command, stdout, stderr, and completion status are saved in the
conversation for the model's next turn, including partial output from interrupted
commands. Each stream retains up to 100,000 characters, with a truncation notice
when it exceeds that limit; all output continues streaming to the terminal.
Running a command does not request a model reply. This context survives
`/resume` without re-running the command; `/new` starts without it. Do not use `!`
for output you do not want saved or shared with the model.

A bare `!` is sent to the agent as an ordinary prompt. Queued `!` lines run in
order with other queued input. `/help` lists the syntax.

## Prompt area

```text
Follow-up: Add tests for the change
Command: /usage
─ Working ⠋ ────────────────────────────────────
/
/resume Browse or restore a saved session; claude or codex imports theirs
/set Change settings; no arguments opens menu
────────────────────────────────────────────────
model | context: ... | running: shell
```

The prompt sits above the footer with one editable line when empty. It grows
for wrapped or pasted text, not to fill the terminal. Text pastes of five or more
lines, or at least 1,000 characters, display as `[paste N lines]`. The full text
is still submitted and saved in input history. Move the cursor inside a folded
paste to reveal it for editing; recalled history shows the full text.
Completion suggestions
appear below the draft, between the top and bottom rules. The rows carry no
side borders and no prompt marker, so they cannot drift out of alignment.
History search stays compact too. The bordered prompt area stays visible below
streamed output while CLAI works, and remains editable. A `Working` label and
animated spinner, in the same pink accent as tool names, appear in the box's top border while a turn or its lifecycle
hooks are active, without adding a row to the input area. The animation uses the editor's existing refresh
cycle and disappears when work finishes, fails, or is cancelled. It is not part
of your draft or submitted message. The editor sits below a Termflow live transcript panel on the alternate screen.
Both partial and completed output paint above it without repainting unchanged
input cells. PageUp/PageDown and the mouse wheel scroll the transcript without
changing the draft. New output leaves a scrolled view in place; returning to the
bottom or submitting follows new output again. Ctrl+L clears the screen, as in
Claude Code: the transcript is forgotten, but the conversation, your draft, and
queued messages stay. During a turn, the response still streaming keeps showing.
Use `/clear` to start a new conversation instead. Drag with the mouse to select
text; releasing the button copies it to the clipboard and the footer says so.
Locally CLAI copies with `pbcopy` on macOS, PowerShell on Windows, and `wl-copy`,
`xclip`, or `xsel` on Linux. Over SSH, or without one of those, it asks the
terminal to copy with OSC 52, which some terminals need enabling first (iTerm2:
*Applications in terminal may access clipboard*; tmux: `set-clipboard on`).
Holding Shift (Option in iTerm2) still uses the terminal's own selection in most
terminals. On exit, CLAI prints the retained
transcript into native terminal scrollback, without output cleared by Ctrl+L. Typing updates the draft row; a
nonblinking highlighted cell marks the cursor. Full-screen menus temporarily hide it along
with the editor. Enter submits a message to an in-memory queue. Pending text appears above the editor as `Follow-up:`
previews, with queued slash commands labeled `Command:`. Previews are shown in
execution order and disappear as each submission starts. Long or multiline messages
have a single-line preview; large queues show a `+N more queued` summary to leave
room for the editor. The original message text is unchanged. The footer also shows
the number waiting. This is a read-only preview, not a queue editor. Control bytes in completion-derived
text are escaped in previews and prompt echoes; the submitted text is unchanged.

Messages and slash commands run in submission order, after the current turn and its cleanup
finish. They do not interrupt or steer the active turn. Bare settings menus such as
`/set` are the exception: they open during the turn (see "Settings and commands"). An unsubmitted draft stays
in the editor as turns finish. Queued messages are not saved as conversation turns
until execution starts, and are discarded on exit or `/reload`.

Full-screen question menus and slash-command menus temporarily take over input.
The editor and its draft return when the menu closes.
Small terminals omit the border to leave room for output.

## Pasting images

Copy a screenshot or image, then press **Ctrl-V** in CLAI to attach it. If your
terminal intercepts Ctrl-V, use **Alt-V** (or press Esc, then V). On macOS, Cmd-V
is the terminal's text paste, not CLAI's clipboard-image shortcut.

You can also paste or drag local image file paths into the prompt. This requires
the terminal's bracketed-paste support. A paste containing only existing image
paths becomes attachments; ordinary pasted text stays text. Quoted paths and
paths containing spaces are supported. UNC and device paths are not read. PNG, JPEG, GIF, WebP, BMP, and TIFF files
are read locally and converted to PNG (the first frame of animated images).

Each image appears as `[image:...]`. Add your question and press Enter, or submit
the image alone. Delete a marker before submitting to remove that attachment.
Images in queued follow-ups belong to that follow-up, not the running turn.
Paste errors appear in the footer without submitting a message or losing the draft.

Clipboard access uses Pillow's native Windows and macOS backends. On Linux,
install `wl-clipboard` for Wayland or `xclip` for X11 and run inside that graphical
session. SSH and headless sessions generally cannot read your local desktop
clipboard; paste a path to an image accessible on the machine running CLAI instead.
The clipboard is read only when you invoke the image shortcut.

Use a model that accepts images. CLAI sends the image bytes to the selected model;
provider-specific size and format restrictions can still apply. Each source file
and encoded attachment is limited to 10 MiB and 25 megapixels, with 32 MiB of
pending image bytes. Accepted images are saved with the conversation and restored
by `/resume`. Input recall stores markers, not image bytes: paste the image again
if you recall an expired marker. If submission is rejected because no model is
selected, the most recently rejected prompt keeps its attachments for retry. Unsubmitted images are discarded on exit or reload.

Pillow is a terminal-only dependency; see the CLAI dependency boundary in
[#875](https://github.com/pydantic/pydantic-ai-harness/issues/875).

## Input history

Submitted prompts and slash commands persist across restarts for Up/Down recall,
including multiline input. They are stored as plaintext in `input-history` next
to `config.db`: `$XDG_CONFIG_HOME/pydantic-clai2/input-history`, or
`~/.config/pydantic-clai2/input-history` by default. On POSIX the file is restricted
to its owner (mode 0600). Avoid entering secrets in the prompt: input history is
not encrypted. Delete this file while CLAI is closed to clear saved input.
`/clear` (alias `/new`), or bare `clear`, starts a new saved conversation, without
deleting the previous one or input recall, and clears the screen back to the startup
banner. Model responses and tool results are not saved to this file.

## CI coverage

The `CLAI coverage` check requires 100% branch coverage for `src/pydantic_clai2`.
It is separate from Harness coverage;
passing CLAI test jobs alone does not mean either coverage gate has passed.
Tracked under [#875](https://github.com/pydantic/pydantic-ai-harness/issues/875).

## Start chatting

Launch `clai2`. The default model is `openai-codex:gpt-6-astra`.
Run `/login openai-codex` to connect your ChatGPT/Codex subscription.
Type `/set model ` and press Tab to pick another provider-qualified model name.
The choice is saved in SQLite and used for the next prompt without restarting.
Until you choose one, the default is the agent's own model, so a plugin
capability that selects a model replaces it. A model you choose with `-m`,
`CLAI_MODEL`, the project file, `/set model`, or `/model` is passed to every
run and wins.

Without installing, run `uvx pydantic-clai2`. The package also installs a
`pydantic-clai2` command that is an alias for `clai2`.

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

## Updating

Install with `uv tool install pydantic-clai2` and CLAI can update itself.
`updates.channel` picks where it looks:

- `stable` (default): the newest release on PyPI.
- `main`: the newest build from the `main` branch. For each `main` commit that changes
  CLAI, harness, or core, CI builds an sdist of each and publishes them to the
  [`clai2-bleeding`](https://github.com/pydantic/pydantic-ai/releases/tag/clai2-bleeding)
  prerelease, whose tag moves with them. CLAI reads `clai2-bleeding.json` there
  to learn the commit, then uses `--overrides` to install CLAI, harness, and core
  from that commit's sdists with their required extras. These are plain release
  downloads, so they need no `git` and no GitHub API calls, which are rate
  limited without a token. The status row names the build by its short commit.
  This channel used to be called `bleeding`. That name still selects it, and
  CLAI saves it under that name so older versions sharing your settings can read it.

```text
/set updates.channel main
/update
```

To use another copy of the release, start CLAI with `CLAI_BLEEDING_URL` set to
the folder that holds `clai2-bleeding.json`. A fork that runs the `CLAI2 bleeding`
workflow publishes its own:

```bash
CLAI_BLEEDING_URL=https://github.com/<you>/pydantic-ai/releases/download/clai2-bleeding clai2
```

To try a local checkout, build the same files and serve them over HTTP:

```bash
src/pydantic_clai2/scripts/build_bleeding.sh /tmp/clai2-bleeding
python -m http.server --directory /tmp/clai2-bleeding 8000 &
CLAI_BLEEDING_URL=http://localhost:8000 clai2
```

When a newer build exists, the status row shows `update <version or commit>: /update`.
CLAI checks once at startup and again when you change the channel, and only for
`uv tool` installs. Offline, it shows nothing. `/update` runs `uv tool install --force`,
shows uv's output, then restarts CLAI as the new build with the same launch options,
resuming the current conversation. Switching back to `stable` offers the latest release.
Run from a source checkout, `/update` installs CLAI as a `uv tool` the same way and
restarts into that.

Windows does not let a program replace files it is running from, so there
`/update` exits first and installs in a new PowerShell window; start `clai2`
again when that window reports success.

The reinstall keeps only CLAI's own packages, so add any extra `--with` packages
again afterwards. Without uv on `PATH`, `/update` prints the complete command to
run yourself, in PowerShell syntax on Windows. For `main` installs, it also
writes the overrides file and includes its path in the printed command.

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

## Your own agent

```bash
clai2 --agent pydantic_ai.main:my_cool_agent
clai2 -a my_agents:reviewer -p "Review the staged diff"
```

`--agent MODULE:ATTR` (short form `-a`) chats with an existing Pydantic AI `Agent`
instance instead of CLAI's default agent. CLAI appends the launch directory to
`sys.path`, after installed packages, so a module next to where you start CLAI
resolves without installing it. `ATTR` must
name an instance, not a class; the agent runs with `deps=None`. For that session
only, no plugins load: no built-ins (including the stock coder tools), no saved or
drop-in user plugins, and no project plugins, and `/plugins` reports that they are
off. Nothing saved changes, so plain `clai2` loads plugins as before. The agent
keeps its own model unless `-m` or `CLAI_MODEL` selects another. `-p`, `--resume`,
and `--worktree` still apply; `config` and `plugins` subcommands reject `--agent`.
To pass deps or plugins, write a launcher that calls `chat` (see
`customization.md`).

## Headless mode

```bash
clai2 -p "Explain this repository" -m anthropic:claude-sonnet-4-6
clai2 -p "Continue the task" --resume SESSION-ID
```

`-p` / `--prompt` requires prompt text as an argument, never reads stdin,
and runs one turn through tools to completion. Only the final answer is printed
to stdout, without Markdown rendering, wrapping, banners, thinking, or tool output.
Errors go to stderr with a nonzero exit status; Ctrl-C exits with status 130.
The turn is saved and can be resumed. With `-p`, `--resume` requires an explicit
session ID; the browser cannot open. Prompt text is literal, not a slash command.

`-m` is the short form of `--model`. It overrides the saved, project, and
`CLAI_MODEL` model for this invocation without changing your saved preference.
It also works in interactive mode.

Headless mode skips the `ask_user` plugin, including saved replacements, without
changing your preferences. Full-screen plugin requests fail rather than waiting
for input. Other enabled plugins and coding tools still run with your permissions.
Trusted third-party plugins must not read input or print directly to stdout;
CLAI cannot enforce that contract on arbitrary Python code. Plugin load failures
abort headless runs. Background session naming is not started.

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
in `clai2`. The file and shell tools work in `workspace`, and `AGENTS.md` or `CLAUDE.md`
is read from it. A capability you pass that supplies a workspace, such as a sandbox,
takes its place, and a run's own `workspace=` replaces it for that run. Commands get
this process's environment minus LLM provider API keys. Model names resolve as in
`clai2`, the agent's own and any a run passes, so `openai-codex:` and `github-copilot:`
use the sign-ins saved with `/login`. CLAI's per-model defaults apply too, such as
Anthropic prompt caching. Without `model`, every run must pass one.

Nothing else you saved for `clai2` applies: no saved, drop-in, or project plugins, no
`.clai/settings.json`, no `/model settings`, and no `chain:` fallback chains. `ask_user`
and the other terminal plugins stay out, as do `observability` and `mcp`, which read
your CLAI configuration, and the CLAI customization guide. Plugins close when the
`async with` block exits.

`plugin_settings` changes a built-in's settings, merged over the stock ones. This
keeps the file tools inside the workspace and loads no agents from disk:

```python
plugin_settings = {'coder': {'unrestricted_filesystem': False, 'agent_folders': []}}
```

`capabilities` adds your own capabilities. They are bound beside the built-ins, so
delegated tasks carry them too.

### Serving it over ACP

Harness [`run_acp_stdio`](../../docs/harness/acp.md) serves the agent to an ACP
client, such as an editor. This script asks the client before every shell command
and file change:

```python
import asyncio
from pathlib import Path

from pydantic_ai_harness.experimental.acp import run_acp_stdio
from pydantic_ai_harness.guardrails import GuardrailResult, ToolCallInfo, ToolGuardrail
from pydantic_clai2 import open_stock_agent


def ask_first(call: ToolCallInfo) -> GuardrailResult:
    if call.name in ('shell', 'write_file', 'edit_file'):
        return GuardrailResult.approve()
    return GuardrailResult.allow()


async def main() -> None:
    async with open_stock_agent(
        workspace=Path.cwd(),
        model='anthropic:claude-opus-5-5',
        capabilities=[ToolGuardrail(guard=ask_first)],
        plugin_settings={'coder': {'sub_agents': False}},
    ) as agent:
        await run_acp_stdio(agent)


asyncio.run(main())
```

Install `pydantic-ai-harness[acp]` for `run_acp_stdio`. Each approval becomes an ACP
permission request. Every session works in `workspace`. To follow the folder each
client session opens instead, return a `workspace` from a `session_config`, as
[Rooting tools at the workspace](../../docs/harness/acp.md#rooting-tools-at-the-workspace)
shows. `sub_agents` is off because a delegated task cannot yet pass an
approval up to the client, so its run would fail
([#4302](https://github.com/pydantic/pydantic-ai/issues/4302)). Where approvals are
answered in your own process, a
[`HandleDeferredToolCalls`](../../docs/deferred-tools.md#resolving-deferred-calls-with-a-handler)
capability in `capabilities` answers them for delegated tasks too.

## Git worktrees

```bash
clai2 --worktree my-task
clai2 -w
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

CLAI prints the path and branch below its logo; a headless `--prompt` run prints
them on stderr instead. A startup error names the kept checkout. A directory at
that path that is not a Git worktree is rejected. If checkout fails, CLAI tries to remove only the branch it just
created, without forcing deletion. If cleanup or the ignore edit fails, the error
names the retained branch or checkout for recovery.

On normal interactive exit, CLAI checks the linked worktree for changes:
uncommitted or untracked files, or a different commit checked out than at launch.
Ignored files do not count. If nothing changed in a worktree this launch created,
CLAI removes it without asking, and deletes its `clai-NAME` branch too when this
launch created that branch and Git agrees it is merged (`git branch -d`). This
still applies after `/update` restarts CLAI in the worktree. An
unchanged worktree that CLAI reopened, or one you launched inside, is kept without
asking.

If the worktree changed, CLAI asks whether to remove its checkout. Enter, Ctrl-C,
or EOF keeps it; only `y` or `yes` confirms removal, and the branch is kept. Git
removal runs without `--force`, so dirty or locked worktrees are kept with an
explanation. The main checkout is never removed. Headless runs, piped input, and
startup errors keep the worktree without prompting. `/new`, `/resume`, and `/reload` do not remove the checkout:
they leave the shell using the same working directory.
Enter a retained directory and run `clai2 --resume` to continue a saved session. `--worktree` cannot be
combined with `--resume`, `config`, or `plugins`.

When you no longer need the checkout, use Git's own cleanup commands from your
original repository root. Without `--force`, Git refuses to remove a dirty worktree:

```bash
git worktree remove .worktrees/my-task
git branch -d clai-my-task
```

A worktree separates working files, not permissions. CLAI's default tools can
still access files outside it. Creation runs before the agent starts and emits
no agent telemetry spans.

## Codex authentication

The built-in model catalog and `/set model` completions include
`openai-codex:gpt-6.1-sol`, `openai-codex:gpt-6-sol`, and `openai-codex:gpt-6-luna`.

`/login openai-codex` opens the browser and uses core's `OpenAICodexOAuthFlow`:
authorization code with PKCE, state validation, and a callback at
`http://localhost:1455/auth/callback`. It times out after five minutes.

If the browser cannot reach that callback on the machine running CLAI (for
example over SSH), the redirect fails in the browser. Copy the URL from the
address bar and paste it at the prompt CLAI shows under the login link; the bare
`code` value works too. CLAI checks the URL's `state` against the current login
before exchanging the code. Whichever arrives first, the callback or the paste,
completes the login.

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

When no keyring backend exists at all (keyring raises `NoKeyringError` or
`InitError`, typical on a headless Linux box or over SSH), credentials go to a
`0600` file in `$XDG_CONFIG_HOME/pydantic-clai2/` (`~/.config/pydantic-clai2/` by
default) instead, named for the account: Codex uses `credentials-openai-codex.json`,
and the GitHub Copilot, vllm and openrouter connections use their own files. Like keyring entries,
these files are per user, so `--database PATH` does not move them. `/login` says so in its confirmation. A locked keyring is not treated as
missing; unlock it instead. Once a keyring becomes available, the next login or
token refresh encrypts the credentials and deletes the plaintext file.

The default Coder shell runs under your OS identity, without a sandbox. Commands
can read files and access credential backends available to that identity, including
CLAI's tokens. Keyring is storage, not isolation from model-controlled commands.
Use a separate OS account or isolated environment for untrusted repositories.

The requested default does not guarantee model availability for a subscription.
Custom agents supplied to `chat` retain their model unless settings explicitly
select an override. `clai2 --agent MODULE:ATTR` keeps the agent's model over saved
and project models; only `-m` or `CLAI_MODEL` replaces it. `/login` is async, and plugin command handlers may also return
an awaitable string.

## GitHub Copilot subscriptions

```bash
uv run clai2
```

In CLAI, run `/login github-copilot`, then open `/model add` and choose
`github-copilot`. The provider menu also starts login when no credentials exist.
You do not need to register an OAuth application or configure a client ID.
CLAI supplies the same [public Copilot OAuth client ID as Pi](https://github.com/earendil-works/pi/blob/fde38ed7c2f64434beffc6c0ec3b9994cb89ae23/packages/ai/src/auth/oauth/github-copilot.ts#L10-L11)
and requests `read:user` access to your GitHub profile. This identifies the existing
Copilot OAuth application, not a separately registered CLAI application.
`GITHUB_COPILOT_CLIENT_ID` remains an optional override for your own device-enabled
OAuth application; unset or blank uses the bundled default.
The workspace temporarily pins the merged Pydantic AI device-flow implementation
until it is released.

Login prints a code and `https://github.com/login/device`, then starts polling.
Open that link on this or another device and approve only the code shown by your
own CLAI session. CLAI does not launch a browser, so a text browser cannot block
login or take over your SSH terminal. Ctrl-C stops polling; GitHub controls the
code's expiry. No localhost callback or pasted token is needed.

GitHub authorization alone does not establish Copilot access. The model menu
queries your account's catalog and lists only picker-enabled models with
`/chat/completions` support. The shared model menu includes details and `Ctrl+S`
settings. Your subscription and organization policy still control inference access.
You can also select a known ID with `/model github-copilot:claude-haiku-4.5`.

Credentials use the existing keyring backend under the `github-copilot` account,
separate from Codex and API keys. Without a keyring, CLAI reports the plaintext
`credentials-github-copilot.json` fallback file, created with mode `0600`.
Tokens and their issuance time stay out of settings, history, and login output.
Expiring tokens require `/login github-copilot` again; CLAI does not refresh them.
A failed or cancelled authorization leaves the previous login unchanged.

Without a saved login, CLAI accepts `GITHUB_COPILOT_API_KEY`,
`GITHUB_COPILOT_API_TOKEN`, or `COPILOT_GITHUB_TOKEN`, in that order.
It does not read `GH_TOKEN`, `GITHUB_TOKEN`, or another application's token files.
Core owns inference and its telemetry; CLAI adds no login-specific spans.
`/login` without a name asks which sign-in to run.

### Several accounts and fallback chains

`/accounts` lists every account you are signed in to, grouped by provider.
Logins you already had show up on their own.

- **a** adds an account: pick a provider and CLAI fills in a free name, such as
  `account-2`. Press Enter to keep it, then sign in as usual (browser, device code,
  connection, or API key).
- **Enter** on an account signs in to it again, for example after it expired.
- **r** gives an account a name to show, such as your email.
- **d** signs out, after asking. For a plugin's account, it only leaves the list;
  the plugin keeps its own sign-in.
- **[** and **]** reorder accounts. The order is the order `@*`, and a model
  without an account, try them in.

Each account shows its current usage as it loads, without holding up the menu:
`5h 92% · 7d 43%` on the row, and a bar with the reset time for each limit on the
right. CLAI reads it for ChatGPT/Codex (the five-hour and weekly windows) and
GitHub Copilot (premium requests); plugins can report their own, as the Claude
Code plugin does. `…` means it is still loading and `?` that the service did not
answer; the right-hand panel says why.

When you pick a model in `/model add` and its provider has more than one account,
CLAI asks which account runs it, or **all accounts**. All accounts saves the model
as `PROVIDER@*:MODEL`: it runs on the first signed-in account and moves to the next
when a request fails with a model API error, such as a rate or usage limit. Signing
in to another account adds it to the rotation, with nothing else to change. If
every account fails, CLAI lists why each one did.

A model that names no account, such as `claude-code:claude-opus-5-5`, works the
same way: once its provider has two or more signed-in accounts, it runs on all of
them in `/accounts` order, as `@*` does. With one, it runs on that one. To run the
default account alone, write `PROVIDER@default:MODEL`; picking an account in
`/model add` saves it that way. `/set accounts.pool false` turns this off, so a
model without an account runs on the default one only, as it did before.

Under the menus, an account other than the default is a profile, written
`PROVIDER@PROFILE`. You can type it instead:

```text
/login openai-codex@work         # a second ChatGPT/Codex account
/login github-copilot@work       # a second GitHub login
/login openrouter@team           # an OpenRouter browser sign-in or key
/login vllm@lab                  # another vLLM server
/login openai@work               # any provider that takes an API key: a /keys entry or a typed key
/model openai-codex@work:gpt-6-astra
/model openai-codex@*:gpt-6-astra  # every signed-in Codex account, in /accounts order
/model openai-codex@default:gpt-6-astra  # the default account alone
```

The default account is the one without `@PROFILE`. For Codex, Copilot,
OpenRouter, and vLLM that is the existing login. Providers such as
`openai` and `anthropic` still read their usual environment variables by default;
a profile uses the key saved at `/login`. Anything else a provider needs, such as an
Azure endpoint or an Ollama URL, still comes from the environment, and `/login`
refuses to save the key until it is set. Plugins can support profiles too, such
as `/login claude@work` for a plugin's `claude-code` models. A profile name is 1
to 32 lowercase letters, digits, hyphens, or underscores; `default` names the
account without a profile, and `/login NAME@default` signs in to it. Each profile's
credentials are stored separately, the same way as the default login.

A fallback chain does the same across different models or providers, such as Codex
first and Claude after it. It tries its models in order. Chains live in the `/model`
picker, under **Fallback chains** below your models (`/model chains` opens it there):

- **New fallback chain...** asks for a name, then for its models: **a** adds a saved
  model or a typed `PROVIDER[@PROFILE]:MODEL` (including `PROVIDER@*:MODEL`), **d**
  removes one, **[** and **]** move one earlier or later, and **Save chain** saves it.
  Esc leaves without saving.
- **Enter** on `chain:NAME` selects it, like any model; `/model chain:NAME` does too.
- **Ctrl+E** changes a chain's models, **Ctrl+R** renames it, and **Ctrl+D** deletes
  it, after asking. The model you are using, or your saved default, cannot be renamed
  or deleted.

A chain uses its first model's
`/model settings` controls and defaults, and saves overrides under
`chain:NAME`, so `/fast` works for a chain that starts with Codex, as it does for
`openai-codex@*`. A chain cannot contain another chain, but it can contain `@*`. A model whose profile is not signed in fails the turn
instead of falling back, so a missing login is not hidden. Older CLAI builds keep
the rest of your settings, but cannot run a profile, `@*`, `@default`, or chain
model, and run a model without an account on the default account only.

## Settings and commands

Preferences live in `$XDG_CONFIG_HOME/pydantic-clai2/config.db`, falling back to
`~/.config/pydantic-clai2/config.db`. Use `--database PATH` to select another database.
A repository can add its own layer with a `.clai/settings.json` file; see
[Project settings](#project-settings). Conversation messages and CLAI's
Codex tokens are not written to the settings database. Plugin settings are arbitrary
JSON stored in plaintext in this database, including secrets if you put them there.
Pass secret references or use plugin-owned credential storage instead of embedding keys.

When you switch versions or branches, CLAI ignores saved setting names it does not
recognize and leaves their stored values unchanged. Missing settings use the current
defaults. New writes still reject unknown names and invalid values. Invalid saved
values for known settings and unsupported database schema versions still cause an error.

```text
/set
/set model <Tab>
/set display.thinking false
/set run.request_limit 10000
/set run.tool_retries 3
/set accounts.pool false
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
`/model chains`, `/accounts`, `/theme`, `/spinner`, `/tasks`, `/keys`, `/login`, `/resume`, and the
`/google_workspace`, `/grain`, and `/pylon` settings menus typed without further
arguments open right away instead of queueing.
The turn keeps running: its output is held while the menu is open
and printed in order when the menu closes. A question from the agent waits for
the menu to close. Model and run settings saved in the menu apply once the
running turn ends, and so do account changes: the running turn keeps the
accounts it started with. `/model settings` edits to the running model apply to its
next model request in the same turn. `/resume` can browse, rename, and delete
sessions mid-turn, but a running conversation cannot be swapped out: picking one
tells you to enter `/resume ID`, which restores it once the turn ends. With
arguments, these commands queue like any other.

`/plugins` runs right away during a turn too, with or without arguments, so
`/plugins disable NAME` does not wait behind the turn or your queued messages.
Commands, status segments, and model providers change at once. The running turn
keeps the tools and hooks it started with, because the agent binds them when a
run begins; the change reaches the agent on your next prompt, and CLAI says so.
A plugin turned off mid-turn finishes its cleanup when the turn ends. While a
delegated task is running, `/plugins` still refuses changes, as between turns.

`run.tool_retries` sets the default retry budget per tool call, starting at `3`.
Use a non-negative integer; `0` disables retries. Changes apply to the next turn.
Explicit per-tool or per-toolset retry limits take precedence. This setting does
not change output-validation or HTTP transport retries.

## Your instructions: `/system_prompt`

```text
/system_prompt
```

`/system_prompt` shows what the model is told and lets you add instructions of
your own. CLAI builds the system prompt from its built-in coding instructions,
the repository's `AGENTS.md` or `CLAUDE.md`, and the instructions of the plugins
you turned on. The menu shows those parts but cannot change them; turn a plugin
off or configure it in `/plugins` instead. Your instructions are sent after them
and replace none of them.

| Row | What it does |
| --- | --- |
| Edit your instructions | Opens your instructions in an editor. Saving an empty text removes them. |
| Append to your instructions | Opens an empty editor; what you write is added as a new paragraph. |
| Reset: remove your instructions | Removes them, so only CLAI's defaults are sent. Greyed out when you have none. |
| View the full system prompt | The instructions sent with the conversation's latest request, yours included. Enter opens them in a scrollable view. |

Editing opens `$VISUAL`, or `$EDITOR` when `$VISUAL` is unset, on a temporary
Markdown file. Save and quit the editor to keep your changes. Unchanged text
changes nothing, and an editor that exits with an error, such as Vim's `:cq`,
cancels. A graphical editor must wait for its window to close, for example
`code --wait`. On Windows, when neither variable is set, or when the editor
cannot start, a built-in editor opens instead: Ctrl-S saves, Esc cancels.

Your instructions are the `run.instructions` setting, saved with your other
preferences, so later sessions and `/fork` use them too. They apply from your
next prompt. During a turn, `/system_prompt` opens right away, and the running
turn keeps the instructions it started with. `/set run.instructions TEXT` sets
them in one go; `/set` on its own leaves them to `/system_prompt`, which keeps
their line breaks.

## Models and their settings

`/model` selects from models you have already added, and from your
[fallback chains](#several-accounts-and-fallback-chains), listed after them.
Choose **Add a model...** to browse providers and select a new model without
leaving the command. This option is available even when no models have been
added. **New fallback chain...** creates a chain in the same picker.

`/model PROVIDER:NAME` switches directly to any model. A model not yet in your
list is added and selected; CLAI does not check that it exists, so a mistyped
name fails on the next prompt with the provider's error. Tab completes the saved
list and the `add`, `settings`, and `chains` subcommands. Model names normally start with a
provider (`openai:gpt-5`), so no real model is called `add`, `settings`, or `chains`.
The currently configured model is kept in the list when upgrading.

| Command | What it does |
| --- | --- |
| `/model` | Pick a saved model or chain, add one, or edit or delete one |
| `/model NAME` | Select `NAME`, adding it first if needed |
| `/model add` | Browse providers and their models |
| `/model add NAME` | The same as `/model NAME` |
| `/model settings` | Choose a saved model to configure |
| `/model settings NAME` | Configure `NAME` |
| `/model chains` | Open the picker on its fallback chains |

`/add_model`, `/model_settings`, and `/chain` still work as deprecated spellings of
`/model add`, `/model settings`, and `/model chains`.

To remove a model you no longer use, highlight it and press **Ctrl+D** or
**Delete**, then confirm **Delete model**. This removes it from the saved list and
Tab completion and deletes its per-model settings. Provider credentials are kept.
The current model and saved default cannot be deleted. Select another model first,
or use `/set model NAME` to change the saved default. **Keep model** or Esc cancels
without changing anything.

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
saves it in your model list and makes it the model for the next prompt. `Ctrl+S` opens that model's settings:
the model-aware request and thinking controls described below. They are
saved per model and passed to every run with that model. Unsupported settings
may be ignored or rejected by the provider; select only settings your provider supports. `/model NAME` sets the model without the menu.

CLAI installs the SDKs for OpenAI and Anthropic. Selecting a model whose provider SDK is
missing from the Python CLAI runs on fails right away, naming the install command, instead
of on the next prompt. TypeSafe's Jev needs the `typesafe` extra:

```bash
pip install "pydantic-clai2[typesafe]"
```

From a pydantic-ai checkout, run CLAI with the extra instead:

```bash
uv run --package pydantic-clai2 --extra typesafe clai2
```

### Model settings and custom parameters

`/model settings` opens a searchable list of added models. Enter configures a
model without changing the active model. Esc returns from settings to this list;
Esc again closes it. `/model settings PROVIDER:NAME` opens that model directly.
Tab completes added models.
`Ctrl+S` in `/model add` opens the same editor. Edits save immediately and apply
from the model's next request, even in a running turn. `r` resets a field; Esc
or Ctrl-C goes back. Fixed choices
open a picker; numeric fields accept typed values, and empty input resets.

CLAI2 enables Anthropic conversation, static-instruction, and tool-schema caching by default:
`anthropic:` and `gateway/anthropic:` use a 5-minute TTL, while `claude-code:` uses 1 hour.
These are CLI defaults only; plain Pydantic AI agents are unchanged. Saved cache settings override
the defaults. Automatic caching advances to the last cacheable block, including tool results.

`/effort` shows the active model's configured reasoning effort and supported values.
`/effort high` (or another listed value) saves it for that model; `/effort reset`
removes the native effort override. Custom parameters stay unchanged: remove any
that override effort with `/model settings` first. Models without an effort
control say so. Like `/fast`, `/effort` waits for the current turn to finish.

First select `openai-codex:gpt-6-astra` with `/model`, then open `/model settings openai-codex:gpt-6-astra`
(or your saved Codex model), then **Service Tier / Fast Mode**. Choose
**Fast (priority)** to request fast processing, or **Standard (default)** to
turn it off. [Codex fast mode](https://developers.openai.com/codex/speed)
uses more ChatGPT credits and depends on model and account availability. It does
not lower reasoning effort. Reset restores the existing model default; it does
not enable fast mode. The stored values remain `service_tier=priority` and
`service_tier=default`, so older CLAI versions can read them. A custom
`service_tier` body parameter still takes precedence.

While the active model starts with `openai-codex:`, `/fast` toggles between
priority and standard processing. `/fast on` and `/fast off` select explicitly.
It saves the active model's service tier for subsequent prompts and sessions,
without changing reasoning effort or other preferences. It is absent from help
and Tab completion on other models, and typing it there reports an unknown command.
If a custom `service_tier` parameter is set, `/fast` asks you to remove it first
with `/model settings` rather than saving an ineffective change.

Model preferences are shared across checkouts. Reading saved preferences ignores
unknown fields, so newer settings do not break an older reader with this
compatibility fix. Editing or resetting a known field preserves unknown fields
in the store. New edits still reject unknown keys and invalid values.
An invalid value in a known field stops that turn with a repair message, not the
shell; use `/model settings` to fix or reset it and try again. CLAI does not silently
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
1024 and below an explicit `max_tokens`. Without one, Pydantic AI
leaves room for the answer beyond the budget. Other unset fields
use the provider default.
Explicit native thinking settings take precedence over generic `thinking`.
GPT-6 and GPT-5.6 families, including provider-qualified and namespaced names,
default to `thinking=true`, `service_tier=default`, reasoning effort `medium`,
context `all_turns`, mode `standard`, summary `detailed`, and verbosity `low`.
Explicit per-model values win; reset restores the family default without saving
it as an override. Other models keep their existing defaults. Family defaults
follow the model each request uses and sit beneath the settings of
capabilities, a plugin's or your agent's, so a capability can change them; an
agent's own `model_settings` stay beneath the defaults. The values you saved for
the session's model (the one you chose, or CLAI's default) are passed to each run
and win over everything else on requests to that model; a request on a model a
capability selected instead gets only that model's family defaults. Provider-specific
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

For `/set` and `/model add`, Tab completes setting names, boolean values, and model names from Pydantic AI's
built-in catalog without network access. Provider prefixes include `openai-codex:`,
which core supports but does not currently include in that model catalog. Complete
the provider prefix, then enter the model identifier; suggestions do not establish
subscription availability. Custom model identifiers are accepted too.

The command registry uses Termflow's `Completer`, `Document`, and `Completion`
types. Completion rows stay visible while replacement suggestions are computed;
stale suggestions cannot be selected. Provider errors appear in the footer
without ending the session. A blocked completion lookup does not hold menus or
shutdown open; its late result is discarded. Lookups use one daemon worker with
one latest queued request, so a stuck provider cannot create a thread per key. Reopening or growing the popup reuses free
space above the editor instead of adding blank transcript lines.
The interactive editor draws its own pinned prompt and completion rows,
using Termflow's layout helpers. It does not run a prompt-toolkit Application or
renderer. The keyboard decoder and history-file backend still come from
prompt-toolkit, preserving bracketed paste and modified-key handling. Redirected
input retains the simple PromptSession path.
`/set SETTING` shows its current value. `/set` changes apply to subsequent prompts
and preserve conversation history; splash changes apply at next startup.

Precedence is defaults, SQLite overrides, the project file, `CLAI_MODEL`, then
explicit CLI flags. Settings are validated before writes. `/set` updates the
active settings snapshot; legacy `/config` writes apply on restart; plugin
changes apply on the next prompt. `--request-limit` controls the full prompt's
model-request budget.

## Project settings

A repository can pin settings for everyone who runs CLAI inside it. Put a
`.clai/settings.json` next to the code; CLAI looks for one in the launch
directory and each parent, stopping at the first directory that contains
`.git`, and uses the nearest file it finds. Nothing is loaded from above the
repository.

```json
{
  "model": "anthropic:claude-sonnet-4-6",
  "thinking": false,
  "request_limit": 50,
  "plugins": [
    {"id": "exa", "factory": "pydantic_ai_harness.exa:ExaSearch", "settings": {"num_results": 8}},
    {"id": "repo_context", "factory": "pydantic_clai2.builtin_plugins.repo_context", "settings": {"inventory_tool": true}}
  ]
}
```

The keys are the field names from `/config show` (`model`, `request_limit`,
`thinking`, `splash`, `shell_lines`, `grep_lines`, `tool_arg_chars`, `smooth_seconds`) and are
validated the same way as `/set`. A bad value stops startup with the file name
and the problem; a key CLAI does not know is reported once at startup and
ignored, so a newer file still works with an older CLAI. Precedence, lowest
first: defaults, your user settings, the project file, `CLAI_MODEL`, CLI flags.

`plugins` takes the module declarations used by `/plugins add NAME module[:Class] [JSON]`: an `id`, a `factory`
(`module` or `module:attr`), an optional `path`, and optional `settings`. A
relative `path` is relative to the folder that holds `.clai`, whichever
subdirectory you launch from. A
repository cannot switch a plugin on for you: plugins are trusted code running
as your user, so every project-declared plugin starts off, CLAI lists the ones
waiting at startup, and `/plugins enable NAME` is your approval. Approval is
remembered in your user settings together with the declaration you approved,
so a later change to the repository's declaration does not run until you
`/plugins remove NAME` (which forgets your approval and restores the project's
current declaration, off) and enable it again. The approval keeps the file's
absolute path, so another repository cannot put its own file at the same
relative path and have it run as the one you approved. An approval saved by an
older CLAI with a relative path does not load; remove it and enable it again
from its repository. Project declarations rank just
above the built-ins: a project may redeclare `coder` or `repo_context` with
other options, and that replacement is also off until you enable it.

The project file is read-only from inside CLAI. `/set KEY VALUE` writes your user
settings and applies for the current session; `/config set` writes your user
settings and applies on restart. Either way the project value returns at the
next start. In the `/set` menu a value the project
sets carries a muted `project` mark after it, and the details panel names the
origin. CLAI prints the file it found when it starts.

`AGENTS.md` or `CLAUDE.md` in the launch directory is read automatically by the
built-in `repo_context` plugin (harness `RepoContext`), which is separate from
the project file. `/plugins disable repo_context` turns it off, for this and
every later session; `/plugins enable repo_context` brings it back. See
[PLUGINS.md](PLUGINS.md#the-built-in-plugins) for its settings.

Interactive commands: `/login`, `/set` (alias `/settings`), `/system_prompt`, `/theme`, `/model`, `/model add`, `/model settings`, `/model chains`, `/help`, `/clear` (alias `/new`), `/resume`, `/exit`, `/config`,
`/plugins`, `/reload`, `/update`, `/usage`, `/cost`, `/fork`, `/forks`, and `/compact` from the built-in `compaction` plugin.
Tab completion suggests commands, settings, boolean values, plugin identifiers,
and paths after `@`. Suggestions match any substring, case-sensitively. While
you type a command name, the best match comes first and is highlighted without
pressing Tab: the exact name, then names that start with what you typed. Enter
fills in the highlighted command, or runs it when it is already typed in full;
Tab and Up/down move the highlight. For paths,
matching applies to the filename within the typed directory. Path completion inserts
a path; it does not attach file contents.
Unknown command-shaped input such as `/missing` still reports an error instead
of reaching the model. Absolute paths such as `/Users/me/Desktop/Screenshot.png`
are prompts, not commands: a slash, dot, or backslash in the first token after
`/` marks path-like input. Quoted paths are also prompts. For an ambiguous
single-component path with spaces, quote it, for example `"/Screen Shot.png"`.
This routing handles path text that the editor has not converted to an attachment.
After trimming surrounding whitespace, that text, including internal spaces and
shell escapes, reaches the agent unchanged. Routing alone does not read the file;
the agent's configured tools determine how it can access it. Separately, bracketed
paste of existing image paths creates attachments as described in
[Pasting images](#pasting-images).

Up/down move through multiline drafts, then recall queued messages and saved
prompt history. Queued messages come first, newest first, because they are
your most recent input; Up then continues into history, skipping the copies of
queued messages that history already holds. Enter on a recalled queued message
rewrites it in place, keeping its position in the queue, and the queue row
shows `(editing)` meanwhile. Clearing the draft and pressing Enter removes the
message from the queue. If the run takes the message before you press Enter,
the edit is queued as a new follow-up. With nothing queued, Up/down only walk
history.
While you walk history, Up/down stay on history even when a recalled slash
command shows completion suggestions. Press Tab to pick a suggestion; Up/down
then move through the suggestions. Editing the recalled text ends the walk, so
suggestions for a prefix you type take Up/down as before. Esc closes the
suggestions, and Tab brings them back.
Enter submits a prompt when idle and queues a separate follow-up turn when busy.
To steer the active run instead, press Alt+Enter (Option+Enter). With a typed
draft, this immediately shows the message in the transcript and enqueues it for
the run's next model request. It does not interrupt the current response or
cancel in-flight tools. Messages already queued stay queued. With an empty
draft, it sends the oldest queued follow-up instead. Each Alt+Enter sends one
message. Slash commands, `!` shell commands, and exit signals are never steered.
A draft that cannot steer, including any draft while idle, is taken as if you
pressed Enter. A queued message that cannot steer stays queued and is not
skipped over. While running with a draft or a queued message, the input box
shows both shortcuts. If Option+Enter queues like Enter, see
[Option keys on macOS](#option-keys-on-macos).
Shift-Enter inserts a newline when the terminal reports it separately from Enter.
Ctrl-J inserts a newline in the editor; plain Enter submits. Some terminals, including
GNOME Terminal/VTE on Ubuntu, send the same input for Shift-Enter and Enter.
Use Ctrl-J there, or a terminal that supports modified-key reporting, such as
Kitty or xterm. CLAI enables xterm and Kitty keyboard reporting only while the
editor is active and releases it for menus and on exit.
Ctrl-R searches history; Enter accepts a search
result without submitting it. Ctrl-D exits when the draft is empty. Ctrl-C at
input clears the line; during a run it cancels the turn and returns to input. No cancelled run is automatically retried.

## Reload CLAI during development

```text
/reload
```

After editing `pydantic_clai2` source, run `/reload` without arguments. It uses
`importlib.reload` on loaded CLAI modules and rebuilds the prompt loop, commands,
and session with the updated code. The Python process, agent, dependencies passed
to `chat`, conversation messages, selected model, and active settings are kept.
The retained transcript is kept too, including when upgrading a running session
from the older scrollback implementation to the live panel.
Enabled plugins unload and load again so their handlers use the refreshed
shell types. Disabled and unapproved project plugins stay off.

If an import or shell rebuild fails, CLAI reports the error and restores the
previous module bindings. Correct the source and retry `/reload`. Plugin
instances and their contributions are recreated. Installed module globals not
overwritten by the new source can survive; initialize mutable state in `__init__`.
Import-time side effects cannot be undone.

Reload ordering follows module-scope imports in the current Python source, so
adding or changing imports between CLAI modules does not require a restart.
Newly referenced local modules and package initializers are included when planning
the dependency order, but Python imports them only if the updated code uses them.
Function-local imports and `TYPE_CHECKING` guards do not create eager dependencies.
Literal guards and direct comparisons of `sys.platform`, `os.name`, and
`sys.version_info` select only the active branch, including imported aliases.
Other conditions are analyzed conservatively and may require a restart if their
alternative imports form a cycle. No guard expression is executed during planning.
A detected import cycle or invalid source reports an error before reloading modules.

Restart for changes to startup code, the custom agent's construction, or dependencies,
including `pydantic_ai_harness`. `/reload` refreshes CLAI only, not Harness or core,
even when their source has changed in the same checkout. If new CLAI code imports
a symbol the running dependency does not have, restart with the same launch options
and `--resume` to continue the saved session. Keep the worktree if asked to remove it.
`/reload` does not rerun the CLI or recursively reload third-party packages. Import-time side effects
still cannot be undone. Use `/plugins reload NAME` when you only want to reload one
plugin.

## Background forks: `/fork`

`/fork` runs a copy of the current conversation in the background, so you can
keep working while it answers. It follows Code Puppy's `/fork`.

```text
/fork write tests for the parser        fork with the current model
/fork @openai:gpt-5 review this         fork with another model
/fork cancel 2                          stop fork #2
/forks                                  list this session's forks
```

The fork copies the retained history at the moment `/fork` runs and continues
from that copy as its own saved session. Later turns in the foreground do not
reach it, and it does not change the foreground history. With no history yet, or
if the copy fails, the fork starts with a fresh context and says so. It uses the
current model, plugins, and model settings unless `@model` names another model.
CLAI has one agent, so unlike Code Puppy there is no `@agent` argument.

While forks run, the editor shows one row per fork above the prompt, like Code
Puppy's sub-agent panel: the fork number, its model, your `/spinner`, the elapsed
time, and what it is doing (`thinking`, `tool: NAME`, `responding`). A fork that
finished while a turn or command was running shows as done until its output prints.

When a fork finishes, CLAI prints a `FORK #N RESPONSE` banner with the model, the response as
Markdown, the elapsed time, and the saved session id. `/resume SESSION-ID`
switches the foreground to that fork's conversation. Output waits while a turn or
command is running, then prints before the next prompt. Commands run between
turns, so a `/fork` typed during a turn is queued like any other command.

Cancelling a turn with Esc or Ctrl-C also cancels running forks. `/exit` and
`/reload` cancel them too.

A fork is a turn for plugins: `turn_start` runs before it starts and can rewrite
or cancel its prompt, which refuses the fork, and `turn_end` reports its outcome
when it finishes. Forks share the foreground's plugin instances, so a tool that
asks you a question can open its picker from a fork.

## Saved sessions and `/resume`

The project pane groups existing Git worktrees and their subdirectories under
one repository name. Session cards show each checkout's current branch, or its
worktree directory name for detached HEAD. These labels are read when the browser
opens, not historical branch names. Missing directories, non-Git workspaces, and
unavailable Git fall back to directory labels. Separate repositories with the
same name remain separate and use paths to distinguish them. Transcript previews
keep the original saved path; resuming does not change directories or migrate
saved data.

CLAI saves accepted prompts before the first model request and saves the retained
history after successful, failed, and cancelled turns. `/compact` commits its
replacement immediately, even if you exit before another prompt. `/new` switches
to a fresh session ID; it does not delete the previous session.

```bash
clai2 --resume                 # browse saved sessions
clai2 --resume SESSION-ID      # restore one session
```

Inside CLAI, `/resume` opens the same browser and `/resume SESSION-ID` restores a
session directly. Opening or restoring a session does not call the coding model
or execute pending tools. Background naming may make a separate, tool-free model
request. Your current model, working directory, credentials, and approved plugins
remain in effect. The saved model name is shown for reference.

Resuming shows the restored conversation in the terminal: the last 10 prompts
with their answers, reasoning, and tool calls, drawn the way they streamed.
Older turns are summarized in one line; the model still receives the whole
history. Inside CLAI, `/resume` replaces the visible conversation, as `/clear`
does: the previous one stays saved but is not shown again. Output that
only streamed, such as shell previews and diffs, is not saved, so those tools
show their call line.

The browser follows Code Puppy's project/session design:

- Projects on the left, with session counts. The current directory is preselected.
  The selected project stays highlighted while browsing its sessions. **SELECT
  PROJECT** or **SELECT SESSION** labels the focused pane, with matching key hints.
- A repository with sessions in more than one checkout (branch or worktree) lists
  them beneath it, indented and dimmed. Select one to see only its sessions.
  Sessions from deleted folders, such as removed worktrees, gather under
  **missing folders** at the bottom.
- Two-line session cards on the right: time, title, subtitle, tags, message and
  token counts. Recent sorting groups cards by local calendar date.
- Enter opens a project or resumes a session. Right opens a scrollable transcript,
  with newer messages first. Left returns to projects.
- `/` searches across projects, including saved user/assistant text. `s` cycles
  recent, message-count, and token-count sorting. `m` loads another 200 summaries.
- `r` sets a manual title, which the namer will not overwrite. `d` asks for
  confirmation before deletion. The active session cannot be deleted.
- Esc goes back; Ctrl-C closes. Narrow screens show one focused pane at a time.
- Selecting a session from another directory resumes immediately, without
  confirmation. It does not change directories or move the saved conversation out
  of its original project group. Direct cross-directory resume asks you to use the browser.

The browser counts loaded summaries, not a separate unbounded catalog. Search
runs against the full catalog before pagination. It does not index tool output,
reasoning, or content removed by compaction.

The resume transcript preview displays at most 24,000 characters of the newest-first
text, with a truncation notice for longer histories. Search is Unicode
case-insensitive and includes text instructions in multimodal prompts.

### Claude Code and Codex sessions

CLAI can continue a session you started in Claude Code or Codex. It reads their
saved transcripts: Claude Code's under `~/.claude/projects` (or
`$CLAUDE_CONFIG_DIR/projects`), and Codex's under `~/.codex/sessions` (or
`$CODEX_HOME/sessions`).

```bash
clai2 --resume-claude              # browse Claude Code sessions
clai2 --resume-claude SESSION-ID   # import and restore one
clai2 --resume-codex               # browse Codex sessions
clai2 --resume-codex SESSION-ID    # import and restore one
```

The session ID is the one Claude Code or Codex shows, such as the ID in
`claude --resume ID` or `codex resume ID`. Inside CLAI, `/resume claude` and
`/resume codex` work the same way, with or without an ID. `-p` accepts both flags
with an ID. Only one of `--resume`, `--resume-claude`, and `--resume-codex` can be
given.

Plain `/resume` lists them too, beside CLAI's own sessions, in the project they
were started in. A card that says **to import** has not been imported yet:
Enter imports and resumes it, and Right previews it. Rename or delete it after
importing it.

Importing copies the conversation into CLAI's saved sessions. The originals are
only read, never changed. CLAI keeps what the model saw: prompts, answers,
reasoning summaries, tool calls, and their results. It skips context the other
agent added itself, such as Claude Code's command notices and Codex's environment
and `AGENTS.md` messages; CLAI adds its own. After compaction, it keeps the
compacted history, and after a Claude Code rewind, it keeps the branch you ended
on. A tool call with no recorded result gets one saying it was interrupted. The
tools in the old calls belong to the other agent; the model sees them as history,
and CLAI's own tools handle what comes next.

Resuming the same session again opens the same copy. Until you continue it in
CLAI, each resume reads the original again, so later turns in Claude Code or
Codex come along. Once you send a prompt in CLAI, the copy is CLAI's own and
later turns in Claude Code or Codex are not merged into it.

### Background names

A saved session immediately gets a fallback title from its first prompt. A
single background worker can replace it with a short title, subtitle, and up to
four topic tags. The browser refreshes names while idle without moving selection.

The worker uses the previous summary plus up to 2,400 characters of recent
user/assistant text. It refreshes generated names after 16 content revisions;
revisions, unlike message offsets, survive compaction. This is a bounded current
summary, not a lossless incremental transcript archive. Opening the browser
backfills up to ten eligible sessions. Queue length, request count, output size,
and a 60-second deadline bound the work. Exiting cancels and joins the worker.

```text
/set sessions.naming false
/set sessions.naming_model openai:gpt-5-mini
/set sessions.naming_model null
```

Naming is enabled by default and uses the current model unless overridden.
It sends conversation excerpts to that model's provider and incurs additional
usage. It has no tools and does not inherit coding plugins. Missing credentials,
timeouts, invalid output, or stale results leave the existing name usable and do
not interrupt foreground work. `/usage` and the browser preview show persisted
naming token counts separately; `/cost` remains retained foreground-history cost.

### Storage and recovery limits

Conversations, metadata, and step records live in `sessions.db` beside the settings
database. `--database` therefore also selects the directory for saved sessions.
New conversation databases use owner-only file permissions where supported. The
contents are **not encrypted**: prompts, replies, tool results, and media may
contain secrets. Do not share the database between machines. An unfinished run
whose recorded process is still alive is treated as busy; revision checks reject
stale writers instead of overwriting another process's work.

Resuming a busy session does not wait for it or take it over. CLAI copies its
newest saved state, including the latest step checkpoint, into a new saved
session titled with `(fork)`, and continues that copy. The original keeps
running in its own process and may still change files; no tools are replayed.

The built-in `persistence` plugin records additional Harness checkpoints before
model requests, after model responses, and at settled tool-cycle boundaries.
`/plugins disable persistence` disables that extra step capture, not conversation
saving. Without it, a hard kill recovers the accepted prompt and preceding saved
history, not the in-flight turn's progress.

An interrupted session is marked `!`. If a process died mid-run, resume loads its
newest available step checkpoint and warns you to inspect external effects. A
completed tool in a partially completed parallel batch may still have no saved
result. An older settled checkpoint does not undo later file writes or commands.
There is no automatic tool replay, side-effect deduplication, workspace rollback,
or restoration of arbitrary plugin state.

Deletion removes the conversation and associated run records in the same SQLite
transaction. Shared content-addressed media is retained; deletion is not secure
erasure. Snapshot retention keeps eight recent checkpoints per run, plus the
latest settled recovery point when needed. There is no whole-session TTL or media
garbage collection.

## Usage and cost

`/usage` prints a table of the current conversation: one row per turn with the
request count, input tokens, cache read and write tokens when a turn used them,
output tokens, and cost, followed by a totals row. `/cost` prints the retained
history's total on one line. Both are derived from the retained messages, so
`/new` resets them, and a cancelled turn's partial responses are counted.
Compaction drops older responses from these totals; summary requests are not
included. These are not lifetime spending totals or a billing ledger.

Prices come from core's genai-prices data for the response's model and provider.
When there is no price data (a local model, `openrouter` and `vllm` models it
does not list, or an unreleased model), the cost cell reads `unknown` rather
than zero, the line under the table names the model, and only the cost total
excludes those responses. Their requests and tokens remain in the usage totals. Cost is shown to four decimal places; a nonzero amount below that
reads `<$0.0001`. There is no spending cap here; use the agent's `UsageLimits`
for that.

## Compacting the conversation

The built-in `compaction` plugin is on by default, with or without `coder`.
It uses harness's `FallbackCompaction` with
`SummarizingCompaction` first and `SlidingWindowCompaction` as the fallback.
It protects the most recent 50,000 tokens (`protected_tokens`) from summarising and
truncation. `coder`'s own tool-result clearing is separate: above 70% of the window it
still empties tool results older than its last three tool calls. `ModelAPIError`,
`FallbackExceptionGroup`, and `UsageLimitExceeded` during summarisation fall
back to truncation; other exceptions propagate. The summary request is billed
to the current model unless `summarization_model` selects another.

The chain runs automatically before requests above `threshold` (85% of the context
window by default). `/compact` runs the same chain between turns regardless of that
threshold, but it protects at most half the conversation, so it always has older
messages to summarise. Add free text to say what the summary must keep, for example
`/compact don't lose the "auth" decisions`. The focus is passed to the summariser
as written, including quotes, backslashes, and line breaks; no shell escaping is needed.
You get one line with the message counts before and after and an estimate of the
tokens saved out of the total. When compacting would not make the conversation
smaller, for example because the summary is longer than what it replaces, the
conversation is left as it was and you are told so. An empty or one-message
conversation sends nothing.

The window comes from genai-prices, the same catalog the `/model add` menu shows
context sizes from. A model it does not list (`test`, a local endpoint) is
assumed to have 200,000 tokens, the harness default. To change any of this,
run `/plugins configure compaction` (or press `c` on its `/plugins` row): each
edit is validated and saved as you make it, and the plugin reloads so the next
turn uses the new settings. A turn already running keeps the old ones.
You can also redeclare the plugin with your own settings;
`/plugins disable compaction` turns it off, `/compact` included:

```text
/plugins add compaction pydantic_clai2.builtin_plugins.compaction '{"threshold": 0.7, "protected_tokens": 20000, "context_window": 200000}'
```

| Key | Default | Does |
|---|---|---|
| `strategy` | `"summarization"` | `"truncation"` skips the summary and only drops older messages |
| `threshold` | `0.85` | fraction of the window above which the fallback chain runs |
| `protected_tokens` | `50000` | tokens of the most recent messages never compacted |
| `context_window` | unset | overrides the catalog when it is wrong or silent for your model |
| `summarization_model` | unset | a cheaper model to write the summary; unset uses the one in use |

The status row shows compact used/max context tokens, such as `128k/1m`.
The maximum comes from the request's model or the `context_window` override.
It stays `?` until the first request of the session or after a model change, and for
a model with no known window. With `compaction` off, the shell reads the window from the request's model itself. An unknown model's fallback
compaction budget is not shown as its maximum. Counts below 1,000 stay unscaled;
larger counts round to whole thousands (`k`) or tenths of a million (`m`).

The context figure turns yellow when a request still exceeds `threshold` after
compaction, for example because the protected tail is too large. For windows smaller
than 50,000 tokens, redeclare the plugin with a smaller `protected_tokens` value;
`/compact` does not override that protection. Run `/compact` to retry the chain,
or `/new` to clear the history. The figure and colour
refresh with the next request; `/compact` alone does not change them.

## Ask CLAI to customize itself

Ask, for example, "Create a plugin with a custom menu" or "Use my model provider".
The default agent has a `read_clai_customization_guide` tool and a short instruction
to read it before advising on CLAI customization. That hint is placed after the coding
guidance the plugins contribute and before the repository instruction file. The guide
is bundled with the installed package and loaded only when the tool is called, not
included in every prompt. No network access or source checkout is needed to read it.
This tool needs no arguments and ignores extra arguments supplied by a model.
Other tools keep their existing validation.

It covers plugin installation and reload, hooks, tools, settings, commands,
rendering, custom TUI menus, and model/provider launchers. It distinguishes plugin
APIs from UI changes that currently need a CLAI source change. This is guidance,
not an automatic installer or a permission boundary: plugins execute trusted Python
as your user. Review generated plugins before enabling them.

Custom agents are unchanged. To offer the same guide, add
`customization_guide()` from `pydantic_clai2.customization` to their capabilities.
See [PLUGINS.md](PLUGINS.md) for the plugin contract.

## Task delegation

The stock `coder` plugin exposes `delegate_task`, which starts a fresh conversation
with the same active plugin tools, instructions, and guardrails. CLAI rebuilds its
stock agent before the next prompt when that capability snapshot changes; session
history is preserved and running tasks keep their original snapshot.

Saved Coder declarations that omit `sub_agents` still default to `false`. Enable
`sub_agents` in `/plugins configure coder` to opt in, or use `/plugins remove coder`
to restore the stock declaration. Explicit `false` remains an opt-out. Custom agents
are not rebuilt and must bind delegation and its accompanying capabilities themselves.

### Managed tasks in the interactive stock CLI

The interactive stock agent runs `SubAgents` inside a harness `DelegationTasks`
owner. `delegate_task(agent_name, task, background=False, resume=None)` starts a
child with its own history. `general-purpose` is an alias for `self`, carrying
all agent-bound tools, instructions, and guardrails. `Explore` and `Plan` use
only filesystem readers and a `ReadOnlyWorkspace`: neither shell commands nor
`run_code` are available. They inherit the current model and are one-shot.
Custom delegates and general-purpose children can resume with the same task ID.
There are three child layers below the main agent by default.

Use foreground delegation when the answer is needed immediately, and background
delegation for independent work. Give each child a self-contained task with
paths, constraints, and expected evidence. A background acceptance receipt is
not a result. CLAI delivers a settled child's report through core's native
queued-message API, marked as automated, untrusted task data. Reports are not
user instructions or permission grants. In the interactive stock CLI, a report
that arrives while idle automatically continues the parent. No extra user prompt
is needed, and your unfinished draft and queued input are preserved. Reports
arriving during a turn use that run's queue; completed reports are not replayed.
Nested children join their descendants and receive their reports before settling.

- **Ctrl+B** moves foreground children of the main run to the background without
  restarting them. The hint appears only while a foreground child can be
  backgrounded. In tmux, send Ctrl+B through to the application or change the
  tmux prefix.

Task rows use the selected `/theme`: foreground and background modes, running
activity, and failures each have their own role colour.

### Agent folders

CLAI loads custom agents from disk, in Claude Markdown (`*.md`) or Codex TOML
(`*.toml`) format. Choose folders in `/plugins configure coder` under
**Agent folders**, as a JSON list. Each entry is a folder name or a path:

- A name such as `agents` searches `.agents/agents`, `.claude/agents`, and
  `.codex/agents`, first in the project, then in your home directory. Project
  definitions win over personal ones with the same name.
- Add more names, for example `["agents", "global"]`, to also load
  `.claude/global` and its siblings.
- A path such as `./team-agents` or `~/my-agents` loads exactly that folder.
- `[]` turns disk agents off.

The stock CLI starts with `["agents"]`, so existing `.claude/agents` and
`.codex/agents` definitions work without setup. If you saved Coder settings
before this existed, disk agents stay off until you set **Agent folders**.
Definition files are read as data and never executed.
- **`/tasks`** opens the live picker during a turn or between turns. Enter opens
  a full-width transcript with in-flight text; Esc returns to the picker, then
  closes it. Arrow keys scroll the transcript; End follows its tail. Completing
  a task does not close its detail view.
- **`b` / `/tasks background ID`** backgrounds the selected child.
- **`x` / `/tasks stop ID`** stops that child and its descendants, leaving siblings
  running. Stopping a child prevents automatic model resume.
- **`/tasks resume ID`** explicitly authorizes resume and queues a request to the
  parent to continue the saved child. Explore and Plan cannot resume. The parent
  uses `delegate_task` with the same ID and agent name; it remains responsible
  for reviewing the child's result.

The editor panel shows the task tree, activity, elapsed time, and descendant
counts. The footer shows the context of the newest foreground child; see
[Status line](#status-line). Successful rows disappear on completion; failed
and stopped rows remain for 30 seconds. The `/tasks` hint also remains for 30 seconds. The picker retains
completed tasks for inspection. Live previews retain the latest 65,536 characters
of an unfinished text part; settled responses retain their full history.
Questions asked by children use the main
terminal and identify the requesting child.

Task metadata and independent histories are stored beside the session database
in `<database-name>.tasks/`. Each database has its own task directory; in-memory
stores do not write task files. Child runs also use the session's `StepPersistence` store.
After restarting, resume the parent session to inspect or continue its children.
Restoring an interrupted child does not execute it. Inspect partial effects
before requesting resume. Delivery acknowledgements are saved to prevent replay
of reports already consumed by a parent.

Background execution is supported for local workspaces. A run-owned remote or
sandbox workspace can still run foreground children, but background requests and
promotion are refused because that workspace can close when its parent exits.
Plugin changes are held while managed children run so their transports and hooks
remain alive. Exit and reload cancel and drain managed children before plugin
shutdown. Cancelling the main turn stops its foreground delegates; children
already in the background remain owned by the session.

Supplied agents and headless invocations keep their existing delegation behavior;
the managed task UI is not installed on them. Saved Coder `sub_agents: false`
and declarations omitting that setting remain opt-outs. CLAI adds no tools when
delegation is disabled.

## Bring an agent

```python
import asyncio
from pydantic_ai import Agent
from pydantic_clai2 import chat

agent = Agent('test')  # No capabilities required.
asyncio.run(chat(agent, deps=None))
```

`Session(agent, deps=..., plugins=..., on_stream_event=...)` is the noninteractive
API. Call `await session.prompt(text)` for each turn, or
`await session.prompt(text, images=[BinaryContent(data=png_bytes, media_type="image/png")])`
for image input (`BinaryContent` comes from `pydantic_ai.messages`). Native `agent.run` drives the
loop through tools to completion. Successful turns retain `result.all_messages()`;
cancelled turns retain the prompt and messages captured by Pydantic AI, including
interrupted responses and tool results. Failed turns leave the previous history
intact. External tool side effects may already have occurred. History is in memory
only. Structured outputs are supported and displayed after completion.

### Themes

```text
/theme
/theme tokyo_night
/set display.theme github_light
/theme default
```

Theme selection and cancellation are silent.

`/theme` opens a searchable picker. Its preview shows a sample conversation with
Markdown, thinking, a tool call, syntax highlighting, warnings, errors, and the
input/status area. Each bundled palette paints the sample's foreground and
background. Browsing does not apply a palette or save a setting. Press `/` to
search the names. Enter confirms; Esc or Ctrl-C keeps your current choice (while
searching, Esc first leaves the search). Narrow terminals show the list alone.

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

The early splash and the `CLAI 2.0` banner keep Pydantic's brand colours under
every palette, except on 16-colour terminals, where the palette owns the ANSI
slots. Code uses the terminal foreground
and ANSI syntax colours. File diffs tint added and removed lines with the palette's
green and red over its background, and light palettes keep diff code dark enough
to read. Theme selection adds no model requests or telemetry.

### Spinners

```text
/spinner
/spinner puppy
/spinner zoomies 0.1
/spinner init
/set display.spinner dots
```

The spinner is the animation in the `Working` title above the prompt while a turn
runs. `working`, the braille CLAI has always shown, is the default. The catalogue
also carries every Code Puppy builtin: `puppy`, `bone`, `zoomies`, `paws`, `dots`,
`dotsWide`, `dots8Bit`, `dotsCircle`, `sand`, `growVertical`, `growHorizontal`,
`noise`, `binary`, `chevrons`, `bouncingBar`, `bouncingBall`, `pong`, `fistBump`,
and `aesthetic`.

`/spinner` opens a searchable picker with an animated preview. `-`/`+` (or
Left/Right) make the highlighted spinner slower or faster in steps of 0.02 seconds;
Enter applies it, Esc keeps your current choice. Press `/` to search the names;
while searching, `-` and `+` filter, Enter keeps the matches, and Esc clears the
search. `/spinner NAME [SECONDS]` applies by name, ignoring case, and Tab
completes the names. The choice is saved as `display.spinner` and shows on the
next frame, with no restart.

A changed speed, from the picker or `SECONDS`, is saved as that spinner's
`interval` in `spinners.json` next to CLAI's settings
(`~/.config/pydantic-clai2/spinners.json`). The file is also where you add your
own; `/spinner init` writes a starter:

```json
{
  "sniffer": {
    "frames": ["( .    ) ", "(  .   ) ", "(   .  ) ", "(    . ) "],
    "interval": 0.1,
    "description": "a very minimalist puppy"
  },
  "zoomies": {"interval": 0.2}
}
```

An entry with `frames` defines a spinner; one without `frames` that names an
existing spinner changes only its `interval` or `description`. Entries in the file
replace builtins and plugin spinners of the same name. Intervals are clamped to
0.02-1 seconds and frames to 40 characters, padded to one width so the title does
not shift. Edits apply on the next frame. `/spinner` lists any entry it skipped
and why. A saved name that no longer exists, such as a removed plugin's, shows
`working`. Plugins add spinners from `get_spinners` (see `PLUGINS.md`).

### Streaming

Streaming uses the defaults from [Code Puppy's smoothing adapters](https://github.com/mpfaffenberger/code_puppy/blob/a862bf478b63822c9d97093f81e4f827e1c53d6e/code_puppy/agents/smooth_stream.py):

- Markdown uses Termflow `SmoothWriter`: 12 ms ticks, 0.5-second catch-up,
  minimum one visible character per tick. Markdown is parsed line-by-line.
- Reasoning runs through the same Markdown pipeline with Termflow's dim
  renderer, at Code Puppy's thinking pace: 20 ms ticks, 0.4-second catch-up,
  minimum two characters per tick. The `Thinking` heading ends without a
  newline, so the first rendered reasoning line continues on the heading's row.
- Both writers feed the live transcript panel. Output paints at up to 60 frames
  per second; changed cells do not clear the input box.
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
The setting applies to the next turn.

Markdown link labels are clickable in terminals that support OSC 8 hyperlinks.
The URL stays visible beside the label for other terminals and redirected output.
Bare `http://` and `https://` URLs, and `<https://...>` autolinks, are shown in the
link colour and are clickable too; URLs inside code spans are left as code.
URLs longer than 2,048 characters are shown without clickable metadata to limit
streaming output size.
Links survive viewport resizing; following one uses your terminal's usual click
modifier (often Cmd-click or Ctrl-click).

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
Long display lines are ellipsized to terminal width. The command itself wraps
under a hanging indent, keeping its own line breaks, up to 10 terminal rows; a
count summarizes any rows beyond that.
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

File diffs are shown regardless of `display.tool_output`. Native capability events
drive specialized output: `FileEditedEvent` renders its bounded unified diff using
Termflow `DiffRenderer`, the same renderer Code Puppy
uses. The default appearance keeps CLAI's existing addition and deletion
backgrounds; bundled palettes use Termflow's defaults. Both use brighter markers.
Code uses the terminal foreground and ANSI syntax colours on the terminal
background. Successful file writes also show the proposed diff from their matching
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

## Speculative execution

Press **Ctrl+X Ctrl+S** to switch speculative execution on or off, the same chord
as Code Puppy. It is off by default and saved as `run.speculative_code_mode`, so
`/set run.speculative_code_mode true` does the same. Between turns the change
applies to your next prompt. During a turn the switch and the pinned row change
at once, but the running turn keeps the tools it started with: the row says
`on from the next prompt` or `off from the next prompt` until that turn ends.
The agent's tools are bound when a run starts, so the new value cannot reach
the turn that is already running.

While it is on, every tool except `write_file` and `edit_file` becomes a
function inside one harness `CodeMode` `run_code` tool, `shell` included. Other
tools that run a program passed as a string, such as a workflow plugin, stay
native as `CodeMode` keeps them by default.

`run_code` is a persistent Python sandbox with isolated environment variables,
the host clock (through `datetime`), and no network. The model writes one snippet that calls many tools, and CLAI runs it while the model
is still writing.

The sandbox's `pathlib` only reaches what the file tools may reach. `pathlib`
calls on a mount skip the `FileSystem` checks, so CLAI mounts the workspace's
working directory at its real path only when a mount can enforce the same limits.
A plugin that supplies a sandbox workspace gets no mount: the host directory is
not the filesystem the file tools act on there.

- **Read-write:** `read_file` and `write_file` registered, no patterns, and not
  `read_only`. This is the built-in `coder` plugin's default. `pathlib` can also
  delete and rename files there, which `write_file` cannot do but could already
  replace the content of.
- **Read-only:** `read_file` registered but `write_file` missing, `read_only`, or
  `read_only_patterns` set.
- **Not mounted:** no `read_file`, `allowed_patterns` or `denied_patterns`, no
  `FileSystem`, more than one, or a plugin that adds a capability function or
  `DynamicCapability` (which could supply one at run time). File access then
  goes through the file tools only.

While the model writes:

- **Eager execution** runs each complete statement as soon as it has streamed,
  so a slow `shell` build or test starts before the snippet is finished.
- **Speculation** starts `list_files`, `read_file`, `grep`, and
  `read_clai_customization_guide` calls whose arguments are all literals the
  moment their line has streamed. Only these
  read-only tools speculate, because an early call may belong to a branch the
  snippet never takes. They must come from CLAI's own file tools and guide: a
  plugin or MCP tool with the same name, or one that declares itself read-only,
  waits like any other call. Writes and shell commands never start speculatively.

The model gets instructions for writing snippets that benefit. On Anthropic
models CLAI also sets `anthropic_eager_input_streaming`, since Anthropic
otherwise sends tool arguments in one burst at the end, which leaves no time to
run anything early.

Tools called from inside `run_code` show the same headers, previews, and shell
output as direct calls, listed under their `run_code` header in the order they
ran. A speculative call is shown only once the snippet uses its result, so a
launch for a branch the snippet never took does not appear.

A pinned row above the footer shows the session's totals while the switch is on:

```text
Speculative Execution  29 hits · 0 misses · 0 wasted    saved ≥ 7.0s
```

- **hits:** sandbox calls that adopted a speculative launch.
- **misses:** speculation-eligible calls that ran without a matching launch.
- **wasted:** launches discarded without being used.
- **saved ≥:** the summed durations of speculative calls that had finished
  before the snippet asked for them, plus the time sandbox calls spent running
  while the `run_code` arguments were still streaming. Calls still running when
  claimed count as hits but add no time; restarted, rejected, or cancelled
  snippets add nothing. It is a lower bound on hidden tool latency, not
  wall-clock speedup, since concurrent calls can overlap.

Counts are coloured only when non-zero, using `/theme` colours. Switching off
hides the row once no running turn speculates; switching back on shows the same
session totals. Headless runs and redirected output use the same tools but show
no row. The `pydantic-monty`
sandbox behind speculative execution is a `pydantic-clai2` dependency; if it
cannot be imported, CLAI prints a warning and runs tools natively.

## Status line

The terminal footer shows the selected model, activity, latest reported
context tokens, and streamed output estimate, including text, thinking, and
string tool-argument deltas. The estimate is characters divided by four, not a
provider tokenizer count. On completion it is replaced by reported run output
usage. Context is the most recent response's reported input plus output tokens,
not cumulative conversation billing or a context-window percentage, over the
request model's context window (`128k/1m`); `?` means unavailable, such as the
window before the first request or for a model with no known window. As each request goes out, the `compaction` plugin replaces it with
that request's estimated size and paints it yellow while the history is
[over its threshold](#compacting-the-conversation); the response's reported
usage takes over when it lands. The retained-history cost (`$0.0123`) follows the
output count, updated after each turn and hidden until a response has price data.
After `/compact`, the footer keeps the previous figure until the next turn;
`/cost` and `/usage` read the retained history immediately.

While the main run waits on a subagent, the footer shows that subagent's own
figures instead: its name and short task ID, the model it is using, its context
over that model's window, its streamed output estimate, and its activity, such
as `Explore [abcd1234]: claude-sonnet-4-5 | context: 8k/200k tokens | ~1,204
streamed tokens | running: grep`. The newest such subagent wins, so a nested
child takes the row from its parent. When it settles, the row returns to the
next one still running, then to the main conversation, with its figures as they
were. Background tasks, and children below one, stay in the task panel and
leave the footer to the main run. A subagent's figures show `?` until its model
reports them.

The shell owns a pinned editor below a Termflow live transcript panel. It uses
Termflow's cell buffer and changed-cell painter, not a second terminal canvas or
a prompt-toolkit renderer. The hardware cursor stays hidden during input; a
nonblinking reverse-video cell marks the editing position. Status checks run ten
times per second. Terminals shorter than six rows or narrower than six columns
omit the border. Below three rows, the draft is retained but hidden until the
terminal grows. Resizing repaints the panel at the new size and preserves the
draft and scroll position. It never erases terminal scrollback.

Assistant text and thinking retain their Markdown source and render again when
the width or theme changes. Tool and command output rewrap at the new width,
and a `/theme` change repaints their colours role by role: muted text, errors,
diff lines, and code take the new palette's shades. Colours no theme assigns
keep their original values, and so does the CLAI logo. Full-screen menus
temporarily leave the panel and detach its keyboard reader. Output arriving during a menu stays in the transcript and appears when
the menu closes. Inline questions borrow the panel. The transcript includes
startup and plugin lifecycle notices and is carried across shell reloads.
It retains up to 10,000 items and four million characters, plus a bounded partial
line. Old output is dropped at those limits; oversized Markdown parts keep their
rendered tail instead of their source. `/clear` and `/new` forget earlier output,
so neither the panel nor the exit printout shows it again. Hyperlinks are
preserved in that printout; Termflow 1.0's live cells do not carry hyperlink metadata.
Redirected output has no live editor or footer.
No model requests or telemetry are added for status reporting.
The status row follows Code Puppy's styling: muted surrounding text, an accented
output-token count, and purple tool names. Colours follow the selected `/theme`;
context warnings keep the warning colour. The same styling applies while idle
and working.

A plugin can append its own fragment to the row from `get_status_segments`, such
as the working directory or a branch name; fragments are muted and dropped when
the plugin unloads. See [PLUGINS.md](PLUGINS.md) for the method and its cost
rules.

## Plugins

Everything beyond the prompt loop is a plugin, including the default coding
tools. Any Pydantic AI capability is a plugin as it is; give the agent web
search from Pydantic AI Harness without writing code. Install the `exa` extra
and set `EXA_API_KEY` first:

```sh
pip install 'pydantic-ai-harness[exa]'
export EXA_API_KEY=...
```

```text
/plugins add exa pydantic_ai_harness.exa:ExaSearch '{"num_results": 8}'
```

For anything beyond one capability, a plugin is a `Plugin` subclass. Like a
Pydantic AI capability, it declares what it contributes by overriding methods,
each defaulting to nothing. A single plugin can do as much as it likes; this one
both adds a capability and reacts to a lifecycle moment, to show two shapes at
once:

```python
from collections.abc import Sequence

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.exa import ExaSearch

from pydantic_clai2.plugins import Plugin, TurnEnd


class Search(Plugin):
    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (ExaSearch(num_results=8),)

    async def on_turn_end(self, event: TurnEnd) -> None:
        self.host.console.bell()
```

Drop the file in `~/.config/pydantic-clai2/plugins/`, register anything
importable with `/plugins add NAME module[:Class] [JSON]`, or clone a trusted
repository with `/plugins add https://github.com/your-org/my-plugin.git`.
Git repositories need an `__init__.py` or `plugin.py` at their root; install
any dependencies in CLAI's Python environment first. See
[where plugins live](PLUGINS.md#where-plugins-live) for supported URLs and checkout management.
The plugin is live for the next prompt; no restart. `/plugins` alone opens a full-screen menu to enable, disable,
reload, and remove: Space toggles the highlighted plugin, and `c`, `r`, and `d`
configure, reload, and remove it. Press `/` to search plugin names; while you
search, every key you type filters, Enter keeps the matches so the keys act on
them, and Esc clears the search. Esc or `q` closes the menu without printing the
plugin list. Use `/plugins list` to print it. Plugins are trusted code running as you.

[PLUGINS.md](PLUGINS.md) has every method, event, and rule.

Every CLAI on your machine shares one settings database, so another worktree or
branch may have saved a plugin setting this one cannot run. When a saved setting
names a feature this build lacks, CLAI skips just that setting, uses the default,
and tells you once at startup:

```text
coder: ignored saved sub_agents (needs stock-bound-delegation); using defaults.
```

Nothing is rewritten; the other build still sees your setting. If a plugin still
rejects its settings when a turn starts, that turn fails with the plugin named, and
CLAI leaves that plugin's capability out of later turns; `/plugins reload NAME` tries
it again. CLAI builds from before this
check apply every saved setting as they always did.

`/plugins enable notion` gives the agent Notion's hosted MCP tools and opens its
settings menu (`/plugins configure notion` reopens it). The token is picked from
`/keys` by name (a new one is saved there as `NOTION_API_KEY`); without one, it
signs in through the browser. Plugin settings never hold the token. See
[PLUGINS.md](PLUGINS.md#notion-workspace-tools).

`/plugins enable linear` gives the agent Linear's hosted MCP tools, read-only by
default, and opens its settings menu (`/plugins configure linear` reopens it). The
key is picked from `/keys` by name (a new one is saved there as `LINEAR_API_KEY`),
or choose browser sign-in. See [PLUGINS.md](PLUGINS.md#linear-issues-and-projects).

## Questions from the model

When the task is ambiguous, the model can call `ask_user_question` instead of
guessing. Questions appear inline, with the conversation still visible above a
compact numbered picker. The question stays pinned above its choices, so output that
streams meanwhile, from a delegated task for instance, scrolls past above it instead
of pushing it away. A delegated task's question names the task in its title. Use Up/Down and Enter, or press an option's number to
select it. For multiple selections, Enter or a number toggles a choice; select
`Done` to submit. At least one choice is required. Esc or Ctrl-C declines the
whole request, which the model is told so it can make a stated choice and carry
on. Several questions show progress in the title. The tool's transcript line
names the questions by header, not their raw JSON. The editor's draft is
preserved, and the question and your picks are printed to the transcript afterwards.

The inline `ask_user_question` picker also offers `Other (type answer)`.
Choose it to type your own answer instead of the suggested options, including for
multi-select questions. Enter submits nonblank text. Esc returns to the choices
and keeps your draft; Ctrl-C declines the whole request. Backspace and arrow keys
edit the text. Multiline paste is inserted as text and waits for Enter; it does
not submit an answer or select choices. The conversation stays visible while you type. Custom answers
appear in the transcript and reach the model as a one-item list under the question's header.

The menu is the built-in `ask_user` plugin around the harness's
[`AskUser`](../../docs/harness/ask-user.md) capability. The capability only knows an
`Answerer`; the terminal menu is one, and [PLUGINS.md](PLUGINS.md#ask_user-questions-answered-from-the-terminal)
shows how to put a different one, a web form for instance, in its place.
`/plugins disable ask_user` removes the tool.

## Telemetry and references

The stock CLI enables the built-in `observability` plugin by default. It adds Pydantic
AI's [`Instrumentation`](https://pydantic.dev/docs/ai/capabilities/overview/)
capability to CLAI turns for agent, model-request, and tool
spans, including timing, token usage, and failures. It adds CLAI's own UI spans
only when `ui_events` is on (see below). HTTP client instrumentation is off by
default; agent instrumentation does not affect unrelated agents globally.

Startup plugin load failures reported in the terminal are also sent through the configured
Logfire instance, including their exception and traceback, even when `ui_events` is off. Failures are
reported after loading finishes, including those that happened before observability
loaded. Disabling the plugin leaves these failures as terminal messages only.

This plugin was previously named `logfire`. Existing enabled/disabled choices,
settings, and saved token references carry over without reconfiguration. Existing
commands and project or drop-in declarations using `logfire` still target this
same plugin; it appears as `observability` in `/plugins`.

Set `LOGFIRE_TOKEN` to a write token for your Logfire project. Alternatively,
place the SDK's `logfire_credentials.json` in your user config directory at
`$XDG_CONFIG_HOME/pydantic-clai2/logfire/` (default
`~/.config/pydantic-clai2/logfire/`). SDK configuration is also read only from
that directory. The plugin ignores repository-local Logfire configuration and
credentials, plus `LOGFIRE_CONFIG_DIR` and `LOGFIRE_CREDENTIALS_DIR`, so a checkout
cannot choose the telemetry destination. Relative `XDG_CONFIG_HOME` values fall
back to `~/.config`. Export uses `send_to_logfire='if-token-present'`: no credentials
means no Logfire export and no interactive project setup. Logfire's
terminal console output is disabled so it does not interfere with the editor.
Standard SDK configuration, including explicitly configured OTLP exporters, still
applies; disable the plugin to stop its instrumentation altogether.

Agent runs and recorded UI interactions nest under a `CLAI session` root span.
Its `agent_session_id` attribute is the saved conversation ID.
`/clear` selects a new root; `/resume` returns to that conversation's root if it
was already opened by this plugin instance. Unloading the plugin ends its roots;
reloading starts new traces with the same saved conversation IDs.

Each session root is tagged with your email, as a Logfire tag and the
`user.email` attribute, never on child spans or logs. `user_tag` picks where it
comes from. The default, `logfire-account`, uses the account you signed in with
when you set up the **Logfire project** (below); that account already has access
to the project, so the tag reveals nothing new to it. With a token from
`LOGFIRE_TOKEN`, the credentials file, a token changed since setup, a setup made
before this setting existed, or a server that does not report your email, roots
are not tagged until you run the setup again. `git-email` uses
`git config user.email` instead (Git is only queried with this choice; a missing
email leaves the tag out), and `false` turns the tag off. Choose **User tag** in
`/plugins configure observability`, or set `user_tag` in the plugin settings.
Everything CLAI records itself (session roots, UI records, and plugin load
failures) uses the `clai2` instrumentation scope.

Prompts, responses, tool arguments/results, and binary image attachments are
included by default, including retained history used by later turns. This can
send source code, file contents, and screenshots to the configured telemetry
destination. Review that destination before supplying
credentials. Keep tokens out of plugin settings, which are saved as plaintext.

```text
/plugins configure observability
/plugins disable observability
/plugins enable observability
/plugins reload observability
/plugins add observability pydantic_clai2.builtin_plugins.logfire '{"include_content": false, "include_binary_content": false}'
```

`/plugins configure observability`, or `c` on `observability` in `/plugins`, opens
its settings menu. Each option below is a row there; each edit saves at once and
applies from the next run. The last command replaces the built-in configuration
instead. Its options are
`service_name` (default `pydantic-clai2`), `include_content` (default `true`),
`include_binary_content` (default `true`), `httpx` (default `false`), and `send_to_logfire` (either
`"if-token-present"` or `false`). The plugin explicitly sets the latter, rather
than taking `LOGFIRE_SEND_TO_LOGFIRE` from the environment. Content flags control
Pydantic AI's prompt/result and standard binary-content capture, not all metadata;
model/tool names and tool definitions may still be recorded. Logfire's normal
scrubbing remains enabled. Set `httpx` to `true` to trace HTTP requests made with
`httpx` and `httpx2` while the plugin is active. With `include_content=true`,
Logfire also captures HTTP headers and request and response bodies; with it off,
those are not captured. HTTP instrumentation is process-wide, unlike agent tracing.

Two more options choose where telemetry goes and what it covers. `token` names a
`/keys` entry holding a Logfire write token, which then takes the place of
`LOGFIRE_TOKEN` and the credential file; a missing key stops export with a warning
rather than falling back. `ui_events` (default `true`; set `false` to opt out) adds spans and logs in the
`clai2` scope for UI interactions: menus, slash commands, `/set`, plugin actions,
`/keys`, prompt submissions, steering, interrupts, completions, and session start,
clear, and resume. They record names and listed choices, never typed values or
secrets. The one exception: while `include_content` is on, a submitted prompt also
carries its text as `prompt`, cut to 64,000 characters. `!` lines record only
that they were shell commands and their length, and slash-command arguments are
never recorded, since both can hold secrets such as `/plugins add` settings.

### Setting up where traces go

Choose **Logfire project** in the settings menu (`/plugins configure observability`, or
`c` on `observability` in `/plugins`):

1. Pick where traces go: Logfire US, Logfire EU, or a self-hosted Logfire URL.
2. Sign in, or sign up, in the browser. CLAI prints the link too, so it works over SSH.
3. Pick one of the projects you can write to.

CLAI then creates a write token for that project, saves it in `/keys` as
`LOGFIRE_TOKEN_<ORG>_<PROJECT>`, and points the plugin's `token` at it; the plugin
reloads and the next turn is traced there. The sign-in itself is not kept, only
your account's email, saved with the key name as `account` to tag session roots. The
URL you picked is saved as the plugin's `base_url`, so `LOGFIRE_BASE_URL` cannot
send the token elsewhere, and sending is turned on if it was off. Choose the row
again to switch projects, or press `R` on it to go back to `LOGFIRE_TOKEN` or the
credentials file.

### Sending UX telemetry to the Pydantic shared project

`@pydantic.dev` staff can send CLAI UX telemetry to the team's shared Logfire
project: run `/plugins configure observability`, choose **Logfire project**, pick
Logfire US, sign in with your Pydantic account, and pick the shared CLAI project.
UI events are recorded by default. If you previously turned them off, set
**UI events** to recorded in the same menu, or:

```text
/plugins add observability pydantic_clai2.builtin_plugins.logfire '{"token": {"name": "LOGFIRE_TOKEN_<ORG>_<PROJECT>"}, "ui_events": true}'
```

using the key name the setup menu printed (Esc closes the menu that `/plugins add`
opens). Run the setup menu again to go back to your personal project. Agent spans
still include content by default; add `"include_content": false,
"include_binary_content": false` if you'd rather share only UX telemetry and
timing with the team.

The plugin owns an isolated Logfire instance. Disable, reload, or exit flushes
and shuts down that instance without shutting down application-global providers.
The existing global propagator is preserved. Logfire's SDK may install shared
executor context-propagation helpers; those SDK hooks are not removed on unload.
The plugin does not mutate a supplied agent. While enabled, its explicit per-run
`Instrumentation` takes precedence over that agent's instrumentation settings;
disabling it leaves the agent's original configuration in effect. Custom
`chat()` launchers only get built-ins when passed `builtin_plugins=DEFAULT_PLUGINS`.

- [Pydantic AI agent execution and events](https://pydantic.dev/docs/ai/core-concepts/agent/)
- [Capability events](https://pydantic.dev/docs/ai/capabilities/overview/)
- [Code Puppy splash](https://github.com/mpfaffenberger/code_puppy/blob/main/code_puppy/splash.py)
- [Code Puppy streaming](https://github.com/mpfaffenberger/code_puppy/blob/main/code_puppy/agents/event_stream_handler.py)
- [Code Puppy command registry](https://github.com/mpfaffenberger/code_puppy/blob/main/code_puppy/command_line/command_registry.py)

See `THIRD_PARTY_NOTICES.md` for attribution.

## vllm connection

Open `/model add`, choose `vllm`, then enter a trusted HTTP(S) server root or `/v1` URL, and optionally a token. CLAI queries `/v1/models` and opens a searchable model picker. HTTP sends tokens unencrypted; use HTTPS outside trusted local networks. The connection is saved like Codex's, see [Codex authentication](#codex-authentication).

## openrouter connection

Open `/model add`, choose `openrouter`, then choose **Sign in with browser** or **Enter API key**. Browser sign-in opens OpenRouter's [PKCE authorization flow](https://openrouter.ai/docs/use-cases/oauth-pkce) and receives an authorization code on a temporary loopback listener. CLAI exchanges the code for a user-controlled API key over HTTPS. If the browser cannot reach CLAI (for example over SSH), paste the final callback URL or authorization code into the terminal. If no browser opens, open the printed authorization URL manually. Login times out after five minutes; Ctrl-C cancels it. You can revoke the generated key on OpenRouter.

Manual entry still accepts a key from https://openrouter.ai/keys in a masked prompt. After either method, select a model from the live catalog. CLAI validates the key with `/api/v1/key` before fetching `/api/v1/models`. Cancelling before model selection leaves the saved connection unchanged.

The connection is saved in the configured Python keyring backend after selection, or in a per-user `0600` file when no keyring backend exists, as described in [Codex authentication](#codex-authentication). Backend security depends on your keyring configuration. Tokens are not stored in SQLite or command history. The selected model persists across restarts. Select the provider again to browse its live models or reconfigure the saved connection. Discovery is explicit and has a 20-second network timeout; redirects are not followed. Agent inference uses Pydantic AI core.


## Saved API keys

Open `/keys` to browse and manage saved API keys in a full-screen menu.
Use `a` to add, Enter to replace a value, `r` to rename, and `d` to delete with
confirmation. Press `/` to search key names; Enter keeps the matches, and Esc
clears the search. Values are masked and are not shown in previews. Changes save
immediately; Esc or Ctrl-C closes the menu. Errors appear inside the menu.

The compatibility command `/set api_key` prompts for a name and masked API key. Names are trimmed
and uppercased automatically, so `my_vllm_key` becomes `MY_VLLM_KEY`. Use letters,
numbers, and underscores, starting with a letter or underscore. Saving an existing
name asks before replacing it. Ctrl-C or Ctrl-D cancels without saving. Do not put
the secret on the command line.

Consumers store a key's name, not its value, and look it up each time they
connect, so replacing a value in `/keys` updates every consumer and a deleted key
makes them fail with an error. Renaming a key that vLLM, OpenRouter, or a
plugin uses is refused until they are pointed at another key. Several
consumers can share one entry: name keys with the conventional variable name for
the service, such as `LINEAR_API_KEY` or `GITHUB_TOKEN`, and every plugin for that
service can pick the same entry. The names are labels only; CLAI does not export
them as environment variables.

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
reference, reconfigure it through `/model add` and select a saved key. Changes do not
alter an already running request or revoke credentials at the provider.
