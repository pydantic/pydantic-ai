# CLAI 2.0

A separately installable terminal client for Pydantic AI. The coding tools,
`Coder(unrestricted_filesystem=True)`, are the built-in `coder` plugin: on by
default, `/plugins disable coder` for a chat-only shell. The model can also ask
you multiple-choice questions mid-run through the built-in `ask_user` plugin;
see [Questions from the model](#questions-from-the-model). The built-in
`repo_context` plugin reads `AGENTS.md` or `CLAUDE.md` from the launch directory
into the agent's instructions; `/plugins disable repo_context` turns that off.
Context management is the built-in `compaction` plugin,
[described below](#compacting-the-conversation).
The `/plugins` menu also lists every other harness capability, disabled by
default. Press Space to enable one. Some need optional packages, credentials,
or constructor settings first; see [optional harness capabilities](PLUGINS.md#optional-harness-capabilities).
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
connect is marked `error` and left out rather than failing the prompt. See
[Connect MCP servers](PLUGINS.md#connect-mcp-servers) for storage, secrets, OAuth,
and project trust. `/plugins disable mcp` removes the command and tools.

Python 3.11+ is required by Termflow. Tracking issue: https://github.com/pydantic/pydantic-ai-harness/issues/875.

CLAI file tools can access paths outside the workspace, including `/tmp`, and do
not protect secret files or repository metadata. OS permissions still apply.
Relative paths use the launch workspace. Use a custom agent with `Coder()` to
retain workspace-scoped file tools.

Tool calls show a single-line summary followed by a blank line by default.
Tool names are pink; their arguments and bullet markers are muted grey. Shell output, exit details and
log paths, grep results, and file diffs stay out of the terminal; the model still
receives full tool results. Long summaries are clipped to the terminal width.
Use `/set display.tool_output true` to show detailed output again, or
`/set display.tool_output false` to return to summaries. In detailed mode,
`display.shell_lines` and `display.grep_lines` limit previews to 20 lines by
default. Plugin-provided rendering, including interactive questions, is unchanged.

## Startup

`clai2 --help` parses arguments without loading the agent or plugins. Interactive
startup defers model menus and provider integrations until you open those menus,
log in, or run a prompt. The first use can therefore take longer. Enabled plugins
still load before the first prompt; their initialization contributes to startup time.
`/login` offers both Codex and GitHub Copilot without loading their integrations for
completion. Copilot requests use your saved login through the lazy provider resolver.

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

## Prompt area

```text
Follow-up: Add tests for the change
Command: /usage
─ Working ⠋ ────────────────────────────────────
/
/resume Browse or restore a saved session
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
of your draft or submitted message. The editor reserves rows below a terminal
scroll region. Both partial and completed output stream directly above it,
without erasing or repainting the input box. Typing updates the draft row; a
nonblinking highlighted cell marks the cursor. Full-screen menus temporarily hide it along
with the editor. Enter submits a message to an in-memory queue. Pending text appears above the editor as `Follow-up:`
previews, with queued slash commands labeled `Command:`. Previews are shown in
execution order and disappear as each submission starts. Long or multiline messages
have a single-line preview; large queues show a `+N more queued` summary to leave
room for the editor. The original message text is unchanged. The footer also shows
the number waiting. This is a read-only preview, not a queue editor. Control bytes in completion-derived
text are escaped in previews and prompt echoes; the submitted text is unchanged.

Messages and slash commands run in submission order, after the current turn and its cleanup
finish. They do not interrupt or steer the active turn. An unsubmitted draft stays
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
`/new` starts a new saved conversation, without deleting the previous one or
input recall. `/clear`, or bare `clear`, is an alias of `/new`. Model responses and tool results are not saved to this file.

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

## Git worktrees

```bash
clai2 --worktree my-task
clai2 -w
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
names the retained branch or checkout for recovery.

On normal interactive exit from a linked worktree, CLAI asks whether to remove
its checkout. Enter, Ctrl-C, or EOF keeps it; only `y` or `yes` confirms removal.
This also applies when launching inside an existing linked worktree. Git removal
runs without `--force`, so dirty or locked worktrees are kept with an explanation.
The branch is kept even when removal succeeds. The main checkout is not offered
for removal. Headless runs, piped input, and startup errors keep the worktree
without prompting. `/new`, `/resume`, and `/reload` do not remove the checkout:
they leave the shell using the same working directory.
Enter a retained directory and run `clai2 --resume` to continue a saved session. `--worktree` cannot be
combined with `--resume`, `config`, or `plugins`.

When you no longer need the checkout, use Git's own cleanup commands from your
original repository root. Without `--force`, Git refuses to remove a dirty worktree:

```bash
git worktree remove .worktrees/my-task
git branch -d clai/my-task
```

A worktree separates working files, not permissions. CLAI's default tools can
still access files outside it. Creation runs before the agent starts and emits
no agent telemetry spans.

## Codex authentication

The built-in model catalog and `/set model` completions include
`openai-codex:gpt-6-sol` and `openai-codex:gpt-6-luna`.

`/login openai-codex` opens the browser and uses core's `OpenAICodexOAuthFlow`:
authorization code with PKCE, state validation, and a callback at
`http://localhost:1455/auth/callback`. It times out after five minutes.

If the browser cannot reach that callback on the machine running CLAI (for
example over SSH), the redirect fails in the browser. Copy the URL from the
address bar and paste it at the prompt CLAI shows under the login link; the bare
`code` value works too. CLAI checks the URL's `state` against the current login
before exchanging the code. Whichever arrives first, the callback or the paste,
completes the login.

Tokens live in the configured Python `keyring` backend under service `pydantic-clai2`,
not in SQLite or `~/.codex/auth.json`. Large token bundles are split across keyring
entries to fit Windows Credential Manager's per-entry size limit. Existing
single-entry logins remain readable. Choose an OS-backed credential store: CLAI
uses the configured backend and does not enforce its encryption or storage policy.
Installing or selecting a plaintext backend can store tokens in plaintext. Core owns
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
token refresh moves the credentials there and deletes the file.

The default Coder shell runs under your OS identity, without a sandbox. Commands
can read files and access credential backends available to that identity, including
CLAI's tokens. Keyring is storage, not isolation from model-controlled commands.
Use a separate OS account or isolated environment for untrusted repositories.

The requested default does not guarantee model availability for a subscription.
Custom agents supplied to `chat` retain their model unless settings explicitly
select an override. `/login` is async, and plugin command handlers may also return
an awaitable string.

## GitHub Copilot subscriptions

```bash
uv run clai2
```

In CLAI, run `/login github-copilot`, then open `/add_model` and choose
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
You can also select a known ID with `/add_model github-copilot:claude-haiku-4.5`.

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
`/login` without a provider continues to sign in to Codex.

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
```

`/set` on its own opens a full-screen menu, the same kind Code Puppy uses: the
settings on the left, details for the highlighted one on the right (current
value, default, what it does). Type to filter. Enter edits: booleans and the
model get a picker (the model list is searchable, with "Type a value..." for
anything not listed), everything else a typed input that validates as you go.
An empty value resets. `R` resets the highlighted setting. Esc closes. Every
edit saves and applies immediately, the same as `/set KEY VALUE`.

`run.tool_retries` sets the default retry budget per tool call, starting at `3`.
Use a non-negative integer; `0` disables retries. Changes apply to the next turn.
Explicit per-tool or per-toolset retry limits take precedence. This setting does
not change output-validation or HTTP transport retries.

## Models and their settings

`/model` selects from models you have already added. Choose **Add a model...**
to browse providers and select a new model without leaving the command. This
option is available even when no models have been added. Tab completion uses
only the saved list. `/model NAME` switches directly to an added model.
The currently configured model is kept in the list when upgrading.

`/add_model` opens a searchable provider list, then a model picker for that provider.
Esc from the model list returns to providers. Providers are unique prefixes from
the merged catalog, including `openai-codex`. Its suggestions include
`gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol`, and `gpt-6-astra`; availability
depends on your account. Unknown prices and context limits are not inferred.

The model catalog combines genai-prices' catalog
filtered to providers Pydantic AI can run, plus core's own model list, plus
whatever you have set now. The left side shows model names and marks the current
model; token counts stay in the details. The right side shows the provider, context window,
prices, and any settings you have saved for that model. Type to filter. Enter
saves it in your model list and makes it the model for the next prompt. `Ctrl+S` opens that model's settings:
the model-aware request and thinking controls described below. They are
saved per model and passed to every run with that model. Unsupported settings
may be ignored or rejected by the provider; select only settings your provider supports. `/add_model NAME` sets the model without the menu.

### Model settings and custom parameters

`/model_settings` opens a searchable list of added models. Enter configures a
model without changing the active model. Esc returns from settings to this list;
Esc again closes it. `/model_settings PROVIDER:NAME` opens that model directly.
Tab completes added models.
`Ctrl+S` in `/add_model` opens the same editor. Edits save immediately and apply
on the next prompt. `r` resets a field; Esc or Ctrl-C goes back. Fixed choices
open a picker; numeric fields accept typed values, and empty input resets.

First add `openai-codex:gpt-6-astra` with `/add_model`, then open `/model_settings openai-codex:gpt-6-astra`
(or your saved Codex model), then **Service Tier / Fast Mode**. Choose
**Fast (priority)** to request fast processing, or **Standard (default)** to
turn it off. [Codex fast mode](https://developers.openai.com/codex/speed)
uses more ChatGPT credits and depends on model and account availability. It does
not lower reasoning effort. Reset restores the existing model default; it does
not enable fast mode. The stored values remain `service_tier=priority` and
`service_tier=default`, so older CLAI versions can read them. A custom
`service_tier` body parameter still takes precedence.

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

For `/set` and `/add_model`, Tab completes setting names, boolean values, and model names from Pydantic AI's
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
    {"id": "repo_context", "factory": "pydantic_clai2.repo_context", "settings": {"inventory_tool": true}}
  ]
}
```

The keys are the field names from `/config show` (`model`, `request_limit`,
`thinking`, `splash`, `shell_lines`, `grep_lines`, `smooth_seconds`) and are
validated the same way as `/set`. A bad value stops startup with the file name
and the problem; a key CLAI does not know is reported once at startup and
ignored, so a newer file still works with an older CLAI. Precedence, lowest
first: defaults, your user settings, the project file, `CLAI_MODEL`, CLI flags.

`plugins` takes the same declarations as `/plugins add`: an `id`, a `factory`
(`module` or `module:attr`), an optional `path`, and optional `settings`. A
repository cannot switch a plugin on for you: plugins are trusted code running
as your user, so every project-declared plugin starts off, CLAI lists the ones
waiting at startup, and `/plugins enable NAME` is your approval. Approval is
remembered in your user settings together with the declaration you approved,
so a later change to the repository's declaration does not run until you
`/plugins remove NAME` (which forgets your approval and restores the project's
current declaration, off) and enable it again. Project declarations rank just
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

Interactive commands: `/login`, `/set`, `/theme`, `/model`, `/add_model`, `/model_settings`, `/help`, `/new`, `/clear`, `/resume`, `/exit`, `/config`,
`/plugins`, `/reload`, `/usage`, `/cost`, and `/compact` from the built-in `compaction` plugin.
Tab completion suggests commands, settings, boolean values, plugin identifiers,
and paths after `@`. Suggestions match any substring, case-sensitively. For paths,
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

Up/down move through multiline drafts, then recall saved prompt history.
Enter submits a prompt when idle and queues a separate follow-up turn when busy.
To steer instead, first queue the message with Enter, then press Alt+Enter
(Option+Enter). This sends the oldest queued follow-up to the active run at its
next opportunity without cancelling in-flight tools or changing your draft.
Each Alt+Enter sends one message. If the run is no longer accepting steering,
the message stays queued. Slash commands and exit signals are not steered or
skipped over. With no queued message, Alt+Enter does nothing.
While running, the input box shows both shortcuts.
Shift-Enter inserts a newline. CLAI requests modified
key reporting while the editor is active and releases it for menus and on exit.
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
Enabled plugins unload and activate again so their handlers use the refreshed
shell types. Disabled and unapproved project plugins stay off.

If an import or shell rebuild fails, CLAI reports the error and restores the
previous module bindings. Correct the source and retry `/reload`. Plugin hosts
and their registrations are recreated. Installed module globals not overwritten
by the new source can survive; initialize mutable state in `activate`.
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

Restart for changes to startup code, the custom agent's construction, or dependencies
loaded dynamically rather than declared by module-scope imports. `/reload` does not
rerun the CLI or recursively reload third-party packages. Import-time side effects
still cannot be undone. Use `/plugins reload NAME` when you only want to reload one
plugin.

## Saved sessions and `/resume`

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

The browser follows Code Puppy's project/session design:

- Projects on the left, with session counts. The current directory is preselected.
  The selected project stays highlighted while browsing its sessions. **SELECT
  PROJECT** or **SELECT SESSION** labels the focused pane, with matching key hints.
- Two-line session cards on the right: time, title, subtitle, tags, message and
  token counts. Recent sorting groups cards by local calendar date.
- Enter opens a project or resumes a session. Right opens a scrollable transcript,
  with newer messages first. Left returns to projects.
- `/` searches across projects, including saved user/assistant text. `s` cycles
  recent, message-count, and token-count sorting. `m` loads another 200 summaries.
- `r` sets a manual title, which the namer will not overwrite. `d` asks for
  confirmation before deletion. The active session cannot be deleted.
- Esc goes back; Ctrl-C closes. Narrow screens show one focused pane at a time.
- Selecting a session from another directory asks for confirmation. It does not
  change directories or move the saved conversation out of its original project
  group. Direct cross-directory resume asks you to use the browser.

The browser counts loaded summaries, not a separate unbounded catalog. Search
runs against the full catalog before pagination. It does not index tool output,
reasoning, or content removed by compaction.

The resume transcript preview displays at most 24,000 characters of the newest-first
text, with a truncation notice for longer histories. Search is Unicode
case-insensitive and includes text instructions in multimodal prompts.

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

The built-in `compaction` plugin uses harness's `FallbackCompaction` with
`SummarizingCompaction` first and `SlidingWindowCompaction` as the fallback.
It protects the most recent 50,000 tokens. `ModelAPIError`,
`FallbackExceptionGroup`, and `UsageLimitExceeded` during summarisation fall
back to truncation; other exceptions propagate. The summary request is billed
to the current model unless `summarization_model` selects another.

The chain runs automatically before requests above `threshold` (85% of the context
window by default). `/compact` runs the same chain between turns regardless of that
threshold. Add words to say what the summary must keep: `/compact the auth refactor, not the CSS`. You get one line
with the message counts before and after and an estimate of the tokens saved.
An empty conversation, or one that fits inside the protected tail, says so and
sends nothing.

The window comes from genai-prices, the same catalog the `/add_model` menu shows
context sizes from. A model it does not list (`test`, a local endpoint) is
assumed to have 200,000 tokens, the harness default. To change any of this,
redeclare the plugin with your own settings; `/plugins disable compaction`
turns it off, `/compact` included:

```text
/plugins add compaction pydantic_clai2.compaction '{"threshold": 0.7, "protected_tokens": 20000, "context_window": 200000}'
```

| Key | Default | Does |
|---|---|---|
| `strategy` | `"summarization"` | `"truncation"` skips the summary and only drops older messages |
| `threshold` | `0.85` | fraction of the window above which the fallback chain runs |
| `protected_tokens` | `50000` | tokens of the most recent messages never compacted |
| `context_window` | unset | overrides the catalog when it is wrong or silent for your model |
| `summarization_model` | unset | a cheaper model to write the summary; unset uses the one in use |

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

The early splash retains its brand colours. Code uses the terminal foreground
and ANSI syntax colours; bundled palettes use Termflow's default diff colours. Theme selection adds no
model requests or telemetry.

### Streaming

Streaming uses the defaults from [Code Puppy's smoothing adapters](https://github.com/mpfaffenberger/code_puppy/blob/a862bf478b63822c9d97093f81e4f827e1c53d6e/code_puppy/agents/smooth_stream.py):

- Markdown uses Termflow `SmoothWriter`: 12 ms ticks, 0.5-second catch-up,
  minimum one visible character per tick. Markdown is parsed line-by-line.
- Reasoning runs through the same Markdown pipeline with Termflow's dim
  renderer, at Code Puppy's thinking pace: 20 ms ticks, 0.4-second catch-up,
  minimum two characters per tick. The `Thinking` heading ends without a
  newline, so the first rendered reasoning line continues on the heading's row.
- Both writers feed the terminal scroll region directly. Partial text does not
  wait for an editor refresh, and a completed line does not clear the input box.
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

Markdown link labels are clickable in terminals that support OSC 8 hyperlinks.
The URL stays visible beside the label for other terminals and redirected output.
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
`/set run.speculative_code_mode true` does the same. The next turn uses the new
value; a turn already running keeps the tools it started with.

While it is on, every tool except `write_file` and `edit_file` becomes a
function inside one harness `CodeMode` `run_code` tool, `shell` included. Other
tools that run a program passed as a string, such as a workflow plugin, stay
native as `CodeMode` keeps them by default.

`run_code` is a persistent Python sandbox with the working directory mounted
read-write at its real path, isolated environment variables, the host clock
(through `datetime`), and no network. The model writes one snippet that calls many tools, and CLAI runs it while the model
is still writing:

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
hides the row; switching back on shows the same session totals. Headless runs
and redirected output use the same tools but show no row. Speculative execution
needs `pydantic-monty` (`pip install "pydantic-ai-harness[code-mode]"`); without
it, CLAI prints a warning and runs tools natively.

## Status line

The terminal footer shows the selected model, activity, latest reported
context tokens, and streamed output estimate, including text, thinking, and
string tool-argument deltas. The estimate is characters divided by four, not a
provider tokenizer count. On completion it is replaced by reported run output
usage. Context is the most recent response's reported input plus output tokens,
not cumulative conversation billing or a context-window percentage; `?` means
unavailable. As each request goes out, the `compaction` plugin replaces it with
that request's estimated size and paints it yellow while the history is
[over its threshold](#compacting-the-conversation); the response's reported
usage takes over when it lands. The retained-history cost (`$0.0123`) follows the
output count, updated after each turn and hidden until a response has price data.
After `/compact`, the footer keeps the previous figure until the next turn;
`/cost` and `/usage` read the retained history immediately.

The shell owns a pinned editor below a VT terminal scroll region, with Termflow
layout and completion helpers. Rich and Termflow transcript output scroll above
it without repainting the editor. The hardware cursor stays hidden during input;
a nonblinking reverse-video cell marks the editing position. Status and resize
checks run ten times per second, writing only changed rows. Terminals shorter
than six rows or narrower than six columns omit the border. Below three rows,
the draft is retained but hidden until the terminal grows. The pinned surface
requires VT scrolling-margin support. While resizing, the visible viewport goes
blank and incoming output is buffered. After 250 ms without another size change,
CLAI redraws recent transcript at the new width and restores the current draft.
It does not clear terminal scrollback or conversation history. The repaint cache
retains up to 2,000 lines and one million characters per editor, plus a bounded
partial line; it includes startup and plugin lifecycle notices and is carried
across shell reloads. Output arriving during resize
is kept separately and flushed in order, spilling to a private temporary file
for large bursts. Full-screen menus release scrolling margins and detach the keyboard reader before taking over.
Redirected output has no live editor or footer.
No model requests or telemetry are added for status reporting.

A plugin can append its own fragment to the row with `host.status_segment`, such
as the working directory or a branch name; fragments are muted and dropped when
the plugin unloads. See [PLUGINS.md](PLUGINS.md) for the registration and its
cost rules.

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

For anything beyond one capability, a plugin is a Python file with an
`activate(host)` function. A single plugin can do as much as it likes; this one
both adds a capability and reacts to a lifecycle hook, to show two shapes at
once:

```python
from pydantic_ai_harness.exa import ExaSearch

from pydantic_clai2.plugins import PluginHost, TurnEnd


def activate(host: PluginHost) -> None:
    host.add(ExaSearch(num_results=8))

    @host.on('turn_end')
    async def ping(event: TurnEnd) -> None:
        host.console.bell()
```

Drop the file in `~/.config/pydantic-clai2/plugins/`, or register anything
importable with `/plugins add NAME module[:attr] [JSON]`. It is live for the
next prompt; no restart. `/plugins` alone opens a full-screen menu to enable, disable,
reload, and remove. Closing the menu returns to the prompt without printing the
plugin list. Use `/plugins list` to print it. Plugins are trusted code running as you.

[PLUGINS.md](PLUGINS.md) has the full list of hooks, events, and rules.

## Questions from the model

When the task is ambiguous, the model can call `ask_user_question` instead of
guessing. Questions appear inline, with the conversation still visible above a
compact numbered picker. Use Up/Down and Enter, or press an option's number to
select it. For multiple selections, Enter or a number toggles a choice; select
`Done` to submit. At least one choice is required. Esc or Ctrl-C declines the
whole request, which the model is told so it can make a stated choice and carry
on. Several questions show progress in the title. The editor's draft is
preserved, and your picks are printed to the transcript afterwards.

The inline `ask_user_question` picker also offers `Other (type answer)`.
Choose it to type your own answer instead of the suggested options, including for
multi-select questions. Enter submits nonblank text. Esc returns to the choices
and keeps your draft; Ctrl-C declines the whole request. Backspace and arrow keys
edit the text. Multiline paste is inserted as text and waits for Enter; it does
not submit an answer or select choices. The conversation stays visible while you type. Custom answers
appear in the transcript and reach the model as a one-item list under the question's header.

The menu is the built-in `ask_user` plugin around the harness's
[`AskUser`](../docs/ask-user.md) capability. The capability only knows an
`Answerer`; the terminal menu is one, and [PLUGINS.md](PLUGINS.md#ask_user-questions-answered-from-the-terminal)
shows how to put a different one, a web form for instance, in its place.
`/plugins disable ask_user` removes the tool.

## Telemetry and references

The stock CLI enables the built-in `logfire` plugin by default. It adds Pydantic
AI's [`Instrumentation`](https://pydantic.dev/docs/ai/capabilities/overview/)
capability to CLAI turns for agent, model-request, and tool
spans, including timing, token usage, and failures. It adds no separate CLAI spans
and does not instrument HTTP clients or unrelated agents globally.

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

Prompts, responses, tool arguments/results, and binary image attachments are
included by default, including retained history used by later turns. This can
send source code, file contents, and screenshots to the configured telemetry
destination. Review that destination before supplying
credentials. Keep tokens out of plugin settings, which are saved as plaintext.

```text
/plugins disable logfire
/plugins enable logfire
/plugins reload logfire
/plugins add logfire pydantic_clai2.logfire '{"include_content": false, "include_binary_content": false}'
```

The last command replaces the built-in configuration. Its options are
`service_name` (default `pydantic-clai2`), `include_content` (default `true`),
`include_binary_content` (default `true`), and `send_to_logfire` (either
`"if-token-present"` or `false`). The plugin explicitly sets the latter, rather
than taking `LOGFIRE_SEND_TO_LOGFIRE` from the environment. Content flags control
Pydantic AI's prompt/result and standard binary-content capture, not all metadata;
model/tool names and tool definitions may still be recorded. Logfire's normal
scrubbing remains enabled.

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
- [Code Puppy splash](https://github.com/code-puppy/code_puppy/blob/main/code_puppy/splash.py)
- [Code Puppy streaming](https://github.com/code-puppy/code_puppy/blob/main/code_puppy/agents/event_stream_handler.py)
- [Code Puppy command registry](https://github.com/code-puppy/code_puppy/blob/main/code_puppy/command_line/command_registry.py)

See `THIRD_PARTY_NOTICES.md` for attribution.

## vllm connection

Open `/add_model`, choose `vllm`, then enter a trusted HTTP(S) server root or `/v1` URL, and optionally a token. CLAI queries `/v1/models` and opens a searchable model picker. HTTP sends tokens unencrypted; use HTTPS outside trusted local networks. The connection is saved like Codex's, see [Codex authentication](#codex-authentication).

## openrouter connection

Open `/add_model`, choose `openrouter`, then choose **Sign in with browser** or **Enter API key**. Browser sign-in opens OpenRouter's [PKCE authorization flow](https://openrouter.ai/docs/use-cases/oauth-pkce) and receives an authorization code on a temporary loopback listener. CLAI exchanges the code for a user-controlled API key over HTTPS. If the browser cannot reach CLAI (for example over SSH), paste the final callback URL or authorization code into the terminal. If no browser opens, open the printed authorization URL manually. Login times out after five minutes; Ctrl-C cancels it. You can revoke the generated key on OpenRouter.

Manual entry still accepts a key from https://openrouter.ai/keys in a masked prompt. After either method, select a model from the live catalog. CLAI validates the key with `/api/v1/key` before fetching `/api/v1/models`. Cancelling before model selection leaves the saved connection unchanged.

The connection is saved in the configured Python keyring backend after selection, or in a per-user `0600` file when no keyring backend exists, as described in [Codex authentication](#codex-authentication). Backend security depends on your keyring configuration. Tokens are not stored in SQLite or command history. The selected model persists across restarts. Select the provider again to browse its live models or reconfigure the saved connection. Discovery is explicit and has a 20-second network timeout; redirects are not followed. Agent inference uses Pydantic AI core.


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
