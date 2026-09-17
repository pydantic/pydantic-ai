# CLAI 2.0

A separately installable terminal client for Pydantic AI. The coding tools,
`Coder(unrestricted_filesystem=True)`, are the built-in `coder` plugin: on by
default, `/plugins disable coder` for a chat-only shell. The built-in
`repo_context` plugin reads `AGENTS.md` or `CLAUDE.md` from the launch directory
into the agent's instructions; `/plugins disable repo_context` turns that off.
Context management is the built-in `compaction` plugin,
[described below](#compacting-the-conversation).
Python 3.11+ is required by Termflow. Tracking issue: https://github.com/pydantic/pydantic-ai-harness/issues/875.

CLAI file tools can access paths outside the workspace, including `/tmp`, and do
not protect secret files or repository metadata. OS permissions still apply.
Relative paths use the launch workspace. Use a custom agent with `Coder()` to
retain workspace-scoped file tools. Shell output is displayed dimly.

## Interrupting a turn

Press Ctrl-C once to cancel the active agent turn and return to input. Tool cleanup
and terminal restoration finish before the next prompt. Press Ctrl-C again within
two seconds to exit, including across the transition back to input. At the prompt,
the first press clears input and the second exits. Ctrl-D and `/exit` also quit.
The interrupted prompt and captured partial responses and tool results stay in
conversation history, so you can follow up with a clarification. No interrupted
run is automatically retried. External application cancellation still propagates,
and completed tool side effects cannot be undone.

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

## Codex authentication

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

When no keyring backend exists at all (keyring raises `NoKeyringError` or
`InitError`, typical on a headless Linux box or over SSH), credentials go to a
`0600` file in `$XDG_CONFIG_HOME/pydantic-clai2/` (`~/.config/pydantic-clai2/` by
default) instead, named for the account: Codex uses `credentials-openai-codex.json`,
and the vllm and openrouter connections use their own files. Like keyring entries,
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

## Settings and commands

Preferences live in `$XDG_CONFIG_HOME/pydantic-clai2/config.db`, falling back to
`~/.config/pydantic-clai2/config.db`. Use `--database PATH` to select another database.
A repository can add its own layer with a `.clai/settings.json` file; see
[Project settings](#project-settings). Conversation messages and CLAI's
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

Interactive commands: `/login`, `/set`, `/model`, `/help`, `/new`, `/exit`, `/config`,
`/plugins`, `/usage`, `/cost`, and `/compact` from the built-in `compaction` plugin.
Tab completion suggests commands, settings, boolean values, plugin identifiers,
and paths after `@`. Path completion inserts a path; it does not attach file contents.
Unknown slash commands are not sent to the model. Up/down recall saved prompt
history. Ctrl-D exits. Ctrl-C at input clears the line; during a run it cancels
the turn and returns to input. No cancelled run is automatically retried.

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

The window comes from genai-prices, the same catalog the `/model` menu shows
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
to read it before advising on CLAI customization. The guide is bundled with the
installed package and loaded only when the tool is called, not included in every
prompt. No network access or source checkout is needed to read it.

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
API. Call `await session.prompt(text)` for each turn. Native `agent.run` drives the
loop through tools to completion. Successful turns retain `result.all_messages()`;
cancelled turns retain the prompt and messages captured by Pydantic AI, including
interrupted responses and tool results. Failed turns leave the previous history
intact. External tool side effects may already have occurred. History is in memory
only. Structured outputs are supported and displayed after completion.

CLAI is painted in the Pydantic brand palette: Lithium magenta for headings,
the banner, and the thing to look at; Calcium for list markers and errors; Aqua
for links and added diff lines; Pydantic AI cyan for guidance; purple with a
magenta-to-white shimmer for the active status line and plain purple while idle;
brand grey for tool previews and hints. On terminals without 24-bit
colour the nearest of the 16 standard colours is used. Every colour lives in
`theme.py`. This is local to CLAI; it does not change your terminal's colours
or Termflow defaults elsewhere. Code block syntax highlighting retains Termflow's
Monokai default.

Streaming matches Code Puppy's separate output and thinking paths:

- Markdown uses Termflow `SmoothWriter`: 12 ms ticks, 0.5-second catch-up,
  minimum one visible character per tick. Markdown is parsed line-by-line.
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
uses. Addition backgrounds are muted teal (`#203c3b`), deletion backgrounds are
muted burgundy (`#432d3b`), and brighter markers distinguish the changes. Code
syntax colors are unchanged. Successful file writes also show the proposed diff from their matching
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
unavailable. As each request goes out, the `compaction` plugin replaces it with
that request's estimated size and paints it yellow while the history is
[over its threshold](#compacting-the-conversation); the response's reported
usage takes over when it lands. The retained-history cost (`$0.0123`) follows the
output count, updated after each turn and hidden until a response has price data.
After `/compact`, the footer keeps the previous figure until the next turn;
`/cost` and `/usage` read the retained history immediately.

While running, the footer reserves the terminal's bottom row using ANSI scrolling
regions. Its text shimmers with a moving highlight at ten frames per second, with no spinner and a
16-colour fallback when truecolour is unavailable. Prompt-toolkit owns the footer while accepting input. The run footer is
disabled for redirected output and restores normal scrolling on cancellation or
failure. The cursor is hidden during runs and restored on completion, failure,
or cancellation. No model requests or telemetry are added for status reporting.

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

Open `/model`, choose `vllm`, then enter a trusted HTTP(S) server root or `/v1` URL, and optionally a token. CLAI queries `/v1/models` and opens a searchable model picker. HTTP sends tokens unencrypted; use HTTPS outside trusted local networks. The connection is saved like Codex's, see [Codex authentication](#codex-authentication).

## openrouter connection

Open `/model`, choose `openrouter`, then choose **Sign in with browser** or **Enter API key**. Browser sign-in opens OpenRouter's [PKCE authorization flow](https://openrouter.ai/docs/use-cases/oauth-pkce) and receives an authorization code on a temporary loopback listener. CLAI exchanges the code for a user-controlled API key over HTTPS. If the browser cannot reach CLAI (for example over SSH), paste the final callback URL or authorization code into the terminal. If no browser opens, open the printed authorization URL manually. Login times out after five minutes; Ctrl-C cancels it. You can revoke the generated key on OpenRouter.

Manual entry still accepts a key from https://openrouter.ai/keys in a masked prompt. After either method, select a model from the live catalog. CLAI validates the key with `/api/v1/key` before fetching `/api/v1/models`. Cancelling before model selection leaves the saved connection unchanged.

The connection is saved in the configured Python keyring backend after selection, or in a per-user `0600` file when no keyring backend exists, as described in [Codex authentication](#codex-authentication). Backend security depends on your keyring configuration. Tokens are not stored in SQLite or command history. The selected model persists across restarts. Select the provider again to browse its live models or reconfigure the saved connection. Discovery is explicit and has a 20-second network timeout; redirects are not followed. Agent inference uses Pydantic AI core.
