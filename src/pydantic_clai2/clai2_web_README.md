# CLAI web UI README content to fold into the README later

<!-- From #10262 -->
## Web UI

Run `/web` in a session to open it in your browser: a local web app with a sessions sidebar,
streaming Markdown answers, collapsible tool calls (shell output, diffs, searches), thinking
blocks, live token counts while a turn streams, a panel with code mode's `run_code` REPL cells, and a composer with `/` command completion, a
model picker, and a stop button. The
terminal and the browser drive the same session: a prompt typed in either runs once, and both
show it. The page follows `/theme`, and the sidebar's footer picks a theme too. `/web stop`
ends the web UI; `/web status` shows its address.

To use only the browser, start CLAI with `--web` (also `uvx pydantic-clai2 --web`). It prints
the address, opens it, and logs the conversation to the terminal until Ctrl-C.

The sidebar lists the sessions `/resume` shows, grouped by project, with search. Click one to
switch to it, or use its menu to rename, fork, or delete it; a session saved in another
directory opens read-only. Keyboard: Enter sends, Shift+Enter adds a line, Esc interrupts,
Cmd/Ctrl+K searches sessions, Cmd/Ctrl+Shift+O starts a new one, Cmd/Ctrl+B hides the sidebar.

The browser can make the agent run commands, so the server listens on 127.0.0.1 only and
every request must carry a token that is new each launch; the printed link contains it. Keep
the link private. The `web` plugin starts disabled; `/web` works once it is enabled with
`/plugins enable web`, and `--web` loads it for that launch only. See
[the plugin guide](PLUGINS.md#web-this-session-in-your-browser) for settings and limits.

<!-- From #10282 -->
`/repl CODE` runs your own cell in that sandbox between turns, with the same tool
functions the model's cells get: `/repl x = 41`, then `/repl print(x)`. It prints
what the cell printed and returned, keeps the state with the session, and stays out
of the conversation; your next prompt tells the model what you ran, since the
variables are in its sandbox now. A cell that fails is reported, not retried. In the
web UI, the REPL panel's input runs `/repl` too.

<!-- From #10289 -->
The Plugins panel (the puzzle button in the sidebar's footer, Cmd/Ctrl+, or `/plugins` in the
composer) turns plugins on and off and edits their settings, saved and applied at once, as
`/plugins` does in the terminal. A plugin that declares its settings menu shows it there, sign-ins
and `/keys` entries included, as dialogs in the page; any other plugin's own menu opens in the
terminal.

<!-- From #10319 -->

```diff
@@ -2384 +2384,2 @@
-default, and opens its settings menu (`/plugins configure linear` reopens it). The
+default, and opens its settings menu (`/plugins configure linear` reopens it, and
+the web UI's plugins panel shows it too). The
```

<!-- From #10328 -->

```diff
@@ -24 +24,2 @@
-`/logfire login`). Reopen it with `/plugins configure logfire_mcp`; see
+`/logfire login`). Reopen it with `/plugins configure logfire_mcp`, or in the web UI's
+plugins panel; see
```

<!-- From #10330 -->

```diff
@@ -17,2 +17,3 @@
-token kept in `/keys`. `/google_workspace` opens its settings menu: the `/keys`
-entry to use (`GOOGLE_ACCESS_TOKEN` by default), products, and read-only tools; see
+token kept in `/keys`. `/google_workspace` opens its settings menu (the web UI's
+plugins panel shows it too): the `/keys` entry to use (`GOOGLE_ACCESS_TOKEN` by
+default), products, and read-only tools; see
```
