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

<!-- From #10333 -->

```diff
@@ -2515,2 +2515,2 @@
-its settings menu. Each option below is a row there; each edit saves at once and
-applies from the next run. The last command replaces the built-in configuration
+its settings menu; the web UI's plugins panel shows the same menu. Each option below
+is a row there; each edit saves at once and applies from the next run. The last command replaces the built-in configuration
```

```diff
@@ -2551,2 +2551,3 @@
-Choose **Logfire project** in the settings menu (`/plugins configure observability`, or
-`c` on `observability` in `/plugins`):
+Choose **Logfire project** in the settings menu (`/plugins configure observability`,
+`c` on `observability` in `/plugins`, or **Change…** on the row in the web UI's plugins
+panel, where each step below is a dialog in the page):
```

```diff
@@ -2559,2 +2560,4 @@
-2. Sign in, or sign up, in the browser. CLAI prints the link too, so it works over SSH.
-3. Pick one of the projects you can write to (`/` to search, Enter to use it).
+2. Sign in, or sign up, in the browser. A waiting screen shows the link too, so it
+   works over SSH; Esc there cancels the sign-in and changes nothing.
+3. Pick one of the projects you can write to (typing filters a list of more than
+   eight; Enter uses it).
```

<!-- From #10334 -->

```diff
@@ -1789 +1789,3 @@
-**Agent folders**. The list saves each change immediately; no JSON editing is
+**Agent folders**, or with its **Change…** button in the
+[web UI](#web-ui)'s plugins panel, where each step below opens as a dialog in
+the page. The list saves each change immediately; no JSON editing is
```

```diff
@@ -1792 +1794 @@
-- **Add directory path** (`a`) accepts one local directory. An unprefixed
+- **Add directory path** accepts one local directory. An unprefixed
```

```diff
@@ -1800,8 +1802,10 @@
-- **Browse local directories** (`b`) includes hidden folders. Type to filter,
-  Enter to open a folder, Left to go to its parent, and **Use this directory**
-  to select it. The choice is saved as an absolute path, used in every project;
-  type a relative path with **Add directory path** to follow each project
-  instead. Ctrl+L opens the full current path for viewing or navigation.
-- Enter on an entry opens its actions. `e` edits its current value; `d` removes
-  it from the search without deleting files. Esc cancels an edit or returns to
-  the previous menu; **Save & close** returns with changes already saved.
+- **Browse local directories** includes hidden folders. Pick a folder to open
+  it, **..** to go to its parent, and **Use this directory** to select it; in
+  the terminal, type to filter a long list. The choice is saved as an absolute
+  path, used in every project; type a relative path with **Add directory path**
+  to follow each project instead. **Go to path** shows the full current path
+  for viewing or navigation.
+- Choosing an entry opens its actions: edit its current value, replace it by
+  browsing, or remove it from the search without deleting files. Esc cancels an
+  edit or returns to the previous menu, and an empty edit changes nothing;
+  **Save & close** returns with changes already saved.
```

<!-- From #10357 -->
Plugins can add a panel on the right, toggled by its button at the top right or Cmd/Ctrl+Shift+L,
resizable by its left edge, and remembered by the browser. The `observability` plugin's shows
Logfire's live view of the current conversation, opened at the latest turn. Logfire does not let
other sites show its pages inside theirs today, so the panel opens it in a window beside CLAI's
instead, and points that window at each new turn. Without a Logfire project the panel says how to
set one up.
