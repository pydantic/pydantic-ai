### [2] CONFIRMED — failed startup leaked acquired media

After `getUserMedia()` succeeded, a rejected `connect()` left `mediaStream`, `video.srcObject`, and the live tracks intact because the caller only changed the status. `start()` now calls `stop()` before rethrowing any startup error, covering both Start and Apply/restart. No JavaScript test harness exists for this example, so no regression test was added.

### [4] CONFIRMED — concurrent startup could acquire two sessions

`running` remained false across both awaited startup operations, so a second click could enter `start()` and overwrite the first stream and socket. `start()` now uses a synchronous `starting` guard and disables the Start button until startup settles. No JavaScript test harness exists for this example, so no regression test was added.

### [6] CONFIRMED — stale socket callbacks could affect the replacement session

Each callback read the mutable global `sock`; after Apply/restart, an old socket's message-level error could call `stop()` and close the new socket, while old close/error callbacks could overwrite its state. `connect()` now captures its `WebSocket` and every callback returns unless that instance is still current. No JavaScript test harness exists for this example, so no regression test was added.

### [3] CONFIRMED — disconnected submissions appeared to be sent

With no open socket, `send()` did nothing but the submit handler still displayed the text and cleared the input. `send()` now reports success, and the handler preserves the input and shows `not connected` unless the frame was sent. No JavaScript test harness exists for this example, so no regression test was added.

### [8] REJECTED — escaped surrogates do not fail the raw-frame size check

A valid text frame containing JSON `"\\ud800"` reaches `_forward_browser_message()` as ASCII characters, so `text.encode()` succeeds. `json.loads()` creates the lone surrogate only later inside `_dispatch_text()`; the claimed pre-dispatch `UnicodeEncodeError` cannot occur from this input. An actual unescaped surrogate cannot be carried by a valid UTF-8 WebSocket text frame. No code or test was changed.

Targeted verification: `node --check` passed for the extracted script; `git diff --check` and Ruff passed. `uv run pytest tests/test_realtime_examples.py -q --inline-snapshot=disable` collected four tests, all skipped because optional example extras are not installed. Pyright was attempted on both assigned Python files and failed because `fastapi` is not installed, producing 54 downstream missing-import/unknown-type errors; neither Python file changed.
