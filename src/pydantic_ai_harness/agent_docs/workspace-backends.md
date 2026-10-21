# Workspace Backends

Read this before adding or changing a sandbox provider (`*_sandbox/`, `ssh_workspace/`). Each rule says
why it exists, so a backend can tell a rule it must follow from a choice it may make differently.

## Start From A Sibling

Copy the backend whose SDK works most like yours, then change only what your SDK forces:

| SDK shape | Sibling |
| --- | --- |
| Local VM, SDK calls that keep running when their caller is cancelled | `boxlite_sandbox`, `smol_sandbox` |
| Remote sandbox, output and exit status as separate streams | `modal_sandbox` |
| Remote sandbox, the SDK collects the output itself | `e2b_sandbox` |
| Remote sandbox with no output stream: output written to files in the sandbox, exit status polled | `daytona_sandbox` |

- `get_sandbox()` does the acquisition itself, as in BoxLite, E2B and Modal. Smol needs `_vm()` because its
  `get_sandbox()` returns a wrapper around the machine the rest of the backend uses; Sprites' `_get_sandbox`
  takes an argument `get_sandbox()` can't. Add such a method only for a reason like these.
- When the backend chooses the sandbox's name (BoxLite, Smol), set `ref` before creation starts, so cleanup
  has it whatever ends the run. When the provider assigns the id (Daytona, E2B, Modal, Sprites), set it as
  soon as the id comes back.
- Any provider-specific code needs a raw-SDK probe that shows why it's needed. Record what the probe
  showed, with the SDK version and date, in the module docstring, as Smol and BoxLite do. Without that,
  the next change can't tell whether the code is still needed.

## Shared Pieces

`pydantic_ai_harness/_workspace_provider.py` holds what every backend needs. Use it, and change it
there, rather than redefining any of it in a backend: each copy drifts on its own.

| Piece | Use |
| --- | --- |
| `command_argv`, `check_env`, `check_timeout`, `absolute_path`, `check_working_dir`, `check_integer` | Argument checks. Smol and BoxLite use all of them; Daytona uses `check_timeout`; E2B, Modal and Sprites still check timeout and env inline, with slightly different errors |
| `running_on_asyncio` | Refusing Trio for an SDK that needs asyncio |
| `stop_shielded` | Stopping a command from a cancelled caller |
| `run_to_completion` | Finishing a provider call whose outcome must be recorded, such as a created sandbox, even if the caller is cancelled |
| `command_deadline` | Deadline, stop and timeout error for a backend that stops reading output when the deadline passes (Daytona, Modal) |
| `MAX_OUTPUT_BYTES`, `OUTPUT_PREVIEW`, `OUTPUT_DRAIN_GRACE` | The output limits below |
| `CommandOutput` | Collecting text output under the cap |
| `Lifecycles`, `settled` | Ordering creations, attachments and deletions whose SDK calls a cancelled caller can't stop |

A backend that keeps reading output after the command exits, under the same deadline (core's local
backend, BoxLite), keeps its own deadline scope. `command_deadline` reports a timeout whenever its
deadline passes, so it would report one for a command that exited while its output was still being read.

## Command Output

1. **Cap.** Stop reading at `MAX_OUTPUT_BYTES` across both streams, stop the command, and raise
   `WorkspaceOutputLimitError` with the first `OUTPUT_PREVIEW` of each stream. The output is held in
   memory until the command ends.
2. **Timeout.** Stop the command and raise `WorkspaceTimeoutError` with all the output collected so far.
   Modal reports only the last `_PARTIAL_OUTPUT_LIMIT` characters, because a cancelled Modal read loses
   what it collected. Daytona reports the last 4 KiB of each output file, cut in the sandbox before it is
   downloaded.
3. **Exit, not end of output.** The protocol asks `run` to return once the command exits, even if a
   background job it started still holds the output open. If the SDK reports the exit apart from the
   output, keep reading for at most `OUTPUT_DRAIN_GRACE` seconds after the exit and then return, as
   BoxLite's `_Output.wait` does. Modal ends its output streams when the command exits, so its longer
   drain is never waited on; Smol reads until its SDK's event stream ends. If the SDK can't report the exit
   apart from the output (E2B, Sprites, SSH), show it with a probe and override
   `can_detect_exit_with_inherited_output_pipes` in the conformance tests with the reason.
4. **Stop on every failure.** Timeout, cancellation, the cap and SDK errors all stop the command. A stop
   that needs a round trip goes through `stop_shielded`, so a cancelled caller still makes it (BoxLite,
   Daytona, E2B, Modal); Smol's `stream.close()` and Sprites' `_close_command` stop it without one. A command whose start
   finishes after its caller was cancelled is stopped too (BoxLite's `_start`, Smol's `_open`).
5. **One error, not an `ExceptionGroup`.** Reader tasks record their failure and cancel their task group;
   the caller raises it after the group ends. AnyIO wraps an error raised inside a group in an
   `ExceptionGroup`. Keep the recorded state on an object (BoxLite's `_Output`), not in variables set
   with `nonlocal` as Modal's `_collect_output` still does: pyright narrows those to their first value
   and silently stops checking the code that reads them.
6. **Classify only after a failure.** Check whether the sandbox is gone only once something has failed, so
   successful commands skip the round trip. If that check itself fails, raise the original error.

## Tests

- Run `WorkspaceBackendSuite` against the fake SDK and, behind the provider's live marker, a real sandbox.
- Test the real limits and previews, as the siblings do, not patched-down copies of them.
- Wait for a deterministic signal from the fake, such as a recorded spawn, rather than for a delay.
