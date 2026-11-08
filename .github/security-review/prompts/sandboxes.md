## This check

Review only. Do not try to run code or demonstrate a breakout.

Hold the changed code to its row in `THREAT_MODEL.md` "Harness and CLI execution boundaries", and to the backend's own security-model documentation under `docs/harness/`.

1. **Controls on every execution path.** An approval, hook, permission, usage limit or path check must cover nested, eager and speculative Code Mode calls, and must run before the call starts.
2. **Work that runs outside the sandbox.** Flag a new step that runs on the host before, after or around a sandboxed command: a login, a startup file, environment setup, cleanup, or a file the host later executes.
3. **Mounts, environment and network.** Flag a mount, environment variable, credential or network permission that widens what the sandbox can reach beyond its configuration and documented defaults.
4. **A changed guarantee.** Flag a change that weakens a guarantee the backend's documentation or `THREAT_MODEL.md` states, unless the same pull request updates that documentation.
