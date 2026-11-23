## This check

Review only. Do not try to run code or demonstrate a breakout.

Hold the changed code to its row in `THREAT_MODEL.md` "Harness and CLI execution boundaries", and to the backend's own security-model documentation under `docs/harness/`.

1. **Controls on every execution path.** A control that guards a call must guard every way the backend can start that call, and must run before the call starts.
2. **Work outside the sandbox.** Flag a change that runs work on the host that the backend's configuration and documentation say runs inside the sandbox.
3. **Reach beyond the configuration.** Flag a change that lets sandboxed code reach host files, credentials, processes or networks that its configuration and documented defaults do not grant.
4. **A changed guarantee.** Flag a change that weakens a guarantee the backend's documentation or `THREAT_MODEL.md` states, unless the same pull request updates that documentation.
5. **Path containment.** For a component that reads or writes under a root, flag a change that makes it open a path other than the one it checked. Flag a change to whether the root itself may be a symlink, or whether the root is resolved once or per call, unless the pull request updates that component's row in `tests/harness/test_rooted_components.py`.
