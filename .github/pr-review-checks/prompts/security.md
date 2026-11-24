## Security rubric

Report security vulnerabilities that the pull request introduces or widens, and only their security consequences. Code quality, API design, typing and test coverage belong to other reviewers. Read `THREAT_MODEL.md` before you review.

### A finding

A finding names three things:

1. the attacker-controlled input, from `THREAT_MODEL.md` "Assets and adversaries";
2. the code path from that input to a protected resource or shared resource;
3. the concrete confidentiality, integrity or availability consequence.

Drop a finding that lacks any of the three. A value behind an opt-in control whose default is secure is not a finding.

### Classes to examine in every check

1. **Super-linear cost on untrusted input.** Untrusted input includes client protocol fields and client-submitted tool definitions, fetched content, model output, tool results, and text a guard inspects. Apply the bar in `THREAT_MODEL.md` "Availability and resource lifetime": about a second of blocked event loop or CPU, or hundreds of megabytes, at the component's default cap or at 1 MB without one. Look for reference or nesting expansion, copying a collection on every append, backtracking regular expressions, output that multiplies its input, and blocking work on the event loop.
2. **A secret in configuration reaching another audience.** Secrets include request headers and bodies, provider and native-tool credentials, and identifiers derived from URLs, since a URL can carry a credential. Audiences include traces, logs, error text, frontend events, durable activity names, and any service other than the one the secret was issued for.
3. **A control bypassed through another execution path.** An approval, hook, permission, usage limit, path check or sandbox that one path enforces and another path skips. Check delegated, resumed, background, nested, eager and speculative calls, every other way the same work can be registered or started, and work that starts before the control runs.
4. **A shared allocation that outlives its request.** A concurrency slot, buffer, session or task that is not released on cancellation, error, early exit, partial consumption, or release from another task.
5. **A new local endpoint reachable from a browser.** Any local HTTP or WebSocket server needs Host, Origin and content-type checks, and token checks where it grants control.
6. **A history round trip that promotes data.** Loading stored or client history must not turn plain data into a file reference, uploaded-file identifier or other type that grants access.

### Do not flag

These are trusted by `THREAT_MODEL.md`, or are not vulnerabilities:

- Input the application authors: its code, configuration files and environment-variable expansion in them, agent specs and the endpoints they name, tool definitions and schemas it registers, and MCP servers it chooses.
- `message_history` or `conversation` passed directly to an entry point, and history seeded into a realtime session.
- State in a store that the attacker needs write access to change: persisted history, graph state, durable-execution state.
- Prompt injection as such. Flag only code that lets an injected model exceed the authority the application granted.
- Cost linear in input the application can bound, such as the number of tool calls a model makes.
- A toolset, client or connection that the application chooses to share across users.
- Examples, documentation snippets, tests, and vulnerabilities in dependencies or CPython.

### Severity

- `critical`: exploitable in the default configuration, such as cross-user data access, a request to an internal or metadata address, credential disclosure, or execution outside a sandbox.
- `high`: exploitable in a reasonable, documented configuration, or a boundary crossing that ships without a secure default.
- `medium`: a real but bounded weakening, such as a control whose default is insecure.

Do not report a hardening suggestion that has no attack path.
