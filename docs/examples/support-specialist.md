Customer-support agent that uses [capabilities on demand](../capabilities.md#deferred-capability-loading) as Python-native runbook skills. The agent can carry many specialist workflows without loading every policy, tool schema, and sensitive procedure into every request.

Demonstrates:

- [capabilities on demand](../capabilities.md#deferred-capability-loading)
- [capabilities](../capabilities.md) bundling instructions, [toolsets](../toolsets.md), [model settings](../agent.md#model-run-settings), [hooks](../hooks.md), and [approval handling](../deferred-tools.md#human-in-the-loop-tool-approval)
- [agent dependencies](../dependencies.md)

The agent advertises runbooks like `orders`, `returns`, and `account-security` by `id` and `description` only. The model sees that compact catalog on its first request and calls `load_capability(id)` when the conversation needs a runbook. The matching runbook's instructions and tools appear on the next model request; runbooks not loaded for that conversation stay out of the context window.

This follows the same progressive-disclosure shape as agent skills: small metadata is always visible, detailed instructions load only when relevant. Pydantic AI capabilities go further because the loaded unit can also bring typed Python tools, per-run model settings, lifecycle hooks, and deferred-tool approval behavior.

The `account-security` runbook shows why this matters. A normal order-status question should not spend context on account-takeover procedures, session-revocation tools, high-reasoning settings, or security workflow guardrails. If the user reports a suspicious login, the model can load that runbook and get all of those pieces together. The runbook's hook keeps `revoke_all_sessions` hidden until the model has first inspected recent logins, then still requires approval before executing it.

Contrast with the [bank support example](./bank-support.md), where a single toolset is always loaded: this example shows the same support-agent pattern scaled to many specialist modes without bloating every request.

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.support_specialist
```

With a `LOGFIRE_TOKEN` set you can watch the model call `load_capability` on each turn in the [Logfire](../logfire.md) dashboard; without one the example still runs and prints each answer.

## Example Code

```snippet {path="/examples/pydantic_ai_examples/support_specialist.py"}```
