# StackOne

Use `StackOne` when an agent needs to work with one of a user's linked business applications, such as BambooHR,
Salesforce, or Zendesk. Each instance is scoped to one linked account, which is one authenticated connection between
[StackOne](https://www.stackone.com) and a provider.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/stackone/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Before you start

Follow the [StackOne docs](https://docs.stackone.com) to:

1. Configure a connector and link an account. For your first test, enable only the read actions you need.
2. Copy the linked account ID from the StackOne dashboard.
3. Create a StackOne API key that can execute actions.

You also need an API key for the model your agent uses.

## Installation

```bash
uv add "pydantic-ai-harness[stackone]" "pydantic-ai-slim[openai,spec]"
```

The `openai` and `spec` extras support the model and agent-spec examples below. Install the provider extra for a
different model provider instead.

Set the credentials in the shell where you will run the example:

```bash
export STACKONE_API_KEY='your-stackone-api-key'
export STACKONE_ACCOUNT_ID='your-linked-account-id'
export OPENAI_API_KEY='your-openai-api-key'
```

`StackOne` reads `STACKONE_API_KEY` automatically. The example reads `STACKONE_ACCOUNT_ID` explicitly so the account
ID is not hard-coded. You can pass `api_key=` directly instead, but keep secrets out of source control.

## Run your first agent

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness import StackOne

agent = Agent(
    'openai:gpt-5',
    capabilities=[
        StackOne(account_id=os.environ['STACKONE_ACCOUNT_ID']),
    ],
)
result = agent.run_sync('List the first 5 employees')
print(result.output)
```

By default, the model receives two tools. It searches for an action that matches the request, then executes the
returned action ID. The final output depends on the linked provider and its data.

## Control available actions

StackOne controls which actions are enabled for the linked account. Treat that configuration as the primary access
control.

Use `actions` when you also want to limit which tools the model sees. Patterns use Python
[`fnmatch`](https://docs.python.org/3/library/fnmatch.html) syntax, where `*` is a wildcard. They ignore case and match
the full `{connector}_{action}_{entity}` tool name:

```python
from pydantic_ai_harness import StackOne

StackOne(account_id='your-linked-account-id', actions=['*_list_*'])            # All matching list tools
StackOne(account_id='your-linked-account-id', actions=['workday_get_worker'])  # One exact tool
```

Passing `actions` selects `individual` mode automatically. Explicitly combining `actions` with
`tool_mode='search_execute'` raises an error because that mode registers only the search and execute tools.

## Choose a tool mode

| Mode | What the model receives | Use it when |
|---|---|---|
| `search_execute` | Two tools: search for an action, then execute it by ID | The account has many enabled actions. This is the default when `actions` is omitted. |
| `individual` | One tool and schema per enabled action | You need to select exact actions or add per-tool behavior. Passing `actions` selects this mode. |

In `search_execute` mode, action IDs are returned by the search tool at runtime and should not be guessed. In
`individual` mode, all selected tool schemas are sent to the model, so filter large action sets with `actions`.

To keep StackOne tools out of the model context until they are needed, pass `defer_loading=True`. The capability uses
`id='stackone'` by default so it can be loaded on demand. Give each instance a distinct `id` when one agent uses
multiple StackOne accounts:

```python
from pydantic_ai_harness import StackOne

StackOne(account_id='your-linked-account-id', defer_loading=True)
```

### Bound large tool results

Provider actions can return large exports. Combine StackOne with the
[Tool Output Limits](../tool_output_limits/README.md) capability to reduce oversized tool returns agent-wide:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import StackOne, ToolOutputLimits

agent = Agent(
    'openai:gpt-5',
    capabilities=[
        StackOne(account_id='your-linked-account-id'),
        ToolOutputLimits(),
    ],
)
```

### Require approval

Approval is not enabled automatically. For operations that need human confirmation, use the public
`StackOneToolset` with Pydantic AI's
[tool approval](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/#requiring-tool-approval):

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.stackone import StackOneToolset

stackone_tools = StackOneToolset(
    account_id=os.environ['STACKONE_ACCOUNT_ID'],
    actions=['workday_create_worker'],
).approval_required()

agent = Agent('openai:gpt-5', toolsets=[stackone_tools])
```

Handle the resulting deferred approval requests as described in the linked guide.

## Define the agent in YAML or JSON

The capability also works with Pydantic AI's
[agent spec](https://pydantic.dev/docs/ai/core-concepts/agent-spec/) format for YAML or JSON. Keep the API key in
`STACKONE_API_KEY` rather than storing it in the file:

```yaml
# agent.yaml
model: openai:gpt-5
capabilities:
  - StackOne:
      account_id: 'your-linked-account-id'
      actions: ['*_list_*']
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import StackOne

agent = Agent.from_file('agent.yaml', custom_capability_types=[StackOne])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate `StackOne`.

Use the lower-level `StackOneToolset` directly when you need
[`Agent(toolsets=[...])`](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/) or other toolset wrappers.

Custom `base_url` and URL-valued `client` values must use HTTPS. The toolset adds auth headers and appends the
`tool-mode` query parameter for URL values when it is absent. It raises an error when the URL's `tool-mode` conflicts
with the configured mode because rewriting would invalidate signed URLs. When using `search_execute` with a signed
URL, include `tool-mode=search_execute` before signing. Prebuilt clients are used as-is; configure their HTTPS
transport, auth, account selection, and tool mode yourself.
