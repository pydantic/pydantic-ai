# Working Directory

Tell the model the workspace's working directory, so it uses relative paths instead of guessing absolute ones. [FileSystem](../file_system/) and [Shell](../shell/) add it themselves; add it yourself when your agent's own tools work in the run's workspace.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/working_directory/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Usage

uv:

```bash
uv add pydantic-ai-harness
```

pip:

```bash
pip install pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import WorkingDirectory

agent = Agent('anthropic:claude-opus-5-5', capabilities=[LocalWorkspace('.'), WorkingDirectory()])
```

The model's instructions then include a line such as ``The working directory is `/home/me/project`.``

## Starting a sandbox

The line asks the workspace for its directory with `working_dir()`. A sandbox that isn't running yet is started to answer, before the first request, even if the model never calls a tool. Without a workspace, or with one that is gone, the line is left out, and the first tool call reports why.

## Composition

`WorkingDirectory` has a fixed `id`, so any number of them on one agent, including the ones `FileSystem` and `Shell` bring, merge into one line. Its instructions are not deferred with a capability that is: a `Shell(defer_loading=True)` still names the directory before its tools load.
