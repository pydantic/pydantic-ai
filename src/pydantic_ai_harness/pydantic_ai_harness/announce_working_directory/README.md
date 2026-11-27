# Announce Working Directory

Tell the model the workspace's working directory, so it uses relative paths instead of guessing absolute ones. [FileSystem](../file_system/) and [Shell](../shell/) add it themselves; add it yourself when your agent's own tools work in the run's workspace.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/announce_working_directory/)

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
from pydantic_ai_harness import AnnounceWorkingDirectory

agent = Agent('anthropic:claude-opus-5-5', capabilities=[LocalWorkspace('.'), AnnounceWorkingDirectory()])
```

The model's instructions then include the line: The working directory is `/home/me/project`.

## Starting a sandbox

The line asks the workspace for its directory with `working_dir()`. A sandbox that isn't running yet is started to answer, before the first request, even if the model never calls a tool. Without a workspace, or with one that is gone, the line is left out, and the first tool call reports why.

## Composition

`AnnounceWorkingDirectory` has a fixed `id`, so any number of them on one agent, including the ones `FileSystem` and `Shell` bring, merge into one line. A `FileSystem` or `Shell` with `defer_loading=True` doesn't bring it, so a sandbox isn't started for tools the model may never load; add `AnnounceWorkingDirectory()` yourself to name the directory anyway.
