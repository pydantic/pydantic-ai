# Skills

Use [Agent Skills](https://agentskills.io/specification) to give an agent
specialized instructions without putting every instruction in its initial
prompt.

Point `Skills` at one or more skill libraries. The model first sees each
skill's name and description. When a skill is useful, the model loads it with
the `load_capability` tool to receive that skill's instructions.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/skills/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Installation

Install the `skills` extra for YAML frontmatter support:

uv:

```bash
uv add "pydantic-ai-harness[skills]"
```

pip:

```bash
pip install "pydantic-ai-harness[skills]"
```

## Quick start

Create a skill library:

```text
.agents/skills/
  code-review/
    SKILL.md
```

Add the skill's description and instructions:

```markdown
---
name: code-review
description: Review a change for correctness and repository conventions.
---

Inspect the change and report findings by severity.
```

Then add the library to your agent, with a
[workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/) to read it from:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import Skills

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[LocalWorkspace('.'), Skills('.agents/skills')],
)
```

`Skills` does not search `.agents`, `.claude`, or your home directory
automatically. Pass each library you want it to load; to layer several the way
coding agents do, see [Layer project and personal skills](#layer-project-and-personal-skills).

> **Note:** `Skills` loads instructions from `SKILL.md` and tells the model
> where the skill's directory is. It does not load bundled resources or run
> scripts itself.

## How it works

At the start of every run, `Skills`:

1. Reads each configured library from the run's workspace. Relative paths
   resolve from its working directory.
2. Scans the immediate child directories and validates the selected `SKILL.md` files.
3. Offers each selected skill as a
   [deferred capability](https://pydantic.dev/docs/ai/capabilities/on-demand/)
   named after it: the model sees its name and description, and loads it with the
   `load_capability` tool.

Loading a skill returns a `# Skill: <name>` heading, a line naming the skill's
directory in the workspace, and the skill's Markdown body. The catalog is the
same on every run over the same files, so it stays in the cached prefix; a new
or renamed skill appears in it on the next run. Without any skills, `Skills`
adds no instructions and no `load_capability` tool.

A run without a workspace fails at its start. To read skills from somewhere
else, such as skills shipped with your application while the agent works in a
sandbox, pass a workspace backend:

```python
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import Skills

skills = Skills('skills', workspace=LocalWorkspaceBackend('/app'))
```

`workspace=` takes a backend, not the `LocalWorkspace` capability. Under a
durable engine such as Temporal, a `workspace=` backend is read in-process, not
through the workflow.

## Choose which skills to expose

By default, all discovered skills are included. Use `include` or `exclude` to
change the catalog for a particular agent:

```python {test="skip"}
from pydantic_ai_harness import Skills

review_skills = Skills(
    '.agents/skills',
    include=['code-review'],
)

release_skills = Skills(
    '.agents/skills',
    exclude=['code-review'],
)
```

| Configuration | Skills in the catalog |
|---|---|
| Neither option | All discovered skills |
| `include=['a', 'b']` | Only `a` and `b` |
| `include=[]` | No skills |
| `exclude=['a', 'b']` | All except `a` and `b` |
| `exclude=[]` | All discovered skills |

`include` and `exclude` cannot be used together. The constructor overloads catch
this in typed code, and runtime validation covers agent specs and untyped
callers. Unknown names fail at run start.

Selection happens before frontmatter is parsed. An unselected skill does not add
instructions or frontmatter validation errors to that `Skills` instance.

These options control catalog exposure. They are not filesystem permissions or
an access-control boundary.

Directory paths choose where discovery starts in the workspace; they do not
create a containment boundary, and normal symlink resolution applies.

A selected `SKILL.md` body becomes model instructions. Load libraries only from
sources you trust, and review repository-provided skills before exposing them.

## Skill format

Each immediate child directory containing `SKILL.md` is a skill:

```text
.agents/skills/
  code-review/
    SKILL.md
  release-notes/
    SKILL.md
```

The loader uses these parts of `SKILL.md`:

| Part | Requirement |
|---|---|
| `name` | Optional. Defaults to the parent directory name. If provided, it must match the directory after Unicode normalization. |
| `description` | Required and non-blank. The Agent Skills limit is 1,024 characters; longer descriptions load with a warning. This appears in the initial catalog. |
| Markdown body | Optional. This is loaded under a generated `# Skill: <name>` heading. |

Skill names and `include` or `exclude` values are normalized with Unicode NFKC
before matching. A normalized name can contain at most 64 lowercase Unicode
letters or numbers, separated by single hyphens. It cannot start or end with a
hyphen.

Only immediate children are discovered. For example,
`code-review/references/SKILL.md` does not create another skill. Ordinary files
and child directories without `SKILL.md` are ignored.

You can pass several libraries:

```python {test="skip"}
from pydantic_ai_harness import Skills

skills = Skills([
    '.agents/skills',
    'company/skills',
])
```

By default, selected skill names must be unique across those libraries.
Repeated references to the same resolved library are scanned once, and a skill
found twice, through a symlinked library or skill directory or as a
byte-identical copy of its `SKILL.md`, counts once.

## Layer project and personal skills

Coding agents read skills from conventional locations that a project may not
have, and let a project's skill take precedence over a personal one with the
same name. Two options give `Skills` the same behavior:

```python {test="skip"}
from pydantic_ai_harness import Skills

skills = Skills(
    ['.agents/skills', '.claude/skills', '/home/me/.agents/skills', '/home/me/.claude/skills'],
    missing_directories='skip',
    duplicate_names='keep_first',
)
```

- `missing_directories='skip'` leaves out a library directory that does not
  exist, instead of failing the run. A path that exists but is not a directory
  still fails.
- `duplicate_names='keep_first'` keeps the skill from the earlier directory when
  two valid skills with different `SKILL.md` files share a name, and skips the
  other with a `UserWarning`. An invalid `SKILL.md` is skipped first, so it does
  not hide a valid skill with its name.

Directories are listed in precedence order. Paths are workspace paths, and `~`
is not expanded, because the workspace may be a sandbox with a home directory of
its own; spell out the absolute path for a library on this machine.

If `.agents/skills` is a symlink to `.claude/skills`, or holds byte-identical
copies of its `SKILL.md` files, as in many repositories, each skill is listed
once. Only `SKILL.md` is compared: the first directory's bundled files are the
ones the model is pointed at.

## Bundled files

Agent Skill packages can contain directories such as `references/`, `assets/`,
and `scripts/`. `Skills` does not enumerate, read, or execute those files.

The loaded instructions name the skill's directory in the run's workspace, so a
model with file or shell tools, such as those from `FileSystem` or `Shell`, can
follow a relative reference like `references/guide.md` or run
`scripts/check.py`. Skills read from a `workspace=` backend are not where those
tools work, so their instructions leave the directory out. Placeholders such as
`${CLAUDE_SKILL_DIR}` remain unchanged in the body.

`Skills` reads the libraries itself: the model does not need `FileSystem` or
`Shell` to load a skill, and adding either does not change which files `Skills`
reads.

## Compatibility with existing skill libraries

The portable `name`, `description`, and Markdown instructions are supported.
`name` may be omitted and derived from the directory.

The following behavioral fields are accepted for compatibility, but their
behavior is not implemented:

```text
agent, allowed-tools, argument-hint, arguments, context, dependencies,
disable-model-invocation, disallowed-tools, effort, hooks, model, paths, shell,
tools, user-invocable, when_to_use
```

If a selected skill uses any of these fields, the run emits one aggregated
`UserWarning` at its start. Fields such as `license`, `compatibility`, and `metadata` are
accepted without changing runtime behavior. Other unknown, non-behavioral fields
are also accepted.

## Invoke a skill yourself

The model decides when to load a skill. A host that also lets a person invoke a
skill, such as a `/code-review src/app.py` command in a terminal client, reads
the same catalog with `load` and renders the skill as a prompt:

```python {test="skip"}
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import Skills

skills = Skills('.agents/skills', missing_directories='skip')
catalog = await skills.load(LocalWorkspaceBackend('.'))
review = next(skill for skill in catalog.skills if skill.name == 'code-review')
result = await agent.run(review.render('src/app.py'))
```

`load` reads the libraries as a run would, without emitting warnings: the
catalog's `skipped` messages name each `SKILL.md` left out, malformed or named
like a skill found earlier, for the host to show. Each `SkillDefinition` has
the skill's `name`, `description`, `body`, `path`, and `directory`.

`render(arguments)` returns what loading the skill returns, with every
`$ARGUMENTS` in the body replaced by `arguments`. A body without `$ARGUMENTS`
gets `ARGUMENTS: <arguments>` appended. Other placeholders, such as Claude
Code's indexed `$0` or `$ARGUMENTS[0]`, are left unchanged. `render()` without
arguments returns exactly what `load_capability` does, including leaving the
directory out for a skill read from `workspace=` (`in_run_workspace` is
`False`).

## Use an agent spec

`Skills` works with Pydantic AI's
[YAML and JSON agent specs](https://pydantic.dev/docs/ai/core-concepts/agent-spec/):

```yaml
model: anthropic:claude-opus-5-5
capabilities:
  - Skills:
      directories: .agents/skills
      include:
        - code-review
        - release-notes
```

Register `Skills` when loading the spec:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import Skills

agent = Agent.from_file('agent.yaml', custom_capability_types=[Skills])
```

The `skills` extra installs PyYAML. It parses `SKILL.md` frontmatter and YAML
agent specs.

## Define capabilities in Python

Use Pydantic AI's core `Capability` when your instructions or tools are defined
in Python instead of a `SKILL.md` package:

```python
from pydantic_ai.capabilities import Capability

refunds = Capability(
    id='refunds',
    description='Use for refund policy questions.',
    instructions='Check the refund policy before answering.',
    defer_loading=True,
)
```

`Skills` loads portable Agent Skill packages. It does not replace the core API
for code-defined capabilities.

## Configuration

```python {test="skip"}
Skills(
    directories: str | Path | Sequence[str | Path],
    *,
    include: Collection[str] | None = None,
    exclude: Collection[str] | None = None,
    missing_directories: Literal['error', 'skip'] = 'error',
    duplicate_names: Literal['error', 'keep_first'] = 'error',
    workspace: WorkspaceBackend | None = None,
)
```

- `directories` accepts one library path or a sequence of paths.
- `include` exposes only the named skills.
- `exclude` omits the named skills from the catalog.
- `missing_directories='skip'` leaves out library directories that do not exist.
- `duplicate_names='keep_first'` keeps the first of two skills with one name, with a warning.
- `workspace` reads the libraries from this backend instead of the run's workspace.

Pass at least one library directory, not the path of an individual skill
package. Malformed frontmatter, invalid UTF-8, and invalid or mismatched names warn and skip that skill. Unknown selections fail at run start, and so do duplicate selected names, missing libraries, and non-directory library paths unless `duplicate_names` or `missing_directories` say otherwise.

Two `Skills` on one agent combine into one catalog.

## Further reading

- [Agent Skills specification](https://agentskills.io/specification)
- [Adding skills support to an agent](https://agentskills.io/client-implementation/adding-skills-support)
- [Pydantic AI workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/)
- [Pydantic AI capabilities overview](https://pydantic.dev/docs/ai/capabilities/overview/)
