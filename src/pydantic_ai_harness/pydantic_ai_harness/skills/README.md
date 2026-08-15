# Skills

Use [Agent Skills](https://agentskills.io/specification) to give an agent
specialized instructions without putting every instruction in its initial
prompt.

Point `Skills` at one or more skill libraries. The model first sees each
skill's name and description. When a skill is useful, the model can call
Pydantic AI's `load_capability` tool to receive that skill's instructions.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/skills/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Installation

Install the `skills` extra for YAML frontmatter support:

```bash
uv add "pydantic-ai-harness[skills]"
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

Then add the library to your agent:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import Skills

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Skills('.agents/skills')],
)
```

`Skills` does not search `.agents`, `.claude`, or your home directory
automatically. Pass each library you want it to load.

> **Note:** `Skills` loads instructions from `SKILL.md`. It does not load
> bundled resources or run scripts.

## How it works

When `Skills(...)` is constructed, it:

1. Scans the immediate child directories of each configured library.
2. Validates the selected `SKILL.md` files.
3. Creates one deferred Pydantic AI capability for each selected skill.

The model initially sees only the skill names and descriptions. Loading a skill
adds instructions headed `# Skill: <name>`, followed by its Markdown body, to
the run through the same `load_capability` flow as other
[on-demand capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/).

Discovery happens once at construction. The catalog and parsed instructions are
a snapshot. Construct a new `Skills` instance to rescan the libraries.

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
callers. Unknown names also fail during construction.

Selection happens before frontmatter is parsed. An unselected skill does not add
instructions or frontmatter validation errors to that `Skills` instance.

These options control catalog exposure. They are not filesystem permissions or
an access-control boundary.

`Skills` reads configured paths through the process filesystem. Relative paths
resolve from the process working directory. Directory paths choose where
discovery starts; they do not create a containment boundary, and normal
filesystem symlink resolution applies. Run the agent in an appropriately
restricted environment if filesystem containment is required.

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

Selected skill names must be unique across those libraries. Repeated references
to the same resolved library are scanned once.

## Bundled files are not loaded

Agent Skill packages can contain directories such as `references/`, `assets/`,
and `scripts/`. `Skills` does not enumerate, read, or execute those files.

Relative paths and placeholders such as `${CLAUDE_SKILL_DIR}` remain unchanged
in the loaded instructions. `Skills` does not provide a model-visible path that
resolves them.

`Skills` also does not infer access from model-facing `FileSystem` or `Shell`
capabilities. Adding either capability does not change which files `Skills`
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

If a selected skill uses any of these fields, construction emits one aggregated
`UserWarning`. Fields such as `license`, `compatibility`, and `metadata` are
accepted without changing runtime behavior. Other unknown, non-behavioral fields
are also accepted.

## Use an agent spec

`Skills` works with Pydantic AI's
[YAML and JSON agent specs](https://pydantic.dev/docs/ai/core-concepts/agent-spec/):

```yaml
model: anthropic:claude-sonnet-4-6
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
)
```

- `directories` accepts one library path or a sequence of paths.
- `include` exposes only the named skills.
- `exclude` omits the named skills from the catalog.

Pass at least one library directory, not the path of an individual skill
package. Malformed frontmatter, invalid or mismatched names, duplicate selected
names, unknown selections, missing libraries, and non-directory library paths
fail during construction.

Every selected skill is deferred. This is part of the `Skills` behavior and is
not configurable.

## Further reading

- [Agent Skills specification](https://agentskills.io/specification)
- [Adding skills support to an agent](https://agentskills.io/client-implementation/adding-skills-support)
- [Pydantic AI on-demand capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/)
- [Pydantic AI capabilities overview](https://pydantic.dev/docs/ai/capabilities/overview/)
