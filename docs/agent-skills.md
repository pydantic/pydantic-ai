# Agent Skills

Agent Skills are structured knowledge files that give AI coding assistants domain-specific expertise. They follow the open [Agent Skills specification](https://agentskills.io/specification), enabling interoperability across different AI coding assistants.

## How Skills Work

Skills are discovered and activated through a three-step process:

1. **Discovery**: The AI coding assistant scans configured skill directories for `SKILL.md` files
2. **Activation**: When a skill's description matches the current task context, the assistant loads the skill
3. **Execution**: The skill's patterns, examples, and guidance are used to complete the task

Each skill contains YAML frontmatter with metadata (name, description, compatibility) followed by markdown content with patterns, code examples, decision trees, and reference information.

## Installation

### Claude Code

Copy the skill to your Claude Code skills directory:

```bash
# Clone and copy
git clone --depth 1 https://github.com/pydantic/pydantic-ai
cp -r pydantic-ai/skills/building-pydantic-ai-agents ~/.claude/skills/

# Or download directly
mkdir -p ~/.claude/skills/building-pydantic-ai-agents
curl -o ~/.claude/skills/building-pydantic-ai-agents/SKILL.md \
  https://raw.githubusercontent.com/pydantic/pydantic-ai/main/skills/building-pydantic-ai-agents/SKILL.md
```

### Other AI Coding Assistants

Agent Skills follow an open specification and work with any compatible assistant. For Claude Code, skills are stored in `~/.claude/skills/`. For other AI coding assistants, consult the assistant's documentation for the skill directory location.

## What the PydanticAI Skill Provides

The `building-pydantic-ai-agents` skill provides:

- **Quick-start patterns** for agent creation, tools, structured output, and dependency injection
- **Task routing table** linking common tasks to relevant documentation
- **Decision trees** for choosing tool registration methods, output modes, multi-agent patterns, and testing approaches
- **Comparison tables** for output modes, model providers, decorators, and agent methods
- **Architecture overview** covering execution flow, generic types, and model string format

See the [SKILL.md](https://github.com/pydantic/pydantic-ai/blob/main/skills/building-pydantic-ai-agents/SKILL.md) for the full skill content.

## Best Practices

For guidance on authoring your own skills, see the [Agent Skills Best Practices](https://agentskills.io/best-practices).
