# Claude Code Integration

PydanticAI includes official skills for AI-assisted development with Claude Code and other AI coding agents.

## Installation

Copy the skill to your Claude skills directory:

```bash
# Clone and copy
git clone --depth 1 https://github.com/pydantic/pydantic-ai
cp -r pydantic-ai/skills/pydantic-ai ~/.claude/skills/

# Or download directly
mkdir -p ~/.claude/skills/pydantic-ai
curl -o ~/.claude/skills/pydantic-ai/SKILL.md \
  https://raw.githubusercontent.com/pydantic/pydantic-ai/main/skills/pydantic-ai/SKILL.md
```

## What's Included

The PydanticAI skill provides:

- Agent creation patterns and best practices
- Tool system reference (decorators, RunContext, docstrings)
- Model provider documentation
- Structured output patterns
- Testing strategies with TestModel
- Decision trees for choosing tools, output modes, and multi-agent patterns
- Comparison tables for providers, decorators, and agent methods

See the [SKILL.md](https://github.com/pydantic/pydantic-ai/blob/main/skills/pydantic-ai/SKILL.md) for the full skill content.
