# Claude Code Integration

PydanticAI includes official skills for AI-assisted development with Claude Code and other AI coding agents.

## Installation

### Option A: Via npx skills (Recommended)

Install to all detected agents (Claude Code, Cursor, Copilot, etc.):

```bash
npx skills add pydantic/pydantic-ai
```

Or install to a universal location shared by all agents:

```bash
npx skills add pydantic/pydantic-ai --universal
```

### Option B: Claude Code Plugin Marketplace

```
/plugin marketplace add pydantic/pydantic-ai
/plugin install pydantic-ai@pydantic-ai-skills
```

### Option C: Manual

```bash
git clone https://github.com/pydantic/pydantic-ai
cp -r pydantic-ai/skills/pydantic-ai ~/.claude/skills/
```

## What's Included

The PydanticAI skill provides:

- Agent creation patterns and best practices
- Tool system reference (decorators, RunContext, docstrings)
- Model provider documentation
- Structured output patterns
- Testing strategies with TestModel
- And more...

See the [SKILL.md](https://github.com/pydantic/pydantic-ai/blob/main/skills/pydantic-ai/SKILL.md) for the full skill content.
