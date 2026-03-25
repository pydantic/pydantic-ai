# Agent Skills

[Agent skills](https://agentskills.io) are packages of instructions and reference material that coding agents load on demand. Pydantic AI maintains a skill for building Pydantic AI applications in the [`pydantic/skills`](https://github.com/pydantic/skills) repository.

With the skill installed, coding agents like Claude Code, Cursor, GitHub Copilot, and OpenAI Codex have access to Pydantic AI patterns, architecture guidance, and common task references covering tools, capabilities, structured output, streaming, testing, multi-agent delegation, hooks, and agent specs.

## Installation

### Claude Code

Add the Pydantic marketplace and install the plugin:

```bash
claude plugin marketplace add pydantic/skills
claude plugin install ai@pydantic-skills
```

### Cross-Agent (agentskills.io)

The [`skills/`](https://github.com/pydantic/skills/tree/main/skills) directory contains standalone `SKILL.md` files compatible with 30+ agents via the [agentskills.io](https://agentskills.io) standard, including Cursor, GitHub Copilot, OpenAI Codex, and Gemini CLI.

Refer to your agent's documentation for how to add skills.

## See Also

- [`pydantic/skills`](https://github.com/pydantic/skills): source repository
- [agentskills.io](https://agentskills.io): the open standard for agent skills
