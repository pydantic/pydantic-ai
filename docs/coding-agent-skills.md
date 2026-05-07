# Coding Agent Skills

If you're building Pydantic AI applications with a coding agent, you can install the Pydantic AI skill from the [`pydantic/skills`](https://github.com/pydantic/skills) repository to give your agent up-to-date framework knowledge.

[Agent skills](https://agentskills.io) are packages of instructions and reference material that coding agents load on demand. With the skill installed, coding agents have access to Pydantic AI patterns, architecture guidance, and common task references covering [tools](tools.md), [capabilities](capabilities.md), [structured output](output.md), [streaming](agent.md#streaming-events-and-final-output), [testing](testing.md), [multi-agent delegation](multi-agent-applications.md), [hooks](hooks.md), and [agent specs](agent-spec.md).

!!! note
    If you want to build agent skills for your Pydantic AI agent, see the [Agent Skills](capabilities.md#agent-skills) entry in the Third-party capabilities section on the Capabilities page.

## Installation

### Claude Code

Add the Pydantic marketplace and install the plugin:

```bash
claude plugin marketplace add pydantic/skills
claude plugin install ai@pydantic-skills
```

### Cross-Agent (agentskills.io)

Install the Pydantic AI skill using the [skills CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add pydantic/skills
```

This works with 30+ agents via the [agentskills.io](https://agentskills.io) standard, including Claude Code, Codex, Cursor, and Gemini CLI.

### Library Skills

Pydantic AI also ships its skill bundled with the package, so you can install it directly from your project's dependencies via [library-skills.io](https://library-skills.io):

```bash
uvx library-skills --all
```

The `--all` flag is required because the skill is bundled in `pydantic-ai-slim`, which is a transitive dependency of the `pydantic-ai` meta-package. Without it, `library-skills` only scans direct dependencies and won't discover the skill.

Add `--claude` to install into `.claude/skills` instead of the default `.agents` directory.

## See Also

- [`pydantic/skills`](https://github.com/pydantic/skills): source repository
- [agentskills.io](https://agentskills.io): the open standard for agent skills
- [library-skills.io](https://library-skills.io): install agent skills bundled with your project's dependencies
