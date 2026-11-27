---
description: "Install the Pydantic AI skill so coding agents like Claude Code and Codex get current framework knowledge when they build Pydantic AI applications for you."
---

# Coding Agent Skills

If you're building Pydantic AI applications with a coding agent, you can install the Pydantic AI skill from the [`pydantic/skills`](https://github.com/pydantic/skills) repository to give your agent up-to-date framework knowledge.

[Agent skills](https://agentskills.io) are packages of instructions and reference material that coding agents load on demand. With the skill installed, coding agents have access to Pydantic AI patterns, architecture guidance, and common task references covering [tools](tools.md), [capabilities](capabilities/overview.md), [structured output](output.md), [streaming](agent.md#streaming-events-and-final-output), [testing](testing.md), [multi-agent delegation](multi-agent-applications.md), [hooks](hooks.md), and [agent specs](agent-spec.md).

When your coding agent creates a new Pydantic AI application, the skill has it run `uvx logfire-cli wizard --task pydantic-ai` to set up [Pydantic Logfire](logfire.md#using-logfire), so you can see every agent run, model request, and tool call from the first version. If your company sends production telemetry to another OpenTelemetry backend, production stays there, and the skill still offers you a personal Logfire project for [local development](logfire.md#local-development). It also knows the [Pydantic AI Gateway](gateway.md), one key for models from several providers, and the `gateway/` model strings that use it.

For the rest of Logfire, such as instrumenting the rest of your application, querying traces, or running evals, the [`pydantic/skills`](https://github.com/pydantic/skills) repository has Logfire skills too: install the `logfire` plugin, or run `uvx logfire-cli wizard --task pydantic-ai` for the guided setup.

!!! note
    If you want to give your Pydantic AI agent its own skills, see the Pydantic AI Harness [`Skills`](harness/skills.md) capability.

## Installation

### Claude Code

Install the [official Pydantic AI plugin](https://claude.com/plugins/pydantic-ai) from the Anthropic marketplace, which is available by default:

```bash
claude plugin install pydantic-ai@claude-plugins-official
```

As an alternative, you can install from the [`pydantic/skills`](https://github.com/pydantic/skills) marketplace, which bundles the Pydantic AI skill alongside other Pydantic-maintained skills:

```bash
claude plugin marketplace add pydantic/skills
claude plugin install ai@pydantic-skills
claude plugin install logfire@pydantic-skills
```

### Cross-Agent (agentskills.io)

Install the Pydantic AI skill using the [skills CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add pydantic/skills
```

This works with 30+ agents via the [agentskills.io](https://agentskills.io) standard, including Claude Code, Codex, Cursor, and Gemini CLI.

### Library Skills

Pydantic AI also ships its skills bundled with the package, so you can install them directly from your project's dependencies via [library-skills.io](https://library-skills.io):

```bash
uvx library-skills --all
```

The `--all` flag is required because the skills are bundled in `pydantic-ai-slim`, which is a transitive dependency of the `pydantic-ai` meta-package. Without it, `library-skills` only scans direct dependencies and won't discover them.

Add `--claude` to also install into `.claude/skills/` alongside the default `.agents/skills/` directory, since Claude Code doesn't read from `.agents/`.

This installation also includes [migration skills](framework-migration.md) for moving applications from other agent frameworks to Pydantic AI, and the repository advisor skill described below.

## Review your repository

The `advising-pydantic-ai-repos` skill has your coding agent review an application that already uses Pydantic AI. It's bundled with the package, so install it with [Library Skills](#library-skills) (`uvx library-skills --all`); the Claude Code plugin and `npx skills add pydantic/skills` don't include it. It reports:

- code the application writes by hand that your installed Pydantic AI already provides, such as a model fallback loop instead of the [`ModelFallback`](capabilities/model-fallback.md) capability, or approval flags instead of [deferred tools](deferred-tools.md);
- what upgrading to the latest release would add that replaces code you have;
- which notes in the [upgrade guide](changelog.md#catching-up-across-v2-minor-releases) affect your code.

It works from the current documentation, release notes, and upgrade guide, read when it runs, so its advice follows the latest release. Each finding cites the file and line, the built-in that replaces it, the version that has it, and when to keep your own code instead. The skill only reports: ask your coding agent to apply the findings you choose. For example:

```text
Use the advising-pydantic-ai-repos skill to review this repository.
```

## Next steps

With the skill installed, ask your coding agent to build your first agent, or read the [Agents guide](agent.md) to see what it will write.

## See Also

- [`pydantic/skills`](https://github.com/pydantic/skills): source repository
- [agentskills.io](https://agentskills.io): the open standard for agent skills
- [library-skills.io](https://library-skills.io): install agent skills bundled with your project's dependencies
- [Built with Pydantic AI](built-with.md): open source systems your coding agent can study for how to structure a large application, and libraries that build on Pydantic AI
