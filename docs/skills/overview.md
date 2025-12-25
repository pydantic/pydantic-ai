# Skills

A standardized, composable framework for building and managing Agent Skills. Skills are modular collections of instructions, scripts, tools, and resources that enable AI agents to progressively discover, load, and execute specialized capabilities for domain-specific tasks.

## What are Agent Skills?

Agent Skills are **modular packages** that extend your agent's capabilities without hardcoding every possible feature into your agent's instructions. Think of them as plugins that agents can discover and load on-demand.

Key benefits:

- **🔍 Progressive Discovery**: Skills are listed in the system prompt; agents load detailed instructions only when needed
- **📦 Modular Design**: Each skill is a self-contained directory with instructions and resources
- **🛠️ Script Execution**: Skills can include executable Python scripts
- **📚 Resource Management**: Support for additional documentation and data files
- **🚀 Easy Integration**: Simple toolset interface that works with any Pydantic AI agent
- **⚡ Automatic Injection**: Skill metadata is automatically added to the agent's system prompt via `get_instructions()`

## Quick Example

```python {title="agent_with_skills.py"}
from pydantic_ai import Agent
from pydantic_ai.toolsets import SkillsToolset

# Initialize Skills Toolset with skill directories
# Can use directories (str/Path), SkillsDirectory instances, or programmatic skills
skills_toolset = SkillsToolset(directories=["./skills"])

# Create agent with skills
# Skills instructions are automatically injected via get_instructions()
agent = Agent(
    model='openai:gpt-4o',
    instructions="You are a helpful research assistant.",
    toolsets=[skills_toolset]
)

# Use agent - skills tools are automatically available
result = await agent.run(
    "What are the last 3 papers on arXiv about machine learning?"
)
print(result.output)
```

!!! note "Alternative Import"
    You can also import `SkillsToolset` from `pydantic_ai.toolsets`:
    ```python
    from pydantic_ai.toolsets import SkillsToolset
    ```

## How It Works

1. **Discovery**: The toolset scans specified directories for skills (folders with `SKILL.md` files) or accepts pre-built `Skill` objects
2. **Automatic Injection**: Skill names and descriptions are automatically injected into the agent's system prompt via the toolset's `get_instructions()` method
3. **Registration**: Four skill management tools are registered with the agent
4. **Progressive Loading**: Agents can:
   - List all available skills with `list_skills()` (optional, as skills are already in system prompt)
   - Load detailed instructions with `load_skill(skill_name)`
   - Read additional resources with `read_skill_resource(skill_name, resource_name)`
   - Execute scripts with `run_skill_script(skill_name, script_name, args)`

The toolset supports multiple initialization modes:

- **Directory-based**: Automatically discover skills from filesystem directories (creates [`LocalSkill`][pydantic_ai.toolsets.skills.LocalSkill] instances)
- **Programmatic**: Pass pre-built [`Skill`][pydantic_ai.toolsets.skills.Skill] objects directly (can be custom implementations)
- **Combined**: Mix both directory-based and programmatic skills
- **SkillsDirectory instances**: Use [`SkillsDirectory`][pydantic_ai.toolsets.skills.SkillsDirectory] objects for fine-grained control over discovery and script execution

## Progressive Disclosure

The toolset implements **progressive disclosure** - exposing information only when needed:

```markdown
┌─────────────────────────────────────────────────────────────┐
│  System Prompt (automatically injected via toolset)         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Available Skills:                                     │  │
│  │ - arxiv-search: Search arXiv for research papers      │  │
│  │ - web-research: Research topics on the web            │  │
│  │ - data-analyzer: Analyze CSV and JSON files           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
          Agent sees skill names & descriptions
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  load_skill("arxiv-search")                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Returns full SKILL.md instructions:                   │  │
│  │ - When to use                                         │  │
│  │ - Step-by-step guide                                  │  │
│  │ - Example invocations                                 │  │
│  │ - Available resources and scripts                     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
      Agent loads detailed instructions when needed
```

This approach:

- **Reduces initial context size** - Only metadata is in the system prompt
- **Lets agents discover capabilities dynamically** - Load what's needed
- **Improves token efficiency** - Don't pay for unused instructions
- **Scales to many skills** - Add hundreds of skills without bloating prompts

## Security Considerations

!!! warning "Use Skills from Trusted Sources Only"

    Skills provide AI agents with new capabilities through instructions and code. While this makes them powerful, it also means a malicious skill can direct agents to invoke tools or execute code in ways that don't match the skill's stated purpose.

    If you must use a skill from an untrusted or unknown source, exercise extreme caution and thoroughly audit it before use. Depending on what access agents have when executing the skill, malicious skills could lead to data exfiltration, unauthorized system access, or other security risks.

The toolset includes security measures:

- **Path traversal prevention**: For filesystem-based skills ([`LocalSkill`][pydantic_ai.toolsets.skills.LocalSkill]), resources and scripts are validated to stay within the skill directory using path resolution to prevent directory traversal attacks
- **Script timeout**: Scripts have a configurable timeout (default: 30 seconds) enforced via `anyio.move_on_after()` to prevent hung processes
- **Subprocess execution**: [`LocalSkillScriptExecutor`][pydantic_ai.toolsets.skills.LocalSkillScriptExecutor] runs scripts in a separate process via `anyio.run_process()`, but with the same OS-level permissions as your agent process
- **Resource depth limit**: Resource discovery is limited to a maximum depth of 3 levels within the skill directory (`max_depth=3`) to prevent excessive file system traversal
- **Flexible execution**: Custom script executors can implement additional security measures through the [`SkillScriptExecutor`][pydantic_ai.toolsets.skills.SkillScriptExecutor] protocol

## Default Directory Behavior

When initializing [`SkillsToolset`][pydantic_ai.toolsets.skills.SkillsToolset] without arguments, it defaults to discovering skills in the `./skills` directory:

```python
# These are equivalent
toolset = SkillsToolset()
toolset = SkillsToolset(directories=["./skills"])
```

**Important:** The default directory is NOT used if you provide programmatic skills:

```python
# No default directory - only programmatic skills
toolset = SkillsToolset(skills=[custom_skill])

# To use both, explicitly specify directories
toolset = SkillsToolset(
    skills=[custom_skill],
    directories=["./skills"]  # Must be explicit
)
```

If the default `./skills` directory doesn't exist, a warning is emitted and no skills are loaded.

## Example Skills

The PydanticAI repository includes example skills you can reference:

- **[arxiv-search](https://github.com/pydantic/pydantic-ai/tree/main/examples/pydantic_ai_examples/skills/arxiv-search)**: Searches arXiv for research papers with Python script
- **[pydanticai-docs](https://github.com/pydantic/pydantic-ai/tree/main/examples/pydantic_ai_examples/skills/pydanticai-docs)**: Documentation skill for PydanticAI

See the [skills_agent.py example](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/skills_agent.py) for how to use them.

## Next Steps

- [Creating Skills](creating-skills.md) - Learn how to build your own skills
- [Using Skills](using-skills.md) - Learn how to integrate and use skills in your agents
- [API Reference](../api/skills.md) - Detailed type and API documentation

## References

This implementation is inspired by:

- [DougTrajano/pydantic-ai-skills](https://github.com/DougTrajano/pydantic-ai-skills/)
- [vstorm-co/pydantic-deepagents](https://github.com/vstorm-co/pydantic-deepagents/)
- [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents/)
- [Introducing Agent Skills | Anthropic](https://www.anthropic.com/news/agent-skills)
- [Using skills with Deep Agents | LangChain](https://blog.langchain.com/using-skills-with-deep-agents/)
