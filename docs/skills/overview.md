# Skills

A standardized, composable framework for building and managing Agent Skills. Skills are modular collections of instructions, scripts, tools, and resources that enable AI agents to progressively discover, load, and execute specialized capabilities for domain-specific tasks.

## What are Agent Skills?

Agent Skills are **modular packages** that extend your agent's capabilities without hardcoding every possible feature into your agent's instructions. Think of them as plugins that agents can discover and load on-demand.

Key benefits:

- **Progressive Discovery**: Skills are listed in the system prompt; agents load detailed instructions only when needed
- **Modular Design**: Each skill is a self-contained directory with instructions and resources
- **Script Execution**: Skills can include executable Python scripts
- **Resource Management**: Support for additional documentation and data files
- **Simple Integration**: Works seamlessly with any Pydantic AI agent and model provider with tool calling support.

## Quick Example

```python {title="agent_with_skills.py"}
from pydantic_ai import Agent
from pydantic_ai.toolsets import SkillsToolset

# Initialize Skills Toolset with skill directories
# Can use directories (str/Path), SkillsDirectory instances, or programmatic skills
skills_toolset = SkillsToolset(directories=["./skills"])

# Create agent with skills
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

1. **Discovery**: The toolset discovers skills from directories or accepts pre-built [`Skill`][pydantic_ai.skills.Skill] objects
2. **Registration**: Skills are listed in the agent's system prompt with four management tools: `list_skills()`, `load_skill()`, `read_skill_resource()`, and `run_skill_script()`
3. **Progressive Loading**: Agents load detailed instructions only when needed, keeping initial context minimal

## Progressive Disclosure

Skills implement **progressive disclosure** - exposing information only when needed. This keeps initial context minimal while allowing agents to discover capabilities dynamically:

- Skills are listed in the system prompt with names and descriptions
- Agents load full instructions using `load_skill()` only when needed
- Additional resources accessed via `read_skill_resource()`
- Scripts executed via `run_skill_script()`

This approach reduces token usage and scales to hundreds of skills without bloating prompts.

## Security Considerations

!!! warning "Use Skills from Trusted Sources Only"

    Skills provide agents with instructions and executable code. Use only skills from trusted sources you control or verify. Malicious skills could misuse agent capabilities or execute harmful code.

The toolset includes security measures:

- **Path validation**: Resources and scripts validated within skill directories
- **Script timeout**: Configurable timeout (default: 30s) prevents hung processes  
- **Subprocess isolation**: Scripts run in separate processes
- **Resource depth limit**: Maximum 3-level depth prevents excessive traversal
- **Custom executors**: Implement [`SkillScriptExecutor`][pydantic_ai.skills.SkillScriptExecutor] for additional security

## Default Directory Behavior

When initializing [`SkillsToolset`][pydantic_ai.toolsets.SkillsToolset] without arguments, it defaults to discovering skills in the `./skills` directory:

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

This implementation follows concepts from [agentskills.io](https://agentskills.io).
