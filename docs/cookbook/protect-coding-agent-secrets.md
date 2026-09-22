---
title: Keep host secrets out of a coding agent
description: Layer filesystem policy, a minimal subprocess environment, and input/output redaction around a local coding agent.
---

# Keep host secrets out of a coding agent

A local coding agent needs workspace and shell access, but it should not inherit every host credential or disclose secrets found in ordinary files. Apply controls at each boundary rather than relying on the system prompt.

```bash
pip/uv-add "pydantic-ai-slim[anthropic]" "pydantic-ai-harness==0.31.0"
export ANTHROPIC_API_KEY=your-api-key
```

```python {call_name="build_agent" dunder_name="not_main" noqa="I001"}
import os
from pathlib import Path

from pydantic_ai_harness import LLM_API_KEY_ENV_PATTERNS, FileSystem, Shell
from pydantic_ai_harness.guardrails import OutputGuardrail, ToolGuardrail
from pydantic_ai_harness.guardrails.detectors import for_text, for_tool_result_text, redact_secrets

from pydantic_ai import Agent
from pydantic_ai.models import Model

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-fable-5')
_SECRET_PATHS = ['.env', '.env.*', '*.pem', '*.key', '**/secrets*']


def build_agent(model: Model | str = DEFAULT_MODEL, *, workspace: Path | None = None) -> Agent[object, str]:
    """Build a local coding agent with layered secret controls."""
    root = (workspace or Path.cwd()).resolve()
    minimal_env = {'PATH': os.environ.get('PATH', '')}
    return Agent(
        model,
        name='secret_safe_coder',
        instructions=(
            'Inspect the project using the available file and shell tools. '
            'Do not expose credentials in tool results or your final answer.'
        ),
        capabilities=[
            FileSystem(
                root_dir=root,
                denied_patterns=_SECRET_PATHS,
                read_only=True,
            ),
            Shell(
                cwd=root,
                allowed_commands=['env'],
                env=minimal_env,
                denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
            ),
            ToolGuardrail(result_guard=for_tool_result_text(redact_secrets)),
            OutputGuardrail(guard=for_text(redact_secrets)),
        ],
    )


def main() -> None:
    # Use a finalized result: OutputGuardrail does not screen partial streamed text.
    request = input('What should the coding agent inspect? ')
    result = build_agent().run_sync(request)
    print(result.output)


if __name__ == '__main__':
    main()
```

The filesystem denylist blocks known credential locations. The shell receives an explicit minimal environment instead of the host environment. Tool-result redaction limits what enters model context, while output redaction is a final defense before returning a non-streamed response. These layers reduce exposure; they do not replace process isolation for hostile code.

## Related

See the Harness guides for [filesystem policy](https://pydantic.dev/docs/ai/harness/filesystem/) and [guardrails](https://pydantic.dev/docs/ai/harness/guardrails/).
