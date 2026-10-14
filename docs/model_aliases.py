"""The concrete models behind the model aliases used throughout `docs/`.

Write `{{ openai:medium }}` in a doc page for the full model string below, like `openai:<model>`,
or `{{ openai:medium.name }}` for the bare model name, like `<model>`.
To move the docs to a newer model, change it here. See `docs/AGENTS.md`.
"""

MODEL_ALIASES = {
    'anthropic:small': 'anthropic:claude-haiku-5-5',
    'anthropic:medium': 'anthropic:claude-sonnet-5-5',
    'anthropic:large': 'anthropic:claude-opus-5-5',
    'openai:small': 'openai:gpt-6-luna',
    'openai:medium': 'openai:gpt-6.1-sol',
    'openai:large': 'openai:gpt-6-astra',
    'google:small': 'google:gemini-3.5-flash-lite',
    'google:medium': 'google:gemini-3.8-flash',
    'google:large': 'google:gemini-4-argon',
    'xai:small': 'xai:grok-4.3',
    'xai:medium': 'xai:grok-4.6',
    'xai:large': 'xai:grok-4.6',
    'bedrock:small': 'bedrock:global.anthropic.claude-haiku-5-5',
    'bedrock:medium': 'bedrock:global.anthropic.claude-sonnet-5-5',
    'bedrock:large': 'bedrock:global.anthropic.claude-opus-5-5',
}
