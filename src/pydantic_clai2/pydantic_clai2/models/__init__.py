"""CLAI's model integrations. Submodules load on demand; this package stays cheap to import."""

CLAI_PROVIDERS = frozenset({'github-copilot', 'openai-codex', 'openrouter', 'vllm'})
"""Prefixes CLAI's own model resolver handles before Pydantic AI sees the name."""
