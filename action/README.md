# Run a Pydantic AI agent

A composite action that runs a [Pydantic AI](https://pydantic.dev/docs/ai/) agent on a GitHub
Actions runner: the [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) by default, or an agent
of your own, named as a `module:variable` pair or as an
[agent spec](https://pydantic.dev/docs/ai/core-concepts/agent-spec/) file.

```yaml
- uses: pydantic/pydantic-ai/action@main
  with:
    model: openai:gpt-5.6-sol
    prompt: Summarize what changed in this pull request.
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

The agent runs on the runner rather than in a sandbox, and the default `Coder` can run commands,
so use this action where you decide what the agent is asked to do. When the prompt can come from
an issue, a comment, or a pull request from a fork, use
[GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) instead.

**Inputs, credentials, observability and the security posture are documented on
[GitHub Actions](https://pydantic.dev/docs/ai/integrations/github-actions/).**
