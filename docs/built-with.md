# Built with Pydantic AI

Open source projects that use Pydantic AI, [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) or [Pydantic Evals](evals.md) in their code. Each entry links to the project's repository and says what it uses them for.

Building something with Pydantic AI in the open? Open a pull request adding it to [`docs/built-with.md`](https://github.com/pydantic/pydantic-ai/blob/main/docs/built-with.md): one line in the matching group, with a link to the repository.

## Applications and platforms

- [Dify](https://github.com/langgenius/dify): a platform for building agentic workflows and RAG pipelines. Its `dify-agent` runtime runs agents on Pydantic AI, with Harness capabilities such as compaction.
- [Stirling PDF](https://github.com/Stirling-Tools/Stirling-PDF): a PDF editing application. Its AI engine builds its agents and models on Pydantic AI.
- [marimo](https://github.com/marimo-team/marimo): a reactive Python notebook. Its AI chat features run on Pydantic AI agents and the Vercel AI UI adapter.
- [Polar](https://github.com/polarsource/polar): a billing platform. Its merchant dashboard assistant is a Pydantic AI agent, and its organization review is evaluated with Pydantic Evals.
- [AgenticOS](https://github.com/vstorm-co/agenticos): a self-hosted platform for building, running and governing a company's agents. Its agent layer is built on Pydantic AI, with Harness capabilities such as compaction, guardrails and planning.

## Libraries and integrations

- [Apache Airflow](https://github.com/apache/airflow): a workflow orchestration platform. Its `common.ai` provider adds Pydantic AI hooks, connections and an agent operator that uses the Harness.
- [FastAPI](https://github.com/fastapi/fastapi): a Python web framework. It translates its own documentation with a Pydantic AI agent.
- [TypeAgent](https://github.com/microsoft/typeagent-py): Microsoft's structured RAG library for ingesting, indexing and querying. It creates its chat and embedding models through Pydantic AI.
- [csp-bot](https://github.com/Point72/csp-bot): a reactive chat bot framework. Its agent commands run Pydantic AI agents with resumable conversation history.
- [pydantic-ai-skills](https://github.com/DougTrajano/pydantic-ai-skills): [Agent Skills](https://agentskills.io) support for Pydantic AI agents, with progressive disclosure and filesystem or programmatic skills.
- [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents): a framework and terminal assistant for deep agents, built on Pydantic AI.
- [ya-mono](https://github.com/Wh1isper/ya-mono): an agent SDK, terminal UI and workspace runtime built on Pydantic AI.
- [Full-Stack AI Agent Template](https://github.com/vstorm-co/full-stack-ai-agent-template): a FastAPI and Next.js project generator whose generated backends can use Pydantic AI agents.

## Agents and tools

- [Code Puppy](https://github.com/mpfaffenberger/code_puppy): a coding agent for the terminal, built on Pydantic AI and the Harness.
- [Conductor](https://github.com/microsoft/conductor): Microsoft's CLI for defining and running multi-agent workflows. Pydantic AI is one of its agent providers, with Harness compaction.
- [autobench](https://github.com/vcoderun/autobench): a YAML-first benchmark framework. It runs and instruments Pydantic AI agents.

## Evals

- [Open Food Facts AI](https://github.com/openfoodfacts/openfoodfacts-ai): Open Food Facts' AI projects. Its `llm-evals` framework evaluates models with Pydantic AI and Pydantic Evals.
- [Systematically Improving RAG](https://github.com/jxnl/systematically-improving-rag): course material on improving RAG applications, with lessons on evaluating them with Pydantic Evals.
- [Consult](https://github.com/i-dot-ai/consult): an application for analysing public consultation responses. Its evaluation framework uses Pydantic Evals as its runner.
- [Excel Agent](https://github.com/SylvianAI/sv-excel-agent): an agent that reads, edits and automates Excel spreadsheets through MCP tools, built on Pydantic AI and evaluated with Pydantic Evals.
