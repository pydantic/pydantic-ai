---
description: "Open source applications, libraries, agents and research projects built with Pydantic AI, Pydantic AI Harness and Pydantic Evals."
---

# Built with Pydantic AI

Open source projects that use Pydantic AI, [Pydantic AI Harness](harness/index.md) or [Pydantic Evals](evals.md). Each entry links to the project's repository and says what it uses them for.

Building something with Pydantic AI in the open? Open a pull request adding it to [`docs/built-with.md`](https://github.com/pydantic/pydantic-ai/blob/main/docs/built-with.md): one line in the matching group, with a link to the repository. A reusable capability package for Pydantic AI agents belongs on [Third-Party Capabilities](capabilities/third-party.md) instead.

## Applications and platforms

- [Dify](https://github.com/langgenius/dify): a platform for building agentic workflows and RAG pipelines. Its `dify-agent` runtime runs agents on Pydantic AI, with Harness capabilities such as compaction.
- [Stirling PDF](https://github.com/Stirling-Tools/Stirling-PDF): a PDF editing application. Its AI engine builds its agents and models on Pydantic AI.
- [marimo](https://github.com/marimo-team/marimo): a reactive Python notebook. Its AI chat features run on Pydantic AI agents and the Vercel AI UI adapter.
- [Polar](https://github.com/polarsource/polar): a billing platform. Its merchant dashboard assistant is a Pydantic AI agent, and its organization review is evaluated with Pydantic Evals.
- [Galaxy](https://github.com/galaxyproject/galaxy): the Galaxy Project's open platform for data-intensive biomedical research. Its AI assistant's agents, such as the router, tool error analysis and page assistant, are built on Pydantic AI and evaluated with Pydantic Evals.
- [Phoenix](https://github.com/Arize-ai/phoenix): Arize's AI observability and evaluation platform. Its in-app assistant is a Pydantic AI agent with custom capabilities, MCP tools and the Vercel AI UI types.
- [La Suite numérique](https://github.com/suitenumerique): the French and German governments' open source workplace suite. Its [Conversations](https://github.com/suitenumerique/conversations) chatbot is a Pydantic AI agent with MCP tools, and the editor AI in [Docs](https://github.com/suitenumerique/docs) runs on Pydantic AI too.
- [Baserow](https://github.com/baserow/baserow): a no-code platform for databases, automations and apps. Its AI assistant is a Pydantic AI agent, and its AI fields call models through Pydantic AI.
- [BlueKing PaaS](https://github.com/TencentBlueKing/blueking-paas): Tencent BlueKing's platform for building and deploying SaaS apps. Its App-Spark coding agent runs on Pydantic AI and AG-UI, with Harness capabilities such as file system, shell and compaction.
- [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot): AI4Finance Foundation's agent platform for financial research. Its desktop research workstation runs its multi-agent workflows on Pydantic AI, streamed through the Vercel AI UI adapter.
- [LTX Desktop](https://github.com/Lightricks/LTX-Desktop): Lightricks' desktop app for generating videos with LTX models. Its assistant, which finds assets and queues generation jobs, is a Pydantic AI agent.
- [Nemesis](https://github.com/SpecterOps/Nemesis): SpecterOps' offensive data enrichment pipeline. Its agents service runs Pydantic AI agents for tasks such as credential analysis and summarization, plus a chatbot that uses tools over MCP.
- [ThreatAtlas](https://github.com/OWASP/www-project-threatatlas): OWASP's collaborative threat modeling platform. Its AI assistant is a Pydantic AI agent.
- [Prebid Sales Agent](https://github.com/prebid/salesagent): Prebid.org's implementation of the Ad Context Protocol sales agent, which lets AI agents buy advertising inventory. Its policy checks, creative review and product ranking agents are built on Pydantic AI.
- [Gramps Web API](https://github.com/gramps-project/gramps-web-api): the backend of Gramps Web, a genealogy application. Its chat assistant is a Pydantic AI agent with tools for querying the family tree.
- [AgenticOS](https://github.com/vstorm-co/agenticos): [Vstorm](https://vstorm.co/)'s self-hosted platform for building, running and governing a company's agents. Its agent layer is built on Pydantic AI, with Harness capabilities such as compaction, guardrails and planning.
- [RPent](https://github.com/RLinf/RPent): agentic infrastructure for the physical world. Its planner runs on Pydantic AI, with Harness compaction.

## Libraries and integrations

- [Apache Airflow](https://github.com/apache/airflow): a workflow orchestration platform. Its `common.ai` provider adds Pydantic AI hooks, connections and an agent operator that uses the Harness.
- [FastAPI](https://github.com/fastapi/fastapi): a Python web framework. It translates its own documentation with a Pydantic AI agent.
- [Marvin](https://github.com/PrefectHQ/marvin): Prefect's library for building AI functions and agents. Its agents run on Pydantic AI and work with any Pydantic AI model or MCP server.
- [dlt](https://github.com/dlt-hub/dlt): dltHub's Python library for loading data. Its workspace deployment agents run on Pydantic AI by default, with MCP tools.
- [angr](https://github.com/angr/angr): a binary analysis platform. Its LLM client, used by its LLM-assisted analysis features, runs on Pydantic AI agents.
- [RD-Agent](https://github.com/microsoft/RD-Agent): Microsoft's agent framework for automating data-driven research and development. Its MCP-backed helper agents, such as documentation lookup and RAG, are Pydantic AI agents.
- [Astronomer Cosmos](https://github.com/astronomer/astronomer-cosmos): Astronomer's library for running dbt projects as Airflow DAGs. Its AI diagnosis of failed dbt tasks is a Pydantic AI agent, run through Airflow's `common.ai` provider.
- [Vizro](https://github.com/mckinsey/vizro): McKinsey's low-code toolkit for data visualization apps. Its experimental dashboard chat agent and its Vizro-AI chart generator run on Pydantic AI.
- [TypeAgent](https://github.com/microsoft/typeagent-py): Microsoft's structured RAG library for ingesting, indexing and querying. It creates its chat and embedding models through Pydantic AI.
- [PowerContext](https://github.com/oceanbase/powercontext): OceanBase's context and memory server for agents. Its built-in inference runs on Pydantic AI models and embedders, and it ships a capability that gives Pydantic AI agents its memory.
- [csp-bot](https://github.com/Point72/csp-bot): a reactive chat bot framework. Its agent commands run Pydantic AI agents with resumable conversation history.
- [Agentic Context Engine](https://github.com/kayba-ai/agentic-context-engine): Kayba's framework for agents that learn from experience. Its agent, reflector and skill manager roles are Pydantic AI agents.
- [haiku.rag](https://github.com/ggozad/haiku.rag): an agentic RAG library and MCP server for local document search. Its agents and capabilities run on Pydantic AI, and it is evaluated with Pydantic Evals. See [Third-Party Capabilities](capabilities/third-party.md#retrieval-rag) for what its capabilities do.
- [pydantic-ai-skills](https://github.com/DougTrajano/pydantic-ai-skills): Doug Trajano's [Agent Skills](https://agentskills.io) library, built on Harness [`Skills`](harness/skills.md). See [Third-Party Capabilities](capabilities/third-party.md#agent-skills) for what it adds.
- [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents): Vstorm's framework and terminal assistant for deep agents, built on Pydantic AI and the Harness.
- [ya-mono](https://github.com/Wh1isper/ya-mono): an agent SDK, terminal UI and workspace runtime built on Pydantic AI.
- [Full-Stack AI Agent Template](https://github.com/vstorm-co/full-stack-ai-agent-template): Vstorm's FastAPI and Next.js project generator whose generated backends can use Pydantic AI agents.

## Agents and tools

- [Code Puppy](https://github.com/mpfaffenberger/code_puppy): a coding agent for the terminal, built on Pydantic AI and the Harness.
- [moyai](https://github.com/BerriAI/moyai): BerriAI's self-hosted coding agent for background work. It runs on Pydantic AI, with Harness capabilities such as planning, ask-user questions and conversation search.
- [oterm](https://github.com/ggozad/oterm): a terminal client for LLMs. Its agents run on Pydantic AI, with Harness file system access, compaction and memory.
- [Conductor](https://github.com/microsoft/conductor): Microsoft's CLI for defining and running multi-agent workflows. Pydantic AI is one of its agent providers, with Harness compaction.
- [Tiger Agents for Work](https://github.com/timescale/tiger-agents-for-work): Tiger Data's library and CLI for Slack-native agents, which runs its own internal Slack agent. Its agents run on Pydantic AI, with MCP servers and Harness subagents.
- [Colin](https://github.com/PrefectHQ/colin): Prefect's context engine that compiles agent skills from live sources and rebuilds them when they go stale. Its LLM blocks run on Pydantic AI agents.
- [OSS-Fuzz MCP](https://github.com/google/oss-fuzz/tree/master/infra/experimental/mcp): Google OSS-Fuzz's experimental agent tooling for fuzzing tasks, such as fixing builds. Its MCP client agent runs on Pydantic AI.
- [Code-Graph-RAG](https://github.com/vitali87/code-graph-rag): a knowledge-graph RAG system for querying and editing multi-language codebases. Its query and editing agents run on Pydantic AI.
- [CodeWiki](https://github.com/FSoft-AI4Code/CodeWiki): FSoft AI4Code's research framework for generating repository-level documentation. Its documentation agents run on Pydantic AI.
- [TimeCopilot](https://github.com/TimeCopilot/timecopilot): a forecasting agent that combines LLMs with time series foundation models. Its forecasting and query agents run on Pydantic AI.
- [Orcha](https://github.com/inveniosoftware/orcha): CERN's Invenio project's AI workflows for research data repositories. Its metadata extraction and funding relevance workflows run on Pydantic AI agents.
- [Funes](https://github.com/opensanctions/funes): OpenSanctions' research project for turning public websites into structured lists of people and the positions they hold. Its discovery and extraction agents run on Pydantic AI and are evaluated with Pydantic Evals.
- [Our World in Data ETL](https://github.com/owid/etl): Our World in Data's pipeline for loading and transforming its data. Its chart critic asks a Pydantic AI agent whether a published chart would mislead readers.
- [Aurelian](https://github.com/monarch-initiative/aurelian): the Monarch Initiative's agents for biomedical literature, ontologies and knowledge bases, built on Pydantic AI.
- [Geodini](https://github.com/developmentseed/geodini): Development Seed's natural language geocoding API over Overture data. Its query routing and geocoding agents are built on Pydantic AI.
- [Prediction Market Agent](https://github.com/gnosis/prediction-market-agent): Gnosis' agents that bet on prediction markets such as Omen. Many of its agents are built on Pydantic AI.
- [GOV.UK AI Graph Tools](https://github.com/alphagov/govuk-ai-graph-tools): the Government Digital Service's proof of concept for finding duplicate and outlier content across GOV.UK. Its content extractor is a Pydantic AI agent on Amazon Bedrock.

## Research and evals

- [Prompt Siren](https://github.com/facebookresearch/prompt-siren): Meta's research workbench for developing and testing prompt injection attacks against LLM agents. The agents it attacks are built on Pydantic AI models and toolsets.
- [ALE-Bench](https://github.com/SakanaAI/ALE-Bench): Sakana AI's benchmark for long-horizon algorithm engineering. Its LLM evaluation tool runs models through Pydantic AI agents.
- [dbt LLM Semantic Layer Benchmark](https://github.com/dbt-labs/dbt-llm-sl-bench): dbt Labs' benchmark of LLM query generation against the dbt Semantic Layer. It runs the models through Pydantic AI agents, with the dbt MCP server as a toolset.
- [Open Food Facts AI](https://github.com/openfoodfacts/openfoodfacts-ai): Open Food Facts' AI projects. Its `llm-evals` framework evaluates models with Pydantic AI and Pydantic Evals.
- [Systematically Improving RAG](https://github.com/jxnl/systematically-improving-rag): course material on improving RAG applications, with lessons on evaluating them with Pydantic Evals.
- [Consult](https://github.com/i-dot-ai/consult): an application for analysing public consultation responses. Its evaluation framework uses Pydantic Evals as its runner.
