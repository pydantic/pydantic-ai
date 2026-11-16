---
description: "Open source applications, developer tools, domain agents, data integrations, libraries and research projects built with Pydantic AI, Pydantic AI Harness and Pydantic Evals, plus case studies of organizations using them in production."
---

# Built with Pydantic AI

Pydantic AI is used by individual developers, startups, large enterprises and governments. The projects below come from organizations such as Microsoft, Google, Walmart, Meta, Tencent, McKinsey, Point72, Lightricks and CERN, and from the French, German and UK governments.

This page lists open source projects that use Pydantic AI, [Pydantic AI Harness](harness/index.md) or [Pydantic Evals](evals.md), grouped by what they built. Each entry links to the project's repository and says what part of it runs on Pydantic AI, so you can find a system shaped like yours and read how it is put together. Organizations using Pydantic AI in production whose code isn't public, such as the UK Ministry of Justice's Justice AI Unit, are in [Case studies](#case-studies).

- [Applications and platforms](#applications-and-platforms): products where agents are one part of a larger system.
- [Coding agents and developer tools](#coding-agents-and-developer-tools): agents and tools for writing, running and analysing code.
- [Domain agents and workflows](#domain-agents-and-workflows): agents built for one field, such as government content, research data or forecasting.
- [Data and infrastructure integrations](#data-and-infrastructure-integrations): data pipelines and orchestrators that run Pydantic AI agents.
- [Libraries and frameworks](#libraries-and-frameworks): packages built on Pydantic AI that you can build on in turn.
- [Research and evals](#research-and-evals): benchmarks and evaluation frameworks.

## Applications and platforms

- [Dify](https://github.com/langgenius/dify): a platform for building agentic workflows and RAG pipelines. Its `dify-agent` runtime runs agents on Pydantic AI, with Harness capabilities such as compaction.
- [La Suite numérique](https://github.com/suitenumerique) by the **French and German governments**: an open source workplace suite. Its [Conversations](https://github.com/suitenumerique/conversations) chatbot is a Pydantic AI agent with MCP tools, and the editor AI in [Docs](https://github.com/suitenumerique/docs) runs on Pydantic AI too.
- [BlueKing PaaS](https://github.com/TencentBlueKing/blueking-paas) by **Tencent BlueKing**: a platform for building and deploying SaaS apps. Its App-Spark coding agent runs on Pydantic AI and AG-UI, with Harness capabilities such as file system, shell and compaction.
- [marimo](https://github.com/marimo-team/marimo): a reactive Python notebook. Its AI chat features run on Pydantic AI agents and the Vercel AI UI adapter.
- [Polar](https://github.com/polarsource/polar): a billing platform. Its merchant dashboard assistant is a Pydantic AI agent, and its organization review is evaluated with Pydantic Evals.
- [Stirling PDF](https://github.com/Stirling-Tools/Stirling-PDF): a PDF editing application. Its AI engine builds its agents and models on Pydantic AI.
- [LTX Desktop](https://github.com/Lightricks/LTX-Desktop) by **Lightricks**: a desktop app for generating videos with LTX models. Its assistant, which finds assets and queues generation jobs, is a Pydantic AI agent.
- [Baserow](https://github.com/baserow/baserow): a no-code platform for databases, automations and apps. Its AI assistant is a Pydantic AI agent, and its AI fields call models through Pydantic AI.
- [Galaxy](https://github.com/galaxyproject/galaxy): the Galaxy Project's open platform for data-intensive biomedical research. Its AI assistant's agents, such as the router, tool error analysis and page assistant, are built on Pydantic AI and evaluated with Pydantic Evals.
- [Prebid Sales Agent](https://github.com/prebid/salesagent) by **Prebid.org**: an implementation of the Ad Context Protocol sales agent, which lets AI agents buy advertising inventory. Its policy checks, creative review and product ranking agents are built on Pydantic AI.
- [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) by the **AI4Finance Foundation**: an agent platform for financial research. Its desktop research workstation runs its multi-agent workflows on Pydantic AI, streamed through the Vercel AI UI adapter.
- [Agent Foundation](https://github.com/converge-ai-labs/agent-foundation) by **Converge AI**: a self-hosted platform and library for building and running agent systems, with managed agents, memory, sandboxes, computer use and durable execution. Its a13n Harness runtime is built on Pydantic AI, with its own capabilities and agent specs.
- [AgenticOS](https://github.com/vstorm-co/agenticos) by **[Vstorm](https://vstorm.co/)**: a self-hosted platform for building, running and governing a company's agents. Its agent layer is built on Pydantic AI, with Harness capabilities such as compaction, guardrails and planning.
- [ThreatAtlas](https://github.com/OWASP/www-project-threatatlas) by **OWASP**: a collaborative threat modeling platform. Its AI assistant is a Pydantic AI agent.
- [Nemesis](https://github.com/SpecterOps/Nemesis) by **SpecterOps**: an offensive data enrichment pipeline. Its agents service runs Pydantic AI agents for tasks such as credential analysis and summarization, plus a chatbot that uses tools over MCP.
- [Gramps Web API](https://github.com/gramps-project/gramps-web-api): the backend of Gramps Web, a genealogy application. Its chat assistant is a Pydantic AI agent with tools for querying the family tree.
- [Phoenix](https://github.com/Arize-ai/phoenix) by **Arize**: an AI observability and evaluation platform. Its in-app assistant is a Pydantic AI agent with custom capabilities, MCP tools and the Vercel AI UI types.

## Coding agents and developer tools

- [Conductor](https://github.com/microsoft/conductor) by **Microsoft**: a CLI for defining and running multi-agent workflows. Pydantic AI is one of its agent providers, with Harness compaction.
- [OSS-Fuzz MCP](https://github.com/google/oss-fuzz/tree/master/infra/experimental/mcp) by **Google**: OSS-Fuzz's experimental agent tooling for fuzzing tasks, such as fixing builds, kept apart from its production fuzzing infrastructure. Its MCP client agent runs on Pydantic AI.
- [Code Puppy](https://github.com/mpfaffenberger/code_puppy) by **Walmart**: a coding agent for the terminal, built at Walmart Global Tech on Pydantic AI and the Harness and used across the company. Its creator has since joined Pydantic.
- [moyai](https://github.com/BerriAI/moyai) by **BerriAI**: a self-hosted coding agent for background work. It runs on Pydantic AI, with Harness capabilities such as planning, ask-user questions and conversation search.
- [Colin](https://github.com/PrefectHQ/colin) by **Prefect**: a context engine that compiles agent skills from live sources and rebuilds them when they go stale. Its LLM blocks run on Pydantic AI agents.
- [Tiger Agents for Work](https://github.com/timescale/tiger-agents-for-work) by **Tiger Data**: a library and CLI for Slack-native agents, which runs Tiger Data's own internal Slack agent. Its agents run on Pydantic AI, with MCP servers and Harness subagents.
- [oterm](https://github.com/ggozad/oterm): a terminal client for LLMs. Its agents run on Pydantic AI, with Harness file system access, compaction and memory.
- [angr](https://github.com/angr/angr): a binary analysis platform. Its optional LLM-assisted decompiler uses Pydantic AI agents to suggest variable names, function names and types.
- [Code-Graph-RAG](https://github.com/vitali87/code-graph-rag): a knowledge-graph RAG system for querying and editing multi-language codebases. Its query and editing agents run on Pydantic AI.
- [CodeWiki](https://github.com/FSoft-AI4Code/CodeWiki) by **FSoft AI4Code**: a research framework for generating repository-level documentation. Its documentation agents run on Pydantic AI.

## Domain agents and workflows

- [GOV.UK AI Graph Tools](https://github.com/alphagov/govuk-ai-graph-tools) by the **UK Government Digital Service**: a proof of concept for finding duplicate and outlier content across GOV.UK. Its content extractor is a Pydantic AI agent on Amazon Bedrock.
- [Orcha](https://github.com/inveniosoftware/orcha) by **CERN**'s Invenio project: AI workflows for research data repositories. Its metadata extraction and funding relevance workflows run on Pydantic AI agents.
- [Our World in Data ETL](https://github.com/owid/etl) by **Our World in Data**: the pipeline that loads and transforms its data. Its chart critic asks a Pydantic AI agent whether a published chart would mislead readers.
- [Funes](https://github.com/opensanctions/funes) by **OpenSanctions**: a research project for turning public websites into structured lists of people and the positions they hold. Its discovery and extraction agents run on Pydantic AI and are evaluated with Pydantic Evals.
- [Aurelian](https://github.com/monarch-initiative/aurelian) by the **Monarch Initiative**: agents for biomedical literature, ontologies and knowledge bases, built on Pydantic AI.
- [Geodini](https://github.com/developmentseed/geodini) by **Development Seed**: a natural language geocoding API over Overture data. Its query routing and geocoding agents are built on Pydantic AI.
- [Prediction Market Agent](https://github.com/gnosis/prediction-market-agent) by **Gnosis**: agents that bet on prediction markets such as Omen. Many of its agents are built on Pydantic AI.
- [TimeCopilot](https://github.com/TimeCopilot/timecopilot): a forecasting agent that combines LLMs with time series foundation models. Its forecasting and query agents run on Pydantic AI.
- [RPent](https://github.com/RLinf/RPent) by **RLinf**: agentic infrastructure for the physical world. Its planner runs on Pydantic AI, with Harness compaction.

## Data and infrastructure integrations

- [Apache Airflow `common.ai`](https://github.com/apache/airflow/tree/main/providers/common/ai): the AI provider for the Airflow workflow orchestration platform. It adds Pydantic AI hooks, connections and an agent operator that runs Pydantic AI agents with the Harness.
- [dlt](https://github.com/dlt-hub/dlt): dltHub's Python library for loading data. Its workspace deployment agents run on Pydantic AI by default, with MCP tools.
- [Astronomer Cosmos](https://github.com/astronomer/astronomer-cosmos) by **Astronomer**: a library for running dbt projects as Airflow DAGs. Its optional AI diagnosis of failed dbt tasks is a Pydantic AI agent, run through Airflow's `common.ai` provider.
- [Vizro](https://github.com/mckinsey/vizro) by **McKinsey**: a low-code toolkit for data visualization apps. Its experimental dashboard chat agent, in `vizro-experimental`, runs on Pydantic AI.
- [PowerContext](https://github.com/oceanbase/powercontext) by **OceanBase**: a context and memory server for agents. Its built-in inference runs on Pydantic AI models and embedders, and it ships a capability that gives Pydantic AI agents its memory.

## Libraries and frameworks

- [Marvin](https://github.com/PrefectHQ/marvin) by **Prefect**: a library for building AI functions and agents. Its agents run on Pydantic AI and work with any Pydantic AI model or MCP server.
- [RD-Agent](https://github.com/microsoft/RD-Agent) by **Microsoft**: an agent framework for automating data-driven research and development. Its MCP-backed helper agents, such as documentation lookup and RAG, are Pydantic AI agents.
- [TypeAgent](https://github.com/microsoft/typeagent-py) by **Microsoft**: a structured RAG library for ingesting, indexing and querying. It creates its chat and embedding models through Pydantic AI.
- [csp-bot](https://github.com/Point72/csp-bot) by **Point72**: a reactive chat bot framework. Its agent commands run Pydantic AI agents with resumable conversation history.
- [PAI Agent SDK](https://github.com/youware-labs/pai-agent-sdk) by **YouWare**: a framework for building agents with Pydantic AI, used to build YouWare's next-generation agent system.
- [Agentic Context Engine](https://github.com/kayba-ai/agentic-context-engine) by **Kayba**: a framework for agents that learn from experience. Its agent, reflector and skill manager roles are Pydantic AI agents.
- [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) by **[Vstorm](https://vstorm.co/)**: a framework and terminal assistant for deep agents, built on Pydantic AI and the Harness.
- [haiku.rag](https://github.com/ggozad/haiku.rag): an agentic RAG library and MCP server for local document search. Its agents and capabilities run on Pydantic AI, and it is evaluated with Pydantic Evals. See [Third-Party Capabilities](capabilities/third-party.md#retrieval-rag) for what its capabilities do.
- [Full-Stack AI Agent Template](https://github.com/vstorm-co/full-stack-ai-agent-template) by **[Vstorm](https://vstorm.co/)**: a FastAPI and Next.js project generator whose generated backends can use Pydantic AI agents.

## Research and evals

- [Prompt Siren](https://github.com/facebookresearch/prompt-siren) by **Meta**: a research workbench for developing and testing prompt injection attacks against LLM agents. The agents it attacks are built on Pydantic AI models and toolsets.
- [ALE-Bench](https://github.com/SakanaAI/ALE-Bench) by **Sakana AI**: a benchmark for long-horizon algorithm engineering. Its LLM evaluation tool runs models through Pydantic AI agents.
- [dbt LLM Semantic Layer Benchmark](https://github.com/dbt-labs/dbt-llm-sl-bench) by **dbt Labs**: a benchmark of LLM query generation against the dbt Semantic Layer. It runs the models through Pydantic AI agents, with the dbt MCP server as a toolset.
- [Consult](https://github.com/i-dot-ai/consult) by **i.AI**, the UK government's AI incubator: an application for analysing public consultation responses. Its evaluation framework uses Pydantic Evals as its runner.
- [Open Food Facts AI](https://github.com/openfoodfacts/openfoodfacts-ai) by **Open Food Facts**: its AI projects. Its `llm-evals` framework evaluates models with Pydantic AI and Pydantic Evals.

## Case studies

Organizations that told their story of using Pydantic AI, Pydantic Evals or [Pydantic Logfire](https://pydantic.dev/logfire) in production, published on [pydantic.dev](https://pydantic.dev/case-studies).

- [Justice AI Unit](https://pydantic.dev/case-studies/justice-ai) of the **UK Ministry of Justice**: the team building AI tools for judges and probation staff already uses Pydantic AI to put contracts around the actions a model proposes, and Pydantic Validation as the contract wherever information changes form.
- [MindsDB](https://pydantic.dev/case-studies/mindsdb): migrated its agents from LangChain to Pydantic AI and got 10x better agent performance in a month.
- [Datalayer](https://pydantic.dev/case-studies/datalayer): moved its agent platform for data science on Jupyter from LangChain to Pydantic AI, with Logfire for observability.
- [Lema AI](https://pydantic.dev/case-studies/lemaai): chose Pydantic AI after evaluating six agent frameworks, and cut its code by 63%.
- [Overjoy](https://pydantic.dev/case-studies/overjoy): replaced LangChain and LangSmith with Pydantic AI and Logfire.
- [Qualio](https://pydantic.dev/case-studies/qualio): ships AI features in a regulated industry with Pydantic AI, gating each release on a Pydantic Evals pass rate that customers can audit.
- [Synera](https://pydantic.dev/case-studies/synera): a text-to-workflow agent on Pydantic AI that turns natural language prompts into engineering workflows.
- [Mixam](https://pydantic.dev/case-studies/mixam): a Pydantic AI agent that guides self-publishing customers through complex printing options.
- [ARIJ](https://pydantic.dev/case-studies/arij): a bilingual chatbot on Pydantic AI that trains journalists across 22 countries.
- [Schmitt-Thompson Clinical Content](https://pydantic.dev/case-studies/stcc): an agentic RAG system on Pydantic AI for nurse triage guidelines, built with [Vstorm](https://vstorm.co/).
- [General Intelligence Company](https://pydantic.dev/case-studies/gic): real-time evaluations and self-correcting autonomous agents with Pydantic AI and Logfire.
- [Evergreen.ai](https://pydantic.dev/case-studies/evergreenai): observability for agents in financial services with Logfire, and agent quality evaluated with Pydantic Evals.
- [Sophos](https://pydantic.dev/case-studies/sophos): end-to-end tracing, alerts and LLM experiments for its SecOps AI agents with Logfire.
- [Boosted.ai](https://pydantic.dev/case-studies/boostedai): monitors more than 50,000 AI investment research workflows with Logfire.
- [Dosu](https://pydantic.dev/case-studies/dosu): debugs its agent runs from its coding agent with Logfire and its MCP server.
- [AutonomyAI](https://pydantic.dev/case-studies/autonomyai): its agents query Logfire to catch their own regressions in production.

## Adding your project

Open a pull request adding it to [`docs/built-with.md`](https://github.com/pydantic/pydantic-ai/blob/main/docs/built-with.md): one line in the matching group, with a link to the repository, saying what part of the project runs on Pydantic AI. Within a group, the projects readers are most likely to know come first; name the maintaining organization in bold when it differs from the project name. A project is listed when it meets all of these:

- Its source is public and open source, and it has had a commit in the past year.
- Pydantic AI, the Harness or Pydantic Evals runs a real part of the project itself, such as its agents, its assistant or its evals. Integrations that let others use Pydantic AI with a product, examples and tutorials don't count.
- It is maintained by a well-known organization (a company, foundation, government or research institution), or it has at least 500 stars on GitHub.

A reusable capability package for Pydantic AI agents belongs on [Third-Party Capabilities](capabilities/third-party.md) instead.
