---
title: Integrations
description: "Pydantic AI Harness capabilities that connect an agent to third-party services: GitHub, Linear, Slack, Notion, Google Workspace, PostHog, Logfire and more."
---

# Integrations

Each integration is a [Pydantic AI Harness](index.md) capability that gives an agent the tools of one outside service.
Most connect to the service's own hosted MCP server, and the credential you pass decides what the agent can reach. Add
one to `capabilities=[...]` like any other capability.

## Code and engineering

- [GitHub](github.md): read and change repositories, issues, pull requests, and other GitHub resources.
- [Linear](linear.md): read and change Linear issues, projects, teams, and comments.
- [Macroscope](macroscope.md): run a local [Macroscope](https://docs.macroscope.com/cli) code review and hand the
  findings to the agent.
- [LocalStack](localstack.md): provision and exercise AWS services in an emulated AWS environment, without a real
  account.

## Documents, mail, and chat

- [Google Workspace](google-workspace.md): use Gmail, Calendar, Drive, and other Google Workspace products.
- [Notion](notion.md): search and change content in a Notion workspace.
- [Slack](slack.md): read and send Slack messages, browse channels, and work with canvases.

## Customers, sales, and meetings

- [Day AI](day-ai.md): search and update CRM records, read meeting context, and draft emails in
  [Day AI](https://day.ai).
- [Grain](grain.md): search and read meetings, transcripts, notes, and deals, and create clips and tag meetings, in
  [Grain](https://grain.com).
- [Pylon](pylon.md): work with support issues, accounts, and contacts in [Pylon](https://www.usepylon.com).
- [StackOne](stackone.md): act on a user's linked business applications, such as BambooHR, Salesforce, or Zendesk,
  through [StackOne](https://www.stackone.com).

## Analytics and telemetry

- [PostHog](posthog.md): query product analytics and manage feature flags, experiments, and dashboards in
  [PostHog](https://posthog.com).
- [Logfire MCP](logfire-mcp.md): query [Logfire](https://pydantic.dev/logfire) telemetry and manage Logfire projects.

## Social media

- [Ordinal](ordinal.md): draft, schedule, and analyze social posts in [Ordinal](https://www.tryordinal.com).

## A service not listed here

If the service runs an MCP server, connect it with the core [MCP capability](../capabilities/mcp.md). For tools from
other libraries, see [Third-Party Tools](../third-party-tools.md).
