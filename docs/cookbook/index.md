---
title: Cookbook
description: Copyable, tested recipes for common Pydantic AI application tasks.
---

# Cookbook

The cookbook contains small, complete solutions to application problems. Start with a recipe when you know the outcome you need; use the guides when you want to understand an API in depth.

Every recipe is executed in the documentation test suite with model and service boundaries replaced by deterministic test implementations. Before using a recipe in production, run it with your chosen model and evaluate it against representative inputs.

## Start here

These recipes form a progressive introduction to the core agent loop:

1. [Extract structured data from text](structured-extraction.md) to receive validated application data.
2. [Give tools typed application context](typed-dependencies.md) to connect an agent to your services.
3. [Stream a response](stream-text.md) when users should see output immediately.
4. [Continue a stored conversation](stored-history.md) across requests or process restarts.
5. [Test an agent without model requests](testing.md) to keep application tests deterministic.

## Build reliable behavior

- [Retry output that fails a business rule](output-retry.md)
- [Put a hard limit on tool execution](usage-limits.md)
- [Fall back when a model provider fails](model-fallback.md)
- [Require approval for a sensitive tool](tool-approval.md)
- [Keep conversation history within a fixed window](bounded-history.md)
- [Evaluate agent behavior in CI](evaluate-agent.md)

## Connect application data and workflows

- [Extract structured data from a PDF](document-extraction.md)
- [Answer analytics questions without exposing arbitrary SQL](safe-sql-analyst.md)
- [Answer from private documents with citations](rag-citations.md)
- [Redact personal data before a model request](redact-pii.md)
- [Review a change with parallel specialists](parallel-review.md)

Each recipe recommends one bounded pattern. It deliberately links to the relevant guide instead of duplicating every configuration option or provider variation.
