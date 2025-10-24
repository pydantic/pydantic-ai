Medical triage and delegation system built with **Pydantic AI**, demonstrating how an orchestrator agent (`triage_agent`) coordinates multiple specialized agents (e.g. cardiology, neurology, and senior clinician).

Demonstrates:
- [Agent delegation and coordination](../multi-agent-applications.md#agent-delegation)
- [structured `output_type`](../output.md#structured-output)
- [tools](../tools.md)

---

## Overview

This example shows how to use **multiple Pydantic AI agents** to simulate a medical triage workflow.

The system includes:
- **General Practitioner, Cardiology, and Neurology agents** — for Level 1 consultation.
- **Senior Doctor agent** — for escalations and treatment planning.
- **Triage Agent (Coordinator)** — which decides which tool to invoke and when to escalate.

The `triage_agent` uses two tools:
1. `consult_specialist` — routes the complaint to a domain specialist.
2. `consult_senior_doctor` — escalates the case for critical or ambiguous scenarios.

Each specialist produces a structured `MedicalReport`, and the senior doctor produces a structured `TreatmentPlan`.
The orchestrator then compiles both into a final `TriageFinalOutput`.

---

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python -m pydantic_ai_examples.medical_agent_delegation

Make sure to set a valid **Cohere API key** or replace the model reference:

```bash
export CO_API_KEY="your-cohere-api-key"
```

You may also switch to an OpenAI or Anthropic model if preferred:

```python
MODEL = "openai:gpt-4o"
```
