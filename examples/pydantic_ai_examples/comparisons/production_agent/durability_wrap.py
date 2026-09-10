"""Durability is a wrapper, not a rewrite (requires an engine to run).

The agent definition below is identical for all three engines; only the
wrapper import changes. This file is a reference, not an offline proof:
run the engine versions in an environment with Temporal/DBOS/Prefect.
"""
from pydantic_ai import Agent

# one agent, unchanged:
def build_agent() -> Agent:
    return Agent('openai:gpt-5.6-luna', deps_type=dict, system_prompt='be terse')


def main() -> None:
    print('same agent source under:')
    print('  - TemporalAgent(agent, task_queue="tq")   # Temporal')
    print('  - DBOSAgent(agent, workflow_name="wf")    # DBOS')
    print('  - PrefectAgent(agent, task_name="t")      # Prefect')
    print('agent definition changes: 0 lines')


if __name__ == '__main__':
    main()
