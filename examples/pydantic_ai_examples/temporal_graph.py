"""Example demonstrating pydantic-graph integration with Temporal workflows.

This example shows how pydantic-graph graphs "just work" inside Temporal workflows,
with TemporalAgent handling model requests and tool calls as durable activities.

The example implements a research workflow that:
1. Breaks down a complex question into simpler sub-questions
2. Researches each sub-question in parallel
3. Synthesizes the results into a final answer

To run this example:
1. Start Temporal server locally:
   ```sh
   brew install temporal
   temporal server start-dev
   ```

2. Run this script:
   ```sh
   uv run python examples/pydantic_ai_examples/temporal_graph.py
   ```
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from pydantic import BaseModel
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import (
    AgentPlugin,
    PydanticAIPlugin,
    TemporalAgent,
)
from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_extend

# ============================================================================
# State and Dependencies
# ============================================================================


@dataclass
class ResearchState:
    """State that flows through the research graph."""

    original_question: str
    sub_questions: list[str] | None = None
    sub_answers: list[str] | None = None
    final_answer: str | None = None


@dataclass
class ResearchDeps:
    """Dependencies for the research workflow (must be serializable for Temporal)."""

    max_sub_questions: int = 3


# ============================================================================
# Output Models
# ============================================================================


class SubQuestions(BaseModel):
    """Model for breaking down a question into sub-questions."""

    sub_questions: list[str]


class Answer(BaseModel):
    """Model for a research answer."""

    answer: str
    confidence: float


# ============================================================================
# Agents
# ============================================================================

# Agent that breaks down complex questions into simpler sub-questions
question_breaker_agent = Agent(
    'openai:gpt-5-mini',
    name='question_breaker',
    instructions=(
        'You are an expert at breaking down complex questions into simpler, '
        'more focused sub-questions that can be researched independently. '
        'Create questions that cover different aspects of the original question.'
    ),
    output_type=SubQuestions,
)

# Agent that researches individual questions
researcher_agent = Agent(
    'openai:gpt-5-mini',
    name='researcher',
    instructions=(
        'You are a research assistant. Provide clear, accurate, and concise answers '
        'to questions based on your knowledge. Include confidence level in your response.'
    ),
    output_type=Answer,
)

# Agent that synthesizes multiple answers into a comprehensive final answer
synthesizer_agent = Agent(
    'openai:gpt-5-mini',
    name='synthesizer',
    instructions=(
        'You are an expert at synthesizing multiple pieces of information into '
        'a coherent, comprehensive answer. Combine the provided answers while '
        'maintaining accuracy and clarity.'
    ),
)

# Wrap all agents with TemporalAgent for durable execution
temporal_question_breaker = TemporalAgent(question_breaker_agent)
temporal_researcher = TemporalAgent(researcher_agent)
temporal_synthesizer = TemporalAgent(synthesizer_agent)


# ============================================================================
# Graph Definition using Beta API
# ============================================================================

# Create the graph builder
g = GraphBuilder(
    name='research_workflow',
    state_type=ResearchState,
    deps_type=ResearchDeps,
    input_type=str,  # Takes a question string as input
    output_type=str,  # Returns final answer as string
    auto_instrument=True,
)


# Step 1: Break down the question into sub-questions
@g.step(node_id='break_down_question', label='Break Down Question')
async def break_down_question(
    ctx: StepContext[ResearchState, ResearchDeps, str],
) -> ResearchState:
    """Break down the original question into sub-questions using an agent."""
    question = ctx.inputs

    # Use the TemporalAgent to break down the question
    result = await temporal_question_breaker.run(
        f'Break down this question into {ctx.deps.max_sub_questions} simpler sub-questions: {question}',
    )

    # Update state with sub-questions
    return ResearchState(
        original_question=question,
        sub_questions=result.output.sub_questions,
    )


# Step 2: Research each sub-question (will run in parallel via map)
@g.step(node_id='research_sub_question', label='Research Sub-Question')
async def research_sub_question(
    ctx: StepContext[ResearchState, ResearchDeps, str],
) -> str:
    """Research a single sub-question using an agent."""
    sub_question = ctx.inputs

    # Use the TemporalAgent to research the sub-question
    result = await temporal_researcher.run(sub_question)

    # Return the answer as a formatted string
    return f'**Q: {sub_question}**\nA: {result.output.answer} (Confidence: {result.output.confidence:.0%})'


# Step 3: Join all research results
research_join = g.join(
    reducer=reduce_list_extend,
    initial=list[str](),
)


# Step 4: Synthesize all answers into a final answer
@g.step(node_id='synthesize_answer', label='Synthesize Answer')
async def synthesize_answer(
    ctx: StepContext[ResearchState, ResearchDeps, list[str]],
) -> ResearchState:
    """Synthesize all research results into a final comprehensive answer."""
    research_results = ctx.inputs

    # Format the research results for the synthesizer
    research_summary = '\n\n'.join(research_results)

    # Use the TemporalAgent to synthesize the final answer
    result = await temporal_synthesizer.run(
        f'Original question: {ctx.state.original_question}\n\n'
        f'Research findings:\n{research_summary}\n\n'
        'Please synthesize these findings into a comprehensive answer to the original question.',
    )

    # Update state with final answer
    state = ctx.state
    state.sub_answers = research_results
    state.final_answer = result.output

    return state


# Build the graph with edges
g.add(
    # Start -> Break down question
    g.edge_from(g.start_node).to(break_down_question),
    # Break down -> Map over sub-questions for parallel research
    g.edge_from(break_down_question)
    .transform(lambda ctx: ctx.inputs.sub_questions or [])
    .map()
    .to(research_sub_question),
    # Research results -> Join
    g.edge_from(research_sub_question).to(research_join),
    # Join -> Synthesize
    g.edge_from(research_join).to(synthesize_answer),
    # Synthesize -> End
    g.edge_from(synthesize_answer)
    .transform(lambda ctx: ctx.inputs.final_answer or '')
    .to(g.end_node),
)

# Build the final graph
research_graph = g.build()


# ============================================================================
# Temporal Workflow
# ============================================================================


@workflow.defn
class ResearchWorkflow:
    """Temporal workflow that executes the research graph with durable execution."""

    @workflow.run
    async def run(self, question: str, deps: ResearchDeps | None = None) -> str:
        """Run the research workflow on a question.

        Args:
            question: The question to research
            deps: Optional dependencies for the workflow

        Returns:
            The final synthesized answer
        """
        if deps is None:
            deps = ResearchDeps()

        # Execute the pydantic-graph graph - it "just works" in Temporal!
        result = await research_graph.run(
            state=ResearchState(original_question=question),
            deps=deps,
            inputs=question,
        )

        return result


# ============================================================================
# Main Execution
# ============================================================================


async def main():
    """Main function to set up worker and execute the workflow."""
    # Monkeypatch uuid.uuid4 to use Temporal's deterministic UUID generation
    # This is necessary because pydantic-graph uses uuid.uuid4 internally for task IDs
    # Connect to Temporal server
    client = await Client.connect(
        'localhost:7233',
        plugins=[PydanticAIPlugin()],
    )

    # Create a worker that will execute workflows and activities
    async with Worker(
        client,
        task_queue='research',
        workflows=[ResearchWorkflow],
        plugins=[
            # Register activities for all three temporal agents
            AgentPlugin(temporal_question_breaker),
            AgentPlugin(temporal_researcher),
            AgentPlugin(temporal_synthesizer),
        ],
    ):
        # Execute the workflow
        question = 'What are the key factors that contributed to the success of the Apollo 11 moon landing?'

        print(f'\n{"=" * 80}')
        print(f'Research Question: {question}')
        print(f'{"=" * 80}\n')

        output = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
            ResearchWorkflow.run,
            args=[question],
            id=f'research-{uuid.uuid4()}',
            task_queue='research',
        )

        print(f'\n{"=" * 80}')
        print('Final Answer:')
        print(f'{"=" * 80}\n')
        print(output)
        print(f'\n{"=" * 80}\n')


if __name__ == '__main__':
    import logfire

    logfire.instrument_pydantic_ai()
    logfire.configure(send_to_logfire=False)

    asyncio.run(main())
