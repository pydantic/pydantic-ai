# LLM-as-a-Judge: The Complete Guide with Pydantic

Evaluating LLM outputs at scale is one of the hardest problems in AI engineering. Human evaluation doesn't scale. Traditional metrics like BLEU and ROUGE miss semantic quality entirely. You need something that understands language, applies nuanced criteria, and runs automatically.

Enter LLM-as-a-Judge: using one language model to evaluate another's output.

This guide covers everything you need to implement LLM-as-a-Judge effectively, including a critical distinction most content misses—the difference between one-size-fits-all and case-specific evaluators. We'll show you when to use each, how to write effective rubrics, and how to integrate evaluation into your development workflow with working code throughout.

## What You'll Learn

- What LLM-as-a-Judge is and why it works
- The critical distinction between one-size-fits-all and case-specific evaluators
- When to use (and when NOT to use) LLM judges
- How to write effective evaluation rubrics
- Development workflows for continuous improvement
- Production monitoring patterns

All examples use [pydantic-evals](https://ai.pydantic.dev/evals/) for implementation and [Logfire](https://pydantic.dev/logfire) for observability.

## TL;DR

If you're short on time, here are the key takeaways:

- **LLM judges work because evaluation is easier than generation.** The judge sees both the question and answer, narrowing the task significantly.
- **Case-specific evaluators outperform generic ones for test suites.** If an LLM could reliably assess quality without case context, it could probably generate good responses in the first place.
- **Combine deterministic checks (fast) with LLM judges (expensive) strategically.** Run type validation and format checks first; save LLM evaluation for semantic quality.
- **Always request reasoning.** It's essential for debugging failures and iterating on rubrics.

---

## What is LLM-as-a-Judge?

LLM-as-a-Judge uses a language model to assess the quality of another LLM's output. Instead of relying on exact string matches or statistical metrics, you give the judge a rubric—a description of what "good" looks like—and it returns a verdict.

The pattern works like this:

1. Your task receives an input and produces an output
2. The judge receives the output (and optionally the input and expected answer)
3. The judge applies your rubric to assess quality
4. The judge returns a verdict: pass/fail, a score, or a label

Judges can return three types of outputs:

- **Assertions** (boolean): Pass/fail decisions. "Does the response answer the question?"
- **Scores** (numeric): Quality on a scale. "Rate helpfulness from 0.0 to 1.0"
- **Labels** (categorical): Classifications. "Is the tone formal, casual, or inappropriate?"

Here's the simplest possible example with pydantic-evals:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

dataset = Dataset(
    cases=[
        Case(
            name='password_reset_help',
            inputs='How do I reset my password?',
        ),
    ],
    evaluators=[
        LLMJudge(rubric='The response provides clear, actionable instructions'),
    ],
)

async def support_bot(query: str) -> str:
    # In a real application, this would call your LLM
    return "To reset your password, click 'Forgot Password' on the login page, enter your email, and follow the link we send you."

async def main():
    report = await dataset.evaluate(support_bot)
    report.print()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```

The `LLMJudge` evaluator sends your rubric and the task output to an LLM (GPT-4o by default), which returns whether the output passes and an explanation of its reasoning.

---

## Why Does LLM-as-a-Judge Work?

The core insight is that **evaluation is easier than generation**.

When generating a response, the LLM faces an enormous solution space. It must satisfy many constraints simultaneously: be accurate, be helpful, match the right tone, stay on topic, avoid harmful content, and more. The model doesn't know what it doesn't know.

When evaluating, the task is much narrower. The judge sees both the question AND the answer. It only needs to assess one dimension at a time. Given a specific rubric like "Does this response answer the user's question?", the judge has clear success criteria.

Research supports this. Studies on MT-Bench and Chatbot Arena show 80%+ agreement between LLM judges and human preferences on many tasks. The agreement is highest for clear-cut criteria and lower for subjective judgments—which is exactly what you'd expect.

Perhaps most surprisingly, **the same model that generated a response can often catch its own mistakes** when asked to evaluate. This works because generation and evaluation are fundamentally different tasks. A model might hallucinate a fact during generation (it doesn't know what it doesn't know), but when you ask it to verify that fact against source material, it can compare claim to evidence—a much simpler task.

---

## The Critical Distinction: One-Size-Fits-All vs. Case-Specific Evaluators

This is the key insight most content about LLM-as-a-Judge misses entirely.

### One-Size-Fits-All Evaluators

A **one-size-fits-all evaluator** uses a single rubric applied uniformly to every test case:

```python
dataset = Dataset(
    cases=[
        Case(name='billing_question', inputs={'query': 'Why was I charged twice?'}),
        Case(name='feature_request', inputs={'query': 'Can you add dark mode?'}),
        Case(name='bug_report', inputs={'query': 'The app crashes when I upload photos'}),
    ],
    evaluators=[
        # This evaluator runs on ALL cases with the same rubric
        LLMJudge(
            rubric='Response is professional, empathetic, and does not blame the user',
            include_input=True,
        ),
    ],
)
```

The rubric must be general enough to apply everywhere. This works well for universal quality dimensions—professionalism, safety, tone—that genuinely apply to all outputs.

### Case-Specific Evaluators

A **case-specific evaluator** has a rubric tailored to an individual test case:

```python
dataset = Dataset(
    cases=[
        Case(
            name='vegetarian_recipe',
            inputs={'request': 'I need a vegetarian dinner recipe'},
            evaluators=[
                LLMJudge(
                    rubric='''
                    Recipe must NOT contain:
                    - Meat (beef, pork, chicken, fish, etc.)
                    - Meat-based broths or stocks
                    - Gelatin or other animal-derived ingredients
                    PASS only if the recipe is fully vegetarian.
                    ''',
                    include_input=True,
                ),
            ],
        ),
        Case(
            name='quick_weeknight_meal',
            inputs={'request': 'I need something I can make in under 30 minutes'},
            evaluators=[
                LLMJudge(
                    rubric='''
                    Recipe must:
                    - Have total prep + cook time under 30 minutes
                    - Use commonly available ingredients
                    - Not require specialized equipment
                    FAIL if the recipe would realistically take longer than 30 minutes.
                    ''',
                    include_input=True,
                ),
            ],
        ),
        Case(
            name='allergy_safe',
            inputs={'request': 'Nut-free dessert for a school event'},
            evaluators=[
                LLMJudge(
                    rubric='''
                    Recipe must NOT contain:
                    - Tree nuts (almonds, walnuts, pecans, etc.)
                    - Peanuts or peanut-derived ingredients
                    - "May contain nuts" warnings should be noted as a concern
                    This is for a child with allergies. FAIL if any nut risk exists.
                    ''',
                    include_input=True,
                ),
            ],
        ),
    ],
    # You can still have universal evaluators that run on all cases
    evaluators=[
        LLMJudge(
            rubric='Recipe instructions are clear and easy to follow',
            include_input=True,
        ),
    ],
)
```

Notice how each case has requirements that would be impossible to capture in a single universal rubric. The vegetarian case doesn't care about cooking time; the quick meal case doesn't care about allergens. Case-specific evaluators let you express exactly what matters for each scenario.

### A Note on Workarounds

Some evaluation frameworks only support one-size-fits-all evaluators at the dataset level. In these systems, you *can* technically achieve case-specific behavior by storing requirements in metadata and writing generic evaluators that conditionally apply logic based on what they find.

This works, but it's awkward:

- The evaluator code becomes complex with lots of conditional logic
- The connection between case requirements and evaluation logic is indirect
- Teams often don't realize this pattern is possible, so they default to generic rubrics
- The friction means case-specific evaluation is underutilized in practice

First-class support for case-specific evaluators—where each case can declare its own evaluators directly—makes this pattern obvious and natural. When the framework supports it natively, teams actually use it.

### The Key Insight

Here's the crucial realization:

> If an LLM were good enough to assess quality reliably across all cases without context, it would likely be good enough to generate good responses in the first place.

Case-specific evaluators sidestep this limitation by providing the context the judge needs to make an accurate assessment.

Think of it this way:
- One-size-fits-all: "Is this a good essay?" (vague, hard to judge)
- Case-specific: "Does this essay argue that renewable energy is cost-effective, using at least three economic studies?" (clear, verifiable)

### When to Use Each Approach

| Scenario | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| Online evaluation (real-time production) | One-size-fits-all | No case-specific context available at runtime |
| Building a test suite for your agent | Case-specific preferred | Capture nuanced expectations per scenario |
| Universal quality checks (tone, safety) | One-size-fits-all | These truly apply to all outputs |
| Regression testing after changes | Case-specific | Verify specific behaviors are preserved |

### When One-Size-Fits-All Still Works

Two exceptions where generic evaluators perform well:

1. **Detection vs. generation asymmetry**: Some problems are easy to detect but hard to solve. Detecting that code won't compile is easier than writing correct code. Though often, the agent itself could do this check.

2. **Model cost arbitrage**: Using a more capable model (GPT-4o) to judge a cheaper production model (GPT-4o-mini). The capability gap can justify the generic approach.

---

## Types of LLM Judges

### Pairwise Comparison

Pairwise comparison presents two outputs and asks the judge to select the better one. This is useful for A/B testing prompts or models during development.

```python
# Pairwise comparison requires a custom evaluator
# since you need to generate two responses and compare them
```

Pairwise judgments are often more reliable than absolute scores because relative comparisons are easier than absolute assessments. However, they require generating two outputs per evaluation, doubling your cost.

### Evaluation by Criteria (Reference-Free)

This is the most common type: score an output on specific dimensions without a reference answer.

The key is to evaluate one dimension per judge:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

dataset = Dataset(
    cases=[
        Case(
            name='explain_concept',
            inputs='Explain how photosynthesis works',
        ),
    ],
    evaluators=[
        LLMJudge(
            rubric='The explanation is scientifically accurate',
            assertion={'evaluation_name': 'accuracy', 'include_reason': True},
        ),
        LLMJudge(
            rubric='The explanation is understandable to a middle school student',
            assertion={'evaluation_name': 'clarity', 'include_reason': True},
        ),
        LLMJudge(
            rubric='The explanation is concise without sacrificing completeness',
            assertion={'evaluation_name': 'conciseness', 'include_reason': True},
        ),
    ],
)
```

Why separate judges? Clearer rubrics lead to more consistent judgments. It's easier to identify which dimension is failing. And you can weight dimensions differently when aggregating results.

### Reference-Based Evaluation

Reference-based evaluation compares the output against something: an expected answer, the original question, retrieved context, or source material.

**Hallucination detection** is a particularly important use case. You provide source documents and ask the judge whether the response stays grounded in them:

```python
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

class RAGInput(BaseModel):
    question: str
    retrieved_context: str

dataset = Dataset(
    cases=[
        Case(
            name='policy_question',
            inputs=RAGInput(
                question='What is the return policy?',
                retrieved_context='''
                Our return policy:
                - Items can be returned within 30 days of purchase
                - Items must be unused and in original packaging
                - Refunds are processed within 5-7 business days
                - Sale items are final sale and cannot be returned
                ''',
            ),
            evaluators=[
                LLMJudge(
                    rubric='''
                    The response must ONLY contain information from the retrieved context.
                    Check for:
                    1. Any facts not present in the context (hallucination)
                    2. Any numbers or timeframes that differ from the context
                    3. Any policies or claims not stated in the context
                    FAIL if the response includes ANY information not in the context.
                    ''',
                    include_input=True,
                    assertion={'evaluation_name': 'grounded', 'include_reason': True},
                ),
            ],
        ),
    ],
)
```

The `include_input=True` parameter is crucial here—it gives the judge access to the retrieved context so it can verify groundedness.

---

## Where LLM Judges Excel

### Groundedness and Hallucination Detection

LLM judges are particularly effective at detecting hallucinations because the task is well-defined: "Is claim X supported by source Y?" Even the model that generated the response can often catch its own hallucinations when explicitly asked to compare against source material.

```python
LLMJudge(
    rubric='''
    Verify that the response is fully grounded in the provided source documents.

    Check for these types of hallucinations:
    1. INVENTED FACTS: Claims that appear nowhere in the sources
    2. WRONG NUMBERS: Statistics, dates, or quantities that differ from sources
    3. EXTRAPOLATIONS: Conclusions or implications not directly stated
    4. ENTITY CONFUSION: Mixing up names, places, or other entities

    PASS: Every factual claim can be traced to the source documents
    FAIL: Any unsupported claim, even if it seems plausible

    Be strict. "Probably true" is not the same as "stated in sources."
    ''',
    include_input=True,
    assertion={'evaluation_name': 'no_hallucination', 'include_reason': True},
)
```

### Hard-to-Generalize Quality Rules

Some quality criteria are easy to articulate for a specific case but hard to write as universal rules. For example:

- "This coding question needs a solution that handles the edge cases the user mentioned"
- "This customer is frustrated, so the response needs to be especially empathetic"
- "This is a follow-up question, so the response shouldn't repeat context already established"

Case-specific LLM judges handle these naturally. You describe what's good or bad about *this specific case*, and the judge applies that description.

### Style and Orthogonal Concerns

Style checks cross-cut other quality dimensions and make excellent one-size-fits-all evaluators:

```python
style_evaluators = [
    LLMJudge(
        rubric='Response uses second person ("you", "your") instead of third person ("the user", "they")',
        assertion={'evaluation_name': 'uses_second_person'},
    ),
    LLMJudge(
        rubric='Response does not start with "I" or make the assistant the subject of the first sentence',
        assertion={'evaluation_name': 'not_self_focused'},
    ),
    LLMJudge(
        rubric='''
        Response avoids corporate buzzwords and jargon including:
        - "leverage", "synergy", "paradigm", "holistic"
        - "circle back", "move the needle", "low-hanging fruit"
        - Unnecessary acronyms without explanation
        ''',
        assertion={'evaluation_name': 'no_jargon'},
    ),
]
```

These are clear, binary criteria that apply universally and are hard to check programmatically (regex would be brittle).

---

## Where NOT to Use LLM Judges

### When Deterministic Checks Suffice

If you can check it with code, check it with code:

- **Type validation**: Use Pydantic models, not LLM judges
- **Format validation**: JSON schema, regex patterns
- **Length constraints**: Character or word counts
- **Required fields**: Check for presence programmatically
- **Exact value matching**: String equality, numeric comparisons

Deterministic checks are faster (milliseconds vs. seconds), cheaper (free vs. API calls), more reliable (100% consistent), and easier to debug.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge, IsInstance

dataset = Dataset(
    cases=[
        Case(
            name='refund_request',
            inputs='I want a refund for my order',
        ),
    ],
    evaluators=[
        # Fast, deterministic check runs first
        IsInstance(type_name='str'),

        # Semantic checks that need LLM judgment
        LLMJudge(
            rubric='Response acknowledges the refund request and provides next steps',
            include_input=True,
        ),
    ],
)
```

Structure your evaluators from cheapest to most expensive. If the response isn't even a string, there's no point in running expensive LLM evaluation.

### The Nuance: Brittle vs. Robust Matching

Checking for specific words *feels* deterministic but is often brittle. Consider "Does the response mention the refund policy?"

A naive check like `'refund' in response` fails because:
- Synonyms: "reimbursement", "money back", "credit"
- Negations: "We cannot offer a refund" contains "refund" but means the opposite
- Context: Mentioning refund policy vs. actually offering a refund

For semantic matching, LLM judges are more robust. But sometimes exact matching IS correct—like verifying a specific calculated value appears verbatim, checking for required legal disclaimers, or ensuring a specific product SKU is mentioned.

### When Speed Matters

LLM judges add 1-3 seconds of latency per evaluation. They're inappropriate for real-time validation in hot paths or high-volume checks. Use them for offline batch evaluation, sampled production monitoring (1-5% of traffic), or development-time testing.

---

## Creating Effective LLM Judges

### Writing Good Rubrics

The quality of your rubric determines the quality of your evaluations. Here's the spectrum:

| Level | Example | Problem |
|-------|---------|---------|
| Bad | "Response is good" | What does "good" mean? |
| Okay | "Response is helpful" | Still vague—helpful how? |
| Better | "Response answers the user's question" | More specific, but what counts? |
| Good | "Response directly addresses the question with actionable next steps" | Clear criteria |
| Best | Includes what to look for AND what constitutes failure | Unambiguous |

Here's a rubric evolution:

```python
# Bad: Too vague
LLMJudge(rubric='Response is helpful')

# Better: More specific
LLMJudge(rubric='Response answers the question helpfully')

# Good: Clear criteria with failure conditions
LLMJudge(
    rubric='''
    The response must:
    1. Directly address the user's question (not a tangential topic)
    2. Provide at least one actionable next step
    3. Not require the user to ask follow-up questions to get basic information

    PASS if all three criteria are met.
    FAIL if any criterion is not met, and explain which one(s).
    ''',
    include_input=True,
    assertion={'evaluation_name': 'helpful', 'include_reason': True},
)
```

### Binary Assertions vs. Numeric Scores

**Use binary (pass/fail)** when:
- A clear threshold exists (policy compliance, safety violations)
- You want simple aggregation ("85% of cases passed")
- The dimension is truly binary (contains PII or doesn't)

**Use numeric scores** when:
- Quality exists on a spectrum
- You want to track improvement over time
- You need to rank or compare responses

```python
# Binary assertion
LLMJudge(
    rubric='Response follows company guidelines',
    assertion={'evaluation_name': 'compliant', 'include_reason': True},
    score=False,
)

# Numeric score
LLMJudge(
    rubric='Rate the helpfulness of the response from 0 to 1',
    score={'evaluation_name': 'helpfulness', 'include_reason': True},
    assertion=False,
)

# Both (threshold + granularity)
LLMJudge(
    rubric='''
    Rate the accuracy from 0 to 1.
    PASS if accuracy >= 0.8, FAIL otherwise.
    ''',
    score={'evaluation_name': 'accuracy_score'},
    assertion={'evaluation_name': 'accuracy_pass'},
)
```

### Providing Context to the Judge

Three context levels:

**Output only** (default): Judge sees just the response. Use for style checks and format validation that don't depend on what was asked.

**Input + Output** (`include_input=True`): Judge sees the question and answer. Use for relevance, completeness, and appropriateness checks.

**Input + Output + Expected** (`include_input=True`, `include_expected_output=True`): Judge sees input, actual output, and reference answer. Use for correctness and semantic equivalence.

```python
# Output only: Style check
LLMJudge(
    rubric='Response uses professional language without slang or emojis',
    include_input=False,
    include_expected_output=False,
)

# Input + Output: Relevance check
LLMJudge(
    rubric='Response directly addresses the question that was asked',
    include_input=True,
)

# Input + Output + Expected: Correctness check
LLMJudge(
    rubric='''
    Compare the response to the expected answer.
    PASS if they are semantically equivalent (same meaning, different wording OK).
    FAIL if they contradict or the response is missing key information.
    ''',
    include_input=True,
    include_expected_output=True,
)
```

Give the judge the minimum context needed. More context isn't always better—it can distract or confuse.

### Model Selection and Settings

The default judge model is `openai:gpt-4o`. For cost-sensitive evaluations, `openai:gpt-4o-mini` is effective for clear-cut criteria. For high-stakes evaluations, consider `anthropic:claude-sonnet-4-5`.

Use low temperature (0.0-0.2) for consistency:

```python
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model

# Set globally for all LLMJudge instances
set_default_judge_model('anthropic:claude-sonnet-4-5')

# Or configure per-evaluator
LLMJudge(
    rubric='...',
    model='openai:gpt-4o',
    model_settings=ModelSettings(temperature=0.0),
)
```

### Always Request Reasoning

Reasoning is essential for debugging and iteration:

```python
LLMJudge(
    rubric='Response is accurate',
    assertion={'evaluation_name': 'accurate', 'include_reason': True},
)
```

When a case fails, the reason tells you *why*. This helps you:
- Debug: Understand what went wrong
- Trust: Verify the judge applied the rubric correctly
- Iterate: Improve rubrics based on the judge's interpretation

---

## Development Workflows with LLM Judges

### Building Your Evaluation Dataset

Test cases come from several sources:

- **Production logs**: Real queries users have asked
- **Manual creation**: Cases designed to test specific behaviors
- **Edge case discovery**: Problems you've encountered and fixed
- **User feedback**: Complaints or issues reported by users

The process is iterative:

1. Start with a few cases (5-10)
2. Run your agent, generate outputs
3. Manually review outputs, identify what's good and bad
4. Write rubrics capturing those observations
5. Run evaluation, check for false positives/negatives
6. Adjust rubrics, add new cases, repeat

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

dataset = Dataset(
    cases=[
        Case(
            name='password_reset',
            inputs={'query': 'How do I reset my password?'},
            metadata={'category': 'account', 'priority': 'high'},
            evaluators=[
                LLMJudge(
                    rubric='''
                    Response must:
                    1. Provide password reset instructions OR link
                    2. Mention email verification step
                    NOT acceptable:
                    - Asking for current password (security risk)
                    ''',
                    include_input=True,
                ),
            ],
        ),
        Case(
            name='billing_dispute',
            inputs={'query': 'I was charged twice for my subscription'},
            metadata={'category': 'billing', 'priority': 'high'},
            evaluators=[
                LLMJudge(
                    rubric='''
                    Response must:
                    1. Acknowledge the double charge concern
                    2. Offer to investigate or escalate
                    3. Provide a timeline for resolution
                    NOT acceptable:
                    - Dismissing the concern
                    - Asking the customer to wait without a timeline
                    ''',
                    include_input=True,
                ),
            ],
        ),
    ],
    evaluators=[
        LLMJudge(rubric='Response is professional and empathetic'),
    ],
)

# Save to YAML for version control
dataset.to_file('evals/support_bot.yaml')
```

Commit your dataset files. They're as valuable as your code.

### Using AI to Improve Your Prompts

Show your evaluation results to a coding agent and ask for prompt improvements:

```python
from pydantic_ai import Agent
from pydantic_evals import Dataset

async def main():
    # Load dataset and run evaluation
    dataset = Dataset.from_file('evals/support_bot.yaml')
    report = await dataset.evaluate(support_bot)

    # Use a more capable model to analyze results
    analyzer = Agent('anthropic:claude-sonnet-4-5')

    result = await analyzer.run(f'''
    You are an expert at improving LLM prompts for customer support bots.

    Here is an evaluation dataset with test cases and their rubrics:
    {dataset}

    Here are the evaluation results (including failures and reasons):
    {report.render()}

    Analyze the failing cases and suggest specific changes to the system prompt
    that would help the bot pass these evaluations. Focus on:
    1. What patterns cause failures?
    2. What instructions should be added to the system prompt?
    3. What existing instructions might be causing problems?

    Provide concrete, actionable suggestions.
    ''')

    print(result.output)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```

**Critical warning**: Don't delete the dataset after improving the prompt. Those cases remain valuable for regression testing. You're building a cumulative test suite, not a one-time fix.

### Generating and Pruning Test Cases

Ask AI to identify gaps in coverage:

- "Given this dataset, what edge cases are missing?"
- "Generate 5 cases where the current prompt is likely to fail"
- "What scenarios would be most valuable for human review?"

And to reduce maintenance burden:

- "Which cases test the same behavior?"
- "If I could only keep 10 cases, which would provide the best coverage?"

Always review generated cases before adding them to your suite.

---

## Production Monitoring with LLM Judges

### Online vs. Offline Evaluation

**Offline evaluation** runs on a fixed test dataset before deployment. The goal is catching regressions and validating new prompts. Run it on every change, in CI/CD.

**Online evaluation** runs on sampled production traffic. The goal is monitoring quality in the wild and catching drift. Run it continuously, sampled at 1-5% of traffic.

You need both. Offline catches known failure modes. Online catches unknown unknowns—real user behavior differs from test cases.

### Sampling Production Traffic

LLM evaluation costs $0.01-0.05 per evaluation. Evaluating every request is cost-prohibitive at scale. Sample 1-5% to capture trends without breaking the bank.

```python
import asyncio
import random
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

# Production evaluators (one-size-fits-all for online use)
PRODUCTION_EVALUATORS = [
    LLMJudge(
        rubric='Response is helpful and addresses the user query',
        include_input=True,
        assertion={'evaluation_name': 'helpful'},
    ),
    LLMJudge(
        rubric='Response does not contain harmful, offensive, or inappropriate content',
        assertion={'evaluation_name': 'safe'},
    ),
]

async def handle_request(query: str) -> str:
    """Production request handler with sampled evaluation."""
    response = await support_bot(query)

    # Sample 5% of traffic for evaluation
    if random.random() < 0.05:
        asyncio.create_task(
            evaluate_production_response(query, response)
        )

    return response

async def evaluate_production_response(query: str, response: str):
    """Evaluate a single production response."""
    dataset = Dataset(
        cases=[Case(name='production', inputs=query)],
        evaluators=PRODUCTION_EVALUATORS,
    )

    async def return_response(_: str) -> str:
        return response

    await dataset.evaluate(return_response)
```

### Viewing Results in Logfire

Configure Logfire once, and results flow automatically:

```python
import asyncio
import logfire
from pydantic_evals import Dataset

# Configure Logfire (do this once at app startup)
logfire.configure()

async def main():
    dataset = Dataset.from_file('evals/support_bot.yaml')

    report = await dataset.evaluate(
        support_bot,
        # Add metadata for filtering in Logfire
        metadata={
            'version': '2.1.0',
            'model': 'gpt-4o-mini',
            'prompt_version': 'empathetic-v3',
        },
    )

    report.print()

if __name__ == '__main__':
    asyncio.run(main())
```

In the Logfire UI, you can:
- View a list of experiments with pass rates
- Compare experiments side-by-side (before/after changes)
- Click into individual cases to see full traces
- Filter and sort by any metric

---

## Relationship with User Feedback

### User Feedback as Evaluation Ground Truth

When users report issues, convert them into evaluation cases:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

# User ticket #12345: "Bot denied my refund, but I bought it 2 weeks ago and it's broken"
# User ticket #12389: "Bot kept asking me to repeat my order number even though I gave it"
# User ticket #12401: "Bot response was rude when I asked about shipping delay"

feedback_cases = [
    Case(
        name='refund_eligibility_false_negative',
        inputs={
            'query': 'I bought this 2 weeks ago and it broke. Can I get a refund?',
        },
        metadata={
            'source': 'user_feedback',
            'ticket_id': '12345',
            'issue': 'incorrectly denied refund',
        },
        evaluators=[
            LLMJudge(
                rubric='''
                Context: Purchase was 2 weeks ago (within 30-day return window).
                Product is defective.

                Response MUST:
                1. Confirm the customer is eligible for a refund
                2. Provide return/refund instructions

                FAIL if refund is denied or eligibility is unclear.
                ''',
                include_input=True,
            ),
        ],
    ),
    Case(
        name='order_number_recognition',
        inputs={
            'query': 'My order number is ABC-12345. Where is my package?',
        },
        metadata={
            'source': 'user_feedback',
            'ticket_id': '12389',
            'issue': 'failed to recognize order number',
        },
        evaluators=[
            LLMJudge(
                rubric='''
                The user has provided their order number (ABC-12345).

                Response MUST:
                1. Acknowledge the order number
                2. Provide tracking or status information

                Response must NOT:
                - Ask for the order number again
                - Say the order number is invalid without checking

                FAIL if the bot asks for information already provided.
                ''',
                include_input=True,
            ),
        ],
    ),
    Case(
        name='empathetic_delay_response',
        inputs={
            'query': 'Why is my order taking so long? This is ridiculous.',
        },
        metadata={
            'source': 'user_feedback',
            'ticket_id': '12401',
            'issue': 'response perceived as rude',
        },
        evaluators=[
            LLMJudge(
                rubric='''
                The customer is frustrated about a shipping delay.

                Response MUST:
                1. Acknowledge the frustration empathetically
                2. Apologize for the delay
                3. Provide information about the order status or next steps

                Response must NOT:
                1. Be defensive or dismissive
                2. Blame the customer or external factors without offering help
                3. Use curt or cold language

                FAIL if the response lacks empathy or feels dismissive.
                ''',
                include_input=True,
            ),
        ],
    ),
]
```

Each complaint becomes a test case. The rubric captures what went wrong. Passing this case proves the issue is fixed.

### Building a Continuous Improvement Loop

The cycle:

1. **Collect**: Gather user feedback (ratings, complaints, support tickets)
2. **Convert**: Turn feedback into test cases with rubrics
3. **Evaluate**: Run against current and candidate prompts
4. **Deploy**: Ship improvements that pass new cases without regressing old ones
5. **Repeat**: Continue collecting, never stop adding cases

Over time, your evaluation suite captures the full range of user expectations. New issues are caught because similar patterns are already tested.

---

## Complete Example: Building a Tested Agent

Let's tie everything together with a complete example: a customer support agent with structured output, tools, dependency injection, and pytest integration.

### Agent Definition

```python
# support_agent.py
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

class SupportQuery(BaseModel):
    query: str
    user_id: str
    account_tier: str  # 'free', 'pro', 'enterprise'

class SupportResponse(BaseModel):
    message: str
    suggested_articles: list[str]
    escalate_to_human: bool

@dataclass
class SupportDeps:
    knowledge_base: 'KnowledgeBase'
    order_service: 'OrderService'

support_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=SupportDeps,
    output_type=SupportResponse,
    system_prompt='''
    You are a helpful customer support agent for TechCorp.

    Guidelines:
    - Be friendly and empathetic
    - For enterprise users, always offer to escalate to human support
    - Search the knowledge base before saying you don't know
    - If the user mentions an order, look it up
    - Never share account details without verification
    ''',
)

@support_agent.tool
async def search_knowledge_base(
    ctx: RunContext[SupportDeps],
    query: str,
) -> list[str]:
    """Search the knowledge base for relevant articles."""
    return await ctx.deps.knowledge_base.search(query)

@support_agent.tool
async def lookup_order(
    ctx: RunContext[SupportDeps],
    order_id: str,
) -> dict:
    """Look up order status and details."""
    return await ctx.deps.order_service.get_order(order_id)
```

### Evaluation Dataset

```python
# evals/support_agent_dataset.py
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge, IsInstance
from support_agent import SupportQuery, SupportResponse

support_dataset = Dataset(
    cases=[
        Case(
            name='enterprise_escalation',
            inputs=SupportQuery(
                query='My integration is broken and I need help urgently',
                user_id='user-123',
                account_tier='enterprise',
            ),
            evaluators=[
                LLMJudge(
                    rubric='''
                    This is an enterprise user with an urgent issue.
                    Response MUST offer to escalate to human support.
                    FAIL if escalation is not offered.
                    ''',
                    include_input=True,
                ),
            ],
        ),
        Case(
            name='free_tier_phone_support',
            inputs=SupportQuery(
                query='Can I get phone support?',
                user_id='user-456',
                account_tier='free',
            ),
            evaluators=[
                LLMJudge(
                    rubric='''
                    This is a free tier user asking about phone support.
                    Phone support is only available for Pro and Enterprise.

                    Response MUST:
                    - Explain that phone support requires an upgrade
                    - Be polite and not dismissive
                    - Offer alternative support options

                    FAIL if it implies free users can get phone support.
                    ''',
                    include_input=True,
                ),
            ],
        ),
        Case(
            name='order_status_inquiry',
            inputs=SupportQuery(
                query='Where is my order ORD-789? It was supposed to arrive yesterday.',
                user_id='user-789',
                account_tier='pro',
            ),
            evaluators=[
                LLMJudge(
                    rubric='''
                    User is asking about order ORD-789.

                    Response MUST:
                    - Reference the specific order number
                    - Provide status information
                    - Acknowledge the delay concern empathetically

                    FAIL if the order is not looked up or the response is generic.
                    ''',
                    include_input=True,
                ),
            ],
        ),
    ],
    evaluators=[
        IsInstance(type_name='SupportResponse'),
        LLMJudge(
            rubric='Response is friendly, professional, and helpful',
            assertion={'evaluation_name': 'professional_tone'},
        ),
        LLMJudge(
            rubric='Response does not reveal sensitive account information without verification',
            assertion={'evaluation_name': 'security_conscious'},
        ),
    ],
)
```

### pytest Integration

```python
# tests/test_support_agent.py
import pytest
from pydantic_evals import Dataset

from support_agent import support_agent, SupportDeps, SupportQuery, SupportResponse
from evals.support_agent_dataset import support_dataset
from tests.mocks import MockKnowledgeBase, MockOrderService

@pytest.fixture
def deps():
    return SupportDeps(
        knowledge_base=MockKnowledgeBase(),
        order_service=MockOrderService(),
    )

@pytest.fixture
def dataset():
    return support_dataset

@pytest.mark.asyncio
async def test_support_agent_evaluations(dataset: Dataset, deps: SupportDeps):
    async def run_agent(query: SupportQuery) -> SupportResponse:
        result = await support_agent.run(query.query, deps=deps)
        return result.output

    report = await dataset.evaluate(run_agent)
    report.print(include_reasons=True)

    # Assert all cases pass
    for case in report.cases:
        for name, assertion in case.assertions.items():
            assert assertion.value, (
                f"Case '{case.name}' failed assertion '{name}': {assertion.reason}"
            )
```

---

## Summary: When to Use What

### Choosing Your Evaluator Type

| Situation | Evaluator Type | Example |
|-----------|---------------|---------|
| Universal quality check | One-size-fits-all `LLMJudge` | "Response is professional" |
| Specific requirements per scenario | Case-specific `LLMJudge` | "Recipe has no gluten" |
| Type/format validation | Deterministic (`IsInstance`) | Output is a Pydantic model |
| Presence check (semantic) | `LLMJudge` | "Mentions the refund policy" |
| Presence check (exact) | Deterministic | Specific value appears |
| Comparing to source material | `LLMJudge` with `include_input` | Hallucination detection |
| Comparing to expected answer | `LLMJudge` with `include_expected_output` | Correctness check |

### Key Principles

1. **Start with case-specific evaluators for your test suite.** Generic rubrics miss nuanced requirements. Each case should capture what makes *that* case pass or fail.

2. **Use one-size-fits-all for production monitoring.** No case-specific context is available at runtime. Focus on universal quality dimensions.

3. **Combine deterministic and LLM evaluators.** Run fast checks first (type, format), expensive LLM checks last.

4. **Always include reasoning.** Essential for debugging failures and iterating on rubrics.

5. **Connect user feedback to evaluation cases.** Every complaint is a potential test case. Build a compounding test suite over time.

---

## Resources and Next Steps

**Documentation**:
- [pydantic-evals documentation](https://ai.pydantic.dev/evals/)
- [pydantic-ai documentation](https://ai.pydantic.dev/)
- [Logfire documentation](https://logfire.pydantic.dev/)

**Getting started**:
1. Install pydantic-evals: `pip install pydantic-ai`
2. Create your first dataset with 5-10 test cases
3. Start with case-specific evaluators capturing your quality requirements
4. Run evaluation and iterate on rubrics based on results
5. Set up Logfire for tracking evaluation results over time

**Further reading**:
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023)
