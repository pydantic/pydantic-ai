# LLM-as-a-Judge: The Complete Guide with Pydantic

## Article Metadata

- **Target word count**: 8,000–10,000 words
- **Target audience**: AI/ML engineers building LLM-powered applications who need to evaluate output quality
- **Core thesis**: LLM-as-a-Judge is powerful, but most implementations miss the critical distinction between one-size-fits-all and case-specific evaluators. This guide shows how to implement both effectively.
- **Differentiator**: While the reference article (Evidently AI) covers theory well, we provide working code throughout and introduce the case-specific vs. one-size-fits-all framework that most content misses.

---

## Section 1: Introduction & TL;DR

**Word count**: ~400 words

### Content

1. **Opening hook**: The challenge of evaluating LLM outputs at scale
   - Human evaluation doesn't scale
   - Traditional metrics (BLEU, ROUGE) miss semantic quality
   - Enter LLM-as-a-Judge: using one LLM to evaluate another

2. **What this guide covers** (bulleted preview):
   - What LLM-as-a-Judge is and why it works
   - The critical distinction between one-size-fits-all and case-specific evaluators
   - When to use (and when NOT to use) LLM judges
   - Practical implementation with working code
   - Development workflows and production monitoring

3. **TL;DR box** (key takeaways for scanners):
   - LLM judges work because evaluation is easier than generation
   - Case-specific evaluators outperform generic ones for test suites
   - Combine deterministic checks (fast) with LLM judges (expensive) strategically
   - Always request reasoning—it's essential for debugging

4. **Tool introduction** (brief):
   - pydantic-evals: The `LLMJudge` class for implementing evaluators
   - pydantic-ai: Building the agents you'll evaluate
   - Logfire: Observability for tracking results over time

### Code Examples

None in this section (conceptual introduction).

---

## Section 2: What is LLM-as-a-Judge?

**Word count**: ~600 words

### Content

1. **Definition and mental model**:
   - Using an LLM to assess the quality of another LLM's output
   - The judge acts as an automated quality reviewer with specific criteria
   - Not a single metric—a flexible technique that adapts to your use case

2. **The evaluation pattern** (step-by-step):
   - Step 1: Task receives input, produces output
   - Step 2: Judge receives output (and optionally input, expected output)
   - Step 3: Judge applies rubric/criteria
   - Step 4: Judge returns verdict (boolean, score, or label) with reasoning

3. **Types of judge outputs**:
   - **Assertions** (boolean): Pass/fail decisions. "Does the response answer the question?"
   - **Scores** (numeric): Quality on a scale. "Rate helpfulness from 0.0 to 1.0"
   - **Labels** (categorical): Classifications. "Is the tone formal, casual, or inappropriate?"

4. **Visual diagram description**:
   - Flow diagram showing: User Query → Agent → Response → LLM Judge → Evaluation Result
   - Annotation showing the judge prompt includes: rubric, output, and optionally input/expected

### Code Example 2.1: Basic LLMJudge Usage

**Purpose**: Show the simplest possible working example

**What it demonstrates**:
- Creating a `Dataset` with a single `Case`
- Adding an `LLMJudge` evaluator with a simple rubric
- Running evaluation and printing results

**Code structure**:
```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

# 1. Define a dataset with one test case
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

# 2. Define the task to evaluate (the agent/function being tested)
async def support_bot(query: str) -> str:
    # In reality, this would call your LLM
    return "To reset your password, click 'Forgot Password' on the login page, enter your email, and follow the link we send you."

# 3. Run evaluation
async def main():
    report = await dataset.evaluate(support_bot)
    report.print()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```

**Expected output to show**:
- Table with case name, assertion result (pass/fail), and reason

---

## Section 3: Why Does LLM-as-a-Judge Work?

**Word count**: ~500 words

### Content

1. **Core insight: Evaluation is easier than generation**
   - Generation: Infinite solution space, must satisfy many constraints simultaneously
   - Evaluation: Binary/limited output, judge one thing at a time
   - Analogy: It's easier to recognize good writing than to write well

2. **The context advantage**:
   - When generating, the LLM must infer what the user wants
   - When evaluating, the LLM sees both the question AND the answer
   - This dramatically narrows the task

3. **Research backing**:
   - Reference: MT-Bench and Chatbot Arena (Zheng et al., 2023)
   - Finding: 80%+ agreement with human preferences on many tasks
   - Caveat: Agreement varies by domain—highest for clear-cut criteria, lower for subjective judgments

4. **Why even the same model can judge itself**:
   - The judge operates on a different, simpler task
   - Given explicit rubric, the model can apply it even if it couldn't have generated a perfect response
   - Particularly effective for detecting hallucinations (model can verify against source material)

### Code Examples

None in this section (conceptual explanation). The point is made through prose, not code.

---

## Section 4: The Critical Distinction: One-Size-Fits-All vs. Case-Specific Evaluators

**Word count**: ~1,000 words

**This is our unique angle—the key insight most content misses.**

### Content

#### 4.1 Defining the Two Approaches (~300 words)

**One-size-fits-all evaluators**:
- A single rubric applied uniformly to every test case
- The rubric must be general enough to apply everywhere
- Example: "Response is professional and helpful"
- Limitation: Generic rubrics can't capture case-specific requirements

**Case-specific evaluators**:
- Rubrics tailored to individual test cases
- Each case can have its own success criteria
- Example: "Recipe must not contain gluten" (for a gluten-free request)
- Advantage: Captures nuanced, context-dependent quality

**A note on workarounds**:

Some evaluation frameworks only support one-size-fits-all evaluators at the dataset level. In these systems, you *can* technically achieve case-specific behavior by:
1. Storing case-specific requirements in metadata (e.g., `{"dietary_restriction": "gluten-free"}`)
2. Writing a generic evaluator that checks for the presence of certain metadata keys
3. Having the evaluator conditionally apply logic based on what it finds in metadata

This works, but it's an awkward pattern:
- The evaluator code becomes complex (lots of conditional logic)
- The connection between case requirements and evaluation logic is indirect
- Teams often don't realize this pattern is possible, so they default to generic rubrics
- The friction means case-specific evaluation is underutilized in practice

First-class support for case-specific evaluators—where each case can declare its own evaluators directly—makes this pattern obvious and natural. When the framework supports it natively, teams actually use it.

**The key realization**:
- If you can write a rubric that works for all cases, that requirement should probably be in your agent's system prompt
- Case-specific evaluators capture what's *unique* about each scenario

#### 4.2 When to Use Each Approach (~300 words)

**Decision matrix**:

| Scenario | Recommended Approach | Why |
|----------|---------------------|-----|
| Online evaluation (real-time production) | One-size-fits-all | No case-specific context available at runtime |
| Building a test suite for your agent | Case-specific preferred | Capture nuanced expectations per scenario |
| Universal quality checks (tone, safety) | One-size-fits-all | These truly apply to all outputs |
| Regression testing after changes | Case-specific | Verify specific behaviors are preserved |

#### 4.3 The Key Insight (~200 words)

> "If an LLM were good enough to assess quality reliably across all cases without context, it would likely be good enough to generate good responses in the first place."

Case-specific evaluators sidestep this limitation by providing the context the judge needs to make an accurate assessment.

**The analogy**:
- One-size-fits-all: "Is this a good essay?" (vague, hard to judge)
- Case-specific: "Does this essay argue that renewable energy is cost-effective, using at least three economic studies?" (clear, verifiable)

#### 4.4 When One-Size-Fits-All Still Works (~200 words)

Two exceptions where generic evaluators perform well:

1. **Detection vs. generation asymmetry**:
   - Some problems are easy to detect but hard to solve
   - Example: Detecting code that won't compile vs. writing correct code
   - Caveat: Often the agent itself could do this check

2. **Model cost arbitrage**:
   - Using a more capable model (GPT-4o) to judge a cheaper production model (GPT-4o-mini)
   - The capability gap justifies the generic approach

### Code Example 4.1: One-Size-Fits-All Evaluator

**Purpose**: Show a universal evaluator applied to all cases

**What it demonstrates**:
- Dataset-level evaluators that run on every case
- A rubric that's genuinely universal (professionalism)

**Code structure**:
```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

dataset = Dataset(
    cases=[
        Case(name='billing_question', inputs={'query': 'Why was I charged twice?'}),
        Case(name='feature_request', inputs={'query': 'Can you add dark mode?'}),
        Case(name='bug_report', inputs={'query': 'The app crashes when I upload photos'}),
    ],
    # This evaluator runs on ALL cases
    evaluators=[
        LLMJudge(
            rubric='Response is professional, empathetic, and does not blame the user',
            include_input=True,
        ),
    ],
)
```

### Code Example 4.2: Case-Specific Evaluators

**Purpose**: Show different rubrics for different cases

**What it demonstrates**:
- Per-case evaluators with tailored rubrics
- Combining case-specific with universal evaluators
- How the rubric captures case-specific requirements

**Code structure**:
```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

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
    # Universal evaluator that still runs on all cases
    evaluators=[
        LLMJudge(
            rubric='Recipe instructions are clear and easy to follow',
            include_input=True,
        ),
    ],
)
```

**Narrative point**: Notice how each case has requirements that would be impossible to capture in a single universal rubric. The vegetarian case doesn't care about cooking time; the quick meal case doesn't care about allergens. Case-specific evaluators let you express exactly what matters for each scenario.

---

## Section 5: Types of LLM Judges

**Word count**: ~800 words

### Content

#### 5.1 Pairwise Comparison (~200 words)

**Definition**: Compare two outputs and select the better one

**Use cases**:
- A/B testing different prompts during development
- Comparing model versions (GPT-4o-mini vs. GPT-4o)
- Preference learning and ranking

**Trade-offs**:
- Pro: Easier than absolute scoring (relative judgments are more reliable)
- Con: Requires generating two outputs per evaluation (2x cost)
- Con: Doesn't give you an absolute quality measure

**Note**: pydantic-evals' `LLMJudge` focuses on single-output evaluation, but pairwise comparison can be implemented with a custom evaluator.

#### 5.2 Evaluation by Criteria (Reference-Free) (~300 words)

**Definition**: Score output on specific dimensions without a reference answer

**Common criteria**:
- **Relevance**: Does the response address the question?
- **Accuracy**: Is the information correct?
- **Clarity**: Is the response easy to understand?
- **Tone**: Is the language appropriate for the context?
- **Safety**: Does the response avoid harmful content?
- **Conciseness**: Is the response appropriately brief?

**Best practice**: Evaluate one dimension per judge

- Bad: "Is the response accurate, clear, and helpful?"
- Good: Three separate judges, one for each dimension

**Why separate judges work better**:
- Clearer rubrics lead to more consistent judgments
- Easier to identify which dimension is failing
- Can weight dimensions differently in aggregation

#### 5.3 Reference-Based Evaluation (~300 words)

**Definition**: Compare output against a reference (expected answer, source material, or context)

**Four sub-types**:

1. **Output vs. Expected Answer** (correctness checking)
   - Use case: Factual Q&A where you know the right answer
   - Example: "Is the response semantically equivalent to the expected answer?"

2. **Output vs. Question** (completeness/relevance)
   - Use case: Ensuring all parts of a multi-part question are addressed
   - Example: "Does the response address all aspects of the user's question?"

3. **Output vs. Retrieved Context** (RAG relevance)
   - Use case: RAG systems where you want to verify the retrieval was useful
   - Example: "Is the retrieved context relevant to answering the question?"

4. **Output vs. Source Material** (hallucination/faithfulness)
   - Use case: Detecting when the model makes up information
   - Example: "Is every claim in the response supported by the source documents?"

### Code Example 5.1: Multi-Criteria Evaluation (Reference-Free)

**Purpose**: Show evaluating multiple dimensions separately

**What it demonstrates**:
- Multiple `LLMJudge` instances, each with a focused rubric
- Using `evaluation_name` to label each dimension
- How results show up separately in the report

**Code structure**:
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
            rubric='The explanation is concise (under 200 words) without sacrificing completeness',
            assertion={'evaluation_name': 'conciseness', 'include_reason': True},
        ),
    ],
)
```

### Code Example 5.2: Reference-Based Evaluation (Hallucination Check)

**Purpose**: Show comparing output to source material

**What it demonstrates**:
- Structured input with context field
- Using `include_input=True` so the judge sees the source material
- A rubric focused on faithfulness to source

**Code structure**:
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

---

## Section 6: Where LLM Judges Excel

**Word count**: ~700 words

### Content

#### 6.1 Groundedness / Hallucination Detection (~250 words)

**Why LLM judges excel here**:
- The task is well-defined: "Is X supported by Y?"
- Even the model that generated the response can often catch its own hallucinations
- The judge has both the claim and the source—it just needs to compare

**Surprising finding**: Models can detect their own hallucinations because:
- Generation: Model doesn't know what it doesn't know
- Evaluation: Model can compare claim to evidence (much easier)

**Best practices for hallucination detection**:
- Always include source material in the judge's context (`include_input=True`)
- Be specific about what counts as hallucination (invented facts, wrong numbers, unsupported claims)
- Consider "partial" verdicts (mostly grounded with minor extrapolations)

#### 6.2 Hard-to-Generalize Quality Rules (~200 words)

**The problem**: Some quality criteria are easy to articulate for a specific case but hard to write as universal rules

**Examples**:
- "This coding question needs a solution that handles edge cases the user mentioned"
- "This customer is frustrated, so the response needs to be especially empathetic"
- "This is a follow-up question, so the response shouldn't repeat context already established"

**Why case-specific LLM judges work**:
- You describe what's good/bad about *this specific case*
- The LLM applies that description as a rubric
- No need to generalize across all possible cases

#### 6.3 Style and Orthogonal Concerns (~200 words)

**Definition**: Quality dimensions that cross-cut other concerns

**Examples**:
- "Response should use second person ('you') not third person ('the user')"
- "Response should not use corporate jargon"
- "Response should not start with 'I'"
- "Response should maintain the persona established in the system prompt"

**Why these are good for LLM judges**:
- Clear, binary criteria
- Apply uniformly across cases (good for one-size-fits-all)
- Hard to check programmatically (regex would be brittle)

### Code Example 6.1: Hallucination Detector

**Purpose**: Show a detailed rubric for catching hallucinations

**Code structure**:
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

### Code Example 6.2: Style Enforcement

**Purpose**: Show orthogonal style checks

**Code structure**:
```python
# These can be one-size-fits-all since they apply universally
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

---

## Section 7: Where NOT to Use LLM Judges

**Word count**: ~500 words

### Content

#### 7.1 When Deterministic Checks Suffice (~200 words)

**Rule of thumb**: If you can check it with code, check it with code

**Examples of deterministic checks**:
- **Type validation**: Use Pydantic models, not LLM judges
- **Format validation**: JSON schema, regex patterns
- **Length constraints**: Character/word counts
- **Required fields**: Check for presence programmatically
- **Exact value matching**: String equality, numeric comparisons

**Why prefer deterministic checks**:
- **Speed**: Milliseconds vs. seconds
- **Cost**: Free vs. API calls
- **Reliability**: 100% consistent vs. probabilistic
- **Debuggability**: Clear pass/fail logic

#### 7.2 The Nuance: Brittle vs. Robust Matching (~150 words)

**The trap**: "I'll just check if the word 'refund' appears in the response"

**Why this is brittle**:
- Synonyms: "reimbursement", "money back", "credit"
- Negations: "We cannot offer a refund" contains "refund" but means the opposite
- Context: Mentioning refund policy vs. actually offering a refund

**When LLM judges are more robust**:
- Semantic matching: "Does the response offer to refund the customer?"
- Intent detection: "Does the response try to help with the user's problem?"
- Nuanced understanding: "Is the tone apologetic without being groveling?"

**Counter-nuance**: Sometimes exact matching IS correct
- Verifying a specific calculated value appears verbatim
- Checking for required legal disclaimers (exact text)
- Ensuring a specific product name or SKU is mentioned

#### 7.3 When Speed Matters (~100 words)

**LLM judge latency**: Typically 1-3 seconds per evaluation

**Inappropriate for**:
- Real-time validation in hot paths
- High-volume checks (millions of requests)
- Synchronous user-facing flows

**Appropriate for**:
- Offline batch evaluation
- Sampled production monitoring (evaluate 1-5% of traffic)
- Development-time testing

### Code Example 7.1: Combining Deterministic and LLM Evaluators

**Purpose**: Show the recommended pattern—fast checks first, expensive checks last

**What it demonstrates**:
- Using `IsInstance` for type checking
- Using `Contains` for basic presence checks
- Using `LLMJudge` only for semantic quality

**Code structure**:
```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge, IsInstance, Contains

dataset = Dataset(
    cases=[
        Case(
            name='refund_request',
            inputs='I want a refund for my order',
        ),
    ],
    evaluators=[
        # Fast, deterministic checks run first
        IsInstance(type_name='str'),

        # Semantic checks that are hard to do with code
        LLMJudge(
            rubric='Response acknowledges the refund request and provides next steps',
            include_input=True,
        ),
        LLMJudge(
            rubric='Response is empathetic and does not blame the customer',
        ),
    ],
)
```

**Narrative point**: The deterministic checks act as fast filters. If the response isn't even a string, there's no point in running expensive LLM evaluation. Structure your evaluators from cheapest to most expensive.

---

## Section 8: Creating Effective LLM Judges

**Word count**: ~1,200 words

### Content

#### 8.1 Writing Good Rubrics (~350 words)

**The spectrum of rubric quality**:

| Level | Example | Problem |
|-------|---------|---------|
| Bad | "Response is good" | What does "good" mean? |
| Okay | "Response is helpful" | Still vague—helpful how? |
| Better | "Response answers the user's question" | More specific, but what counts as an answer? |
| Good | "Response directly addresses the question with actionable next steps" | Clear criteria |
| Best | Includes what to look for AND what constitutes failure | Unambiguous |

**Rubric writing principles**:

1. **Be specific about what you're checking**
   - Bad: "Response is accurate"
   - Good: "Response contains no factual errors about [specific domain]"

2. **Define the threshold for pass/fail**
   - Bad: "Response is mostly accurate"
   - Good: "FAIL if any factual claim is incorrect. Minor omissions are acceptable."

3. **List what to look for (and what to look against)**
   - Provide examples of passing and failing behaviors
   - Call out edge cases explicitly

4. **Keep rubrics focused on one dimension**
   - If you're checking multiple things, use multiple judges

#### 8.2 Binary Assertions vs. Numeric Scores (~250 words)

**When to use binary (pass/fail)**:
- Clear threshold exists (policy compliance, safety violations)
- You want simple aggregation ("85% of cases passed")
- The dimension is truly binary (contains PII or doesn't)

**When to use numeric scores (0.0–1.0)**:
- Quality exists on a spectrum (helpfulness, clarity)
- You want to track improvement over time (average score trending up)
- You need to rank or compare responses

**Hybrid approach**: Use both when you need a threshold AND a score
- Score: "Rate the helpfulness from 0 to 1"
- Assertion: "PASS if score >= 0.7"

#### 8.3 Providing Context to the Judge (~250 words)

**Three context levels**:

1. **Output only** (default): Judge sees just the response
   - Use for: Style checks, format validation
   - Example: "Is the response in a professional tone?"

2. **Input + Output** (`include_input=True`): Judge sees what the user asked and the response
   - Use for: Relevance, completeness, appropriateness
   - Example: "Does the response address the user's question?"

3. **Input + Output + Expected** (`include_expected_output=True`): Judge sees input, actual output, and reference answer
   - Use for: Correctness, semantic equivalence
   - Example: "Is the response semantically equivalent to the expected answer?"

**Rule of thumb**: Give the judge the minimum context needed to make the judgment. More context isn't always better—it can distract or confuse.

#### 8.4 Model Selection and Settings (~200 words)

**Default**: `openai:gpt-4o` (good balance of quality and cost)

**Options**:
- Cost-sensitive: `openai:gpt-4o-mini` (cheaper, still effective for clear-cut criteria)
- High-stakes: `anthropic:claude-sonnet-4-5` or `openai:gpt-4o` (more nuanced judgment)

**Temperature setting**:
- Use low temperature (0.0–0.2) for consistency
- Higher temperature adds variance, rarely helpful for evaluation

**Setting defaults globally**:
```python
from pydantic_evals.evaluators import set_default_judge_model
set_default_judge_model('anthropic:claude-sonnet-4-5')
```

#### 8.5 Always Request Reasoning (~150 words)

**Why reasoning matters**:
- **Debugging**: Understand why a case failed
- **Trust**: Verify the judge applied the rubric correctly
- **Iteration**: Improve rubrics based on judge's interpretation
- **Documentation**: Reasoning explains the verdict to humans

**How to enable**:
```python
assertion={'evaluation_name': 'accuracy', 'include_reason': True}
```

### Code Example 8.1: Rubric Evolution

**Purpose**: Show the progression from bad to good rubrics

**Code structure**:
```python
# ❌ Too vague - what does "helpful" mean?
LLMJudge(rubric='Response is helpful')

# ❌ Still vague - helpful to whom? how?
LLMJudge(rubric='Response answers the question helpfully')

# ✅ Specific criteria with clear pass/fail
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

### Code Example 8.2: Score vs. Assertion Configuration

**Purpose**: Show different output configurations

**Code structure**:
```python
from pydantic_ai.settings import ModelSettings

# Assertion only (pass/fail)
LLMJudge(
    rubric='Response follows company guidelines',
    assertion={'evaluation_name': 'compliant', 'include_reason': True},
    score=False,  # Don't output a score
)

# Score only (0.0 to 1.0)
LLMJudge(
    rubric='Rate the helpfulness of the response on a scale from 0 to 1',
    score={'evaluation_name': 'helpfulness', 'include_reason': True},
    assertion=False,  # Don't output pass/fail
)

# Both score and assertion
LLMJudge(
    rubric='''
    Rate the accuracy of the response from 0 to 1.
    PASS if accuracy >= 0.8, FAIL otherwise.
    ''',
    score={'evaluation_name': 'accuracy_score'},
    assertion={'evaluation_name': 'accuracy_pass'},
)

# With model settings for consistency
LLMJudge(
    rubric='...',
    model='openai:gpt-4o',
    model_settings=ModelSettings(temperature=0.0),
)
```

### Code Example 8.3: Context Configuration

**Purpose**: Show when to include input and expected output

**Code structure**:
```python
# Output only: Style check (doesn't need to know what was asked)
LLMJudge(
    rubric='Response uses professional language without slang or emojis',
    include_input=False,
    include_expected_output=False,
)

# Input + Output: Relevance check (needs to know the question)
LLMJudge(
    rubric='Response directly addresses the question that was asked',
    include_input=True,
    include_expected_output=False,
)

# Input + Output + Expected: Correctness check (needs reference answer)
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

---

## Section 9: Development Workflows with LLM Judges

**Word count**: ~1,000 words

### Content

#### 9.1 Building Your Evaluation Dataset (~350 words)

**The challenge**: Where do test cases come from?

**Sources of test cases**:
1. **Production logs**: Real queries users have asked
2. **Manual creation**: Cases you design to test specific behaviors
3. **Edge case discovery**: Problems you've encountered and fixed
4. **User feedback**: Complaints or issues reported by users

**The iterative process**:
1. Start with a few cases (5-10)
2. Run your agent, generate outputs
3. Manually review outputs, identify what's good/bad
4. Write rubrics capturing those observations
5. Run evaluation, check for false positives/negatives
6. Adjust rubrics, add new cases, repeat

**Dataset file format** (YAML):
```yaml
cases:
  - name: password_reset
    inputs:
      query: "How do I reset my password?"
    evaluators:
      - LLMJudge:
          rubric: "Response includes link or instructions for password reset"
          include_input: true
```

**Version control**: Commit your dataset files. They're as valuable as your code.

#### 9.2 Using AI to Improve Your Prompts (~250 words)

**The pattern**: Show your evaluation results to a coding agent, ask for prompt improvements

**Workflow**:
1. Run evaluation, identify failing cases
2. Share the dataset (cases + rubrics) and results with an AI assistant
3. Ask: "What changes to the system prompt would help with these failing cases?"
4. Apply suggestions, re-run evaluation
5. Verify improvements without regressions

**Critical warning**: Don't delete the dataset after improving the prompt
- Those cases remain valuable for regression testing
- You're building a cumulative test suite, not a one-time fix

#### 9.3 Generating Additional Test Cases (~200 words)

**The pattern**: Ask AI to identify gaps in your test coverage

**Prompts that work**:
- "Given this dataset, what edge cases are missing?"
- "Generate 5 cases where the current prompt is likely to fail"
- "What scenarios would be most valuable for human review?"

**Quality control**: Always review generated cases before adding to your suite

#### 9.4 Pruning Redundant Cases (~200 words)

**The problem**: Datasets grow over time, increasing evaluation cost and maintenance burden

**Signs of redundancy**:
- Multiple cases testing the same behavior
- Cases that always pass/fail together
- Similar inputs with overlapping rubrics

**The pattern**: Ask AI to identify redundancy
- "Which cases test the same behavior?"
- "If I could only keep 10 cases, which would provide the best coverage?"

### Code Example 9.1: Building a Dataset File

**Purpose**: Show the recommended file-based workflow

**Code structure**:
```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

# Build dataset programmatically
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

# Load from file (in your test suite)
loaded_dataset = Dataset.from_file('evals/support_bot.yaml')
```

### Code Example 9.2: AI-Assisted Prompt Improvement

**Purpose**: Show how to use eval results to improve prompts

**Code structure**:
```python
from pydantic_ai import Agent
from pydantic_evals import Dataset

async def main():
    # Load dataset and run evaluation
    dataset = Dataset.from_file('evals/support_bot.yaml')
    report = await dataset.evaluate(support_bot)

    # Use a more capable model to analyze results and suggest improvements
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

---

## Section 10: Production Monitoring with LLM Judges

**Word count**: ~800 words

### Content

#### 10.1 Online vs. Offline Evaluation (~200 words)

**Offline evaluation** (batch):
- Run on a fixed test dataset before deployment
- Goal: Catch regressions, validate new prompts
- Frequency: On every change, in CI/CD

**Online evaluation** (real-time):
- Run on sampled production traffic
- Goal: Monitor quality in the wild, catch drift
- Frequency: Continuous, sampled (1-10% of traffic)

**Why you need both**:
- Offline catches known failure modes
- Online catches unknown unknowns (real user behavior differs from test cases)

#### 10.2 Sampling Production Traffic (~200 words)

**The economics**:
- LLM evaluation is expensive ($0.01-0.05 per evaluation)
- Evaluating every request is cost-prohibitive at scale
- Sampling 1-5% captures trends without breaking the bank

**What to sample**:
- Random sampling: Baseline quality measurement
- Stratified sampling: Ensure coverage of different query types
- Triggered sampling: Evaluate when certain conditions are met (long responses, low confidence)

#### 10.3 What to Monitor and Alert On (~200 words)

**Key metrics**:
1. **Pass rate**: Percentage of evaluations that pass (trending down = problem)
2. **Average scores**: Mean quality scores over time
3. **Failure patterns**: Are the same rubrics failing repeatedly?

**Alert conditions**:
- Pass rate drops below threshold (e.g., < 90%)
- Rolling average score decreases significantly
- Specific rubric starts failing that previously passed

#### 10.4 Viewing Results in Logfire (~200 words)

**Integration**:
- Configure Logfire once, results flow automatically
- Each `dataset.evaluate()` call creates an "experiment"
- Experiments are named automatically or can be named explicitly

**What you can do in the Logfire UI**:
- View list of experiments with pass rates
- Compare experiments side-by-side (before/after changes)
- Click into individual cases to see full traces
- Filter and sort by any metric

### Code Example 10.1: Logfire Integration

**Purpose**: Show end-to-end Logfire setup for evaluation

**Code structure**:
```python
import asyncio
import logfire
from pydantic_evals import Dataset

# Configure Logfire (do this once at app startup)
logfire.configure()

async def main():
    # Load your evaluation dataset
    dataset = Dataset.from_file('evals/support_bot.yaml')

    # Run evaluation - results automatically sent to Logfire
    report = await dataset.evaluate(
        support_bot,
        # Optional: add metadata for filtering in Logfire
        experiment_metadata={
            'version': '2.1.0',
            'model': 'gpt-4o-mini',
            'prompt_version': 'empathetic-v3',
        },
    )

    # Print local summary
    report.print()

    # In Logfire UI:
    # 1. Navigate to Evals tab
    # 2. See experiment with pass rate, duration, metrics
    # 3. Click to see individual case results
    # 4. Select multiple experiments to compare

if __name__ == '__main__':
    asyncio.run(main())
```

### Code Example 10.2: Production Sampling Pattern

**Purpose**: Show how to sample production traffic for evaluation

**Code structure**:
```python
import random
import asyncio
from pydantic_evals import Dataset, Case
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
    # Generate response
    response = await support_bot(query)

    # Sample 5% of traffic for evaluation
    if random.random() < 0.05:
        # Run evaluation asynchronously (don't block the response)
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

    # Task that returns the already-generated response
    async def return_response(_: str) -> str:
        return response

    await dataset.evaluate(return_response)
```

---

## Section 11: Relationship with User Feedback

**Word count**: ~600 words

### Content

#### 11.1 User Feedback as Evaluation Ground Truth (~200 words)

**Types of user feedback**:
- **Explicit**: Thumbs up/down, star ratings, written complaints
- **Implicit**: Retry behavior, follow-up questions, abandonment

**The value of negative feedback**:
- Each complaint is a potential test case
- User's words describe what went wrong (the rubric!)
- Fixing this case proves you addressed their concern

#### 11.2 Converting Feedback to Test Cases (~200 words)

**The pattern**:
1. User reports issue: "The bot told me I couldn't get a refund, but I should have been eligible"
2. Extract the input: "I bought this 2 weeks ago and it broke, can I get a refund?"
3. Write the rubric: "Given a purchase within 30 days with a defect, response MUST confirm refund eligibility"
4. Add to dataset with metadata linking to the ticket

**Why this works**:
- Real failure mode, not hypothetical
- Rubric is grounded in actual user expectations
- Passing this case proves the issue is fixed

#### 11.3 Building a Continuous Improvement Loop (~200 words)

**The cycle**:
1. **Collect**: Gather user feedback (ratings, complaints, support tickets)
2. **Convert**: Turn feedback into test cases with rubrics
3. **Evaluate**: Run against current and candidate prompts
4. **Deploy**: Ship improvements that pass new cases without regressing old ones
5. **Repeat**: Continue collecting, never stop adding cases

**The compounding effect**:
- Each round adds more coverage
- Over time, your eval suite captures the full range of user expectations
- New issues are caught because similar patterns are already tested

### Code Example 11.1: Converting User Feedback to Test Cases

**Purpose**: Show the feedback-to-test-case workflow

**Code structure**:
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

# Add to existing dataset
dataset = Dataset.from_file('evals/support_bot.yaml')
for case in feedback_cases:
    dataset.add_case(**case.__dict__)
dataset.to_file('evals/support_bot.yaml')
```

---

## Section 12: Complete Example: Building a Tested Agent

**Word count**: ~800 words

### Content

**Goal**: Walk through a complete, end-to-end example that ties everything together

**The example**:
- A customer support agent with structured output
- Multiple tools (knowledge base search, order lookup)
- Dependency injection for database access
- pytest integration for running evaluations

#### Structure

1. **Agent definition** (support_agent.py)
   - Pydantic models for input/output
   - Agent with system prompt
   - Tools for knowledge base and order lookup

2. **Evaluation dataset** (evals/support_agent.yaml)
   - Mix of case-specific and universal evaluators
   - Cases covering different scenarios

3. **Test file** (tests/test_support_agent.py)
   - pytest fixture for dataset
   - Async test that runs evaluation
   - Assertions on results

### Code Example 12.1: Agent Definition

**Purpose**: Show a realistic agent to be evaluated

**Code structure**:
```python
# support_agent.py
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

# Structured input
class SupportQuery(BaseModel):
    query: str
    user_id: str
    account_tier: str  # 'free', 'pro', 'enterprise'

# Structured output
class SupportResponse(BaseModel):
    message: str
    suggested_articles: list[str]
    escalate_to_human: bool

# Dependencies
@dataclass
class SupportDeps:
    knowledge_base: 'KnowledgeBase'
    order_service: 'OrderService'

# Agent definition
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

### Code Example 12.2: Evaluation Dataset

**Purpose**: Show a comprehensive test suite

**Code structure**:
```python
# evals/support_agent_dataset.py
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge, IsInstance
from support_agent import SupportQuery, SupportResponse

support_dataset = Dataset(
    cases=[
        # Case 1: Enterprise escalation (case-specific)
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

        # Case 2: Free tier limitations (case-specific)
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
                    - Offer alternative support options (chat, email, docs)

                    FAIL if it implies free users can get phone support.
                    ''',
                    include_input=True,
                ),
            ],
        ),

        # Case 3: Order inquiry (case-specific)
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
                    - Provide status information (or explain why it can't)
                    - Acknowledge the delay concern empathetically

                    FAIL if the order is not looked up or the response is generic.
                    ''',
                    include_input=True,
                ),
            ],
        ),
    ],

    # Universal evaluators (run on all cases)
    evaluators=[
        # Type check: output must be the expected type
        IsInstance(type_name='SupportResponse'),

        # Universal quality checks
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

### Code Example 12.3: pytest Integration

**Purpose**: Show how to run evaluations in a test suite

**Code structure**:
```python
# tests/test_support_agent.py
import pytest
from pydantic_evals import Dataset

from support_agent import support_agent, SupportDeps, SupportQuery, SupportResponse
from evals.support_agent_dataset import support_dataset
from tests.mocks import MockKnowledgeBase, MockOrderService

@pytest.fixture
def deps():
    """Create mock dependencies for testing."""
    return SupportDeps(
        knowledge_base=MockKnowledgeBase(),
        order_service=MockOrderService(),
    )

@pytest.fixture
def dataset():
    """Load the evaluation dataset."""
    return support_dataset

@pytest.mark.asyncio
async def test_support_agent_evaluations(dataset: Dataset, deps: SupportDeps):
    """Run full evaluation suite on the support agent."""

    async def run_agent(query: SupportQuery) -> SupportResponse:
        result = await support_agent.run(
            query.query,
            deps=deps,
            # Pass additional context the agent needs
            message_history=[],  # Could include conversation history
        )
        return result.output

    # Run evaluations
    report = await dataset.evaluate(run_agent)

    # Print detailed results (useful for debugging)
    report.print(include_reasons=True)

    # Assert all cases pass
    for case in report.cases:
        for name, assertion in case.assertions.items():
            assert assertion.value, (
                f"Case '{case.name}' failed assertion '{name}': {assertion.reason}"
            )

@pytest.mark.asyncio
async def test_support_agent_pass_rate(dataset: Dataset, deps: SupportDeps):
    """Ensure overall pass rate meets threshold."""

    async def run_agent(query: SupportQuery) -> SupportResponse:
        result = await support_agent.run(query.query, deps=deps)
        return result.output

    report = await dataset.evaluate(run_agent)

    # Calculate pass rate
    total_assertions = 0
    passed_assertions = 0
    for case in report.cases:
        for name, assertion in case.assertions.items():
            total_assertions += 1
            if assertion.value:
                passed_assertions += 1

    pass_rate = passed_assertions / total_assertions if total_assertions > 0 else 0

    # Assert minimum pass rate
    assert pass_rate >= 0.9, f"Pass rate {pass_rate:.1%} is below 90% threshold"
```

---

## Section 13: Summary: When to Use What

**Word count**: ~400 words

### Content

**Quick reference decision tree**:

#### Choose Your Evaluator Type

| Situation | Evaluator Type | Example |
|-----------|---------------|---------|
| Universal quality check | One-size-fits-all `LLMJudge` | "Response is professional" |
| Specific requirements per scenario | Case-specific `LLMJudge` | "Recipe has no gluten" |
| Type/format validation | Deterministic (`IsInstance`) | Output is a Pydantic model |
| Presence check (semantic) | `LLMJudge` | "Mentions the refund policy" |
| Presence check (exact) | Deterministic (`Contains`) | Specific value appears |
| Comparing to source material | `LLMJudge` with `include_input` | Hallucination detection |
| Comparing to expected answer | `LLMJudge` with `include_expected_output` | Correctness check |

#### Choose Your Workflow

| Goal | Approach |
|------|----------|
| Pre-deployment validation | Offline batch evaluation on test dataset |
| Production monitoring | Online sampling (1-5%) with one-size-fits-all evaluators |
| Debugging a specific issue | Add case-specific test case, run targeted evaluation |
| Comparing model versions | Run same dataset with different models, compare in Logfire |

#### Key Principles Recap

1. **Start with case-specific evaluators for your test suite**
   - Generic rubrics miss nuanced requirements
   - Each case should capture what makes *that* case pass or fail

2. **Use one-size-fits-all for production monitoring**
   - No case-specific context available at runtime
   - Focus on universal quality dimensions

3. **Combine deterministic and LLM evaluators**
   - Run fast checks first (type, format, length)
   - Run expensive LLM checks last

4. **Always include reasoning**
   - Essential for debugging failures
   - Helps iterate on rubrics

5. **Connect user feedback to evaluation cases**
   - Every complaint is a potential test case
   - Build a compounding test suite over time

### No Code Examples

This section is a summary/reference. The code examples are in previous sections.

---

## Section 14: Resources and Next Steps

**Word count**: ~200 words

### Content

**Links to documentation**:
- pydantic-evals documentation: [link]
- pydantic-ai documentation: [link]
- Logfire documentation: [link]
- Example repository with all code from this article: [link]

**Further reading**:
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023)
- Research on evaluation bias and mitigation strategies

**Next steps**:
1. Install pydantic-evals: `pip install pydantic-evals`
2. Create your first dataset with 5-10 test cases
3. Start with case-specific evaluators capturing your quality requirements
4. Run evaluation and iterate on rubrics based on results
5. Set up Logfire for tracking evaluation results over time

### No Code Examples

This section is a closing with resources, not a code-heavy section.

---

## Appendix: Comparison with Reference Article

### What We Cover That They Don't

1. **One-size-fits-all vs. case-specific distinction** (Sections 4, 9, 11, 12)
   - This is our unique angle, mentioned nowhere in the reference
   - Includes concrete guidance on when to use each approach

2. **Working code examples throughout** (15 code examples vs. pseudo-code snippets)
   - Every concept has runnable pydantic-evals code
   - Complete end-to-end example with pytest integration

3. **Development workflows** (Section 9)
   - Using AI to improve prompts based on eval results
   - Generating and pruning test cases
   - Building datasets incrementally

4. **User feedback integration** (Section 11)
   - Converting complaints to test cases
   - Building a continuous improvement loop

5. **Complete pytest integration example** (Section 12)
   - How to run evaluations in CI/CD
   - Assertion patterns for test suites

### What They Cover That We Match

1. **Types of judges** (Section 5) — pairwise, criteria-based, reference-based
2. **Prompt engineering techniques** (Section 8) — rubric writing, context, model settings
3. **Production monitoring** (Section 10) — online vs. offline, sampling, alerting
4. **Pros and cons** (Sections 6, 7) — where judges excel and where they don't
5. **When NOT to use** (Section 7) — deterministic alternatives

### Our Structural Advantages

- More concrete, runnable code (15 examples vs. conceptual descriptions)
- The case-specific angle is novel and highly actionable
- Integrated tooling story: pydantic-evals → Logfire → iterate
- pytest integration shows how to fit evals into existing workflows
