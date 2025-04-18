from rich.pretty import pprint

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('o4-mini')
agent = Agent(model=model)
result = agent.run_sync(
    'Tell me the steps to cross the street!',
    model_settings=OpenAIResponsesModelSettings(
        openai_reasoning_effort='high',
        openai_reasoning_generate_summary='detailed',
    ),
)
pprint(result.all_messages())

# anthropic_agent = Agent('anthropic:claude-3-7-sonnet-latest')
# result = anthropic_agent.run_sync(
#     'Now make analogous steps to cross the river!',
#     model_settings=AnthropicModelSettings(
#         anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024},
#     ),
#     message_history=result.all_messages(),
# )
# pprint(result.all_messages())

# groq_agent = Agent('groq:deepseek-r1-distill-llama-70b')
# result = groq_agent.run_sync(
#     'Tell me the steps to cross the ocean!',
#     model_settings=GroqModelSettings(groq_reasoning_format='raw'),
#     message_history=result.all_messages(),
# )
# pprint(result.all_messages())

bedrock_agent = Agent('bedrock:us.deepseek.r1-v1:0')
result = bedrock_agent.run_sync(
    'Tell me the steps to cross the ocean!',
    # message_history=result.all_messages(),
)
pprint(result.all_messages())


# deepseek_agent = Agent('deepseek:deepseek-reasoner')
# result = deepseek_agent.run_sync(
#     'Tell me the steps to cross the ocean!',
#     model_settings=OpenAIModelSettings(openai_reasoning_effort='high'),
#     # message_history=result.all_messages(),
# )
# pprint(result.all_messages())
