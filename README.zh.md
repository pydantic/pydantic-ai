<div align="center">
  <a href="https://ai.pydantic.dev/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://ai.pydantic.dev/img/pydantic-ai-dark.svg">
      <img src="https://ai.pydantic.dev/img/pydantic-ai-light.svg" alt="Pydantic AI">
    </picture>
  </a>
</div>
<div align="center">
  <h3>GenAI 智能体框架，Pydantic 风格</h3>
</div>
<div align="center">
  <a href="https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push" alt="CI"></a>
  <a href="https://coverage-badge.samuelcolvin.workers.dev/redirect/pydantic/pydantic-ai"><img src="https://coverage-badge.samuelcolvin.workers.dev/pydantic/pydantic-ai.svg" alt="Coverage"></a>
  <a href="https://pypi.python.org/pypi/pydantic-ai"><img src="https://img.shields.io/pypi/v/pydantic-ai.svg" alt="PyPI"></a>
  <a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/pypi/pyversions/pydantic-ai.svg" alt="versions"></a>
  <a href="https://github.com/pydantic/pydantic-ai/blob/main/LICENSE"><img src="https://img.shields.io/github/license/pydantic/pydantic-ai.svg?v" alt="license"></a>
  <a href="https://logfire.pydantic.dev/docs/join-slack/"><img src="https://img.shields.io/badge/Slack-加入_Slack-4A154B?logo=slack" alt="Join Slack" /></a>
</div>

---

**官方文档**: [ai.pydantic.dev](https://ai.pydantic.dev/)

---

<p align="center">
  <strong>简体中文</strong> | <a href="README.md">English</a>
</p>

### <em>Pydantic AI 是一个 Python 智能体（Agent）框架，旨在帮助您快速、自信且轻松地利用生成式 AI 构建生产级应用和工作流。</em>


FastAPI 凭借其创新的、符合人体工程学的设计，在 [Pydantic 校验 (Validation)](https://docs.pydantic.dev) 和现代 Python 特性（如类型提示）的基础上，彻底改变了 Web 开发。

然而，尽管几乎每一个 Python 智能体框架和 LLM 库都在使用 Pydantic 校验，当我们在 [Pydantic Logfire](https://pydantic.dev/logfire) 中开始使用大模型时，却找不到任何一个能带给我们同样开发体验的工具。

我们构建 Pydantic AI 的初衷很简单：将 FastAPI 的那种开发爽感带到生成式 AI 应用和智能体的开发中。

## 为什么选择 Pydantic AI

1. **由 Pydantic 团队打造**：
[Pydantic 校验](https://docs.pydantic.dev/latest/) 是 OpenAI SDK、Google SDK、Anthropic SDK、LangChain、LlamaIndex、AutoGPT、Transformers、CrewAI、Instructor 等众多项目的校验层。_既然能直接使用原汁原味的技术，为何还要使用其衍生品呢？_ :smiley:

2. **模型无关 (Model-agnostic)**：
支持几乎所有的[模型](https://ai.pydantic.dev/models/overview)和供应商：OpenAI, Anthropic, Gemini, DeepSeek, Grok, Cohere, Mistral, 和 Perplexity；以及 Azure AI Foundry, Amazon Bedrock, Google Vertex AI, Ollama, LiteLLM, Groq, OpenRouter, Together AI, Fireworks AI, Cerebras, Hugging Face, GitHub, Heroku, Vercel, Nebius, OVHcloud, 阿里云, SambaNova, 和 Outlines。如果您喜爱的模型或供应商未列出，也可以轻松实现[自定义模型](https://ai.pydantic.dev/models/overview#custom-models)。

3. **无缝的可观测性**：
与 [Pydantic Logfire](https://pydantic.dev/logfire) 紧密[集成](https://ai.pydantic.dev/logfire)。Logfire 是我们的通用 OpenTelemetry 可观测性平台，支持实时调试、基于评估（Evals）的性能监控，以及行为追踪和成本追踪。如果您已有支持 OTel 的可观测性平台，也可以[直接使用](https://ai.pydantic.dev/logfire#alternative-observability-backends)。

4. **完全类型安全**：
旨在为您的 IDE 或 AI 编程助手提供尽可能多的上下文，用于自动补全和[类型检查](https://ai.pydantic.dev/agents#static-type-checking)。这将整类错误从运行期提前到了编写期，带给您一种类似 Rust 的“只要编译通过，就能运行成功”的体验。

5. **强大的评估系统 (Evals)**：
使您能够系统地测试和[评估](https://ai.pydantic.dev/evals)所构建智能体系统的性能和准确性，并在 Pydantic Logfire 中持续监控性能变化。

6. **支持 MCP, A2A 和 UI**：
集成了[模型上下文协议 (MCP)](https://ai.pydantic.dev/mcp/overview)、[智能体间互操作 (Agent2Agent)](https://ai.pydantic.dev/a2a) 以及多种 [UI 事件流](https://ai.pydantic.dev/ui/overview) 标准。这让您的智能体能够访问外部工具和数据，与其他智能体协同工作，并构建具备流式事件通信的交互式应用。

7. **人在回路 (Human-in-the-Loop) 工具审批**：
支持轻松标记某些工具调用[需要审批](https://ai.pydantic.dev/deferred-tools#human-in-the-loop-tool-approval)后才能执行。审批逻辑可以基于工具参数、对话历史或用户偏好动态决定。

8. **持久化执行 (Durable Execution)**：
支持构建[持久化智能体](https://ai.pydantic.dev/durable_execution/overview/)。它们能在瞬时 API 故障、应用错误或重启后保留进度，并以生产级的可靠性处理长时运行的异步工作流和“人在回路”流程。

9. **流式输出**：
具备持续[流式传输](https://ai.pydantic.dev/output#streamed-results)结构化输出的能力，并支持即时校验，确保能实时访问生成的数据。

10. **图支持 (Graph Support)**：
提供了一种通过类型提示定义[图](https://ai.pydantic.dev/graph)的高效方式，适用于标准控制流可能演变为“面条代码”的复杂应用。

不过说实话，再多的列表也比不上[亲自动手尝试](#后续步骤)，感受一下它带给您的开发体验！

## Hello World 示例

这是一个 Pydantic AI 的极简示例：

```python
from pydantic_ai import Agent

# 定义一个极简智能体，包括要使用的模型。您也可以在运行智能体时再设置模型。
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    # 使用关键字参数为智能体注册静态指令（instructions）。
    # 对于更复杂的动态生成指令，请参阅下方的示例。
    instructions='回复要简练，用一句话回答。',
)

# 同步运行智能体，与 LLM 进行对话。
result = agent.run_sync('"Hello world" 这个词是怎么来的？')
print(result.output)
"""
"Hello, world" 最早为人所知的用法源于 1974 年一本关于 C 语言的教科书。
"""
```

_(此示例是完整的，假设您已 [安装了 `pydantic_ai` 包](https://ai.pydantic.dev/install)，即可直接运行)_

这次交互非常简短：Pydantic AI 将指令和用户提示词发送给 LLM，模型返回文本响应。

目前看起来还不算太惊艳，但我们可以轻松添加[工具](https://ai.pydantic.dev/tools)、[动态指令](https://ai.pydantic.dev/agents#instructions)和[结构化输出](https://ai.pydantic.dev/output)来构建更强大的智能体。

## 工具与依赖注入示例

这是一个使用 Pydantic AI 构建银行支持智能体的简洁示例：

**(更详尽的文档示例请见 [官方文档](https://ai.pydantic.dev/#tools-dependency-injection-example))**

```python
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn


# SupportDependencies 用于将运行指令和工具函数所需的数据、连接和逻辑传递给模型。
# 依赖注入为自定义智能体行为提供了一种类型安全的方式。
@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


# 此 Pydantic 模型定义了智能体返回的输出结构。
class SupportOutput(BaseModel):
    support_advice: str = Field(description='返回给客户的建议')
    block_card: bool = Field(description='是否冻结客户的卡片')
    risk: int = Field(description='查询的风险等级', ge=0, le=10)


# 此智能体将作为银行的一线支持。
# 智能体在其接受的依赖类型和返回的输出类型上是泛型的。
# 在本例中，支持智能体的类型为 `Agent[SupportDependencies, SupportOutput]`。
support_agent = Agent(
    'openai:gpt-5.2',
    deps_type=SupportDependencies,
    # 智能体的响应保证符合 SupportOutput 结构，
    # 如果校验失败，智能体会收到提示并重试。
    output_type=SupportOutput,
    instructions=(
        '您是我们银行的支持智能体，请为客户提供支持并'
        '判断其查询的风险等级。'
    ),
)


# 动态指令可以使用依赖注入。
# 依赖项通过 `RunContext` 参数传递，该参数已由上方的 `deps_type` 参数化。
# 如果此处的类型标注错误，静态类型检查器将会捕获它。
@support_agent.instructions
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"客户的名字是 {customer_name!r}"


# `tool` 装饰器允许您注册 LLM 在回复用户时可能调用的函数。
# 同样，依赖项通过 `RunContext` 传递，任何其他参数都会成为传递给 LLM 的工具模式（schema）。
# Pydantic 用于校验这些参数，错误会返回给 LLM 以便其重试。
@support_agent.tool
async def customer_balance(
        ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """返回客户当前的账户余额。"""
    # 工具的 docstring 也会作为工具描述发送给 LLM。
    # 参数描述会从 docstring 中提取并添加到发送给 LLM 的参数模式中。
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return balance


...  # 在实际场景中，您会添加更多工具和更长的系统提示词


async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    # 异步运行智能体，与 LLM 持续对话直到获得最终响应。
    # 即便在这个相对简单的案例中，随着工具被调用以获取输出，智能体也会与 LLM 交换多条消息。
    result = await support_agent.run('我的余额是多少？', deps=deps)
    # `result.output` 会通过 Pydantic 校验，确保其为 `SupportOutput`。
    # 由于智能体是泛型的，它也会被标注为 `SupportOutput` 类型，以辅助静态类型检查。
    print(result.output)
    """
    support_advice='您好 John，您当前的账户余额（包括待处理交易）为 $123.45。' block_card=False risk=1
    """

    result = await support_agent.run('我的卡丢了！', deps=deps)
    print(result.output)
    """
    support_advice='非常抱歉听到这个消息，John。我们正临时冻结您的卡片以防止未经授权的交易。' block_card=True risk=8
    """
```

## 后续步骤

若想亲自尝试 Pydantic AI，请[安装它](https://ai.pydantic.dev/install)并按照[示例中的说明](https://ai.pydantic.dev/examples/setup)进行操作。

阅读[文档](https://ai.pydantic.dev/agents/)以了解更多关于使用 Pydantic AI 构建应用的信息。

阅读 [API 参考](https://ai.pydantic.dev/api/agent/) 以深入了解 Pydantic AI 的接口。

如有任何问题，欢迎加入 [Slack](https://logfire.pydantic.dev/docs/join-slack/) 或在 [GitHub](https://github.com/pydantic/pydantic-ai/issues) 上提交 Issue。
