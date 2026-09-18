"""Top models, neutral tool returns. (a) mixed-tools process, text allowed, 3 retries; (b) four-way judgment, pairwise vs Jev."""
from __future__ import annotations
import asyncio, statistics, sys, time
from decimal import Decimal
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.typesafe import TypeSafeModel
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import bench_process3 as b
from bench_process import Ticket
from cases import CASES

TOP = ['anthropic:claude-opus-5', 'openai:gpt-5.6-sol', 'openai:gpt-5.6-luna']
N = len(CASES)


def process_agent(model):
    calls: list[str] = []
    def escalate_to_human() -> str:
        """Hand the ticket to a person on the support team."""
        calls.append('escalate_to_human'); return 'Escalated: case #4821 opened.'
    def search_docs() -> str:
        """Look the answer up in the product documentation."""
        calls.append('search_docs'); return 'Found: docs.example.com/help/article-217'
    def refund(ctx: RunContext[None], amount: float) -> str:
        """Return a payment to the customer."""
        calls.append(f'refund({amount})'); return f'Refund of {amount} queued.'
    typed_only = isinstance(model, FallbackModel)
    return Agent(model, output_type=Ticket if typed_only else [Ticket, str], tools=[escalate_to_human, search_docs, refund], retries=3), calls


async def process(label, model):
    sem = asyncio.Semaphore(6)
    async def one(text, urgent, area):
        agent, calls = process_agent(model)
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await agent.run(text); out = r.output; error = None
                cost = getattr(r.usage, 'cost', None); by = r.response.model_name or ''
            except Exception as e:  # noqa: BLE001
                out, error, cost, by = None, f'{type(e).__name__}', None, ''
            dt = time.perf_counter() - t0
        typed = isinstance(out, Ticket)
        return {'dt': dt, 'calls': calls, 'error': error, 'typed': typed, 'u': typed and out.urgent == urgent, 'a': typed and out.area == area, 'cost': cost, 'by': by}
    rows = await asyncio.gather(*(one(t, u, a) for t, u, a in CASES))
    dts = sorted(r['dt'] for r in rows)
    print(f'{label:<34} median {statistics.median(dts)*1000:>5.0f} ms  p95 {dts[int(0.95*N)-1]*1000:>6.0f} ms  total {sum(dts):>6.1f} s  typed {sum(r["typed"] for r in rows):>3}/{N}  urgent {sum(r["u"] for r in rows):>3}  area {sum(r["a"] for r in rows):>3}  tool calls {sum(len(r["calls"]) for r in rows):>3}  errors {sum(1 for r in rows if r["error"]):>2}  cost ${sum((r["cost"] or Decimal()) for r in rows):.3f}  by jev {sum(1 for r in rows if r["by"].startswith("jev"))}')


async def main() -> None:
    print('## Mixed-tools process, neutral tool returns, text allowed for LLMs, 3 retries')
    for m in TOP:
        await process(m, m)
    await process('jev (0.8), sol behind', FallbackModel(TypeSafeModel('jev-latest'), 'openai:gpt-5.6-sol'))

    print('\n## Four-way judgment (ticket / escalate / refund(amount) / draft_reply -> LLM)')
    wants = [b.want(u, a) for _, u, a in CASES]; kinds = ('ticket', 'escalate_to_human', 'refund', 'draft_reply')
    picks = {}
    for label, model in [(m, m) for m in TOP[:2]] + [
        ('jev 0.8, sol behind', FallbackModel(TypeSafeModel('jev-latest'), 'openai:gpt-5.6-sol')),
        ('jev 0.6, sol behind', FallbackModel(TypeSafeModel('jev-latest', settings={'typesafe_tool_call_threshold': 0.6}), 'openai:gpt-5.6-sol')),  # type: ignore[arg-type]
    ]:
        rows = await b.run_all(model); picks[label] = [r['pick'] for r in rows]
        dts = sorted(r['dt'] for r in rows)
        print(f'{label:<26} agreed with rule {sum(p == w for p, w in zip(picks[label], wants)):>3}/{N}  median {statistics.median(dts)*1000:>5.0f} ms  cost ${sum((r["cost"] or Decimal()) for r in rows):.3f}  errors {sum(1 for r in rows if r["error"])}  picked ' + '  '.join(f'{k}={picks[label].count(k)}' for k in kinds))
    labels = list(picks)
    print('\npairwise agreement on the pick:')
    for i, x in enumerate(labels):
        for y in labels[i+1:]:
            print(f'  {x:<22} vs {y:<22} {sum(p == q for p, q in zip(picks[x], picks[y]))}/{N}')

asyncio.run(main())
