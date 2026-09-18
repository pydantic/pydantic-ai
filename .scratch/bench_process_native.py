"""The mixed-tools process with NativeOutput(Ticket): the model's own structured-output mode, typed only, neutral tool returns."""
from __future__ import annotations
import asyncio, statistics, sys, time
from decimal import Decimal
from pydantic_ai import Agent, NativeOutput, RunContext
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from bench_process import Ticket
from cases import CASES
N = len(CASES)

def make_agent(model):
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
    return Agent(model, output_type=NativeOutput(Ticket), tools=[escalate_to_human, search_docs, refund], retries=3), calls

async def run(model):
    sem = asyncio.Semaphore(6)
    async def one(text, urgent, area):
        agent, calls = make_agent(model)
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await agent.run(text); out = r.output; error = None; cost = getattr(r.usage, 'cost', None)
            except Exception as e:  # noqa: BLE001
                out, error, cost = None, f'{type(e).__name__}: {str(e)[:50]}', None
            dt = time.perf_counter() - t0
        return {'dt': dt, 'calls': calls, 'error': error, 'u': out is not None and out.urgent == urgent, 'a': out is not None and out.area == area, 'cost': cost}
    rows = await asyncio.gather(*(one(t, u, a) for t, u, a in CASES))
    dts = sorted(r['dt'] for r in rows)
    errs = [r['error'] for r in rows if r['error']]
    print(f'{model:<28} NativeOutput  median {statistics.median(dts)*1000:>5.0f} ms  p95 {dts[int(0.95*N)-1]*1000:>6.0f} ms  total {sum(dts):>6.1f} s  urgent {sum(r["u"] for r in rows):>3}  area {sum(r["a"] for r in rows):>3}  tool calls {sum(len(r["calls"]) for r in rows):>3}  errors {len(errs):>2}  cost ${sum((r["cost"] or Decimal()) for r in rows):.3f}' + (f'   e.g. {errs[0]}' if errs else ''))

async def main():
    for m in ('openai:gpt-5.6-luna', 'openai:gpt-5.6-sol', 'anthropic:claude-opus-5'):
        await run(m)
asyncio.run(main())
