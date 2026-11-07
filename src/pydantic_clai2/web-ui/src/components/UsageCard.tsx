import { compact } from '../format'
import type { TurnUsage, UsageReport } from '../protocol'
import type { NoticeEntry } from '../state'

const number = (value: number) => value.toLocaleString()

/** What `/usage` or `/cost` reported, as a card: the totals, and for `/usage` each turn, as its terminal table lists them. */
export function UsageCard({ notice, usage }: { notice: NoticeEntry; usage: UsageReport }) {
  const command = notice.command ?? ''
  const perTurn = command.split(/\s+/)[0] === '/usage'
  // The first line is the cost line the figures replace; the rest, such as background naming, stays as text.
  const notes = notice.text.split('\n').slice(1).filter(Boolean)
  const { total } = usage
  const cached = usage.turns.some((turn) => turn.cache_read_tokens || turn.cache_write_tokens)
  return (
    <section className="usage-card" aria-label={perTurn ? 'Session usage' : 'Session cost'}>
      <header className="usage-card-header">
        <span className="usage-card-title">{perTurn ? 'Usage' : 'Cost'}</span>
        <span className="usage-card-scope">retained history</span>
        {command && <code className="notice-command">{command}</code>}
      </header>
      {usage.turns.length === 0 ? (
        <p className="usage-card-empty">No usage yet.</p>
      ) : (
        <>
          <dl className="usage-stats">
            <div>
              <dt>Cost</dt>
              <dd>{total.cost ?? 'unknown'}</dd>
            </div>
            <div>
              <dt>Tokens</dt>
              <dd title={`${number(total.input_tokens + total.output_tokens)} tokens`}>
                {compact(total.input_tokens + total.output_tokens)}
              </dd>
              <span className="usage-split">
                ↑ {compact(total.input_tokens)} · ↓ {compact(total.output_tokens)}
              </span>
            </div>
            <div>
              <dt>Requests</dt>
              <dd>{number(total.requests)}</dd>
              <span className="usage-split">
                {usage.turns.length} {usage.turns.length === 1 ? 'turn' : 'turns'}
              </span>
            </div>
          </dl>
          {perTurn && (
            <div className="usage-table-wrap">
              <table className="usage-table">
                <thead>
                  <tr>
                    <th>Turn</th>
                    <th>Requests</th>
                    <th>Input</th>
                    {cached && <th>Cache read</th>}
                    {cached && <th>Cache write</th>}
                    <th>Output</th>
                    <th>Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {usage.turns.map((turn, index) => (
                    <Row key={index} label={String(index + 1)} turn={turn} cached={cached} />
                  ))}
                </tbody>
                <tfoot>
                  <Row label="Total" turn={total} cached={cached} />
                </tfoot>
              </table>
            </div>
          )}
        </>
      )}
      {usage.unpriced.length > 0 && (
        <p className="usage-note">
          No price data for {usage.unpriced.join(', ')}; those responses are not counted in costs.
        </p>
      )}
      {usage.unmeasured_requests > 0 && (
        <p className="usage-note">
          Token and cost totals are lower bounds because {number(usage.unmeasured_requests)}{' '}
          {usage.unmeasured_requests === 1 ? 'response' : 'responses'} omitted usage information.
        </p>
      )}
      {notes.map((note) => (
        <p key={note} className="usage-note">
          {note}
        </p>
      ))}
    </section>
  )
}

function Row({ label, turn, cached }: { label: string; turn: TurnUsage; cached: boolean }) {
  return (
    <tr>
      <th scope="row">{label}</th>
      <td>{number(turn.requests)}</td>
      <td>{number(turn.input_tokens)}</td>
      {cached && <td>{number(turn.cache_read_tokens)}</td>}
      {cached && <td>{number(turn.cache_write_tokens)}</td>}
      <td>{number(turn.output_tokens)}</td>
      <td>{turn.cost ?? 'unknown'}</td>
    </tr>
  )
}
