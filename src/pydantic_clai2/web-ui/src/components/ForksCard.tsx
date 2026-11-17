import { duration } from '../format'
import type { ForkEntry } from '../protocol'
import type { NoticeEntry } from '../state'
import { Markdown } from './Markdown'

/** What `/forks` reported, as a card: its totals, and each fork as its terminal table lists it. */
export function ForksCard({ notice, forks }: { notice: NoticeEntry; forks: ForkEntry[] }) {
  return (
    <section className="usage-card" aria-label="Forks">
      <header className="usage-card-header">
        <span className="usage-card-title">Forks</span>
        <span className="usage-card-scope">{notice.text}</span>
        {notice.command && <code className="notice-command">{notice.command}</code>}
      </header>
      <div className="usage-table-wrap">
        <table className="usage-table forks-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Model</th>
              <th>Status</th>
              <th>Time</th>
              <th>Session</th>
              <th>Prompt</th>
            </tr>
          </thead>
          <tbody>
            {forks.map((fork) => (
              <tr key={fork.id}>
                <th scope="row">{fork.id}</th>
                <td>{fork.model}</td>
                <td className={`fork-status ${fork.status}`}>{fork.status}</td>
                <td>{duration(fork.elapsed)}</td>
                <td>{fork.session_id ?? ''}</td>
                <td title={fork.prompt}>
                  <span className="fork-prompt">{fork.prompt.split(/\s+/).join(' ')}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

/** A background `/fork` that ended, as the terminal prints it: its answer, or why it failed or that it was cancelled. */
export function ForkResult({ notice, fork }: { notice: NoticeEntry; fork: ForkEntry }) {
  const ended = fork.status === 'done' ? 'finished' : fork.status
  return (
    <section className={`usage-card fork-result ${fork.status}`} aria-label={`Fork #${fork.id}`}>
      <header className="usage-card-header">
        <span className="usage-card-title">Fork #{fork.id}</span>
        <span className="usage-card-scope">{fork.model}</span>
        <code className="notice-command">/fork {fork.prompt.split(/\s+/).join(' ')}</code>
      </header>
      {fork.status === 'done' ? (
        <Markdown className="assistant" text={notice.text} />
      ) : (
        notice.text && <pre className="fork-error">{notice.text}</pre>
      )}
      <p className="usage-note">
        {ended} in {duration(fork.elapsed)}
        {fork.session_id && <> · continue it with /resume {fork.session_id}</>}
      </p>
    </section>
  )
}
