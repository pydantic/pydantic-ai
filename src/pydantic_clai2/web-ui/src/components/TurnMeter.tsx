import { useEffect, useState } from 'react'
import { compact, duration } from '../format'
import type { StatusInfo } from '../protocol'

/** The running turn's clock and token counts, ticking as it streams. */
export function TurnMeter({ status, startedAt }: { status: StatusInfo | null; startedAt: number | null }) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 250)
    return () => window.clearInterval(timer)
  }, [])
  const seconds = startedAt === null ? null : Math.max(0, (now - startedAt) / 1000)
  const streamed = status?.streamed_tokens ?? 0
  const rate = seconds && seconds > 1 && streamed ? Math.round(streamed / seconds) : null
  const input = status?.run?.input_tokens
  return (
    <span className="turn-meter" title="Output is estimated from the streamed text until the turn reports it">
      {seconds !== null && <span className="meter-time">{duration(seconds)}</span>}
      <span>
        <span className="meter-arrow">↓</span>~{compact(streamed)}
      </span>
      {rate !== null && <span className="meter-rate">{rate} tok/s</span>}
      {input ? (
        <span title="Input tokens of the requests finished so far">
          <span className="meter-arrow">↑</span>
          {compact(input)}
        </span>
      ) : null}
    </span>
  )
}
