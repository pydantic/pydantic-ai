import { useState } from 'react'
import type { AnswerBody, ApprovalInfo, Questions } from '../protocol'
import { DiffView } from './DiffView'
import { ShieldIcon } from './icons'

interface Props {
  request: Questions
  approval: ApprovalInfo
  asker: string
  answer: (id: string, answers: AnswerBody[] | null) => Promise<void>
  onError: (message: string) => void
}

/** A tool call the session's permission mode holds: approve it once or for the session, or deny it with a reason. */
export function ApprovalCard({ request, approval, asker, answer, onError }: Props) {
  const [reason, setReason] = useState('')
  const [busy, setBusy] = useState(false)
  const question = request.questions[0]
  const header = question?.header ?? ''
  // The server's question offers approve, approve for this session, and deny, in that order.
  const [approve = '', approveSession = '', deny = ''] = (question?.options ?? []).map((option) => option.label)

  const decide = async (label: string) => {
    setBusy(true)
    const typed = label === deny ? reason.trim() : ''
    const body: AnswerBody = { header, selected: typed ? [] : [label], custom: typed || null }
    try {
      await answer(request.id, [body])
    } catch (error) {
      onError(String(error instanceof Error ? error.message : error))
      setBusy(false)
    }
  }

  const what = approval.action === 'command' ? 'run a command' : 'write a file'
  return (
    <div className="question-card approval-card" role="group" aria-label="Approval">
      <div className="question-title">
        <ShieldIcon size={16} />
        {request.asker ?? asker} wants to {what}
        <span className="approval-tool">{approval.tool}</span>
      </div>
      <p className="approval-reason">{approval.reason}</p>
      <pre className="approval-subject">{approval.subject}</pre>
      {approval.diff && <DiffView diff={approval.diff} preview={12} />}
      <input
        className="option-custom"
        placeholder="Reason for denying (optional), sent to the model"
        aria-label="Reason for denying"
        value={reason}
        disabled={busy}
        onChange={(event) => setReason(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && reason.trim()) void decide(deny)
        }}
      />
      <div className="question-actions">
        <button className="ghost" onClick={() => void decide(deny)} disabled={busy}>
          {deny}
        </button>
        <button className="ghost" onClick={() => void decide(approveSession)} disabled={busy}>
          {approveSession}
        </button>
        <button className="primary" onClick={() => void decide(approve)} disabled={busy}>
          {approve}
        </button>
      </div>
    </div>
  )
}
