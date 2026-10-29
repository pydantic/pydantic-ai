import { useContext, useState } from 'react'
import { SessionContext } from '../api'
import type { AnswerBody, Questions } from '../protocol'
import { CheckIcon, QuestionIcon } from './icons'

interface Props {
  request: Questions
  /** Who asks when the request names no one: CLAI, or the orchestrator. */
  asker?: string
  /** Where the answer goes: the shown session's questions by default. */
  onAnswer?: (id: string, answers: AnswerBody[] | null) => Promise<void>
  onError: (message: string) => void
}

/** The model's multiple-choice questions, answerable here or in the terminal, whichever comes first. */
export function QuestionCard({ request, asker = 'CLAI', onAnswer, onError }: Props) {
  const session = useContext(SessionContext)
  const answer = onAnswer ?? session.answer
  const [picked, setPicked] = useState<Record<string, string[]>>({})
  const [custom, setCustom] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const complete = request.questions.every((q) => (picked[q.header]?.length ?? 0) > 0 || (custom[q.header] ?? '').trim())

  const toggle = (header: string, label: string, multi: boolean) => {
    setCustom((current) => ({ ...current, [header]: '' }))
    setPicked((current) => {
      const chosen = current[header] ?? []
      if (!multi) return { ...current, [header]: [label] }
      return { ...current, [header]: chosen.includes(label) ? chosen.filter((l) => l !== label) : [...chosen, label] }
    })
  }

  const submit = async (decline: boolean) => {
    setBusy(true)
    const answers: AnswerBody[] | null = decline
      ? null
      : request.questions.map((q) => {
          const typed = (custom[q.header] ?? '').trim()
          return { header: q.header, selected: typed ? [] : (picked[q.header] ?? []), custom: typed || null }
        })
    try {
      await answer(request.id, answers)
    } catch (error) {
      onError(String(error instanceof Error ? error.message : error))
      setBusy(false)
    }
  }

  return (
    <div className="question-card">
      <div className="question-title">
        <QuestionIcon size={16} />
        {request.asker ?? asker} needs your input
      </div>
      {request.questions.map((q) => (
        <fieldset key={q.header} className="question">
          <legend>
            <span className="question-header">{q.header}</span>
            {q.question}
          </legend>
          <div className="options">
            {q.options.map((option, index) => {
              const selected = picked[q.header]?.includes(option.label) ?? false
              return (
                <button
                  key={option.label}
                  className={`option ${selected ? 'selected' : ''}`}
                  onClick={() => toggle(q.header, option.label, q.multi_select)}
                  disabled={busy}
                >
                  <span className="option-key">{selected ? <CheckIcon size={12} /> : index + 1}</span>
                  <span>
                    <span className="option-label">{option.label}</span>
                    {option.description && <span className="option-description">{option.description}</span>}
                  </span>
                </button>
              )
            })}
          </div>
          <input
            className="option-custom"
            placeholder="Or type your own answer"
            value={custom[q.header] ?? ''}
            disabled={busy}
            onChange={(event) => {
              const value = event.target.value
              setCustom((current) => ({ ...current, [q.header]: value }))
              if (value) setPicked((current) => ({ ...current, [q.header]: [] }))
            }}
          />
        </fieldset>
      ))}
      <div className="question-actions">
        <button className="ghost" onClick={() => void submit(true)} disabled={busy}>
          Decline
        </button>
        <button className="primary" onClick={() => void submit(false)} disabled={busy || !complete}>
          Answer
        </button>
      </div>
    </div>
  )
}
