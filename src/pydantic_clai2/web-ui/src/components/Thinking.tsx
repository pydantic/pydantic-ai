import { memo, useState } from 'react'
import { BrainIcon, ChevronIcon } from './icons'
import { Markdown } from './Markdown'

export const Thinking = memo(function Thinking({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(false)
  if (!text.trim()) return null
  const shown = open || live
  return (
    <div className={`thinking ${shown ? 'open' : ''}`}>
      <button className="row-toggle" onClick={() => setOpen(!open)} aria-expanded={shown}>
        <BrainIcon size={14} />
        <span className={live ? 'shimmer' : ''}>{live ? 'Thinking' : 'Thought'}</span>
        <ChevronIcon size={12} className="chevron" />
      </button>
      {shown && <Markdown className="thinking-body" text={text} live={live} />}
    </div>
  )
})
