import { memo, useCallback, useMemo } from 'react'
import { copyText } from '../clipboard'
import { cachedMarkdown, renderMarkdown } from '../markdown'

/** Rendered, sanitized Markdown; a code block's Copy button copies its code. `live` text is still streaming. */
export const Markdown = memo(function Markdown({
  text,
  className,
  live = false,
}: {
  text: string
  className?: string
  live?: boolean
}) {
  const html = useMemo(() => (live ? renderMarkdown(text) : cachedMarkdown(text)), [text, live])
  const onClick = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-copy]')
    const code = button?.closest('.code-block')?.querySelector('code')
    if (button && code) void copyText(code.textContent ?? '', button)
  }, [])
  return <div className={`markdown ${className ?? ''}`} onClick={onClick} dangerouslySetInnerHTML={{ __html: html }} />
})
