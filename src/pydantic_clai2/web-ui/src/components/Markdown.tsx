import { memo, useCallback, useMemo } from 'react'
import { renderMarkdown } from '../markdown'

/** Rendered, sanitized Markdown; a code block's Copy button copies its code. */
export const Markdown = memo(function Markdown({ text, className }: { text: string; className?: string }) {
  const html = useMemo(() => renderMarkdown(text), [text])
  const onClick = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-copy]')
    const code = button?.closest('.code-block')?.querySelector('code')
    if (button && code) {
      const show = (label: string) => {
        button.textContent = label
        window.setTimeout(() => (button.textContent = 'Copy'), 1200)
      }
      navigator.clipboard.writeText(code.textContent ?? '').then(
        () => show('Copied'),
        () => show('Copy failed'),
      )
    }
  }, [])
  return <div className={`markdown ${className ?? ''}`} onClick={onClick} dangerouslySetInnerHTML={{ __html: html }} />
})
