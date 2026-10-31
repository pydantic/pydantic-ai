/** Copy on the browser's device and briefly show the result on the triggering button. */
export async function copyText(text: string, button: HTMLButtonElement): Promise<void> {
  try {
    await navigator.clipboard.writeText(text)
    button.textContent = 'Copied'
  } catch {
    button.textContent = 'Copy failed'
  }
  window.setTimeout(() => (button.textContent = 'Copy'), 1200)
}
