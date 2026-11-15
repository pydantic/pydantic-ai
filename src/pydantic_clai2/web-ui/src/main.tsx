import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { preconnect } from './api'
import { App } from './App'
import './styles.css'

// The snapshot of the session to show is on its way while React renders the page.
preconnect()

const root = document.getElementById('root')
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
}
