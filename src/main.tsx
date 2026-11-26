import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import { App } from './ui/App.tsx'

const root = document.getElementById('root')
if (!root) throw new Error('index.html must contain <div id="root"></div>')
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
