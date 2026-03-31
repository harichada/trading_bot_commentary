import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api'

const PAGE_MAP: Record<string, string> = {
  '1': '/', '2': '/scanner', '3': '/trades', '4': '/backtest',
  '5': '/strategies', '6': '/config', '7': '/database', '8': '/tracker', '9': '/chat',
}

export function useKeyboardShortcuts() {
  const navigate = useNavigate()
  const [showHelp, setShowHelp] = useState(false)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const inInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT'

      // Ctrl+K — command palette (handled by CommandPalette component)
      // Ctrl+S — save config
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault()
        // Dispatch custom event for config page to catch
        window.dispatchEvent(new CustomEvent('rudra:save-config'))
      }

      // Ctrl+Z — undo config
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        window.dispatchEvent(new CustomEvent('rudra:undo-config'))
      }

      // Ctrl+Shift+Z — redo config
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
        window.dispatchEvent(new CustomEvent('rudra:redo-config'))
      }

      // Ctrl+E — emergency stop
      if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault()
        if (confirm('EMERGENCY STOP — Close all positions and halt trading?')) {
          api.stop()
        }
      }

      // Escape — close modals (handled by individual components)
      if (e.key === 'Escape') {
        setShowHelp(false)
      }

      // ? — show help
      if (e.key === '?' && !inInput) {
        e.preventDefault()
        setShowHelp(h => !h)
      }

      // / — focus config search
      if (e.key === '/' && !inInput) {
        e.preventDefault()
        navigate('/config')
        setTimeout(() => {
          (document.querySelector('[data-config-search]') as HTMLInputElement)?.focus()
        }, 100)
      }

      // 1-9 — page navigation (only when not in input)
      if (!inInput && !e.ctrlKey && !e.metaKey && PAGE_MAP[e.key]) {
        navigate(PAGE_MAP[e.key])
      }
    }

    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [navigate])

  return { showHelp, setShowHelp }
}

export const SHORTCUTS = [
  { keys: '1-9', desc: 'Navigate to page' },
  { keys: 'Ctrl+K', desc: 'Command palette' },
  { keys: 'Ctrl+S', desc: 'Save config' },
  { keys: 'Ctrl+Z', desc: 'Undo config change' },
  { keys: 'Ctrl+Shift+Z', desc: 'Redo config change' },
  { keys: 'Ctrl+E', desc: 'Emergency stop' },
  { keys: '/', desc: 'Jump to config search' },
  { keys: '?', desc: 'Show keyboard shortcuts' },
  { keys: 'Esc', desc: 'Close modal / panel' },
]
