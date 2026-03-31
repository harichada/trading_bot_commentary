import { SHORTCUTS } from '../hooks/useKeyboardShortcuts'
import s from './StatusBar.module.css' // reuse modal styles

export default function ShortcutsModal({ onClose }: { onClose: () => void }) {
  return (
    <div className={s.modalOverlay} onClick={onClose}>
      <div className={s.modal} onClick={e => e.stopPropagation()} style={{ width: 400 }}>
        <div className={s.modalHeader}>
          <span>Keyboard Shortcuts</span>
          <button className={s.modalClose} onClick={onClose}>✕</button>
        </div>
        <div className={s.modalBody}>
          {SHORTCUTS.map(sc => (
            <div key={sc.keys} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <span style={{ color: '#bdbdbd', fontSize: 12 }}>{sc.desc}</span>
              <kbd style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#00FFBB', background: 'rgba(0,255,187,0.06)', padding: '2px 8px', borderRadius: 2 }}>{sc.keys}</kbd>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
