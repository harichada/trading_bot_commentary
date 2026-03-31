import { Construction } from 'lucide-react'

export default function Placeholder({ name }: { name: string }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      gap: 'var(--space-4)',
      color: 'var(--text-muted)',
    }}>
      <Construction size={48} strokeWidth={1} />
      <div style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-secondary)' }}>{name}</div>
      <div style={{ fontSize: 13 }}>Coming in Phase 2+</div>
    </div>
  )
}
