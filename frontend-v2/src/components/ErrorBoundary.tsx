import { Component, type ReactNode } from 'react'

interface Props { children: ReactNode; fallback?: ReactNode }
interface State { error: Error | null }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error) { return { error } }

  componentDidCatch(error: Error) { console.error('Page error:', error) }

  render() {
    if (this.state.error) {
      return this.props.fallback ?? (
        <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>
          <div style={{ fontSize: 16, marginBottom: 8, color: '#FF4D6A' }}>Something went wrong</div>
          <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: '#555', marginBottom: 16 }}>{this.state.error.message}</div>
          <button onClick={() => this.setState({ error: null })} style={{ padding: '8px 16px', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)', color: '#bdbdbd', cursor: 'pointer', fontSize: 12, fontFamily: 'inherit' }}>Retry</button>
        </div>
      )
    }
    return this.props.children
  }
}
