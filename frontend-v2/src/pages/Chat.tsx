import { useState, useRef, useEffect } from 'react'
import s from './Chat.module.css'

interface Message { role: 'user' | 'ai'; text: string }

const SUGGESTIONS = [
  'What gaps look good today?',
  'Explain the exhaustion entry',
  'Show my best performing strategy',
  'Why was TSLA rejected?',
  'What is the current regime?',
]

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const send = async (text?: string) => {
    const msg = text ?? input.trim()
    if (!msg) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', text: msg }])
    setLoading(true)
    try {
      const res = await fetch('/api/llm/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, { role: 'ai', text: data.reply ?? data.response ?? data.error ?? 'No response' }])
    } catch {
      setMessages(prev => [...prev, { role: 'ai', text: 'Error: Could not reach Rudra. Check if LLM is enabled.' }])
    }
    setLoading(false)
  }

  return (
    <div className={s.layout}>
      {/* Header */}
      <div className={s.header}>
        <div className={s.headerDot} />
        <span className={s.headerTitle}>Rudra</span>
        <span className={s.headerSub}>AI Trading Assistant</span>
      </div>

      {/* Messages */}
      <div className={s.messages}>
        {messages.length === 0 && (
          <div className={s.welcome}>
            <div className={s.welcomeTitle}>Ask Rudra anything about your trades</div>
            <div className={s.welcomeSub}>Market analysis, trade review, strategy questions</div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`${s.msg} ${m.role === 'user' ? s.msgUser : s.msgAi}`}>
            <div className={`${s.bubble} ${m.role === 'user' ? s.bubbleUser : s.bubbleAi}`}>
              {m.text}
            </div>
          </div>
        ))}
        {loading && (
          <div className={`${s.msg} ${s.msgAi}`}>
            <div className={`${s.bubble} ${s.bubbleAi}`}>
              <span className="thinking-dots"><span /><span /><span /></span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggestions */}
      {messages.length === 0 && (
        <div className={s.suggestions}>
          {SUGGESTIONS.map(q => (
            <button key={q} className={s.sugBtn} onClick={() => send(q)}>{q}</button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className={s.inputArea}>
        <input className={s.input} placeholder="Ask Rudra..." value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()} />
        <button className={s.sendBtn} onClick={() => send()} disabled={loading || !input.trim()}>→</button>
      </div>
    </div>
  )
}
