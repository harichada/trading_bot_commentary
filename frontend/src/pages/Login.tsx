import { useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'

// Auth API base — goes through Vite proxy to gap_fade_app on port 8003
const AUTH_BASE = '/gf/api/auth'

export default function Login() {
  const navigate = useNavigate()
  const [checking, setChecking] = useState(true)
  const [authEnabled, setAuthEnabled] = useState(false)

  // Check if already logged in
  useEffect(() => {
    fetch(`${AUTH_BASE}/me`, { credentials: 'include' })
      .then(r => r.json())
      .then(data => {
        if (data.user) {
          navigate('/') // Already authenticated
        }
        setAuthEnabled(data.auth_enabled !== false)
        setChecking(false)
      })
      .catch(() => {
        setAuthEnabled(false)
        setChecking(false)
      })
  }, [navigate])

  const loginWith = (provider: string) => {
    // Redirect to the OAuth login endpoint on the backend
    // The backend handles the OAuth flow and redirects back with a session cookie
    window.location.href = `${AUTH_BASE}/login/${provider}`
  }

  if (checking) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-black">
        <div className="text-[#8a7535] text-sm">Checking authentication...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-[#1a1500] to-[#0a0a00]">
      {/* Logo */}
      <div className="mb-8 text-center">
        <div className="w-16 h-16 mx-auto mb-4 border-2 border-[#c9a84c] rounded-lg flex items-center justify-center">
          <svg viewBox="0 0 40 40" className="w-10 h-10 text-[#c9a84c]" fill="currentColor">
            <path d="M20 4L8 10v4h24v-4L20 4zm-8 8v14h4V12h-4zm6 0v14h4V12h-4zm6 0v14h4V12h-4zM6 28v4h28v-4H6z" />
          </svg>
        </div>
        <h1 className="text-2xl font-bold text-[#c9a84c] tracking-[0.3em] uppercase">RUDRA</h1>
        <p className="text-xs text-[#8a7535] tracking-[0.2em] uppercase mt-1">Institutional Trading Engine</p>
      </div>

      {/* Login Card */}
      <div className="w-96 bg-[#1a1508]/80 rounded-xl border border-[#3a2f10] p-6 backdrop-blur-sm">
        <h2 className="text-center text-lg font-semibold text-[#c9a84c] mb-6">Secure Gateway Access</h2>

        {!authEnabled && (
          <div className="mb-4 p-3 rounded-lg bg-yellow-900/20 border border-yellow-700/30 text-yellow-500 text-xs text-center">
            Auth not configured on backend. Click "Continue" to proceed without login.
          </div>
        )}

        <div className="space-y-3">
          {/* Google */}
          <button
            onClick={() => loginWith('google')}
            className="w-full flex items-center justify-between px-4 py-3 rounded-lg bg-[#0d0a00] border border-[#3a2f10] text-[#c9a84c] hover:border-[#c9a84c]/50 transition-colors"
          >
            <span className="text-sm font-bold tracking-wider">GOOGLE</span>
            <span className="text-xs text-[#8a7535]">LOGIN WITH GOOGLE &rarr;</span>
          </button>

          {/* GitHub */}
          <button
            onClick={() => loginWith('github')}
            className="w-full flex items-center justify-between px-4 py-3 rounded-lg bg-[#0d0a00] border border-[#3a2f10] text-[#c9a84c] hover:border-[#c9a84c]/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <svg viewBox="0 0 16 16" className="w-4 h-4" fill="currentColor">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
              </svg>
              <span className="text-sm font-bold tracking-wider">LOGIN WITH GITHUB</span>
            </div>
            <span className="text-xs text-[#8a7535]">&rarr;</span>
          </button>

          {/* Discord */}
          <button
            onClick={() => loginWith('discord')}
            className="w-full flex items-center justify-between px-4 py-3 rounded-lg bg-[#0d0a00] border border-[#3a2f10] text-[#c9a84c] hover:border-[#c9a84c]/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <svg viewBox="0 0 16 16" className="w-4 h-4 text-[#5865F2]" fill="currentColor">
                <path d="M13.55 3.15A13.3 13.3 0 0010.3 2a9.7 9.7 0 00-.43.87 12.4 12.4 0 00-3.74 0A9.4 9.4 0 005.7 2a13.4 13.4 0 00-3.26 1.15A14.6 14.6 0 00.1 12.5a13.5 13.5 0 004.1 2.1 10 10 0 00.9-1.47 8.7 8.7 0 01-1.4-.68l.33-.26a9.6 9.6 0 008.17 0l.34.26a8.8 8.8 0 01-1.41.68 10.1 10.1 0 00.9 1.47 13.4 13.4 0 004.1-2.1A14.5 14.5 0 0013.55 3.15zM5.35 10.7c-.87 0-1.59-.8-1.59-1.8s.7-1.8 1.59-1.8 1.6.81 1.59 1.8-.7 1.8-1.59 1.8zm5.3 0c-.87 0-1.59-.8-1.59-1.8s.7-1.8 1.59-1.8 1.6.81 1.59 1.8c0 1-.7 1.8-1.59 1.8z" />
              </svg>
              <span className="text-sm font-bold tracking-wider">LOGIN WITH DISCORD</span>
            </div>
            <span className="text-xs text-[#8a7535]">&rarr;</span>
          </button>
        </div>

        <div className="my-5 flex items-center gap-3">
          <div className="flex-1 h-px bg-[#3a2f10]" />
          <span className="text-[10px] text-[#8a7535] uppercase tracking-wider">Or Direct Entry</span>
          <div className="flex-1 h-px bg-[#3a2f10]" />
        </div>

        {/* Continue Button — skip login (works when auth is disabled) */}
        <button
          onClick={() => navigate('/')}
          className="w-full py-3 rounded-lg bg-gradient-to-r from-[#8a6914] to-[#c9a84c] text-black font-bold text-sm hover:opacity-90 transition-opacity"
        >
          CONTINUE TO TERMINAL
        </button>

        <div className="text-center mt-4">
          <button
            onClick={() => navigate('/')}
            className="text-[10px] text-[#8a7535] hover:text-[#c9a84c] transition-colors"
          >
            &#10005; Skip Authentication
          </button>
        </div>
      </div>

      {/* Security Badge */}
      <div className="mt-6 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-green" />
        <span className="text-[10px] text-[#8a7535] uppercase tracking-wider">Institutional Grade Security Protocol Active</span>
      </div>

      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 py-4 flex items-center justify-between px-8 text-[10px] text-[#5a4d20]">
        <span>&copy; 2024 RUDRA TRADING ENGINE. INSTITUTIONAL GRADE SECURITY.</span>
        <div className="flex gap-6">
          <button className="hover:text-[#8a7535] transition-colors">PRIVACY POLICY</button>
          <button className="hover:text-[#8a7535] transition-colors">TERMS OF SERVICE</button>
          <button className="hover:text-[#8a7535] transition-colors">SECURITY ARCHITECTURE</button>
          <button className="hover:text-[#8a7535] transition-colors">CONTACT SUPPORT</button>
        </div>
      </footer>
    </div>
  )
}
