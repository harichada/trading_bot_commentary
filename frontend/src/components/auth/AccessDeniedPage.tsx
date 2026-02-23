export function AccessDeniedPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0A0E17] p-4">
      <div
        className="w-full max-w-sm rounded-2xl border border-[#FF3B5C]/30 p-8 text-center"
        style={{
          background: 'rgba(255,59,92,0.04)',
          backdropFilter: 'blur(20px)',
          boxShadow: '0 0 60px rgba(255,59,92,0.08)',
        }}
      >
        {/* Shield icon */}
        <div className="flex justify-center mb-4">
          <svg
            viewBox="0 0 24 24"
            className="w-12 h-12 text-[#FF3B5C]"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
        </div>

        <h1 className="text-xl font-bold text-white mb-2">Access Denied</h1>
        <p className="text-[#64748b] text-sm mb-6">
          Your account is not authorized to access this dashboard.
          Contact the administrator to request access.
        </p>

        <a
          href="/api/auth/login/google"
          className="inline-block text-sm font-medium text-[#00D4FF] hover:underline"
        >
          Try a different account
        </a>
      </div>
    </div>
  )
}
