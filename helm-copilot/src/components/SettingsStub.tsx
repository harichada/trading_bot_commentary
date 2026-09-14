import { Settings, Server, Palette, Bell, Shield, Database } from 'lucide-react'
import { Card, CardContent } from './ui/Card'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:9000'

export function SettingsStub() {
  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <Settings className="w-5 h-5 text-cyan-400" />
        <h2 className="text-lg font-semibold text-white">Settings</h2>
      </div>

      {/* Connection Info */}
      <Card className="mb-4">
        <CardContent>
          <div className="flex items-center gap-3 mb-4">
            <Server className="w-5 h-5 text-cyan-400" />
            <h3 className="font-medium text-white">API Connection</h3>
          </div>
          
          <div className="space-y-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-zinc-400">API Base URL</span>
              <code className="text-xs bg-helm-elevated px-2 py-1 rounded text-zinc-300">
                {API_BASE}
              </code>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-zinc-400">WebSocket URL</span>
              <code className="text-xs bg-helm-elevated px-2 py-1 rounded text-zinc-300">
                {API_BASE.replace(/^http/, 'ws')}/ws
              </code>
            </div>
            
            <p className="text-xs text-zinc-500 mt-4">
              Configure <code className="bg-helm-elevated px-1 rounded">VITE_API_BASE</code> and 
              <code className="bg-helm-elevated px-1 rounded ml-1">VITE_API_KEY</code> in your 
              <code className="bg-helm-elevated px-1 rounded ml-1">.env</code> file to change the API endpoint.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Placeholder Settings */}
      <Card>
        <CardContent className="py-8">
          <h3 className="text-lg font-medium text-white mb-6 text-center">
            Settings Coming Soon
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-helm-elevated/50 border border-helm-border">
              <Palette className="w-5 h-5 text-cyan-400 mb-2" />
              <h4 className="text-sm font-medium text-white mb-1">Appearance</h4>
              <p className="text-xs text-zinc-500">
                Theme, font size, and display preferences
              </p>
            </div>
            
            <div className="p-4 rounded-lg bg-helm-elevated/50 border border-helm-border">
              <Bell className="w-5 h-5 text-cyan-400 mb-2" />
              <h4 className="text-sm font-medium text-white mb-1">Notifications</h4>
              <p className="text-xs text-zinc-500">
                Alert sounds, push notifications, and email
              </p>
            </div>
            
            <div className="p-4 rounded-lg bg-helm-elevated/50 border border-helm-border">
              <Shield className="w-5 h-5 text-cyan-400 mb-2" />
              <h4 className="text-sm font-medium text-white mb-1">Risk Controls</h4>
              <p className="text-xs text-zinc-500">
                Position limits, confirmation requirements
              </p>
            </div>
            
            <div className="p-4 rounded-lg bg-helm-elevated/50 border border-helm-border">
              <Database className="w-5 h-5 text-cyan-400 mb-2" />
              <h4 className="text-sm font-medium text-white mb-1">Data & Privacy</h4>
              <p className="text-xs text-zinc-500">
                Cache settings, data retention, export
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Disclaimer */}
      <div className="mt-6 p-4 rounded-lg bg-amber-500/10 border border-amber-500/20">
        <p className="text-sm text-amber-400">
          <strong>Important:</strong> Helm is an informational/educational tool only. 
          It does not provide personalized financial advice. All trading decisions 
          are made at your own risk. Past performance is not indicative of future results.
        </p>
      </div>
    </div>
  )
}
