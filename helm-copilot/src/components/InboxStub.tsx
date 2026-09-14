import { Inbox, Bell, BellOff } from 'lucide-react'
import { Card, CardContent } from './ui/Card'

export function InboxStub() {
  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <Inbox className="w-5 h-5 text-cyan-400" />
        <h2 className="text-lg font-semibold text-white">Inbox</h2>
      </div>

      {/* Placeholder */}
      <Card>
        <CardContent className="py-16 text-center">
          <div className="w-16 h-16 rounded-full bg-helm-elevated flex items-center justify-center mx-auto mb-4">
            <Bell className="w-8 h-8 text-zinc-500" />
          </div>
          <h3 className="text-lg font-medium text-white mb-2">
            Inbox Coming Soon
          </h3>
          <p className="text-sm text-zinc-400 max-w-md mx-auto">
            This is where you'll see alerts, notifications, and messages from the bot — 
            including trade confirmations, risk warnings, and important market events.
          </p>
          
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4 text-left">
            <div className="p-4 rounded-lg bg-helm-elevated/50 border border-helm-border">
              <Bell className="w-5 h-5 text-cyan-400 mb-2" />
              <h4 className="text-sm font-medium text-white mb-1">Trade Alerts</h4>
              <p className="text-xs text-zinc-500">
                Entry/exit notifications and confirmation requests
              </p>
            </div>
            
            <div className="p-4 rounded-lg bg-helm-elevated/50 border border-helm-border">
              <Bell className="w-5 h-5 text-amber-400 mb-2" />
              <h4 className="text-sm font-medium text-white mb-1">Risk Warnings</h4>
              <p className="text-xs text-zinc-500">
                Circuit breaker triggers and position sizing alerts
              </p>
            </div>
            
            <div className="p-4 rounded-lg bg-helm-elevated/50 border border-helm-border">
              <BellOff className="w-5 h-5 text-zinc-400 mb-2" />
              <h4 className="text-sm font-medium text-white mb-1">Muted Symbols</h4>
              <p className="text-xs text-zinc-500">
                Manage which symbols you've silenced
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
