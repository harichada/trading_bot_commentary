import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import CommandPalette from './components/CommandPalette'
import ShortcutsModal from './components/ShortcutsModal'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import Dashboard from './pages/Dashboard'
import Scanner from './pages/Scanner'
import TradeLog from './pages/TradeLog'
import Intraday from './pages/Intraday'
import Tracker from './pages/Tracker'
import Backtest from './pages/Backtest'
import WalkForward from './pages/WalkForward'
import IntradayWF from './pages/IntradayWF'
import Strategies from './pages/Strategies'
import Config from './pages/Config'
import Database from './pages/Database'
import Chat from './pages/Chat'
import Guide from './pages/Guide'
import IntradayLab from './pages/IntradayLab'
import Replay from './pages/Replay'

function AppInner() {
  const { showHelp, setShowHelp } = useKeyboardShortcuts()

  return (
    <>
      <CommandPalette />
      {showHelp && <ShortcutsModal onClose={() => setShowHelp(false)} />}
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="/scanner" element={<Scanner />} />
          <Route path="/trades" element={<TradeLog />} />
          <Route path="/intraday" element={<Intraday />} />
          <Route path="/tracker" element={<Tracker />} />
          <Route path="/backtest" element={<Backtest />} />
          <Route path="/walkforward" element={<WalkForward />} />
          <Route path="/intraday-wf" element={<IntradayWF />} />
          <Route path="/strategies" element={<Strategies />} />
          <Route path="/config" element={<Config />} />
          <Route path="/database" element={<Database />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/guide" element={<Guide />} />
          <Route path="/lab" element={<IntradayLab />} />
          <Route path="/replay" element={<Replay />} />
        </Route>
      </Routes>
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppInner />
    </BrowserRouter>
  )
}
