import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Charting from './pages/Charting'
import Trade from './pages/Trade'
import Positions from './pages/Positions'
import Strategies from './pages/Strategies'
import Analytics from './pages/Analytics'
import AIChat from './pages/AIChat'
import Config from './pages/Config'
import Login from './pages/Login'

function PlaceholderPage({ title }: { title: string }) {
  return (
    <div className="flex items-center justify-center h-96">
      <div className="text-center">
        <h1 className="text-2xl font-bold text-text-primary mb-2">{title}</h1>
        <p className="text-text-secondary">This page is under construction.</p>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/*" element={
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/markets" element={<Charting />} />
            <Route path="/trade" element={<Trade />} />
            <Route path="/positions" element={<Positions />} />
            <Route path="/orders" element={<Positions />} />
            <Route path="/strategies" element={<Strategies />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/logs" element={<Analytics />} />
            <Route path="/config" element={<Config />} />
            <Route path="/tracker" element={<PlaceholderPage title="Position Tracker" />} />
            <Route path="/ai" element={<AIChat />} />
          </Routes>
        </Layout>
      } />
    </Routes>
  )
}
