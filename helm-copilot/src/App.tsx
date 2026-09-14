import { Routes, Route } from 'react-router-dom'
import { 
  Shell, 
  LivePulse, 
  DecisionCard, 
  PortfolioRadar, 
  InboxStub, 
  SettingsStub 
} from './components'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Shell />}>
        <Route index element={<LivePulse />} />
        <Route path="decision/:id" element={<DecisionCard />} />
        <Route path="radar" element={<PortfolioRadar />} />
        <Route path="inbox" element={<InboxStub />} />
        <Route path="settings" element={<SettingsStub />} />
      </Route>
    </Routes>
  )
}
