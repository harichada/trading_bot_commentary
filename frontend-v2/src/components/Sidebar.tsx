import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, ScanSearch, Receipt, CandlestickChart, Activity,
  BarChart2, GitBranch, Layers, Settings2, Database, Bot, BookOpen,
  PanelLeftClose, PanelLeftOpen, X, FlaskConical, Play,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import styles from './Sidebar.module.css'

interface NavItem {
  path: string
  icon: LucideIcon
  label: string
  dot?: boolean  // live indicator
}

interface NavSection {
  label: string
  items: NavItem[]
}

const navSections: NavSection[] = [
  {
    label: 'TRADING',
    items: [
      { path: '/',           icon: LayoutDashboard,  label: 'Dashboard', dot: true },
      { path: '/scanner',    icon: ScanSearch,        label: 'Scanner' },
      { path: '/trades',     icon: Receipt,           label: 'Trade Log' },
      { path: '/intraday',   icon: CandlestickChart,  label: 'Intraday' },
      { path: '/tracker',    icon: Activity,          label: 'Tracker' },
    ],
  },
  {
    label: 'ANALYSIS',
    items: [
      { path: '/backtest',     icon: BarChart2,   label: 'Backtest' },
      { path: '/walkforward',  icon: GitBranch,   label: 'Walk-Forward' },
      { path: '/intraday-wf',  icon: GitBranch,   label: 'Intraday WF' },
      { path: '/lab',          icon: FlaskConical, label: 'Strategy Lab' },
      { path: '/replay',      icon: Play,         label: 'Visual Replay' },
    ],
  },
  {
    label: 'SYSTEM',
    items: [
      { path: '/strategies', icon: Layers,    label: 'Strategies' },
      { path: '/config',     icon: Settings2, label: 'Config' },
      { path: '/database',   icon: Database,  label: 'Database' },
      { path: '/chat',       icon: Bot,       label: 'Rudra Chat' },
      { path: '/guide',      icon: BookOpen,  label: 'Guide' },
    ],
  },
]

interface SidebarProps {
  open: boolean
  collapsed: boolean
  onClose: () => void
  onToggleCollapse: () => void
  botLive?: boolean
}

function RudraLogo() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <polygon points="14,2 26,22 2,22" stroke="#00FFBB" strokeWidth="1.5" fill="none" />
      <polygon points="14,8 21,20 7,20" fill="#00FFBB" opacity="0.3" />
      <circle cx="14" cy="14" r="2" fill="#00FFBB" />
    </svg>
  )
}

export default function Sidebar({ open, collapsed, onClose, onToggleCollapse, botLive }: SidebarProps) {
  return (
    <>
      {open && <div className={styles.overlay} onClick={onClose} />}

      <aside className={`${styles.sidebar} ${open ? styles.open : ''} ${collapsed ? styles.collapsed : ''}`}>
        {/* Brand */}
        <div className={styles.brand}>
          <div className={styles.logoMark}><RudraLogo /></div>
          <div className={styles.brandText}>
            <div className={styles.brandName}>RUDRA</div>
            <div className={styles.brandSub}>Gap Fade Terminal</div>
          </div>
          <button className={styles.mobileClose} onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        {/* Automate button (like Obside) */}
        <button className={styles.autoBtn} onClick={() => {
          fetch('/api/start', { method: 'POST' })
        }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
          <span className={styles.autoLabel}>Automate</span>
        </button>

        {/* Scrollable nav */}
        <div className={styles.navScroll}>
          {navSections.map((section) => (
            <div key={section.label}>
              <div className={styles.sectionLabel}>{section.label}</div>
              {section.items.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === '/'}
                  onClick={onClose}
                  className={({ isActive }) =>
                    `${styles.navItem} ${isActive ? styles.active : ''}`
                  }
                >
                  <item.icon size={16} className={styles.navIcon} />
                  <span className={styles.navLabel}>{item.label}</span>
                  {item.dot && botLive && <span className={styles.navDot} />}
                </NavLink>
              ))}
            </div>
          ))}
        </div>

        {/* Collapse toggle */}
        <div className={styles.toggle} onClick={onToggleCollapse}>
          {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          <span className={styles.toggleLabel}>{collapsed ? 'Expand' : 'Collapse'}</span>
        </div>
      </aside>
    </>
  )
}
