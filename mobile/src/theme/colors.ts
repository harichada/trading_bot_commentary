export const colors = {
  // Backgrounds
  bg: '#0A0E17',
  bgCard: '#0F1521',
  bgCardHover: '#141B2B',
  bgInput: '#111827',
  bgOverlay: 'rgba(10, 14, 23, 0.85)',

  // Accents
  cyan: '#00D4FF',
  cyanDim: 'rgba(0, 212, 255, 0.06)',
  cyanMuted: 'rgba(0, 212, 255, 0.15)',
  red: '#FF3B5C',
  redDim: 'rgba(255, 59, 92, 0.06)',
  green: '#10B981',
  greenDim: 'rgba(16, 185, 129, 0.10)',
  yellow: '#F59E0B',
  yellowDim: 'rgba(245, 158, 11, 0.10)',
  purple: '#8B5CF6',
  orange: '#F97316',

  // Text
  textPrimary: '#E2E8F0',
  textSecondary: '#94A3B8',
  textMuted: '#64748B',
  textDisabled: '#475569',

  // Borders
  border: 'rgba(255, 255, 255, 0.08)',
  borderActive: 'rgba(0, 212, 255, 0.3)',

  // Status
  statusTrading: '#10B981',
  statusPaused: '#F59E0B',
  statusStopped: '#64748B',
  statusScanning: '#00D4FF',
  statusHalted: '#FF3B5C',

  // Gradients (rgba arrays for LinearGradient)
  gradientGreen: ['rgba(16, 185, 129, 0.25)', 'rgba(16, 185, 129, 0.02)'] as const,
  gradientRed: ['rgba(255, 59, 92, 0.25)', 'rgba(255, 59, 92, 0.02)'] as const,
  gradientCyan: ['rgba(0, 212, 255, 0.20)', 'rgba(0, 212, 255, 0.02)'] as const,
  gradientPurple: ['rgba(139, 92, 246, 0.20)', 'rgba(139, 92, 246, 0.02)'] as const,
  gradientCard: ['rgba(15, 21, 33, 0.9)', 'rgba(15, 21, 33, 0.7)'] as const,
} as const;

export type ColorName = keyof typeof colors;
