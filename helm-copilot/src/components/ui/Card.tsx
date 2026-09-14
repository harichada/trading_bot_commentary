import { clsx } from 'clsx'

interface CardProps {
  children: React.ReactNode
  className?: string
  interactive?: boolean
  onClick?: () => void
}

export function Card({ children, className, interactive, onClick }: CardProps) {
  const Component = onClick ? 'button' : 'div'
  
  return (
    <Component
      className={clsx(
        'bg-helm-surface border border-helm-border rounded-lg',
        interactive && 'transition-all duration-150 hover:bg-helm-elevated/50 hover:border-helm-border/80 cursor-pointer',
        onClick && 'text-left w-full',
        className
      )}
      onClick={onClick}
    >
      {children}
    </Component>
  )
}

export function CardHeader({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={clsx('px-4 py-3 border-b border-helm-border', className)}>
      {children}
    </div>
  )
}

export function CardContent({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={clsx('px-4 py-3', className)}>
      {children}
    </div>
  )
}

export function CardFooter({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={clsx('px-4 py-3 border-t border-helm-border', className)}>
      {children}
    </div>
  )
}
