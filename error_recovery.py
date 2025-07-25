"""
Error recovery and state management for handling failures gracefully
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ErrorRecoveryManager:
    """Manages error recovery state and coordinates recovery actions"""
    
    def __init__(self, state_file: str = "error_recovery_state.json"):
        self.state_file = Path(state_file)
        self.state = self._load_state()
        
    def _load_state(self) -> Dict:
        """Load recovery state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load recovery state: {e}")
        
        return {
            'failed_orders': {},
            'circuit_breakers': {},
            'error_counts': {},
            'last_errors': {},
            'recovery_actions': []
        }
    
    def _save_state(self):
        """Persist recovery state to disk"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save recovery state: {e}")
    
    def record_order_failure(self, order_id: str, symbol: str, error: Exception, order_details: Dict):
        """Record a failed order for potential retry"""
        self.state['failed_orders'][order_id] = {
            'symbol': symbol,
            'error': str(error),
            'error_type': type(error).__name__,
            'timestamp': datetime.now().isoformat(),
            'order_details': order_details,
            'retry_count': 0,
            'status': 'pending_retry'
        }
        self._save_state()
    
    def get_pending_retries(self) -> List[Dict]:
        """Get orders that should be retried"""
        pending = []
        for order_id, details in self.state['failed_orders'].items():
            if details['status'] == 'pending_retry' and details['retry_count'] < 3:
                # Check if enough time has passed (exponential backoff)
                last_attempt = datetime.fromisoformat(details['timestamp'])
                wait_time = timedelta(minutes=2 ** details['retry_count'])
                if datetime.now() - last_attempt > wait_time:
                    pending.append({
                        'order_id': order_id,
                        **details
                    })
        return pending
    
    def mark_retry_attempted(self, order_id: str):
        """Update retry count for an order"""
        if order_id in self.state['failed_orders']:
            self.state['failed_orders'][order_id]['retry_count'] += 1
            self.state['failed_orders'][order_id]['timestamp'] = datetime.now().isoformat()
            self._save_state()
    
    def mark_order_recovered(self, order_id: str):
        """Mark an order as successfully recovered"""
        if order_id in self.state['failed_orders']:
            self.state['failed_orders'][order_id]['status'] = 'recovered'
            self.state['failed_orders'][order_id]['recovered_at'] = datetime.now().isoformat()
            self._save_state()
    
    def record_circuit_breaker_open(self, component: str, reason: str):
        """Record when a circuit breaker opens"""
        self.state['circuit_breakers'][component] = {
            'status': 'open',
            'reason': reason,
            'opened_at': datetime.now().isoformat(),
            'expected_recovery': (datetime.now() + timedelta(minutes=5)).isoformat()
        }
        self._save_state()
    
    def is_component_available(self, component: str) -> bool:
        """Check if a component is available (circuit breaker closed)"""
        if component not in self.state['circuit_breakers']:
            return True
        
        breaker = self.state['circuit_breakers'][component]
        if breaker['status'] == 'open':
            recovery_time = datetime.fromisoformat(breaker['expected_recovery'])
            if datetime.now() > recovery_time:
                # Auto-close after recovery time
                breaker['status'] = 'closed'
                breaker['closed_at'] = datetime.now().isoformat()
                self._save_state()
                return True
            return False
        return True
    
    def increment_error_count(self, error_type: str, symbol: Optional[str] = None):
        """Track error frequency for monitoring"""
        key = f"{error_type}:{symbol}" if symbol else error_type
        if key not in self.state['error_counts']:
            self.state['error_counts'][key] = {
                'count': 0,
                'first_seen': datetime.now().isoformat(),
                'last_seen': None
            }
        
        self.state['error_counts'][key]['count'] += 1
        self.state['error_counts'][key]['last_seen'] = datetime.now().isoformat()
        
        # Record last error details
        self.state['last_errors'][error_type] = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol
        }
        
        self._save_state()
    
    def get_error_summary(self) -> Dict:
        """Get summary of recent errors"""
        summary = {
            'total_errors': sum(e['count'] for e in self.state['error_counts'].values()),
            'error_types': {},
            'recent_errors': []
        }
        
        # Group by error type
        for key, data in self.state['error_counts'].items():
            error_type = key.split(':')[0]
            if error_type not in summary['error_types']:
                summary['error_types'][error_type] = 0
            summary['error_types'][error_type] += data['count']
        
        # Get recent errors (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        for error_type, details in self.state['last_errors'].items():
            if datetime.fromisoformat(details['timestamp']) > cutoff:
                summary['recent_errors'].append({
                    'type': error_type,
                    **details
                })
        
        return summary
    
    def suggest_recovery_action(self, error: Exception) -> Optional[str]:
        """Suggest recovery action based on error type"""
        from trading_exceptions import (
            InsufficientFundsException, RateLimitException,
            DailyLossLimitException, CircuitBreakerException,
            AuthenticationException
        )
        
        if isinstance(error, InsufficientFundsException):
            return "Reduce position size or wait for funds to settle"
        elif isinstance(error, RateLimitException):
            return f"Wait {error.retry_after} seconds before retrying"
        elif isinstance(error, DailyLossLimitException):
            return "Stop trading for the day - loss limit reached"
        elif isinstance(error, CircuitBreakerException):
            return f"System protection activated - wait {error.cooldown_minutes} minutes"
        elif isinstance(error, AuthenticationException):
            return "Check API credentials and re-authenticate"
        else:
            return None
    
    def clear_old_errors(self, days: int = 7):
        """Clean up old error records"""
        cutoff = datetime.now() - timedelta(days=days)
        
        # Clear old failed orders
        old_orders = []
        for order_id, details in self.state['failed_orders'].items():
            if datetime.fromisoformat(details['timestamp']) < cutoff:
                old_orders.append(order_id)
        
        for order_id in old_orders:
            del self.state['failed_orders'][order_id]
        
        # Reset error counts older than cutoff
        for key in list(self.state['error_counts'].keys()):
            if datetime.fromisoformat(self.state['error_counts'][key]['last_seen']) < cutoff:
                del self.state['error_counts'][key]
        
        self._save_state()
        logger.info(f"Cleared {len(old_orders)} old error records")