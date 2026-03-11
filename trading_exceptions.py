"""
Custom exception classes for the trading bot to handle different error scenarios appropriately.
"""
from datetime import datetime


class TradingBotException(Exception):
    """Base exception for all trading bot errors"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()


# API and Network Errors
class SchwabAPIException(TradingBotException):
    """Schwab API specific errors"""
    pass

class RateLimitException(SchwabAPIException):
    """API rate limit exceeded"""
    def __init__(self, retry_after: int = None):
        super().__init__(
            "API rate limit exceeded",
            error_code="RATE_LIMIT",
            details={"retry_after": retry_after}
        )
        self.retry_after = retry_after

class AuthenticationException(SchwabAPIException):
    """Authentication failures - fatal, requires user intervention"""
    pass

class NetworkException(TradingBotException):
    """Network connectivity issues - usually retryable"""
    pass


# Trading and Order Errors
class OrderException(TradingBotException):
    """Base class for order-related errors"""
    pass

class InsufficientFundsException(OrderException):
    """Not enough buying power for the order"""
    def __init__(self, required: float, available: float):
        super().__init__(
            f"Insufficient funds: required ${required:.2f}, available ${available:.2f}",
            error_code="INSUFFICIENT_FUNDS",
            details={"required": required, "available": available}
        )

class OrderRejectionException(OrderException):
    """Order rejected by broker"""
    def __init__(self, reason: str, order_details: dict = None):
        super().__init__(
            f"Order rejected: {reason}",
            error_code="ORDER_REJECTED",
            details={"reason": reason, "order": order_details}
        )

class PositionNotFoundException(OrderException):
    """Trying to close a position that doesn't exist"""
    pass

class MarketClosedException(OrderException):
    """Market is closed for trading"""
    pass


# Risk Management Errors
class RiskLimitException(TradingBotException):
    """Risk limits exceeded"""
    pass

class DailyLossLimitException(RiskLimitException):
    """Daily loss limit reached - stop trading"""
    def __init__(self, loss: float, limit: float):
        super().__init__(
            f"Daily loss limit reached: ${loss:.2f} (limit: ${limit:.2f})",
            error_code="DAILY_LOSS_LIMIT",
            details={"loss": loss, "limit": limit}
        )

class PositionLimitException(RiskLimitException):
    """Maximum position count reached"""
    pass

class CircuitBreakerException(RiskLimitException):
    """Circuit breaker triggered - emergency stop"""
    def __init__(self, reason: str, cooldown_minutes: int = 30):
        super().__init__(
            f"Circuit breaker triggered: {reason}",
            error_code="CIRCUIT_BREAKER",
            details={"reason": reason, "cooldown_minutes": cooldown_minutes}
        )
        self.cooldown_minutes = cooldown_minutes


# Data and State Errors
class DataException(TradingBotException):
    """Data-related errors"""
    pass

class InvalidSymbolException(DataException):
    """Invalid or unsupported symbol"""
    pass

class StaleDataException(DataException):
    """Market data is too old"""
    def __init__(self, data_age_seconds: int, max_age: int = 60):
        super().__init__(
            f"Data is {data_age_seconds}s old (max: {max_age}s)",
            error_code="STALE_DATA",
            details={"age": data_age_seconds, "max_age": max_age}
        )

class StateCorruptionException(DataException):
    """State file is corrupted or invalid"""
    pass


# ML Model Errors
class ModelException(TradingBotException):
    """Machine learning model errors"""
    pass

class ModelNotTrainedException(ModelException):
    """Model needs training before use"""
    pass

class FeatureExtractionException(ModelException):
    """Failed to extract features for prediction"""
    pass


# Recovery Functions
def is_retryable_error(exception: Exception) -> bool:
    """Determine if an error should be retried"""
    return isinstance(exception, (
        NetworkException,
        RateLimitException,
        StaleDataException
    ))

def get_retry_delay(exception: Exception, attempt: int) -> float:
    """Calculate retry delay based on error type and attempt number"""
    if isinstance(exception, RateLimitException) and exception.retry_after:
        return float(exception.retry_after)
    
    # Exponential backoff with jitter
    base_delay = 1.0
    max_delay = 60.0
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = delay * 0.1 * (2 * np.random.random() - 1)
    return delay + jitter

def should_stop_trading(exception: Exception) -> bool:
    """Determine if trading should be halted"""
    return isinstance(exception, (
        AuthenticationException,
        DailyLossLimitException,
        CircuitBreakerException,
        StateCorruptionException
    ))