"""Tests for structured logging."""
import json
import logging
import pytest

from core.config import JSONFormatter


class TestJSONFormatter:
    def test_basic_format(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            'TradingBot', logging.INFO, 'test.py', 42,
            'Test message', (), None
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert data['message'] == 'Test message'
        assert data['level'] == 'INFO'
        assert data['logger'] == 'TradingBot'
        assert data['line'] == 42
        assert 'timestamp' in data

    def test_extra_trade_fields(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            'TradingBot', logging.INFO, 'engine.py', 100,
            'Order placed', (), None
        )
        record.symbol = 'AAPL'
        record.action = 'BUY'
        record.order_id = '12345'
        record.quantity = 10
        record.price = 150.0

        output = fmt.format(record)
        data = json.loads(output)
        assert data['symbol'] == 'AAPL'
        assert data['action'] == 'BUY'
        assert data['order_id'] == '12345'
        assert data['quantity'] == 10
        assert data['price'] == 150.0

    def test_exception_included(self):
        fmt = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            'TradingBot', logging.ERROR, 'test.py', 1,
            'Error occurred', (), exc_info
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert 'exception' in data
        assert 'ValueError' in data['exception']

    def test_missing_extra_fields_omitted(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            'TradingBot', logging.INFO, 'test.py', 1,
            'No extras', (), None
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert 'symbol' not in data
        assert 'order_id' not in data
