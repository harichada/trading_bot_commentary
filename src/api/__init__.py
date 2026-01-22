"""
API Module

FastAPI application and endpoints for the trading bot.
"""

from .app import create_app, get_app

__all__ = ['create_app', 'get_app']
