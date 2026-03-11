import logging
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from core.models import CommentaryType
from core.commentary import TradingCommentary

logger = logging.getLogger('TradingBot')


class AnomalyDetector:
    """Detects anomalies in market data and trading behavior"""

    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self.last_price = {}
        self.last_volume = {}

    def check_data_anomalies(self, symbol: str, data: pd.DataFrame) -> List[Dict]:
        """Check for anomalies in OHLCV data"""
        anomalies = []

        # Check for large price gaps
        if self.last_price.get(symbol):
            gap = abs(data['Open'].iloc[0] - self.last_price[symbol]) / self.last_price[symbol]
            if gap > 0.1:  # 10% gap
                anomalies.append({
                    'type': 'price_gap',
                    'message': f"Large price gap of {gap:.1%} detected for {symbol}",
                    'level': 'high'
                })

        # Check for extreme volume
        avg_volume = data['Volume'].mean()
        if data['Volume'].iloc[-1] > avg_volume * 5:
            anomalies.append({
                'type': 'volume_spike',
                'message': f"Unusual volume spike detected for {symbol}",
                'level': 'medium'
            })

        # Update last known values
        self.last_price[symbol] = data['Close'].iloc[-1]
        self.last_volume[symbol] = data['Volume'].iloc[-1]

        # Add commentary for anomalies
        for anomaly in anomalies:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.ANOMALY,
                symbol=symbol,
                title=f"Anomaly Detected: {anomaly['type']}",
                message=anomaly['message'],
                importance=8 if anomaly['level'] == 'high' else 6
            ))

        return anomalies


class DataValidator:
    """Validates data quality and detects anomalies"""

    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self.data_quality_history = {}

    def validate_market_data(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate market data quality"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'quality_score': 1.0
        }

        if data.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Empty dataset")
            validation_results['quality_score'] = 0.0
            return validation_results

        # Check for missing values
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            validation_results['issues'].append(f"Missing data: {missing_data.to_dict()}")
            validation_results['quality_score'] *= 0.8

        # Check for price anomalies
        if 'close' in data.columns:
            price_changes = data['close'].pct_change().abs()
            extreme_changes = price_changes > 0.5  # 50% price changes

            if extreme_changes.sum() > 0:
                validation_results['issues'].append(f"Extreme price changes detected: {extreme_changes.sum()} instances")
                validation_results['quality_score'] *= 0.7

        # Check for volume anomalies
        if 'volume' in data.columns:
            zero_volume = (data['volume'] == 0).sum()
            if zero_volume > len(data) * 0.1:  # More than 10% zero volume
                validation_results['issues'].append(f"High number of zero volume periods: {zero_volume}")
                validation_results['quality_score'] *= 0.9

        # Check for stale data
        if 'timestamp' in data.columns:
            latest_time = pd.to_datetime(data['timestamp'].max())
            current_time = pd.Timestamp.now()
            time_diff = (current_time - latest_time).total_seconds() / 3600  # hours

            if time_diff > 24:  # Data older than 24 hours
                validation_results['issues'].append(f"Stale data: {time_diff:.1f} hours old")
                validation_results['quality_score'] *= 0.5

        # Check for price gaps
        if 'close' in data.columns and len(data) > 1:
            gaps = []
            for i in range(1, len(data)):
                gap = abs(data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                if gap > 0.1:  # 10% gap
                    gaps.append(gap)

            if gaps:
                validation_results['issues'].append(f"Price gaps detected: {len(gaps)} instances")
                validation_results['quality_score'] *= 0.8

        # Update history
        self.data_quality_history[symbol] = {
            'timestamp': datetime.now(),
            'quality_score': validation_results['quality_score'],
            'issues': validation_results['issues']
        }

        # Alert on poor data quality
        if validation_results['quality_score'] < 0.7:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title="Data Quality Issues",
                message=f"Data quality score: {validation_results['quality_score']:.2f}. "
                       f"Issues: {', '.join(validation_results['issues'])}",
                data=validation_results,
                importance=6
            ))

        return validation_results

    def cross_validate_sources(self, symbol: str, data_sources: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Cross-validate data from multiple sources"""
        if len(data_sources) < 2:
            return {'is_consistent': True, 'discrepancies': []}

        # Compare closing prices across sources
        closing_prices = {}
        for source, data in data_sources.items():
            if not data.empty and 'close' in data.columns:
                closing_prices[source] = data['close'].iloc[-1]

        if len(closing_prices) < 2:
            return {'is_consistent': True, 'discrepancies': []}

        prices = list(closing_prices.values())
        mean_price = np.mean(prices)
        max_deviation = max(abs(price - mean_price) / mean_price for price in prices)

        discrepancies = []
        if max_deviation > 0.01:  # 1% deviation
            for source, price in closing_prices.items():
                deviation = abs(price - mean_price) / mean_price
                if deviation > 0.01:
                    discrepancies.append({
                        'source': source,
                        'price': price,
                        'deviation': deviation
                    })

        return {
            'is_consistent': max_deviation <= 0.01,
            'max_deviation': max_deviation,
            'discrepancies': discrepancies,
            'mean_price': mean_price
        }
