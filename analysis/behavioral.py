import logging
from typing import Dict, List, Any

import pandas as pd

logger = logging.getLogger('TradingBot')


class BehavioralAnalyzer:
    """Analyzes trading behavior and psychological patterns"""

    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self.trade_patterns = []
        self.emotional_state = {
            'confidence': 0.5,
            'fear': 0.0,
            'greed': 0.0,
            'patience': 0.5
        }

    def analyze_trading_patterns(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze trading behavior for psychological patterns"""
        if not trades:
            return {}

        df = pd.DataFrame(trades)

        # Calculate time between trades
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df['time_between_trades'] = df['timestamp'].diff().dt.total_seconds() / 3600  # hours

        # Detect patterns
        patterns = {
            'overtrading': self._detect_overtrading(df),
            'revenge_trading': self._detect_revenge_trading(df),
            'risk_tolerance_changes': self._detect_risk_changes(df),
            'timing_patterns': self._analyze_timing(df),
            'position_sizing_patterns': self._analyze_position_sizing(df)
        }

        # Update emotional state
        self._update_emotional_state(patterns, df)

        return patterns

    def _detect_overtrading(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect overtrading patterns"""
        if 'time_between_trades' not in df.columns:
            return {}

        avg_time_between = df['time_between_trades'].mean()
        recent_trades = df.tail(10)
        recent_avg_time = recent_trades['time_between_trades'].mean()

        return {
            'is_overtrading': recent_avg_time < avg_time_between * 0.5,
            'avg_time_between_trades': avg_time_between,
            'recent_avg_time': recent_avg_time,
            'trade_frequency_increase': recent_avg_time < avg_time_between * 0.7
        }

    def _detect_revenge_trading(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect revenge trading after losses"""
        if len(df) < 3:
            return {}

        # Look for increased position sizes after losses
        df['pnl'] = df.get('pnl', 0)
        df['position_size'] = df.get('position_size', 1)

        revenge_patterns = []
        for i in range(1, len(df)):
            prev_trade = df.iloc[i-1]
            current_trade = df.iloc[i]

            # Check if previous trade was a loss and current position is larger
            if (prev_trade['pnl'] < 0 and
                current_trade['position_size'] > prev_trade['position_size'] * 1.5):
                revenge_patterns.append({
                    'index': i,
                    'previous_loss': prev_trade['pnl'],
                    'size_increase': current_trade['position_size'] / prev_trade['position_size']
                })

        return {
            'revenge_trading_detected': len(revenge_patterns) > 0,
            'revenge_patterns': revenge_patterns,
            'total_revenge_instances': len(revenge_patterns)
        }

    def _detect_risk_changes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect changes in risk tolerance"""
        if len(df) < 10:
            return {}

        # Calculate rolling average of position sizes
        df['position_size'] = df.get('position_size', 1)
        df['rolling_avg_size'] = df['position_size'].rolling(window=5).mean()

        recent_avg = df['rolling_avg_size'].tail(5).mean()
        earlier_avg = df['rolling_avg_size'].head(5).mean()

        risk_change = (recent_avg - earlier_avg) / (earlier_avg + 1e-8)

        return {
            'risk_increasing': risk_change > 0.2,
            'risk_decreasing': risk_change < -0.2,
            'risk_change_percent': risk_change * 100,
            'recent_avg_position_size': recent_avg,
            'earlier_avg_position_size': earlier_avg
        }

    def _analyze_timing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading timing patterns"""
        if 'timestamp' not in df.columns:
            return {}

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Find most active trading hours
        hour_counts = df['hour'].value_counts()
        most_active_hour = hour_counts.index[0] if len(hour_counts) > 0 else None

        # Check for weekend trading (should be minimal)
        weekend_trades = len(df[df['day_of_week'] >= 5])

        return {
            'most_active_hour': most_active_hour,
            'weekend_trades': weekend_trades,
            'trading_hours_distribution': hour_counts.to_dict()
        }

    def _analyze_position_sizing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze position sizing patterns"""
        if 'position_size' not in df.columns:
            return {}

        df['position_size'] = df.get('position_size', 1)

        # Check for consistency in position sizing
        size_std = df['position_size'].std()
        size_mean = df['position_size'].mean()
        size_cv = size_std / (size_mean + 1e-8)  # Coefficient of variation

        return {
            'position_size_consistency': size_cv < 0.5,  # Low CV = more consistent
            'size_coefficient_variation': size_cv,
            'avg_position_size': size_mean,
            'position_size_std': size_std
        }

    def _update_emotional_state(self, patterns: Dict, df: pd.DataFrame):
        """Update emotional state based on patterns"""
        # Update confidence based on recent performance
        if len(df) > 0:
            recent_pnl = df.tail(5).get('pnl', pd.Series([0])).sum()
            if recent_pnl > 0:
                self.emotional_state['confidence'] = min(1.0, self.emotional_state['confidence'] + 0.1)
            else:
                self.emotional_state['confidence'] = max(0.0, self.emotional_state['confidence'] - 0.1)

        # Update fear/greed based on patterns
        if patterns.get('overtrading', {}).get('is_overtrading', False):
            self.emotional_state['fear'] = min(1.0, self.emotional_state['fear'] + 0.2)

        if patterns.get('revenge_trading', {}).get('revenge_trading_detected', False):
            self.emotional_state['greed'] = min(1.0, self.emotional_state['greed'] + 0.3)

        # Decay emotions over time
        self.emotional_state['fear'] *= 0.95
        self.emotional_state['greed'] *= 0.95

    def get_behavioral_recommendations(self) -> List[str]:
        """Get recommendations based on behavioral analysis"""
        recommendations = []

        if self.emotional_state['fear'] > 0.7:
            recommendations.append("High fear detected - consider reducing position sizes")

        if self.emotional_state['greed'] > 0.7:
            recommendations.append("High greed detected - review risk management")

        if self.emotional_state['confidence'] < 0.3:
            recommendations.append("Low confidence - consider taking a trading break")

        return recommendations
