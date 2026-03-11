import logging
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats
import ta
from sklearn.preprocessing import StandardScaler, RobustScaler

try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False

logger = logging.getLogger('TradingBot')


class MarketRegimeDetector:
    """Detects market regimes for adaptive model behavior"""

    def __init__(self):
        self.regime_history = []
        self.current_regime = "normal"

    def detect_regime(self, market_data: Dict[str, float]) -> str:
        """Detect current market regime based on volatility and trend"""
        vix = market_data.get('vix', 20)
        trend_strength = market_data.get('trend_strength', 0)

        if vix > 30:
            regime = "high_volatility"
        elif vix < 15:
            regime = "low_volatility"
        elif abs(trend_strength) > 0.7:
            regime = "trending"
        else:
            regime = "normal"

        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime,
            'vix': vix
        })

        self.current_regime = regime
        return regime

class ConceptDriftDetector:
    """Monitors for concept drift in model performance"""

    def __init__(self, warning_level: float = 0.95, drift_level: float = 0.99):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.error_rate = 0.0
        self.error_std = 0.0
        self.n_samples = 0
        self.drift_detected = False
        self.warning_detected = False

    def update(self, prediction_correct: bool) -> Tuple[bool, bool]:
        """Update drift detector with new prediction result"""
        error = 0 if prediction_correct else 1

        if self.n_samples == 0:
            self.error_rate = error
            self.error_std = 0
        else:
            # Update error rate with exponential weighted average
            alpha = 2 / (self.n_samples + 1)
            self.error_rate = alpha * error + (1 - alpha) * self.error_rate
            self.error_std = np.sqrt(self.error_rate * (1 - self.error_rate) / (self.n_samples + 1))

        self.n_samples += 1

        # Check for drift
        if self.n_samples > 30:  # Need minimum samples
            drift_threshold = self.error_rate + self.error_std * self.drift_level
            warning_threshold = self.error_rate + self.error_std * self.warning_level

            current_error_rate = error

            if current_error_rate > drift_threshold:
                self.drift_detected = True
            elif current_error_rate > warning_threshold:
                self.warning_detected = True

        return self.warning_detected, self.drift_detected

class AdvancedFeatureEngineer:
    """Multi-timeframe feature engineering with microstructure and wavelets"""

    def __init__(self):
        self.timeframes = ['1min', '5min', '15min', '1hour']
        self.feature_names = []

    def extract_multi_timeframe_features(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Extract features from multiple timeframes"""
        all_features = []

        for tf in self.timeframes:
            if tf in data and not data[tf].empty:
                tf_features = self._extract_timeframe_features(data[tf], tf)
                all_features.extend(tf_features)

        return np.array(all_features)

    def _extract_timeframe_features(self, df: pd.DataFrame, timeframe: str) -> List[float]:
        """Extract features for a single timeframe"""
        features = []

        if len(df) < 50:
            return [0] * 26  # Return zeros if insufficient data

        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values

        # Price-based features
        features.extend([
            # Returns at different intervals
            (close[-1] - close[-2]) / close[-2] if close[-2] != 0 else 0,
            (close[-1] - close[-5]) / close[-5] if close[-5] != 0 else 0,
            (close[-1] - close[-20]) / close[-20] if close[-20] != 0 else 0,

            # Moving averages
            close[-20:].mean(),
            close[-50:].mean() if len(close) >= 50 else close.mean(),

            # Price position
            (close[-1] - low[-20:].min()) / (high[-20:].max() - low[-20:].min() + 1e-10),

            # Volatility
            np.std(close[-20:]) / (close[-20:].mean() + 1e-10),

            # Volume features
            volume[-1] / (volume[-20:].mean() + 1e-10),
            np.std(volume[-20:]) / (volume[-20:].mean() + 1e-10),
        ])

        # Technical indicators
        features.extend(self._calculate_technical_indicators(close, high, low, volume))

        # Microstructure features
        features.extend(self._calculate_microstructure_features(df))

        # Statistical moments
        returns = np.diff(close[-20:]) / close[-21:-1]
        features.extend([
            stats.skew(returns),
            stats.kurtosis(returns),
            np.percentile(returns, 75) - np.percentile(returns, 25)  # IQR
        ])

        # Wavelet features if available
        if WAVELET_AVAILABLE:
            features.extend(self._extract_wavelet_features(close[-50:]))

        return features

    def _calculate_technical_indicators(self, close: np.ndarray, high: np.ndarray,
                                      low: np.ndarray, volume: np.ndarray) -> List[float]:
        """Calculate technical indicators"""
        features = []

        # RSI
        rsi = self._calculate_rsi(close, 14)
        features.append(rsi)

        # MACD
        if len(close) >= 26:
            macd, signal, hist = self._calculate_macd(close)
            features.extend([macd, signal, hist])
        else:
            features.extend([0, 0, 0])

        # ATR
        atr = self._calculate_atr(high, low, close, 14)
        features.append(atr)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(pd.Series(close))
        features.extend([
            (close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10),
            (bb_upper - bb_lower) / (bb_middle + 1e-10)  # BB width
        ])

        return features

    def _calculate_microstructure_features(self, df: pd.DataFrame) -> List[float]:
        """Calculate market microstructure features"""
        features = []

        # Bid-ask spread proxy (high-low as proxy)
        spread = (df['High'] - df['Low']).iloc[-20:]
        features.extend([
            spread.mean(),
            spread.std(),
            spread.iloc[-1] / spread.mean() if spread.mean() > 0 else 1
        ])

        # Order flow imbalance proxy (using volume and price direction)
        price_direction = np.sign(df['Close'].diff())
        signed_volume = df['Volume'] * price_direction

        features.extend([
            signed_volume.iloc[-20:].sum() / (df['Volume'].iloc[-20:].sum() + 1e-10),
            signed_volume.iloc[-5:].sum() / (df['Volume'].iloc[-5:].sum() + 1e-10)
        ])

        return features

    def _extract_wavelet_features(self, signal: np.ndarray) -> List[float]:
        """Extract wavelet transform features"""
        features = []

        try:
            # Discrete Wavelet Transform
            coeffs = pywt.wavedec(signal, 'db4', level=3)

            # Energy of each level
            for i, coeff in enumerate(coeffs):
                energy = np.sum(coeff**2)
                features.append(energy)

            # Approximate to detail ratio
            if len(coeffs) > 1:
                approx_energy = np.sum(coeffs[0]**2)
                detail_energy = sum(np.sum(c**2) for c in coeffs[1:])
                features.append(approx_energy / (detail_energy + 1e-10))

        except Exception:
            features = [0] * 5  # Default features if wavelet fails

        return features

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period

        if down == 0:
            return 100.0

        rs = up / down
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD"""
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        return macd.iloc[-1], signal.iloc[-1], hist.iloc[-1]

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, period: int = 14) -> float:
        """Calculate ATR"""
        if len(high) < period:
            return 0.0

        tr = np.maximum(high - low,
                       np.abs(high - np.roll(close, 1)),
                       np.abs(low - np.roll(close, 1)))[1:]

        return np.mean(tr[-period:])

    def _calculate_bollinger_bands(self, prices: pd.Series,
                                  period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        bollinger = ta.volatility.BollingerBands(prices, window=period, window_dev=std_dev)
        return bollinger.bollinger_hband().iloc[-1], bollinger.bollinger_mavg().iloc[-1], bollinger.bollinger_lband().iloc[-1]


@dataclass
class MLTrainingRecord:
    """Record of training data for consistency"""
    timestamp: datetime
    symbol: str
    features: Dict[str, float]
    label: int
    market_conditions: Dict[str, Any]

@dataclass
class MLPrediction:
    """Structured ML prediction result"""
    signal: int  # 0 or 1
    confidence: float
    reasoning: Dict[str, Any]
    feature_importance: Dict[str, float]
    timestamp: datetime


# ============================================================================
# FEATURE ENGINEERING THAT MATCHES EXISTING TECHNICAL ANALYZER
# ============================================================================

class MLFeatureExtractor:
    """Extract features consistently using same data as TechnicalAnalyzer"""

    def __init__(self):
        # Define feature groups for organization and debugging
        self.feature_groups = {
            'price_momentum': ['returns_1', 'returns_5', 'returns_20', 'price_vs_sma20', 'price_vs_sma50'],
            'technical_indicators': ['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'bb_width'],
            'volume_analysis': ['volume_ratio', 'volume_std_ratio', 'volume_trend'],
            'volatility': ['atr_ratio', 'high_low_ratio', 'std_dev_ratio'],
            'support_resistance': ['pivot_distance', 'r1_distance', 's1_distance'],
            'market_microstructure': ['spread_proxy', 'volume_imbalance']
        }

        # Flattened feature list for consistent ordering
        self.feature_names = []
        for group_features in self.feature_groups.values():
            self.feature_names.extend(group_features)

    def extract_from_dataframe(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract features from OHLCV DataFrame - matches data available during live trading"""
        features = {}

        # Validate input
        if df is None or df.empty or len(df) < 50:
            # Return zero features if insufficient data
            return {name: 0.0 for name in self.feature_names}

        try:
            # Ensure numeric types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop any NaN rows
            df = df.dropna()

            if len(df) < 50:
                return {name: 0.0 for name in self.feature_names}

            # Extract price series
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            volume = df['Volume'].values

            # Price momentum features
            features['returns_1'] = (close[-1] - close[-2]) / close[-2] if close[-2] != 0 else 0
            features['returns_5'] = (close[-1] - close[-5]) / close[-5] if close[-5] != 0 else 0
            features['returns_20'] = (close[-1] - close[-20]) / close[-20] if close[-20] != 0 else 0
            features['price_vs_sma20'] = close[-1] / np.mean(close[-20:]) - 1
            features['price_vs_sma50'] = close[-1] / np.mean(close[-50:]) - 1 if len(close) >= 50 else features['price_vs_sma20']

            # Technical indicators (matching TechnicalAnalyzer calculations)
            features['rsi'] = self._calculate_rsi(pd.Series(close)) / 100.0  # Normalize to 0-1

            # MACD
            macd, signal, hist = self._calculate_macd(pd.Series(close))
            features['macd'] = macd / close[-1] if close[-1] != 0 else 0  # Normalize by price
            features['macd_signal'] = signal / close[-1] if close[-1] != 0 else 0
            features['macd_hist'] = hist / close[-1] if close[-1] != 0 else 0

            # Bollinger Bands
            bb_middle = np.mean(close[-20:])
            bb_std = np.std(close[-20:])
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
            features['bb_position'] = (close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0

            # Volume analysis
            features['volume_ratio'] = volume[-1] / (np.mean(volume[-20:]) + 1e-10)
            features['volume_std_ratio'] = np.std(volume[-20:]) / (np.mean(volume[-20:]) + 1e-10)
            features['volume_trend'] = (np.mean(volume[-5:]) - np.mean(volume[-20:])) / (np.mean(volume[-20:]) + 1e-10)

            # Volatility features
            atr = self._calculate_atr(pd.Series(high), pd.Series(low), pd.Series(close))
            features['atr_ratio'] = atr / close[-1] if close[-1] != 0 else 0
            features['high_low_ratio'] = (high[-1] - low[-1]) / close[-1] if close[-1] != 0 else 0
            features['std_dev_ratio'] = np.std(close[-20:]) / np.mean(close[-20:])

            # Support/Resistance
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            features['pivot_distance'] = (close[-1] - pivot) / close[-1] if close[-1] != 0 else 0
            features['r1_distance'] = ((2 * pivot - low[-1]) - close[-1]) / close[-1] if close[-1] != 0 else 0
            features['s1_distance'] = (close[-1] - (2 * pivot - high[-1])) / close[-1] if close[-1] != 0 else 0

            # Market microstructure
            features['spread_proxy'] = (high[-1] - low[-1]) / close[-1] if close[-1] != 0 else 0
            price_direction = np.sign(np.diff(close[-20:]))
            signed_volume = volume[-19:] * price_direction
            features['volume_imbalance'] = np.sum(signed_volume) / (np.sum(volume[-19:]) + 1e-10)

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Return zero features on error
            features = {name: 0.0 for name in self.feature_names}

        # Ensure all features are present and handle NaN/Inf
        for name in self.feature_names:
            if name not in features:
                features[name] = 0.0
            elif np.isnan(features[name]) or np.isinf(features[name]):
                features[name] = 0.0

        return features

    def extract_from_indicators(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Extract features from pre-calculated indicators (for compatibility)"""
        features = {}

        # Map indicator values to our feature names
        mapping = {
            'rsi': 'rsi',
            'macd': 'macd',
            'macd_signal': 'macd_signal',
            'macd_histogram': 'macd_hist',
            'volume_ratio': 'volume_ratio',
            'atr': 'atr_ratio',
            'high_low_ratio': 'high_low_ratio',
            'returns': 'returns_1'
        }

        # Fill in what we can from indicators
        for indicator_name, feature_name in mapping.items():
            if indicator_name in indicators:
                if feature_name == 'rsi':
                    features[feature_name] = indicators[indicator_name] / 100.0
                elif feature_name in ['macd', 'macd_signal', 'macd_hist']:
                    # These should already be normalized in indicators
                    features[feature_name] = indicators[indicator_name]
                else:
                    features[feature_name] = indicators[indicator_name]

        # Fill missing features with defaults
        for name in self.feature_names:
            if name not in features:
                features[name] = 0.0

        return features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI - matches TechnicalAnalyzer implementation"""
        return ta.momentum.rsi(prices, window=period).iloc[-1]

    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float, float]:
        """Calculate MACD - matches TechnicalAnalyzer implementation"""
        macd = ta.trend.MACD(prices)
        return macd.macd().iloc[-1], macd.macd_signal().iloc[-1], macd.macd_diff().iloc[-1]

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate ATR"""
        return ta.volatility.average_true_range(high, low, close, window=period).iloc[-1]
