"""
Industry-Standard ML Model for Scalping Day Trading
Optimized for high-frequency trading with focus on microstructure and order flow
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
import json
from collections import deque
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None

# Import version control
try:
    from model_version_control import ModelVersionControl
    HAS_VERSION_CONTROL = True
except ImportError:
    HAS_VERSION_CONTROL = False
    logging.warning("Model version control not available")
import ta
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ScalpingFeatureConfig:
    """Configuration for scalping-specific features"""
    # Microstructure features
    use_order_flow_imbalance: bool = True
    use_bid_ask_spread: bool = True
    use_trade_intensity: bool = True
    use_price_impact: bool = True
    
    # High-frequency patterns
    use_tick_patterns: bool = True
    use_microstructure_noise: bool = True
    use_liquidity_measures: bool = True
    
    # Time-based features
    use_intraday_seasonality: bool = True
    use_market_hours: bool = True
    use_time_to_events: bool = True
    
    # Advanced technical
    use_order_book_imbalance: bool = True
    use_vpin: bool = True  # Volume-synchronized Probability of Informed Trading
    use_realized_volatility: bool = True
    
    # Multi-timeframe
    timeframes: List[str] = field(default_factory=lambda: ['1min', '5min', '15min', '30min'])
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

class ScalpingFeatureEngineer:
    """Advanced feature engineering for scalping strategies"""
    
    def __init__(self, config: ScalpingFeatureConfig = None):
        self.config = config or ScalpingFeatureConfig()
        self.feature_names = []
        self.feature_groups = {
            'microstructure': [],
            'order_flow': [],
            'volatility': [],
            'momentum': [],
            'liquidity': [],
            'time_based': [],
            'pattern': [],
            'statistical': []
        }
        
    def extract_features(self, data: pd.DataFrame, quote_data: Optional[Dict] = None,
                        market_data: Optional[Dict] = None) -> Dict[str, float]:
        """Extract comprehensive features for scalping"""
        features = {}
        
        # Validate data
        if data is None or data.empty or len(data) < 50:
            return self._get_default_features()
            
        try:
            # Basic price/volume features
            features.update(self._extract_basic_features(data))
            
            # Microstructure features
            if self.config.use_bid_ask_spread and quote_data:
                features.update(self._extract_microstructure_features(data, quote_data))
            
            # Order flow features
            if self.config.use_order_flow_imbalance:
                features.update(self._extract_order_flow_features(data))
            
            # Volatility features
            features.update(self._extract_volatility_features(data))
            
            # Momentum features
            features.update(self._extract_momentum_features(data))
            
            # Liquidity features
            if self.config.use_liquidity_measures:
                features.update(self._extract_liquidity_features(data))
            
            # Pattern features
            if self.config.use_tick_patterns:
                features.update(self._extract_pattern_features(data))
            
            # Time-based features
            if self.config.use_intraday_seasonality:
                features.update(self._extract_time_features(data))
            
            # Statistical features
            features.update(self._extract_statistical_features(data))
            
            # Multi-timeframe features
            if market_data:
                features.update(self._extract_multitimeframe_features(market_data))
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return self._get_default_features()
        
        # Ensure all features are valid
        return self._validate_features(features)
    
    def _extract_basic_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract basic price and volume features"""
        features = {}
        
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values
        
        # Price returns at multiple horizons
        for period in [1, 2, 5, 10, 20]:
            if len(close) > period:
                features[f'return_{period}'] = (close[-1] - close[-period-1]) / close[-period-1]
            
        # Price position
        features['price_position'] = (close[-1] - low[-20:].min()) / (high[-20:].max() - low[-20:].min() + 1e-10)
        
        # Volume features
        features['volume_ratio'] = volume[-1] / (volume[-20:].mean() + 1e-10)
        features['volume_trend'] = (volume[-5:].mean() - volume[-20:].mean()) / (volume[-20:].mean() + 1e-10)
        
        # High-low range
        features['hl_ratio'] = (high[-1] - low[-1]) / close[-1] if close[-1] > 0 else 0
        
        self.feature_groups['momentum'].extend(['return_1', 'return_5', 'return_10'])
        self.feature_groups['liquidity'].extend(['volume_ratio', 'volume_trend'])
        
        return features
    
    def _extract_microstructure_features(self, data: pd.DataFrame, quote_data: Dict) -> Dict[str, float]:
        """Extract market microstructure features"""
        features = {}
        
        # Bid-ask spread
        bid = quote_data.get('bid', 0)
        ask = quote_data.get('ask', 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else data['Close'].iloc[-1]
        
        features['spread'] = (ask - bid) / mid if mid > 0 else 0
        features['spread_relative'] = features['spread'] / data['Close'].std() if data['Close'].std() > 0 else 0
        
        # Price impact measures
        close = data['Close'].values
        volume = data['Volume'].values
        
        # Kyle's lambda (price impact coefficient)
        if len(close) > 10 and len(volume) > 10:
            returns = np.diff(close[-10:]) / close[-11:-1]
            signed_volume = volume[-10:] * np.sign(returns)
            if np.std(signed_volume) > 0:
                features['price_impact'] = np.abs(np.corrcoef(returns, signed_volume[:-1])[0, 1])
            else:
                features['price_impact'] = 0
        
        # Effective spread
        features['effective_spread'] = 2 * np.abs(close[-1] - mid) / mid if mid > 0 else 0
        
        self.feature_groups['microstructure'].extend(['spread', 'spread_relative', 'price_impact'])
        
        return features
    
    def _extract_order_flow_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract order flow imbalance features"""
        features = {}
        
        close = data['Close'].values
        volume = data['Volume'].values
        high = data['High'].values
        low = data['Low'].values
        
        # Order flow imbalance
        price_direction = np.sign(np.diff(close))
        signed_volume = volume[1:] * price_direction
        
        features['order_flow_imbalance'] = signed_volume[-20:].sum() / (volume[-20:].sum() + 1e-10)
        features['order_flow_imbalance_5'] = signed_volume[-5:].sum() / (volume[-5:].sum() + 1e-10)
        
        # VPIN (Volume-synchronized Probability of Informed Trading)
        if self.config.use_vpin and len(data) > 50:
            features['vpin'] = self._calculate_vpin(data)
        
        # Buy/Sell pressure
        # Using tick rule: if close > mid of bar, it's buy pressure
        mid_price = (high + low) / 2
        buy_volume = volume[close > mid_price].sum()
        sell_volume = volume[close <= mid_price].sum()
        total_volume = buy_volume + sell_volume + 1e-10
        
        features['buy_pressure'] = buy_volume / total_volume
        features['sell_pressure'] = sell_volume / total_volume
        features['net_pressure'] = (buy_volume - sell_volume) / total_volume
        
        self.feature_groups['order_flow'].extend(['order_flow_imbalance', 'buy_pressure', 'net_pressure'])
        
        return features
    
    def _extract_volatility_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract volatility-based features"""
        features = {}
        
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        
        # Realized volatility at different frequencies
        returns = np.diff(close) / close[:-1]
        
        features['volatility_1min'] = np.std(returns[-5:]) if len(returns) >= 5 else 0
        features['volatility_5min'] = np.std(returns[-20:]) if len(returns) >= 20 else 0
        features['volatility_ratio'] = features['volatility_1min'] / (features['volatility_5min'] + 1e-10)
        
        # Parkinson volatility (using high-low)
        if len(high) >= 20:
            parkinson = np.sqrt(1 / (4 * np.log(2)) * np.mean(np.log(high[-20:] / low[-20:])**2))
            features['parkinson_vol'] = parkinson
        
        # Garman-Klass volatility
        if len(data) >= 20:
            features['gk_volatility'] = self._calculate_garman_klass_vol(data.tail(20))
        
        # ATR-based volatility
        atr = ta.volatility.average_true_range(
            pd.Series(high), pd.Series(low), pd.Series(close), window=14
        )
        features['atr_ratio'] = atr.iloc[-1] / close[-1] if close[-1] > 0 else 0
        
        self.feature_groups['volatility'].extend(['volatility_1min', 'volatility_ratio', 'atr_ratio'])
        
        return features
    
    def _extract_momentum_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract momentum indicators"""
        features = {}
        
        close = pd.Series(data['Close'].values)
        high = pd.Series(data['High'].values)
        low = pd.Series(data['Low'].values)
        volume = pd.Series(data['Volume'].values)
        
        # RSI at multiple periods
        for period in [7, 14, 21]:
            if len(close) > period:
                rsi = ta.momentum.rsi(close, window=period)
                features[f'rsi_{period}'] = rsi.iloc[-1] / 100.0  # Normalize
        
        # Stochastic oscillator
        stoch = ta.momentum.stoch(high, low, close, window=14)
        features['stochastic'] = stoch.iloc[-1] / 100.0
        
        # MACD
        if len(close) >= 26:
            macd = ta.trend.MACD(close)
            features['macd'] = macd.macd().iloc[-1] / close.iloc[-1] if close.iloc[-1] > 0 else 0
            features['macd_signal'] = macd.macd_signal().iloc[-1] / close.iloc[-1] if close.iloc[-1] > 0 else 0
            features['macd_diff'] = macd.macd_diff().iloc[-1] / close.iloc[-1] if close.iloc[-1] > 0 else 0
        
        # Williams %R
        williams = ta.momentum.williams_r(high, low, close)
        features['williams_r'] = (williams.iloc[-1] + 100) / 100.0  # Normalize to 0-1
        
        # Money Flow Index
        mfi = ta.volume.money_flow_index(high, low, close, volume)
        features['mfi'] = mfi.iloc[-1] / 100.0
        
        self.feature_groups['momentum'].extend(['rsi_14', 'macd', 'stochastic', 'mfi'])
        
        return features
    
    def _extract_liquidity_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract liquidity measures"""
        features = {}
        
        close = data['Close'].values
        volume = data['Volume'].values
        high = data['High'].values
        low = data['Low'].values
        
        # Amihud illiquidity ratio
        returns = np.abs(np.diff(close) / close[:-1])
        if len(returns) > 0 and volume[-len(returns):].sum() > 0:
            features['amihud_illiquidity'] = np.mean(returns / (volume[-len(returns):] + 1e-10))
        
        # Roll's implicit spread estimator
        if len(close) > 20:
            price_changes = np.diff(close[-20:])
            if len(price_changes) > 1:
                autocov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
                if autocov < 0:
                    features['roll_spread'] = 2 * np.sqrt(-autocov) / close[-1]
                else:
                    features['roll_spread'] = 0
        
        # Kyle's lambda (already calculated in microstructure)
        
        # Turnover ratio
        features['turnover_ratio'] = volume[-1] / volume[-20:].mean() if volume[-20:].mean() > 0 else 1
        
        self.feature_groups['liquidity'].extend(['amihud_illiquidity', 'turnover_ratio'])
        
        return features
    
    def _extract_pattern_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract price pattern features"""
        features = {}
        
        close = data['Close'].values
        
        # Tick patterns (up/down sequences)
        if len(close) > 10:
            price_changes = np.sign(np.diff(close[-10:]))
            
            # Run length encoding
            runs = []
            current_run = 1
            for i in range(1, len(price_changes)):
                if price_changes[i] == price_changes[i-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)
            
            features['max_run_length'] = max(runs) if runs else 0
            features['avg_run_length'] = np.mean(runs) if runs else 0
            features['num_reversals'] = len(runs) - 1
        
        # Support/Resistance levels
        if len(data) > 50:
            # Find local maxima/minima
            highs = data['High'].values[-50:]
            lows = data['Low'].values[-50:]
            
            # Simple peak detection
            high_peaks = signal.find_peaks(highs, distance=5)[0]
            low_peaks = signal.find_peaks(-lows, distance=5)[0]
            
            if len(high_peaks) > 0:
                resistance = highs[high_peaks[-1]]
                features['distance_to_resistance'] = (resistance - close[-1]) / close[-1]
            
            if len(low_peaks) > 0:
                support = lows[low_peaks[-1]]
                features['distance_to_support'] = (close[-1] - support) / close[-1]
        
        self.feature_groups['pattern'].extend(['max_run_length', 'num_reversals'])
        
        return features
    
    def _extract_time_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract time-based features"""
        features = {}
        
        # Assuming data has datetime index
        if hasattr(data.index, 'hour'):
            current_time = data.index[-1]
            
            # Time of day features
            features['hour_of_day'] = current_time.hour / 24.0
            features['minute_of_hour'] = current_time.minute / 60.0
            
            # Market session features (US market hours)
            market_open = 9.5  # 9:30 AM
            market_close = 16  # 4:00 PM
            current_decimal_hour = current_time.hour + current_time.minute / 60.0
            
            features['is_market_hours'] = 1.0 if market_open <= current_decimal_hour <= market_close else 0.0
            features['time_since_open'] = max(0, (current_decimal_hour - market_open) / (market_close - market_open))
            features['time_to_close'] = max(0, (market_close - current_decimal_hour) / (market_close - market_open))
            
            # Intraday seasonality
            features['morning_session'] = 1.0 if 9.5 <= current_decimal_hour <= 11.5 else 0.0
            features['lunch_session'] = 1.0 if 11.5 <= current_decimal_hour <= 13.5 else 0.0
            features['afternoon_session'] = 1.0 if 13.5 <= current_decimal_hour <= 16 else 0.0
            
        self.feature_groups['time_based'].extend(['hour_of_day', 'is_market_hours', 'time_since_open'])
        
        return features
    
    def _extract_statistical_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract statistical features"""
        features = {}
        
        close = data['Close'].values
        returns = np.diff(close) / close[:-1]
        
        if len(returns) > 20:
            # Distribution moments
            features['skewness'] = stats.skew(returns[-20:])
            features['kurtosis'] = stats.kurtosis(returns[-20:])
            
            # Percentiles
            features['percentile_25'] = np.percentile(returns[-20:], 25)
            features['percentile_75'] = np.percentile(returns[-20:], 75)
            features['iqr'] = features['percentile_75'] - features['percentile_25']
            
            # Autocorrelation
            if len(returns) > 30:
                features['autocorr_1'] = np.corrcoef(returns[-30:-1], returns[-29:])[0, 1]
                features['autocorr_5'] = np.corrcoef(returns[-30:-5], returns[-25:])[0, 1]
            
            # Hurst exponent (simplified)
            features['hurst'] = self._calculate_hurst_exponent(close[-50:]) if len(close) >= 50 else 0.5
        
        self.feature_groups['statistical'].extend(['skewness', 'kurtosis', 'autocorr_1'])
        
        return features
    
    def _extract_multitimeframe_features(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Extract features from multiple timeframes"""
        features = {}
        
        for tf in self.config.timeframes:
            if tf in market_data and not market_data[tf].empty:
                tf_data = market_data[tf]
                
                # Get key indicators for each timeframe
                if len(tf_data) > 20:
                    close = tf_data['Close'].values
                    
                    # Trend strength
                    sma20 = np.mean(close[-20:])
                    features[f'{tf}_trend'] = (close[-1] - sma20) / sma20 if sma20 > 0 else 0
                    
                    # Volatility
                    features[f'{tf}_volatility'] = np.std(close[-20:]) / np.mean(close[-20:])
                    
                    # RSI
                    if len(close) > 14:
                        rsi = ta.momentum.rsi(pd.Series(close), window=14)
                        features[f'{tf}_rsi'] = rsi.iloc[-1] / 100.0
        
        return features
    
    def _calculate_vpin(self, data: pd.DataFrame) -> float:
        """Calculate Volume-synchronized Probability of Informed Trading"""
        try:
            close = data['Close'].values
            volume = data['Volume'].values
            
            # Bucket trades by volume
            bucket_size = volume[-50:].sum() / 50  # 50 buckets
            
            # Classify volume as buy or sell
            price_changes = np.diff(close[-51:])
            buy_volume = volume[-50:][price_changes > 0].sum()
            sell_volume = volume[-50:][price_changes <= 0].sum()
            
            # VPIN = |Buy Volume - Sell Volume| / Total Volume
            total_volume = buy_volume + sell_volume + 1e-10
            vpin = np.abs(buy_volume - sell_volume) / total_volume
            
            return vpin
            
        except Exception:
            return 0.5
    
    def _calculate_garman_klass_vol(self, data: pd.DataFrame) -> float:
        """Calculate Garman-Klass volatility estimator"""
        try:
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            open_price = data['Open'].values
            
            # GK formula
            term1 = 0.5 * np.log(high / low) ** 2
            term2 = (2 * np.log(2) - 1) * np.log(close / open_price) ** 2
            
            gk_vol = np.sqrt(np.mean(term1 - term2))
            return gk_vol
            
        except Exception:
            return 0
    
    def _calculate_hurst_exponent(self, ts: np.ndarray) -> float:
        """Calculate Hurst exponent for time series"""
        try:
            # Simplified R/S analysis
            lags = range(2, min(len(ts) // 2, 20))
            tau = []
            
            for lag in lags:
                # Calculate R/S for this lag
                rs_values = []
                for start in range(0, len(ts) - lag):
                    subset = ts[start:start + lag]
                    mean = np.mean(subset)
                    deviations = subset - mean
                    Z = np.cumsum(deviations)
                    R = np.max(Z) - np.min(Z)
                    S = np.std(subset, ddof=1)
                    
                    if S != 0:
                        rs_values.append(R / S)
                
                if rs_values:
                    tau.append(np.mean(rs_values))
            
            if len(tau) > 2:
                # Fit log(tau) = H * log(lag) + c
                log_lags = np.log(list(lags)[:len(tau)])
                log_tau = np.log(tau)
                
                # Linear regression
                H = np.polyfit(log_lags, log_tau, 1)[0]
                return H
            
        except Exception:
            pass
        
        return 0.5  # Random walk
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values"""
        # Create a flat list of all possible feature names
        all_features = []
        for period in [1, 2, 5, 10, 20]:
            all_features.append(f'return_{period}')
        
        all_features.extend([
            'price_position', 'volume_ratio', 'volume_trend', 'hl_ratio',
            'spread', 'spread_relative', 'price_impact', 'effective_spread',
            'order_flow_imbalance', 'order_flow_imbalance_5', 'vpin',
            'buy_pressure', 'sell_pressure', 'net_pressure',
            'volatility_1min', 'volatility_5min', 'volatility_ratio',
            'parkinson_vol', 'gk_volatility', 'atr_ratio',
            'rsi_7', 'rsi_14', 'rsi_21', 'stochastic', 'macd', 'macd_signal',
            'macd_diff', 'williams_r', 'mfi', 'amihud_illiquidity',
            'roll_spread', 'turnover_ratio', 'max_run_length', 'avg_run_length',
            'num_reversals', 'distance_to_resistance', 'distance_to_support',
            'hour_of_day', 'minute_of_hour', 'is_market_hours', 'time_since_open',
            'time_to_close', 'morning_session', 'lunch_session', 'afternoon_session',
            'skewness', 'kurtosis', 'percentile_25', 'percentile_75', 'iqr',
            'autocorr_1', 'autocorr_5', 'hurst'
        ])
        
        # Add multi-timeframe features
        for tf in self.config.timeframes:
            all_features.extend([f'{tf}_trend', f'{tf}_volatility', f'{tf}_rsi'])
        
        return {feature: 0.0 for feature in all_features}
    
    def _validate_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Validate and clean features"""
        validated = {}
        
        for name, value in features.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    validated[name] = 0.0
                else:
                    validated[name] = float(value)
            else:
                validated[name] = 0.0
        
        return validated
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        if not self.feature_names:
            # Generate feature names from a dummy extraction
            dummy_data = pd.DataFrame({
                'Open': [100] * 100,
                'High': [101] * 100,
                'Low': [99] * 100,
                'Close': [100] * 100,
                'Volume': [1000] * 100
            })
            dummy_features = self.extract_features(dummy_data)
            self.feature_names = sorted(dummy_features.keys())
        
        return self.feature_names


class ScalpingMLModel:
    """Industry-standard ML model for scalping with ensemble methods"""
    
    def __init__(self, commentary_system=None):
        self.commentary = commentary_system
        self._setup_commentary()
        self.feature_engineer = ScalpingFeatureEngineer()
        self.feature_names = self.feature_engineer.get_feature_names()
        
        # Model ensemble
        self.models = self._initialize_models()
        self.meta_model = None
        self.feature_selector = None
        self.scaler = RobustScaler()
        
        # Model state
        self.is_trained = False
        self.selected_features = None
        self.feature_importance = {}
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.model_weights = {name: 1.0 for name in self.models.keys()}
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        
        # Online learning
        self.online_buffer = deque(maxlen=500)
        self.last_retrain_time = None
        self.retrain_threshold = 100  # samples
        
        # Paths
        self.model_path = Path("scalping_ml_model.pkl")
        
        # Version control
        if HAS_VERSION_CONTROL:
            self.version_control = ModelVersionControl()
            self.current_version_id = None
        else:
            self.version_control = None
        
    def _setup_commentary(self):
        """Setup commentary helper"""
        self._commentary_available = False
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from trading_bot_commentary_updated import TradingCommentary, CommentaryType
            self.TradingCommentary = TradingCommentary
            self.CommentaryType = CommentaryType
            self._commentary_available = True
        except ImportError:
            logger.debug("TradingCommentary not available")
    
    def _add_commentary(self, comment_data: Dict):
        """Add commentary with proper object type"""
        if not self.commentary:
            return
            
        if self._commentary_available:
            commentary = self.TradingCommentary(
                timestamp=comment_data.get('timestamp', datetime.now()),
                type=self.CommentaryType.MARKET_ANALYSIS,
                symbol=comment_data.get('symbol'),
                title=comment_data.get('title', 'ML Model Update'),
                message=comment_data.get('message', ''),
                data=comment_data.get('data', comment_data),
                confidence=comment_data.get('confidence'),
                importance=comment_data.get('importance', 5)
            )
            self.commentary.add_commentary(commentary)
        else:
            # Fallback
            self.commentary.add_commentary(comment_data)
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize ensemble of models optimized for scalping"""
        models = {
            'xgb_scalp': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,  # Shallow for fast inference
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                min_child_weight=5,
                objective='multi:softprob',
                num_class=3,  # Buy, Hold, Sell
                tree_method='hist',  # Faster training
                random_state=42,
                n_jobs=-1
            ),
            
            'lgb_scalp': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                min_data_in_leaf=20,
                lambda_l1=0.1,
                lambda_l2=0.1,
                objective='multiclass',
                num_class=3,
                metric='multi_logloss',
                verbose=-1,
                random_state=42,
                n_jobs=-1
            ),
            
            
            'rf_fast': RandomForestClassifier(
                n_estimators=100,  # Fewer trees for speed
                max_depth=8,
                min_samples_split=50,
                min_samples_leaf=20,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            ),
            
            'et_fast': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=50,
                min_samples_leaf=20,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )
        }
        
        return models
    
    def train(self, X: np.ndarray, y: np.ndarray, optimize_hyperparams: bool = False):
        """Train the ensemble model with feature selection"""
        if len(X) < 200:
            logger.warning(f"Insufficient samples for training: {len(X)}")
            return False
            
        try:
            # Feature selection
            self._select_features(X, y)
            X_selected = X[:, self.selected_features]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_selected)
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train and evaluate each model
            model_scores = {}
            
            for name, model in self.models.items():
                scores = []
                
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model
                    if 'xgb' in name:
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                                early_stopping_rounds=50, verbose=False)
                    elif 'lgb' in name:
                        # LightGBM uses callbacks for early stopping
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                    else:
                        model.fit(X_train, y_train)
                    
                    # Evaluate
                    val_pred = model.predict(X_val)
                    score = f1_score(y_val, val_pred, average='weighted')
                    scores.append(score)
                
                model_scores[name] = np.mean(scores)
                
                # Final training on all data
                model.fit(X_scaled, y)
            
            # Update model weights based on performance
            self._update_model_weights(model_scores)
            
            # Train meta-model (stacking)
            self._train_meta_model(X_scaled, y)
            
            # Calculate feature importance
            self._calculate_feature_importance()
            
            # Update state
            self.is_trained = True
            self.last_retrain_time = datetime.now()
            
            # Save model
            self.save_model()
            
            if self.commentary:
                self._add_commentary({
                    'type': 'MODEL_TRAINING',
                    'message': f'Scalping model trained on {len(X)} samples',
                    'scores': model_scores,
                    'selected_features': len(self.selected_features)
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def predict(self, data: pd.DataFrame, quote_data: Optional[Dict] = None,
               market_data: Optional[Dict] = None) -> Tuple[int, float, Dict]:
        """Make prediction with confidence and explanation"""
        if not self.is_trained:
            return 0, 0.5, {'error': 'Model not trained'}
        
        try:
            # Extract features
            features = self.feature_engineer.extract_features(data, quote_data, market_data)
            
            # Convert to array and select features
            X = np.array([features[name] for name in self.feature_names])
            X_selected = X[self.selected_features].reshape(1, -1)
            X_scaled = self.scaler.transform(X_selected)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0]
                predictions[name] = pred
                probabilities[name] = prob
            
            # Meta prediction
            if self.meta_model:
                meta_features = self._get_meta_features(X_scaled)
                final_pred = self.meta_model.predict(meta_features)[0]
                final_prob = self.meta_model.predict_proba(meta_features)[0]
            else:
                # Weighted voting
                final_pred = self._weighted_vote(predictions)
                final_prob = self._weighted_average_proba(probabilities)
            
            # Convert to trading signal
            signal_map = {0: -1, 1: 0, 2: 1}  # Sell, Hold, Buy
            signal = signal_map[final_pred]
            confidence = float(np.max(final_prob))
            
            # Generate explanation
            explanation = self._generate_explanation(features, predictions, 
                                                   final_pred, confidence)
            
            # Store for online learning
            self.online_buffer.append({
                'features': X_selected[0],
                'prediction': final_pred,
                'timestamp': datetime.now()
            })
            
            return signal, confidence, explanation
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0, 0.5, {'error': str(e)}
    
    def update_online(self, features: np.ndarray, true_label: int, reward: float):
        """Update model with new observation (online learning)"""
        self.online_buffer.append({
            'features': features,
            'label': true_label,
            'reward': reward,
            'timestamp': datetime.now()
        })
        
        # Update performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'reward': reward,
            'label': true_label
        })
        
        # Check if retraining needed
        if len(self.online_buffer) >= self.retrain_threshold:
            self._retrain_online()
    
    def _select_features(self, X: np.ndarray, y: np.ndarray):
        """Select most important features"""
        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=50)
        selector.fit(X, y)
        
        # Get selected feature indices
        self.selected_features = selector.get_support(indices=True)
        self.feature_selector = selector
        
        # Store feature scores
        feature_scores = dict(zip(range(len(self.feature_names)), selector.scores_))
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        
        logger.info(f"Selected {len(self.selected_features)} features")
        logger.info(f"Top features: {[self.feature_names[i] for i, _ in top_features[:10]]}")
    
    def _train_meta_model(self, X: np.ndarray, y: np.ndarray):
        """Train meta-model for stacking"""
        # Get predictions from base models
        meta_features = self._get_meta_features(X)
        
        # Train simple meta-model
        self.meta_model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        
        self.meta_model.fit(meta_features, y)
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Get meta-features from base model predictions"""
        meta_features = []
        
        for name, model in self.models.items():
            # Get probability predictions
            proba = model.predict_proba(X)
            meta_features.append(proba)
        
        return np.hstack(meta_features)
    
    def _update_model_weights(self, scores: Dict[str, float]):
        """Update model weights based on performance"""
        # Normalize scores
        total_score = sum(scores.values())
        
        for name, score in scores.items():
            self.model_weights[name] = score / total_score
    
    def _weighted_vote(self, predictions: Dict[str, int]) -> int:
        """Weighted voting for ensemble prediction"""
        vote_counts = {0: 0, 1: 0, 2: 0}
        
        for name, pred in predictions.items():
            vote_counts[pred] += self.model_weights[name]
        
        return max(vote_counts, key=vote_counts.get)
    
    def _weighted_average_proba(self, probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average of probability predictions"""
        weighted_proba = np.zeros(3)
        
        for name, proba in probabilities.items():
            weighted_proba += proba * self.model_weights[name]
        
        return weighted_proba / sum(self.model_weights.values())
    
    def _calculate_feature_importance(self):
        """Calculate aggregate feature importance"""
        importance_scores = {}
        
        # Get importance from tree-based models
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                for idx, importance in enumerate(importances):
                    if idx in self.selected_features:
                        feature_idx = self.selected_features[idx]
                        feature_name = self.feature_names[feature_idx]
                        
                        if feature_name not in importance_scores:
                            importance_scores[feature_name] = []
                        
                        importance_scores[feature_name].append(importance)
        
        # Average importances
        self.feature_importance = {
            name: np.mean(scores) 
            for name, scores in importance_scores.items()
        }
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), 
                   key=lambda x: x[1], reverse=True)[:20]
        )
    
    def _generate_explanation(self, features: Dict[str, float], 
                            predictions: Dict[str, int],
                            final_pred: int, confidence: float) -> Dict[str, Any]:
        """Generate detailed explanation for prediction"""
        explanation = {
            'prediction': final_pred,
            'confidence': confidence,
            'signal_strength': 'strong' if confidence > 0.7 else 'moderate' if confidence > 0.55 else 'weak',
            'model_votes': predictions,
            'model_weights': self.model_weights,
            'top_features': {}
        }
        
        # Add top feature values
        for feature_name in list(self.feature_importance.keys())[:10]:
            if feature_name in features:
                explanation['top_features'][feature_name] = features[feature_name]
        
        # Add specific insights
        if features.get('order_flow_imbalance', 0) > 0.3:
            explanation['insight'] = "Strong buy pressure detected"
        elif features.get('order_flow_imbalance', 0) < -0.3:
            explanation['insight'] = "Strong sell pressure detected"
        elif features.get('volatility_ratio', 1) > 2:
            explanation['insight'] = "High short-term volatility"
        elif features.get('spread_relative', 0) > 0.002:
            explanation['insight'] = "Wide spread - lower liquidity"
        else:
            explanation['insight'] = "Normal market conditions"
        
        return explanation
    
    def _retrain_online(self):
        """Retrain model with online data"""
        # Convert buffer to training data
        online_data = list(self.online_buffer)
        
        if len(online_data) < 100:
            return
        
        X_online = np.array([d['features'] for d in online_data if 'label' in d])
        y_online = np.array([d['label'] for d in online_data if 'label' in d])
        
        if len(X_online) < 50:
            return
        
        # Partial fit for models that support it
        for name, model in self.models.items():
            if hasattr(model, 'partial_fit'):
                model.partial_fit(X_online, y_online, classes=[0, 1, 2])
        
        # Clear buffer
        self.online_buffer.clear()
        
        logger.info("Online model update completed")
    
    def save_model(self, create_version: bool = True):
        """Save model to disk with optional version control
        
        Args:
            create_version: Whether to create a versioned backup
        """
        model_data = {
            'models': self.models,
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_weights': self.model_weights,
            'is_trained': self.is_trained,
            'last_retrain_time': self.last_retrain_time,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_profit': self.total_profit,
            # CRITICAL: Save the online buffer so we don't lose training data!
            'online_buffer': list(self.online_buffer),  # Convert deque to list for pickling
            'performance_history': list(self.performance_history)  # Save performance history too
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Log buffer save
        if self.online_buffer:
            logger.info(f"Saved {len(self.online_buffer)} training samples in buffer")
        
        # Create version if enabled (rest of versioning code stays the same)
    
    def load_model(self, version_id: Optional[str] = None):
        """Load model from disk, optionally loading a specific version
        
        Args:
            version_id: Specific version to load (None for latest/best)
        
        Returns:
            Success status
        """
        # Try version control first if available
        if self.version_control and version_id:
            model_path = self.version_control.load_version(version_id)
            if model_path:
                self.model_path = model_path
        elif self.version_control and not self.model_path.exists():
            # Try to load best version if main model doesn't exist
            model_path = self.version_control.load_version()
            if model_path:
                self.model_path = model_path
        
        if not self.model_path.exists():
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.meta_model = model_data['meta_model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.selected_features = model_data['selected_features']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data['feature_importance']
            self.model_weights = model_data['model_weights']
            self.is_trained = model_data['is_trained']
            self.last_retrain_time = model_data['last_retrain_time']
            
            # Load performance metrics if available
            self.total_trades = model_data.get('total_trades', 0)
            self.winning_trades = model_data.get('winning_trades', 0)
            self.total_profit = model_data.get('total_profit', 0.0)
            
            # CRITICAL: Restore the online buffer if available
            if 'online_buffer' in model_data:
                buffer_data = model_data['online_buffer']
                self.online_buffer = deque(buffer_data, maxlen=500)
                logger.info(f"Restored {len(self.online_buffer)} training samples from buffer")
            
            # Restore performance history if available
            if 'performance_history' in model_data:
                history_data = model_data['performance_history']
                self.performance_history = deque(history_data, maxlen=1000)
                logger.info(f"Restored {len(self.performance_history)} performance records")
            
            logger.info(f"Model loaded successfully (Version: {version_id or 'latest'})")
            
            if self.commentary:
                win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
                self._add_commentary({
                    'type': 'MODEL_LOADED',
                    'message': f'Model loaded: {self.total_trades} trades, {win_rate:.1f}% win rate, ${self.total_profit:.2f} profit',
                    'importance': 6
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _calculate_recent_accuracy(self):
        """Calculate accuracy of recent predictions"""
        if len(self.performance_history) < 10:
            return 0.5  # Default accuracy
        
        recent = list(self.performance_history)[-100:]  # Last 100 predictions
        correct = sum(1 for p in recent if p.get('correct', False))
        return correct / len(recent)


class ModelValidator:
    """Validate model performance and detect issues"""
    
    def __init__(self):
        self.validation_metrics = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def validate_predictions(self, predictions: List[Dict], 
                           true_labels: List[int]) -> Dict[str, float]:
        """Validate model predictions"""
        pred_labels = [p['prediction'] for p in predictions]
        
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='weighted'),
            'recall': recall_score(true_labels, pred_labels, average='weighted'),
            'f1': f1_score(true_labels, pred_labels, average='weighted')
        }
        
        # Check for prediction bias
        pred_distribution = pd.Series(pred_labels).value_counts(normalize=True)
        metrics['prediction_entropy'] = -sum(p * np.log(p + 1e-10) for p in pred_distribution)
        
        # Check confidence calibration
        confidences = [p['confidence'] for p in predictions]
        metrics['avg_confidence'] = np.mean(confidences)
        metrics['confidence_std'] = np.std(confidences)
        
        self.validation_metrics = metrics
        return metrics
    
    def detect_feature_drift(self, current_features: np.ndarray, 
                           historical_features: np.ndarray) -> bool:
        """Detect if feature distribution has drifted"""
        if len(historical_features) < 100:
            return False
        
        # Fit anomaly detector on historical data
        self.anomaly_detector.fit(historical_features)
        
        # Check if current features are anomalous
        anomaly_scores = self.anomaly_detector.decision_function(current_features)
        anomaly_rate = np.mean(anomaly_scores < 0)
        
        return anomaly_rate > 0.3  # 30% anomaly rate threshold


# Integration function for existing trading bot
def create_scalping_ml_predictor(commentary_system=None) -> ScalpingMLModel:
    """Create and initialize scalping ML model"""
    model = ScalpingMLModel(commentary_system)
    
    # Try to load existing model
    if model.load_model():
        logger.info("Loaded existing scalping ML model")
    else:
        logger.info("No existing model found, will train on first batch of data")
    
    return model