"""
Training and validation pipeline for Scalping ML Model
Includes backtesting, walk-forward analysis, and performance monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from scalping_ml_model import ScalpingMLModel, ScalpingFeatureEngineer
from scalping_ml_integration import ScalpingMLPredictor

logger = logging.getLogger(__name__)

class ScalpingModelTrainer:
    """Advanced training pipeline with validation and monitoring"""
    
    def __init__(self, data_provider, commentary_system=None):
        self.data_provider = data_provider
        self.commentary = commentary_system
        self._setup_commentary()
        self.model = ScalpingMLModel(commentary_system)
        self.feature_engineer = ScalpingFeatureEngineer()
        
        # Training configuration
        self.training_config = {
            'min_samples': 1000,
            'validation_split': 0.2,
            'n_splits': 5,  # For time series CV
            'lookback_days': 30,
            'prediction_horizon': 10,  # bars
            'profit_threshold': 0.003,  # 0.3% for scalping
            'risk_reward_ratio': 1.5
        }
        
        # Performance tracking
        self.training_history = []
        self.validation_results = {}
        self.backtest_results = []
        
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
            # Map types
            type_mapping = {
                'DATA_COLLECTION': self.CommentaryType.MARKET_ANALYSIS,
                'MODEL_VALIDATION': self.CommentaryType.TECHNICAL,
                'BACKTEST_COMPLETE': self.CommentaryType.DECISION,
            }
            
            comment_type = type_mapping.get(
                comment_data.get('type', ''), 
                self.CommentaryType.MARKET_ANALYSIS
            )
            
            commentary = self.TradingCommentary(
                timestamp=comment_data.get('timestamp', datetime.now()),
                type=comment_type,
                symbol=comment_data.get('symbol'),
                title=comment_data.get('title', ''),
                message=comment_data.get('message', ''),
                data=comment_data.get('data', {}),
                confidence=comment_data.get('confidence'),
                importance=comment_data.get('importance', 5)
            )
            self.commentary.add_commentary(commentary)
        else:
            self.commentary.add_commentary(comment_data)
        
    def collect_training_data(self, symbols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Collect and prepare training data from multiple symbols"""
        all_features = []
        all_labels = []
        all_metadata = []
        
        # Parallel data collection
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def collect_symbol_data(symbol):
            """Collect data for a single symbol"""
            logger.info(f"Collecting data for {symbol}")
            try:
                # Fetch both timeframes in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_1min = executor.submit(
                        self.data_provider.get_market_data,
                        symbol,
                        period_type='month',
                        period=1,
                        frequency_type='minute',
                        frequency=1
                    )
                    future_5min = executor.submit(
                        self.data_provider.get_market_data,
                        symbol,
                        period_type='month',
                        period=1,
                        frequency_type='minute',
                        frequency=5
                    )
                    
                    data_1min = future_1min.result()
                    data_5min = future_5min.result()
                
                if data_5min.empty or len(data_5min) < 100:
                    return symbol, [], [], []
                
                # Generate samples
                samples = self._generate_labeled_samples(
                    {'1min': data_1min, '5min': data_5min},
                    symbol
                )
                
                features = [s['features'] for s in samples]
                labels = [s['label'] for s in samples]
                metadata = [s['metadata'] for s in samples]
                
                return symbol, features, labels, metadata
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                return symbol, [], [], []
        
        # Fetch all symbols in parallel
        with ThreadPoolExecutor(max_workers=min(len(symbols), 5)) as executor:
            futures = {executor.submit(collect_symbol_data, symbol): symbol 
                      for symbol in symbols}
            
            for future in as_completed(futures):
                symbol, features, labels, metadata = future.result()
                if features:
                    all_features.extend(features)
                    all_labels.extend(labels)
                    all_metadata.extend(metadata)
                    logger.info(f"Collected {len(features)} samples from {symbol}")
        
        logger.info(f"Collected {len(all_features)} training samples")
        
        if self.commentary:
            self._add_commentary({
                'timestamp': datetime.now(),
                'type': 'DATA_COLLECTION',
                'title': '📊 Training Data Collected',
                'message': f'Collected {len(all_features)} samples from {len(symbols)} symbols',
                'importance': 7
            })
        
        return np.array(all_features), np.array(all_labels), all_metadata
    
    def _generate_labeled_samples(self, market_data: Dict[str, pd.DataFrame], 
                                symbol: str) -> List[Dict]:
        """Generate labeled training samples"""
        samples = []
        primary_df = market_data['5min']
        
        # Use sliding window
        for i in range(50, len(primary_df) - self.training_config['prediction_horizon']):
            try:
                # Extract features at time i
                window_data = primary_df.iloc[:i+1]
                features = self.feature_engineer.extract_features(
                    window_data,
                    market_data=market_data
                )
                
                # Create label based on future price movement
                current_price = primary_df['Close'].iloc[i]
                future_window = primary_df.iloc[i+1:i+1+self.training_config['prediction_horizon']]
                
                label, metadata = self._create_scalping_label(
                    current_price, future_window
                )
                
                # Convert features to array
                feature_array = [features[name] for name in self.feature_engineer.get_feature_names()]
                
                samples.append({
                    'features': feature_array,
                    'label': label,
                    'metadata': {
                        'symbol': symbol,
                        'timestamp': primary_df.index[i],
                        'price': current_price,
                        **metadata
                    }
                })
                
            except Exception as e:
                logger.debug(f"Sample generation error: {e}")
                continue
        
        return samples
    
    def _create_scalping_label(self, entry_price: float, 
                             future_window: pd.DataFrame) -> Tuple[int, Dict]:
        """Create label optimized for scalping strategies"""
        high_prices = future_window['High'].values
        low_prices = future_window['Low'].values
        close_prices = future_window['Close'].values
        
        # Calculate potential gains and losses
        max_gain = (high_prices.max() - entry_price) / entry_price
        max_loss = (entry_price - low_prices.min()) / entry_price
        
        # Actual close price at end of window
        final_return = (close_prices[-1] - entry_price) / entry_price
        
        # Risk-reward based labeling
        profit_threshold = self.training_config['profit_threshold']
        rr_ratio = self.training_config['risk_reward_ratio']
        
        metadata = {
            'max_gain': max_gain,
            'max_loss': max_loss,
            'final_return': final_return
        }
        
        # Label logic optimized for scalping
        if max_gain > profit_threshold and max_gain > max_loss * rr_ratio:
            # Good long opportunity
            label = 2  # Buy
            metadata['signal_quality'] = 'strong' if max_gain > profit_threshold * 2 else 'moderate'
        elif max_loss > profit_threshold and max_loss > max_gain * rr_ratio:
            # Good short opportunity
            label = 0  # Sell
            metadata['signal_quality'] = 'strong' if max_loss > profit_threshold * 2 else 'moderate'
        else:
            # No clear opportunity
            label = 1  # Hold
            metadata['signal_quality'] = 'weak'
        
        return label, metadata
    
    def train_with_validation(self, X: np.ndarray, y: np.ndarray, 
                            metadata: List[Dict]) -> Dict[str, Any]:
        """Train model with comprehensive validation"""
        logger.info("Starting model training with validation")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.training_config['n_splits'])
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{self.training_config['n_splits']}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            fold_model = ScalpingMLModel()
            fold_model.train(X_train, y_train)
            
            # Validate
            val_results = self._validate_fold(fold_model, X_val, y_val, 
                                            [metadata[i] for i in val_idx])
            val_results['fold'] = fold
            cv_results.append(val_results)
            
            # Log progress
            logger.info(f"Fold {fold + 1} - Accuracy: {val_results['accuracy']:.3f}, "
                       f"Sharpe: {val_results['sharpe_ratio']:.2f}")
        
        # Train final model on all data
        self.model.train(X, y, optimize_hyperparams=True)
        
        # Aggregate results
        final_results = self._aggregate_cv_results(cv_results)
        self.validation_results = final_results
        
        # Save training history
        self._save_training_history(final_results)
        
        if self.commentary:
            self._add_commentary({
                'timestamp': datetime.now(),
                'type': 'MODEL_VALIDATION',
                'title': '✅ Model Training Complete',
                'message': f'Avg Accuracy: {final_results["avg_accuracy"]:.1%}\n'
                          f'Avg Sharpe: {final_results["avg_sharpe"]:.2f}',
                'data': final_results,
                'importance': 9
            })
        
        return final_results
    
    def _validate_fold(self, model: ScalpingMLModel, X_val: np.ndarray, 
                      y_val: np.ndarray, metadata: List[Dict]) -> Dict[str, Any]:
        """Validate a single fold with trading simulation"""
        predictions = []
        
        # Make predictions
        for i, (features, true_label) in enumerate(zip(X_val, y_val)):
            # Simulate real-time prediction
            pred_signal, confidence, explanation = model.predict(
                pd.DataFrame(),  # Empty df as we're using features directly
                quote_data=None,
                market_data=None
            )
            
            predictions.append({
                'true': true_label,
                'pred': pred_signal,
                'confidence': confidence,
                'metadata': metadata[i]
            })
        
        # Calculate metrics
        y_pred = [p['pred'] for p in predictions]
        
        # Classification metrics
        accuracy = np.mean(np.array(y_pred) == y_val)
        report = classification_report(y_val, y_pred, output_dict=True)
        
        # Trading metrics
        trading_results = self._simulate_trading(predictions)
        
        return {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            **trading_results
        }
    
    def _simulate_trading(self, predictions: List[Dict]) -> Dict[str, float]:
        """Simulate trading based on predictions"""
        trades = []
        
        for pred in predictions:
            if pred['pred'] != 1 and pred['confidence'] > 0.6:  # Not Hold and confident
                # Simulate trade
                entry_price = pred['metadata']['price']
                max_gain = pred['metadata']['max_gain']
                max_loss = pred['metadata']['max_loss']
                final_return = pred['metadata']['final_return']
                
                # Simple PnL calculation
                if pred['pred'] == 2:  # Buy
                    pnl = final_return
                elif pred['pred'] == 0:  # Sell
                    pnl = -final_return
                else:
                    pnl = 0
                
                trades.append({
                    'pnl': pnl,
                    'return': pnl,
                    'confidence': pred['confidence']
                })
        
        if not trades:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # Calculate metrics
        returns = [t['return'] for t in trades]
        winning_trades = sum(1 for r in returns if r > 0)
        
        metrics = {
            'num_trades': len(trades),
            'win_rate': winning_trades / len(trades),
            'avg_return': np.mean(returns),
            'sharpe_ratio': self._calculate_sharpe(returns),
            'max_drawdown': self._calculate_max_drawdown(returns)
        }
        
        return metrics
    
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0) -> float:
        """Calculate Sharpe ratio"""
        if not returns or np.std(returns) == 0:
            return 0
        
        excess_returns = np.array(returns) - risk_free_rate
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
    
    def _aggregate_cv_results(self, cv_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate cross-validation results"""
        metrics = defaultdict(list)
        
        for result in cv_results:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    metrics[key].append(value)
        
        aggregated = {}
        for key, values in metrics.items():
            aggregated[f'avg_{key}'] = np.mean(values)
            aggregated[f'std_{key}'] = np.std(values)
        
        # Add detailed breakdown
        aggregated['fold_results'] = cv_results
        
        return aggregated
    
    def backtest_model(self, test_data: Dict[str, pd.DataFrame], 
                      initial_capital: float = 10000) -> Dict[str, Any]:
        """Comprehensive backtesting of the trained model"""
        logger.info("Starting model backtest")
        
        results = {
            'trades': [],
            'equity_curve': [initial_capital],
            'timestamps': [],
            'metrics': {}
        }
        
        capital = initial_capital
        position = None
        
        # Generate predictions for test period
        for i in range(50, len(test_data['5min']) - 10):
            window_data = test_data['5min'].iloc[:i+1]
            
            # Get prediction
            features = self.feature_engineer.extract_features(window_data, market_data=test_data)
            feature_array = np.array([features[name] for name in self.feature_engineer.get_feature_names()])
            
            signal, confidence, explanation = self.model.predict(
                window_data, 
                quote_data=None,
                market_data=test_data
            )
            
            current_price = window_data['Close'].iloc[-1]
            timestamp = window_data.index[-1]
            
            # Execute trades
            if signal != 0 and confidence > 0.6 and position is None:
                # Open position
                position = {
                    'type': 'long' if signal == 1 else 'short',
                    'entry_price': current_price,
                    'entry_time': timestamp,
                    'size': capital * 0.1 / current_price,  # 10% position size
                    'confidence': confidence
                }
            elif position is not None:
                # Check exit conditions
                if position['type'] == 'long':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                # Exit on stop loss, take profit, or opposing signal
                if pnl_pct < -0.002 or pnl_pct > 0.003 or (signal != 0 and signal != (1 if position['type'] == 'long' else -1)):
                    # Close position
                    pnl = pnl_pct * position['size'] * position['entry_price']
                    capital += pnl
                    
                    results['trades'].append({
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'confidence': position['confidence']
                    })
                    
                    position = None
            
            results['equity_curve'].append(capital)
            results['timestamps'].append(timestamp)
        
        # Calculate final metrics
        results['metrics'] = self._calculate_backtest_metrics(results, initial_capital)
        
        self.backtest_results = results
        
        if self.commentary:
            self._add_commentary({
                'timestamp': datetime.now(),
                'type': 'BACKTEST_COMPLETE',
                'title': '📈 Backtest Results',
                'message': f"Total Return: {results['metrics']['total_return']:.1%}\n"
                          f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}\n"
                          f"Win Rate: {results['metrics']['win_rate']:.1%}",
                'data': results['metrics'],
                'importance': 8
            })
        
        return results
    
    def _calculate_backtest_metrics(self, results: Dict, initial_capital: float) -> Dict[str, float]:
        """Calculate comprehensive backtest metrics"""
        trades = results['trades']
        equity_curve = np.array(results['equity_curve'])
        
        if not trades:
            return {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # Basic metrics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        # Returns calculation
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics = {
            'total_return': (equity_curve[-1] - initial_capital) / initial_capital,
            'num_trades': len(trades),
            'win_rate': len(winning_trades) / len(trades),
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'profit_factor': abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades else 0,
            'sharpe_ratio': self._calculate_sharpe(returns.tolist()),
            'max_drawdown': self._calculate_max_drawdown(returns.tolist()),
            'avg_trade_duration': np.mean([(t['exit_time'] - t['entry_time']).total_seconds() / 60 for t in trades])  # minutes
        }
        
        return metrics
    
    def generate_performance_report(self, output_path: str = "scalping_model_report.html"):
        """Generate comprehensive performance report"""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity Curve
        if self.backtest_results:
            ax = axes[0, 0]
            equity = self.backtest_results['equity_curve']
            ax.plot(equity)
            ax.set_title('Equity Curve')
            ax.set_xlabel('Trade Number')
            ax.set_ylabel('Capital')
            ax.grid(True)
        
        # 2. Feature Importance
        if self.model.feature_importance:
            ax = axes[0, 1]
            top_features = list(self.model.feature_importance.items())[:15]
            features, importances = zip(*top_features)
            ax.barh(features, importances)
            ax.set_title('Top 15 Feature Importances')
            ax.set_xlabel('Importance')
        
        # 3. Confusion Matrix
        if self.validation_results and 'fold_results' in self.validation_results:
            ax = axes[1, 0]
            # Use last fold's results for visualization
            last_fold = self.validation_results['fold_results'][-1]
            if 'confusion_matrix' in last_fold:
                sns.heatmap(last_fold['confusion_matrix'], annot=True, fmt='d', ax=ax)
                ax.set_title('Confusion Matrix (Last Fold)')
        
        # 4. Returns Distribution
        if self.backtest_results and self.backtest_results['trades']:
            ax = axes[1, 1]
            returns = [t['pnl_pct'] * 100 for t in self.backtest_results['trades']]
            ax.hist(returns, bins=30, alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_title('Trade Returns Distribution')
            ax.set_xlabel('Return (%)')
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('scalping_model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate HTML report
        html_content = self._generate_html_report()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance report saved to {output_path}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML performance report"""
        html = f"""
        <html>
        <head>
            <title>Scalping ML Model Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
                .good {{ color: green; }}
                .bad {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Scalping ML Model Performance Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Model Information</h2>
            <div class="metric">
                <p><strong>Model Version:</strong> Scalping ML v1.0</p>
                <p><strong>Training Samples:</strong> {len(self.training_history) if self.training_history else 'N/A'}</p>
                <p><strong>Selected Features:</strong> {len(self.model.selected_features) if self.model.selected_features else 'N/A'}</p>
            </div>
            
            <h2>Validation Results</h2>
        """
        
        if self.validation_results:
            html += """
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Average</th>
                    <th>Std Dev</th>
                </tr>
            """
            
            for metric in ['accuracy', 'sharpe_ratio', 'win_rate', 'max_drawdown']:
                avg_key = f'avg_{metric}'
                std_key = f'std_{metric}'
                if avg_key in self.validation_results:
                    avg_val = self.validation_results[avg_key]
                    std_val = self.validation_results.get(std_key, 0)
                    
                    # Format based on metric type
                    if metric in ['accuracy', 'win_rate', 'max_drawdown']:
                        avg_str = f"{avg_val:.1%}"
                        std_str = f"{std_val:.1%}"
                    else:
                        avg_str = f"{avg_val:.2f}"
                        std_str = f"{std_val:.2f}"
                    
                    # Color coding
                    if metric in ['accuracy', 'win_rate', 'sharpe_ratio']:
                        class_name = 'good' if avg_val > 0.5 else 'bad'
                    else:  # max_drawdown
                        class_name = 'good' if avg_val < 0.1 else 'bad'
                    
                    html += f"""
                    <tr>
                        <td>{metric.replace('_', ' ').title()}</td>
                        <td class="{class_name}">{avg_str}</td>
                        <td>{std_str}</td>
                    </tr>
                    """
            
            html += "</table>"
        
        if self.backtest_results and 'metrics' in self.backtest_results:
            html += """
            <h2>Backtest Results</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            """
            
            metrics = self.backtest_results['metrics']
            for key, value in metrics.items():
                if key in ['total_return', 'win_rate', 'max_drawdown']:
                    value_str = f"{value:.1%}"
                elif key in ['sharpe_ratio', 'profit_factor']:
                    value_str = f"{value:.2f}"
                elif key == 'avg_trade_duration':
                    value_str = f"{value:.1f} min"
                else:
                    value_str = f"{value:.2f}"
                
                html += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td>{value_str}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add performance chart
        html += """
            <h2>Performance Charts</h2>
            <img src="scalping_model_performance.png" alt="Performance Charts" style="max-width: 100%;">
            
            <h2>Top Features</h2>
            <ul>
        """
        
        if self.model.feature_importance:
            for feature, importance in list(self.model.feature_importance.items())[:10]:
                html += f"<li>{feature}: {importance:.4f}</li>"
        
        html += """
            </ul>
        </body>
        </html>
        """
        
        return html
    
    def _save_training_history(self, results: Dict[str, Any]):
        """Save training history to file"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'config': self.training_config,
            'model_params': {
                'n_features': len(self.model.feature_names),
                'n_selected': len(self.model.selected_features) if self.model.selected_features else 0
            }
        }
        
        self.training_history.append(history_entry)
        
        # Save to file
        history_path = Path("scalping_training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)


def run_training_pipeline(data_provider, symbols: List[str], 
                         commentary_system=None) -> ScalpingModelTrainer:
    """Run complete training pipeline"""
    trainer = ScalpingModelTrainer(data_provider, commentary_system)
    
    # Collect data
    X, y, metadata = trainer.collect_training_data(symbols)
    
    if len(X) < trainer.training_config['min_samples']:
        logger.error(f"Insufficient training data: {len(X)} samples")
        return trainer
    
    # Train with validation
    validation_results = trainer.train_with_validation(X, y, metadata)
    
    # Run backtest on recent data
    test_symbols = symbols[:3]  # Test on subset
    for symbol in test_symbols:
        try:
            test_data = {
                '1min': data_provider.get_market_data(
                    symbol, period_type='day', period=5,
                    frequency_type='minute', frequency=1
                ),
                '5min': data_provider.get_market_data(
                    symbol, period_type='day', period=5,
                    frequency_type='minute', frequency=5
                )
            }
            
            trainer.backtest_model(test_data)
            break  # One backtest for demonstration
            
        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}")
            continue
    
    # Generate report
    trainer.generate_performance_report()
    
    return trainer