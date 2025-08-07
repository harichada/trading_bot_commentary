#!/usr/bin/env python3
"""
Asynchronous ML training wrapper to prevent blocking the main trading loop
"""

import asyncio
import threading
import queue
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AsyncMLTrainer:
    """Handles ML model training in a separate thread to prevent blocking"""
    
    def __init__(self, model_path: Path = Path("ml_model_integrated.pkl")):
        self.model_path = model_path
        self.training_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_training = False
        self.training_thread = None
        self.last_training_time = None
        self.min_training_interval = 3600  # 1 hour minimum between trainings
        
    def can_train(self) -> bool:
        """Check if enough time has passed since last training"""
        if not self.last_training_time:
            return True
        
        elapsed = (datetime.now() - self.last_training_time).total_seconds()
        return elapsed >= self.min_training_interval
        
    async def train_async(self, ml_predictor, data_provider, symbols: List[str]) -> bool:
        """
        Asynchronously train the model without blocking the main loop
        """
        if self.is_training:
            logger.warning("ML training already in progress, skipping")
            return False
            
        if not self.can_train():
            logger.info("Too soon since last training, skipping")
            return False
            
        # Start training in separate thread
        self.is_training = True
        training_data = {
            'ml_predictor': ml_predictor,
            'data_provider': data_provider,
            'symbols': symbols,
            'timestamp': datetime.now()
        }
        
        # Run training in thread pool
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, self._train_in_thread, training_data)
        
        # Add commentary about starting training
        if ml_predictor.commentary:
            ml_predictor.commentary.add_commentary({
                'timestamp': datetime.now(),
                'type': 'MARKET_ANALYSIS',
                'symbol': None,
                'title': '🔧 ML Training Started (Async)',
                'message': f'Training model in background with {len(symbols)} symbols. Trading continues uninterrupted.',
                'importance': 7
            })
        
        # Don't await - let it run in background
        asyncio.create_task(self._monitor_training(future, ml_predictor))
        return True
        
    def _train_in_thread(self, training_data: Dict) -> Dict:
        """Execute training in separate thread"""
        try:
            start_time = datetime.now()
            ml_predictor = training_data['ml_predictor']
            data_provider = training_data['data_provider']
            symbols = training_data['symbols']
            
            # Collect training data
            X, y = ml_predictor.model.collect_training_data(data_provider, symbols)
            
            if len(X) > 0:
                # Train model
                success = ml_predictor.model.train(X, y)
                
                # Save model
                if success:
                    ml_predictor.model.save_model()
                
                duration = (datetime.now() - start_time).total_seconds()
                
                return {
                    'success': success,
                    'samples': len(X),
                    'duration': duration,
                    'timestamp': datetime.now()
                }
            else:
                return {
                    'success': False,
                    'error': 'Insufficient training data',
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            logger.error(f"Error in ML training thread: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
        finally:
            self.is_training = False
            
    async def _monitor_training(self, future, ml_predictor):
        """Monitor training completion"""
        try:
            result = await future
            
            if result['success']:
                self.last_training_time = datetime.now()
                
                # Add success commentary
                if ml_predictor.commentary:
                    ml_predictor.commentary.add_commentary({
                        'timestamp': datetime.now(),
                        'type': 'MARKET_ANALYSIS',
                        'symbol': None,
                        'title': '✅ ML Training Completed',
                        'message': f"Model trained on {result['samples']} samples in {result['duration']:.1f}s",
                        'data': {
                            'samples': result['samples'],
                            'duration': result['duration']
                        },
                        'importance': 7
                    })
                    
                # Update brain
                if ml_predictor.brain:
                    ml_predictor.brain.remember_trade(
                        symbol="MODEL",
                        pattern="async_retraining",
                        outcome="completed",
                        pnl_percent=0,
                        context={
                            'samples': result['samples'],
                            'duration': result['duration'],
                            'timestamp': datetime.now().isoformat()
                        }
                    )
            else:
                # Add error commentary
                if ml_predictor.commentary:
                    ml_predictor.commentary.add_commentary({
                        'timestamp': datetime.now(),
                        'type': 'WARNING',
                        'symbol': None,
                        'title': '⚠️ ML Training Failed',
                        'message': f"Error: {result.get('error', 'Unknown error')}",
                        'importance': 8
                    })
                    
        except Exception as e:
            logger.error(f"Error monitoring ML training: {e}")
            self.is_training = False


# Global instance
ml_trainer = AsyncMLTrainer()


def patch_trading_engine():
    """
    Patch the existing TradingEngine to use async ML training
    """
    import trading_bot_commentary_updated
    
    # Store original method
    original_retrain = trading_bot_commentary_updated.MLPredictor.retrain_model
    
    # Create new async method
    async def async_retrain_model(self, data_provider, symbols: List[str]):
        """Async wrapper for ML training"""
        await ml_trainer.train_async(self, data_provider, symbols)
    
    # Replace method
    trading_bot_commentary_updated.MLPredictor.retrain_model = async_retrain_model
    
    logger.info("Patched MLPredictor with async training")