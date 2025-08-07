#!/usr/bin/env python3
"""
Real-Time Trading Bot Launch Script
Starts the complete real-time trading system with streaming data
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class RealTimeTradingLauncher:
    """Launcher for the real-time trading system"""
    
    def __init__(self):
        self.processes = []
        self.is_running = False
        
    async def start_system(self):
        """Start the complete real-time trading system"""
        try:
            logger.info("🚀 Starting Real-Time Trading System...")
            
            # Print startup banner
            self._print_startup_banner()
            
            # Check system requirements
            await self._check_requirements()
            
            # Start Redis (if not running)
            await self._start_redis()
            
            # Start the real-time API
            await self._start_realtime_api()
            
            # Start the dashboard
            await self._start_dashboard()
            
            self.is_running = True
            
            logger.info("✅ Real-Time Trading System started successfully!")
            self._print_access_info()
            
            # Keep the system running
            await self._monitor_system()
            
        except Exception as e:
            logger.error(f"❌ Failed to start Real-Time Trading System: {e}")
            await self.stop_system()
            sys.exit(1)
    
    def _print_startup_banner(self):
        """Print startup banner"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                           REAL-TIME TRADING BOT SYSTEM v2.0                                          ║
║                                                                                                      ║
║  🚀 Advanced Algorithmic Trading with Real-Time Streaming Data                                      ║
║  📊 Intelligent Decision Making with Machine Learning                                              ║
║  🔄 Live Market Data Processing and Instant Signal Generation                                      ║
║  💰 Risk Management and Portfolio Optimization                                                     ║
║  📈 Real-Time Performance Monitoring and Analytics                                                 ║
║                                                                                                      ║
║  Features:                                                                                          ║
║  • Real-time streaming data from Schwab                                                            ║
║  • Instant signal generation and execution                                                         ║
║  • Live commentary and analysis                                                                    ║
║  • WebSocket real-time updates                                                                     ║
║  • Advanced risk management                                                                        ║
║  • Machine learning enhanced predictions                                                           ║
║                                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
        """)
    
    async def _check_requirements(self):
        """Check system requirements"""
        logger.info("🔍 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        
        # Check required packages
        required_packages = [
            'fastapi', 'uvicorn', 'websockets', 'aiohttp', 'numpy', 
            'pandas', 'redis', 'streamlit', 'plotly'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"❌ Missing required packages: {missing_packages}")
            logger.info("💡 Install missing packages with: pip install -r requirements_realtime.txt")
            raise RuntimeError(f"Missing packages: {missing_packages}")
        
        logger.info("✅ System requirements check passed")
    
    async def _start_redis(self):
        """Start Redis server if not running"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            logger.info("✅ Redis is already running")
        except:
            logger.info("🔄 Starting Redis server...")
            # Note: In production, you should have Redis running as a service
            # This is just for development
            logger.warning("⚠️  Please ensure Redis is running on localhost:6379")
    
    async def _start_realtime_api(self):
        """Start the real-time API server"""
        logger.info("🌐 Starting Real-Time Trading API...")
        
        try:
            import subprocess
            import sys
            
            # Start the API server
            cmd = [
                sys.executable, "realtime_api_endpoints.py"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.processes.append(("Real-Time API", process))
            
            # Wait a moment for the server to start
            await asyncio.sleep(3)
            
            # Check if the server started successfully
            if process.returncode is None:
                logger.info("✅ Real-Time Trading API started successfully")
            else:
                raise RuntimeError("Failed to start Real-Time Trading API")
                
        except Exception as e:
            logger.error(f"❌ Failed to start Real-Time Trading API: {e}")
            raise
    
    async def _start_dashboard(self):
        """Start the Streamlit dashboard"""
        logger.info("📊 Starting Trading Dashboard...")
        
        try:
            import subprocess
            import sys
            
            # Start the Streamlit dashboard
            cmd = [
                sys.executable, "-m", "streamlit", "run", "trading_dashboard.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.processes.append(("Dashboard", process))
            
            # Wait a moment for the dashboard to start
            await asyncio.sleep(5)
            
            # Check if the dashboard started successfully
            if process.returncode is None:
                logger.info("✅ Trading Dashboard started successfully")
            else:
                raise RuntimeError("Failed to start Trading Dashboard")
                
        except Exception as e:
            logger.error(f"❌ Failed to start Trading Dashboard: {e}")
            raise
    
    def _print_access_info(self):
        """Print access information"""
        print("""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    ACCESS INFORMATION                                                 ║
║                                                                                                      ║
║  🌐 Real-Time Trading API:     http://localhost:8000                                                ║
║  📚 API Documentation:         http://localhost:8000/docs                                           ║
║  📊 Trading Dashboard:         http://localhost:8501                                                ║
║  📱 WebSocket Real-time:       ws://localhost:8000/ws/realtime                                      ║
║  💬 WebSocket Commentary:      ws://localhost:8000/ws/commentary                                    ║
║                                                                                                      ║
║  🔧 API Endpoints:                                                                                  ║
║  • Start Trading:              POST /api/realtime/start                                             ║
║  • Stop Trading:               POST /api/realtime/stop                                              ║
║  • Get Status:                 GET /api/realtime/status                                             ║
║  • Get Positions:              GET /api/realtime/positions                                          ║
║  • Get Signals:                GET /api/realtime/signals                                            ║
║  • Get Performance:            GET /api/realtime/performance                                        ║
║  • Get Dashboard:              GET /api/realtime/dashboard                                          ║
║                                                                                                      ║
║  ⚠️  IMPORTANT: Before starting live trading, ensure:                                               ║
║  • Schwab API credentials are properly configured                                                   ║
║  • Trading mode is set to simulation first                                                          ║
║  • Risk management parameters are reviewed                                                          ║
║                                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
        """)
    
    async def _monitor_system(self):
        """Monitor the running system"""
        logger.info("👀 Monitoring system... Press Ctrl+C to stop")
        
        try:
            while self.is_running:
                # Check if all processes are still running
                for name, process in self.processes:
                    if process.returncode is not None:
                        logger.error(f"❌ {name} process has stopped unexpectedly")
                        await self.stop_system()
                        return
                
                # Print status every 30 seconds
                await asyncio.sleep(30)
                logger.info("✅ System is running normally...")
                
        except KeyboardInterrupt:
            logger.info("🛑 Received stop signal...")
            await self.stop_system()
    
    async def stop_system(self):
        """Stop the real-time trading system"""
        logger.info("🛑 Stopping Real-Time Trading System...")
        
        self.is_running = False
        
        # Stop all processes
        for name, process in self.processes:
            try:
                logger.info(f"🛑 Stopping {name}...")
                process.terminate()
                await process.wait()
                logger.info(f"✅ {name} stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
        
        self.processes.clear()
        logger.info("✅ Real-Time Trading System stopped")

async def main():
    """Main function"""
    launcher = RealTimeTradingLauncher()
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(launcher.stop_system())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await launcher.start_system()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 