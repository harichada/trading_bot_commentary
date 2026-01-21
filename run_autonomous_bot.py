#!/usr/bin/env python3
"""
Autonomous Trading Bot Runner
==============================
Complete autonomous operation with:
- Self-healing and recovery
- Real-time monitoring
- Automatic restarts
- Health checks
- Graceful shutdown

Usage:
    python run_autonomous_bot.py --paper     # Paper trading (default)
    python run_autonomous_bot.py --live      # Live trading (requires confirmation)
    python run_autonomous_bot.py --status    # Check status
    python run_autonomous_bot.py --stop      # Stop gracefully
"""

import os
import sys
import json
import time
import signal
import asyncio
import threading
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('autonomous_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AutonomousBot')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class HealthMonitor:
    """Monitors bot health and triggers recovery actions"""

    def __init__(self, health_file: str = "bot_health.json"):
        self.health_file = Path(health_file)
        self._last_heartbeat = datetime.now()
        self._consecutive_failures = 0
        self._max_failures = 5
        self._is_healthy = True

    def heartbeat(self, status: Dict = None):
        """Record a heartbeat with status"""
        self._last_heartbeat = datetime.now()
        self._consecutive_failures = 0
        self._is_healthy = True

        health_data = {
            'last_heartbeat': self._last_heartbeat.isoformat(),
            'status': status or {},
            'consecutive_failures': self._consecutive_failures,
            'is_healthy': self._is_healthy
        }

        with open(self.health_file, 'w') as f:
            json.dump(health_data, f, indent=2)

    def record_failure(self, error: str):
        """Record a failure"""
        self._consecutive_failures += 1
        self._is_healthy = self._consecutive_failures < self._max_failures

        logger.warning(f"Health check failure {self._consecutive_failures}/{self._max_failures}: {error}")

    def is_healthy(self) -> bool:
        """Check if bot is healthy"""
        # Check heartbeat age
        heartbeat_age = (datetime.now() - self._last_heartbeat).total_seconds()
        if heartbeat_age > 300:  # 5 minutes without heartbeat
            return False

        return self._is_healthy and self._consecutive_failures < self._max_failures

    def get_status(self) -> Dict:
        """Get current health status"""
        if self.health_file.exists():
            try:
                with open(self.health_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {'status': 'unknown'}


class ProcessManager:
    """Manages the trading bot process with restart capability"""

    def __init__(self):
        self._process = None
        self._restart_count = 0
        self._max_restarts = 10
        self._restart_delay = 30  # seconds
        self._shutdown_requested = False

    def start(self, mode: str = 'paper') -> bool:
        """Start the trading bot process"""
        if self._process and self._process.poll() is None:
            logger.warning("Bot already running")
            return False

        logger.info(f"Starting trading bot in {mode} mode...")

        env = os.environ.copy()
        env['PAPER_TRADING'] = 'true' if mode == 'paper' else 'false'
        env['TRADING_ENV'] = mode

        try:
            # Run the institutional integration
            self._process = subprocess.Popen(
                [sys.executable, '-u', 'institutional_integration.py',
                 '--paper' if mode == 'paper' else '--live'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            logger.info(f"Bot started with PID {self._process.pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            return False

    def stop(self, timeout: int = 30) -> bool:
        """Stop the trading bot gracefully"""
        self._shutdown_requested = True

        if not self._process:
            return True

        logger.info("Stopping trading bot...")

        # Try graceful shutdown first
        self._process.terminate()

        try:
            self._process.wait(timeout=timeout)
            logger.info("Bot stopped gracefully")
            return True
        except subprocess.TimeoutExpired:
            logger.warning("Graceful shutdown failed, killing process...")
            self._process.kill()
            self._process.wait()
            return True

    def restart(self, mode: str = 'paper') -> bool:
        """Restart the bot with backoff"""
        if self._shutdown_requested:
            return False

        self._restart_count += 1

        if self._restart_count > self._max_restarts:
            logger.error(f"Max restarts ({self._max_restarts}) exceeded")
            return False

        # Exponential backoff
        delay = min(self._restart_delay * (2 ** (self._restart_count - 1)), 300)
        logger.info(f"Restart {self._restart_count}/{self._max_restarts} in {delay}s...")
        time.sleep(delay)

        self.stop(timeout=10)
        return self.start(mode)

    def is_running(self) -> bool:
        """Check if process is running"""
        return self._process and self._process.poll() is None

    def get_output(self) -> Optional[str]:
        """Get process output (non-blocking)"""
        if not self._process or not self._process.stdout:
            return None

        try:
            return self._process.stdout.readline()
        except Exception:
            return None


class AutonomousRunner:
    """
    Main autonomous runner that coordinates everything:
    - Process management
    - Health monitoring
    - Auto-recovery
    - Status reporting
    """

    def __init__(self, mode: str = 'paper'):
        self.mode = mode
        self.health_monitor = HealthMonitor()
        self.process_manager = ProcessManager()
        self._running = False
        self._status_file = Path("bot_status.json")

    def _update_status(self, status: str, details: Dict = None):
        """Update status file for external monitoring"""
        status_data = {
            'status': status,
            'mode': self.mode,
            'timestamp': datetime.now().isoformat(),
            'pid': self.process_manager._process.pid if self.process_manager._process else None,
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds() if hasattr(self, '_start_time') else 0,
            'restart_count': self.process_manager._restart_count,
            'details': details or {}
        }

        with open(self._status_file, 'w') as f:
            json.dump(status_data, f, indent=2)

    def _check_trading_hours(self) -> bool:
        """Check if market is open or near open"""
        now = datetime.now()

        # Check if weekend
        if now.weekday() >= 5:
            return False

        # Market hours (EST) - simplified
        hour = now.hour
        # Extended hours: 4 AM - 8 PM
        return 4 <= hour < 20

    def _monitor_loop(self):
        """Background monitoring loop"""
        check_interval = 60  # seconds

        while self._running:
            try:
                # Check if process is running
                if not self.process_manager.is_running():
                    if not self.process_manager._shutdown_requested:
                        logger.warning("Bot process died unexpectedly")
                        self.health_monitor.record_failure("Process died")

                        if self.health_monitor.is_healthy():
                            self.process_manager.restart(self.mode)
                        else:
                            logger.error("Too many failures, stopping auto-restart")
                            self._running = False
                else:
                    # Healthy heartbeat
                    self.health_monitor.heartbeat({
                        'process_running': True,
                        'restart_count': self.process_manager._restart_count
                    })

                # Read and log process output
                output = self.process_manager.get_output()
                if output:
                    output = output.strip()
                    if output:
                        logger.info(f"[BOT] {output}")

                # Update status
                self._update_status(
                    'running' if self.process_manager.is_running() else 'stopped',
                    self.health_monitor.get_status()
                )

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(check_interval)

    def start(self):
        """Start the autonomous bot"""
        logger.info("=" * 60)
        logger.info("  AUTONOMOUS TRADING BOT")
        logger.info("=" * 60)
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        self._start_time = datetime.now()
        self._running = True

        # Handle signals for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("\nShutdown signal received...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the bot
        if not self.process_manager.start(self.mode):
            logger.error("Failed to start bot")
            return False

        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()

        # Main loop - just keep running
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        self.stop()
        return True

    def stop(self):
        """Stop the autonomous bot"""
        logger.info("Stopping autonomous bot...")
        self._running = False
        self.process_manager.stop()
        self._update_status('stopped')
        logger.info("Autonomous bot stopped")


def check_prerequisites() -> list:
    """Check if all prerequisites are met"""
    errors = []

    # Check for .env or environment variables
    if not os.getenv('SCHWAB_API_KEY') and not Path('.env').exists():
        errors.append("No Schwab API credentials found. Run setup_wizard.py first.")

    # Check for token file
    if not Path('token_1.json').exists():
        errors.append("No token_1.json found. Run setup_wizard.py to authenticate with Schwab.")

    # Check for required files
    required_files = [
        'institutional_integration.py',
        'trading_bot_commentary_updated.py',
    ]
    for f in required_files:
        if not Path(f).exists():
            errors.append(f"Missing required file: {f}")

    return errors


def print_status():
    """Print current bot status"""
    status_file = Path("bot_status.json")
    health_file = Path("bot_health.json")

    print("\n" + "=" * 60)
    print("  TRADING BOT STATUS")
    print("=" * 60)

    if status_file.exists():
        with open(status_file) as f:
            status = json.load(f)

        print(f"\nStatus: {status.get('status', 'unknown').upper()}")
        print(f"Mode: {status.get('mode', 'unknown')}")
        print(f"PID: {status.get('pid', 'N/A')}")
        print(f"Uptime: {status.get('uptime_seconds', 0) / 3600:.1f} hours")
        print(f"Restarts: {status.get('restart_count', 0)}")
        print(f"Last Update: {status.get('timestamp', 'unknown')}")
    else:
        print("\nBot is not running (no status file found)")

    if health_file.exists():
        with open(health_file) as f:
            health = json.load(f)

        print(f"\nHealth: {'HEALTHY' if health.get('is_healthy', False) else 'UNHEALTHY'}")
        print(f"Failures: {health.get('consecutive_failures', 0)}")
        print(f"Last Heartbeat: {health.get('last_heartbeat', 'unknown')}")

    # Check if process is actually running
    try:
        import subprocess
        result = subprocess.run(
            ['pgrep', '-f', 'institutional_integration'],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split('\n') if result.stdout.strip() else []
        if pids and pids[0]:
            print(f"\nRunning processes: {', '.join(pids)}")
        else:
            print("\nNo trading bot processes found")
    except Exception:
        pass

    print("=" * 60 + "\n")


def stop_bot():
    """Stop the running bot"""
    import subprocess

    print("Stopping trading bot...")

    # Try to kill gracefully
    try:
        result = subprocess.run(
            ['pkill', '-TERM', '-f', 'institutional_integration'],
            capture_output=True
        )
        time.sleep(2)

        # Check if still running
        result = subprocess.run(
            ['pgrep', '-f', 'institutional_integration'],
            capture_output=True
        )
        if result.stdout.strip():
            print("Forcing kill...")
            subprocess.run(['pkill', '-9', '-f', 'institutional_integration'])

        print("Bot stopped")
    except Exception as e:
        print(f"Error stopping bot: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Autonomous Trading Bot Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_autonomous_bot.py --paper     Start in paper trading mode
  python run_autonomous_bot.py --live      Start in live trading mode
  python run_autonomous_bot.py --status    Check current status
  python run_autonomous_bot.py --stop      Stop the bot
        """
    )
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode (default)')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--stop', action='store_true', help='Stop the bot')

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.stop:
        stop_bot()
        return

    # Determine mode
    if args.live:
        mode = 'live'
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE")
        print("  Real money will be at risk!")
        print("!" * 60)
        confirm = input("\nType 'LIVE' to confirm: ")
        if confirm != 'LIVE':
            print("Aborted.")
            return
    else:
        mode = 'paper'

    # Check prerequisites
    errors = check_prerequisites()
    if errors:
        print("\n" + "=" * 60)
        print("  SETUP INCOMPLETE")
        print("=" * 60)
        for error in errors:
            print(f"  - {error}")
        print("\nRun: python setup_wizard.py")
        print("=" * 60 + "\n")
        return

    # Start the bot
    runner = AutonomousRunner(mode=mode)
    runner.start()


if __name__ == "__main__":
    main()
