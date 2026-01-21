#!/usr/bin/env python3
"""
System Validation Script
========================
Comprehensive validation of all trading system components.
Run this to verify everything is working correctly.

Usage:
    python validate_system.py           # Run all validations
    python validate_system.py --quick   # Run quick checks only
    python validate_system.py --fix     # Attempt to fix issues
"""

import os
import sys
import json
import importlib
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_pass(text: str):
    print(f"  {Colors.GREEN}[PASS]{Colors.END} {text}")


def print_fail(text: str, error: str = None):
    print(f"  {Colors.RED}[FAIL]{Colors.END} {text}")
    if error:
        print(f"         {Colors.RED}{error}{Colors.END}")


def print_warn(text: str):
    print(f"  {Colors.YELLOW}[WARN]{Colors.END} {text}")


def print_info(text: str):
    print(f"  {Colors.BLUE}[INFO]{Colors.END} {text}")


class ValidationResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []

    def add_pass(self, name: str):
        self.passed += 1
        print_pass(name)

    def add_fail(self, name: str, error: str = None):
        self.failed += 1
        self.errors.append((name, error))
        print_fail(name, error)

    def add_warn(self, name: str):
        self.warnings += 1
        print_warn(name)


def validate_python_version() -> bool:
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True
    return False


def validate_dependencies(result: ValidationResult):
    """Check all required dependencies"""
    print_header("DEPENDENCY VALIDATION")

    required_packages = [
        ('numpy', 'Core numerical operations'),
        ('pandas', 'Data manipulation'),
        ('scipy', 'Statistical functions'),
        ('sklearn', 'Machine learning'),
        ('requests', 'HTTP client'),
        ('fastapi', 'Web framework'),
        ('uvicorn', 'ASGI server'),
        ('websockets', 'WebSocket support'),
        ('yaml', 'Configuration parsing'),
    ]

    optional_packages = [
        ('schwab', 'Schwab API client'),
        ('xgboost', 'XGBoost ML'),
        ('lightgbm', 'LightGBM ML'),
        ('torch', 'PyTorch deep learning'),
        ('ta', 'Technical analysis'),
        ('textblob', 'Sentiment analysis'),
    ]

    # Check required
    for package, description in required_packages:
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn')
            elif package == 'yaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            result.add_pass(f"{package}: {description}")
        except ImportError as e:
            result.add_fail(f"{package}: {description}", str(e))

    # Check optional
    print_info("Optional packages:")
    for package, description in optional_packages:
        try:
            importlib.import_module(package)
            result.add_pass(f"{package}: {description}")
        except ImportError:
            result.add_warn(f"{package}: {description} (optional)")


def validate_files(result: ValidationResult):
    """Check all required files exist"""
    print_header("FILE VALIDATION")

    required_files = [
        ('trading_bot_commentary_updated.py', 'Main trading engine'),
        ('institutional_integration.py', 'Institutional layer'),
        ('profitable_strategies.py', 'Profitable strategies'),
        ('run_autonomous_bot.py', 'Autonomous runner'),
        ('setup_wizard.py', 'Setup wizard'),
        ('requirements.txt', 'Dependencies list'),
    ]

    optional_files = [
        ('institutional_core.py', 'Core institutional components'),
        ('risk_intelligence.py', 'Risk management'),
        ('ml_pipeline.py', 'ML pipeline'),
        ('observability.py', 'Observability'),
        ('infrastructure.py', 'Infrastructure'),
        ('backtesting_engine.py', 'Backtesting'),
        ('strategy_system.py', 'Strategy system'),
    ]

    for filename, description in required_files:
        if Path(filename).exists():
            result.add_pass(f"{filename}: {description}")
        else:
            result.add_fail(f"{filename}: {description}", "File not found")

    print_info("Optional files:")
    for filename, description in optional_files:
        if Path(filename).exists():
            result.add_pass(f"{filename}: {description}")
        else:
            result.add_warn(f"{filename}: {description} (optional)")


def validate_syntax(result: ValidationResult):
    """Validate Python syntax of all modules"""
    print_header("SYNTAX VALIDATION")

    python_files = list(Path('.').glob('*.py'))

    for py_file in python_files:
        try:
            proc = subprocess.run(
                [sys.executable, '-m', 'py_compile', str(py_file)],
                capture_output=True,
                text=True
            )
            if proc.returncode == 0:
                result.add_pass(f"{py_file.name}: Syntax OK")
            else:
                result.add_fail(f"{py_file.name}: Syntax error", proc.stderr.strip())
        except Exception as e:
            result.add_fail(f"{py_file.name}: Validation failed", str(e))


def validate_imports(result: ValidationResult):
    """Validate that key modules can be imported"""
    print_header("IMPORT VALIDATION")

    modules_to_test = [
        ('institutional_integration', 'InstitutionalTradingBot'),
        ('profitable_strategies', 'StrategyEnsemble'),
    ]

    for module_name, class_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                result.add_pass(f"{module_name}.{class_name}: Importable")
            else:
                result.add_fail(f"{module_name}.{class_name}: Class not found")
        except Exception as e:
            result.add_fail(f"{module_name}: Import failed", str(e))


def validate_strategies(result: ValidationResult):
    """Validate trading strategies"""
    print_header("STRATEGY VALIDATION")

    try:
        import numpy as np
        import pandas as pd
        from profitable_strategies import (
            AdaptiveTrendStrategy,
            MeanReversionWithRegimeStrategy,
            OpeningRangeBreakoutStrategy,
            VWAPReversionStrategy,
            StrategyEnsemble,
            create_profitable_strategies
        )

        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=300, freq='5min')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(300) * 0.5)

        test_df = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.rand(300) * 0.5,
            'low': prices - np.random.rand(300) * 0.5,
            'close': prices + np.random.randn(300) * 0.2,
            'volume': np.random.randint(100000, 1000000, 300)
        }, index=dates)
        test_df.attrs['symbol'] = 'TEST'

        # Test each strategy
        strategies = [
            ('AdaptiveTrendStrategy', AdaptiveTrendStrategy()),
            ('MeanReversionWithRegimeStrategy', MeanReversionWithRegimeStrategy()),
            ('VWAPReversionStrategy', VWAPReversionStrategy()),
        ]

        for name, strategy in strategies:
            try:
                signals = strategy.analyze(test_df, {})
                result.add_pass(f"{name}: Generated {len(signals)} signal(s)")
            except Exception as e:
                result.add_fail(f"{name}: Analysis failed", str(e))

        # Test ensemble
        try:
            ensemble = create_profitable_strategies()
            signals = ensemble.analyze(test_df, {})
            result.add_pass(f"StrategyEnsemble: Generated {len(signals)} consensus signal(s)")
        except Exception as e:
            result.add_fail("StrategyEnsemble: Analysis failed", str(e))

    except ImportError as e:
        result.add_fail("Strategy module import failed", str(e))
    except Exception as e:
        result.add_fail("Strategy validation failed", str(e))


def validate_institutional_components(result: ValidationResult):
    """Validate institutional components"""
    print_header("INSTITUTIONAL COMPONENTS VALIDATION")

    try:
        from institutional_integration import (
            TradingState,
            STATE_TRANSITIONS,
            AtomicStateManager,
            IdempotentOrderManager,
            PositionReconciler,
            RiskEngine,
            AuditTrail,
        )

        # Test state machine
        result.add_pass(f"TradingState: {len(TradingState)} states defined")
        result.add_pass(f"STATE_TRANSITIONS: {len(STATE_TRANSITIONS)} transitions defined")

        # Test state manager
        try:
            sm = AtomicStateManager(
                state_file="/tmp/test_state.json",
                wal_file="/tmp/test_wal.json"
            )
            sm.update({'test_key': 'test_value'})
            assert sm.get('test_key') == 'test_value'
            result.add_pass("AtomicStateManager: WAL persistence working")
            # Cleanup
            Path("/tmp/test_state.json").unlink(missing_ok=True)
            Path("/tmp/test_wal.json").unlink(missing_ok=True)
        except Exception as e:
            result.add_fail("AtomicStateManager: Test failed", str(e))

        # Test idempotency
        try:
            im = IdempotentOrderManager(persistence_file="/tmp/test_idempotency.json")
            key = im.generate_key("AAPL", "BUY", 100, "test", datetime.now())
            assert im.check_and_mark(key) == True
            assert im.check_and_mark(key) == False  # Duplicate
            result.add_pass("IdempotentOrderManager: Deduplication working")
            Path("/tmp/test_idempotency.json").unlink(missing_ok=True)
        except Exception as e:
            result.add_fail("IdempotentOrderManager: Test failed", str(e))

        # Test position reconciler
        try:
            pr = PositionReconciler()
            internal = {'AAPL': {'quantity': 100}}
            broker = {'AAPL': {'quantity': 100}}
            report = pr.reconcile(internal, broker)
            assert report['status'] == 'OK'
            result.add_pass("PositionReconciler: Reconciliation working")
        except Exception as e:
            result.add_fail("PositionReconciler: Test failed", str(e))

        # Test risk engine
        try:
            re = RiskEngine()
            snapshot = re.calculate_snapshot({}, 100000)
            assert snapshot.total_exposure == 0
            result.add_pass("RiskEngine: Risk calculation working")
        except Exception as e:
            result.add_fail("RiskEngine: Test failed", str(e))

        # Test audit trail
        try:
            at = AuditTrail(file_path="/tmp/test_audit.jsonl")
            at.record("TEST_EVENT", {"test": "data"})
            is_valid, msg = at.verify_integrity()
            assert is_valid
            result.add_pass("AuditTrail: Tamper-evident logging working")
            Path("/tmp/test_audit.jsonl").unlink(missing_ok=True)
        except Exception as e:
            result.add_fail("AuditTrail: Test failed", str(e))

    except ImportError as e:
        result.add_fail("Institutional module import failed", str(e))
    except Exception as e:
        result.add_fail("Institutional validation failed", str(e))


def validate_configuration(result: ValidationResult):
    """Validate configuration files"""
    print_header("CONFIGURATION VALIDATION")

    # Check .env
    if Path('.env').exists():
        result.add_pass(".env: Configuration file exists")

        # Check for required variables
        with open('.env') as f:
            content = f.read()

        required_vars = ['SCHWAB_API_KEY', 'SCHWAB_APP_SECRET']
        for var in required_vars:
            if var in content and f'{var}=""' not in content:
                result.add_pass(f".env: {var} is set")
            else:
                result.add_warn(f".env: {var} not set (required for live trading)")
    else:
        result.add_warn(".env: Not found (run setup_wizard.py)")

    # Check token
    if Path('token_1.json').exists():
        try:
            with open('token_1.json') as f:
                token = json.load(f)
            if 'access_token' in token or 'refresh_token' in token:
                result.add_pass("token_1.json: Valid token structure")
            else:
                result.add_warn("token_1.json: May be invalid")
        except Exception as e:
            result.add_fail("token_1.json: Invalid JSON", str(e))
    else:
        result.add_warn("token_1.json: Not found (required for Schwab API)")

    # Check config directory
    config_dir = Path('config')
    if config_dir.exists():
        for config_file in config_dir.glob('*.yaml'):
            result.add_pass(f"config/{config_file.name}: Found")
    else:
        result.add_warn("config/: Directory not found")


def validate_api_connection(result: ValidationResult):
    """Validate API connections"""
    print_header("API CONNECTION VALIDATION")

    # Check if we can connect to Schwab (if credentials exist)
    if not Path('token_1.json').exists():
        result.add_warn("Schwab API: Cannot test (no token)")
        return

    try:
        from schwab import auth
        import os

        api_key = os.getenv('SCHWAB_API_KEY')
        app_secret = os.getenv('SCHWAB_APP_SECRET')

        if not api_key or not app_secret:
            result.add_warn("Schwab API: Credentials not in environment")
            return

        client = auth.client_from_token_file(
            'token_1.json',
            api_key,
            app_secret
        )

        # Try to get account numbers
        response = client.get_account_numbers()
        if response.status_code == 200:
            accounts = response.json()
            result.add_pass(f"Schwab API: Connected ({len(accounts)} account(s))")
        elif response.status_code == 401:
            result.add_fail("Schwab API: Authentication failed (token expired?)")
        else:
            result.add_fail(f"Schwab API: Error {response.status_code}")

    except ImportError:
        result.add_warn("Schwab API: schwab-py not installed")
    except Exception as e:
        result.add_fail("Schwab API: Connection failed", str(e))


def run_validation(quick: bool = False) -> ValidationResult:
    """Run all validations"""
    result = ValidationResult()

    print_header("TRADING SYSTEM VALIDATION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")

    # Always run these
    validate_files(result)
    validate_syntax(result)

    if not quick:
        validate_dependencies(result)
        validate_imports(result)
        validate_strategies(result)
        validate_institutional_components(result)
        validate_configuration(result)
        validate_api_connection(result)

    # Summary
    print_header("VALIDATION SUMMARY")

    total = result.passed + result.failed
    print(f"  {Colors.GREEN}Passed:{Colors.END} {result.passed}/{total}")
    print(f"  {Colors.RED}Failed:{Colors.END} {result.failed}/{total}")
    print(f"  {Colors.YELLOW}Warnings:{Colors.END} {result.warnings}")

    if result.failed > 0:
        print(f"\n{Colors.RED}SYSTEM NOT READY{Colors.END}")
        print("\nFailed checks:")
        for name, error in result.errors:
            print(f"  - {name}")
            if error:
                print(f"    {error}")
    else:
        print(f"\n{Colors.GREEN}SYSTEM READY{Colors.END}")
        if result.warnings > 0:
            print(f"  (with {result.warnings} warning(s))")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate trading system')
    parser.add_argument('--quick', action='store_true', help='Quick validation only')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues')
    args = parser.parse_args()

    result = run_validation(quick=args.quick)

    # Return appropriate exit code
    sys.exit(0 if result.failed == 0 else 1)


if __name__ == "__main__":
    main()
