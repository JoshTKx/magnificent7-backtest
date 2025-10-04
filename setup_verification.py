#!/usr/bin/env python3
"""
Setup verification script for Magnificent 7 Backtesting System.

This script verifies that all dependencies are installed and the system
is ready to run. Execute this before running the main analysis.
"""

import sys
import importlib
from typing import List, Tuple

def check_dependencies() -> List[Tuple[str, bool, str]]:
    """Check if all required dependencies are available."""
    dependencies = [
        ('pandas', 'Data manipulation and analysis'),
        ('numpy', 'Numerical computations'),
        ('yfinance', 'Yahoo Finance data fetching'),
        ('matplotlib', 'Plotting and visualization'),
        ('seaborn', 'Statistical visualization'),
        ('scipy', 'Scientific computing'),
        ('IPython', 'Interactive Python for notebooks')
    ]
    
    results = []
    for dep, description in dependencies:
        try:
            importlib.import_module(dep)
            results.append((dep, True, description))
        except ImportError:
            results.append((dep, False, description))
    
    return results

def check_source_modules() -> List[Tuple[str, bool, str]]:
    """Check if all source modules can be imported."""
    import os
    import sys
    
    # Add current directory and src to path for testing
    current_dir = os.getcwd()
    src_dir = os.path.join(current_dir, 'src')
    
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    modules = [
        ('src.constants', 'Configuration constants'),
        ('src.data_loader', 'Yahoo Finance data fetching'),
        ('src.indicator', 'RSI technical indicator'),
        ('src.portfolio', 'Portfolio management'),
        ('src.backtest', 'Main backtesting engine'),
        ('src.sensitivity_analysis', 'Robustness testing')
    ]
    
    results = []
    for module, description in modules:
        try:
            importlib.import_module(module)
            results.append((module.split('.')[-1], True, description))
        except ImportError as e:
            results.append((module.split('.')[-1], False, f"{description} - Error: {str(e)[:50]}..."))
    
    return results

def main():
    """Run complete system verification."""
    print("MAGNIFICENT 7 BACKTESTING SYSTEM VERIFICATION")
    print("=" * 60)
    
    # Check dependencies
    print("\nCHECKING DEPENDENCIES:")
    print("-" * 30)
    dep_results = check_dependencies()
    all_deps_ok = True
    
    for dep, status, desc in dep_results:
        status_icon = "[OK]" if status else "[FAIL]"
        print(f"{status_icon} {dep:<12} - {desc}")
        if not status:
            all_deps_ok = False
    
    # Check source modules
    print("\nCHECKING SOURCE MODULES:")
    print("-" * 30)
    module_results = check_source_modules()
    all_modules_ok = True
    
    for module, status, desc in module_results:
        status_icon = "[OK]" if status else "[FAIL]"
        print(f"{status_icon} {module:<15} - {desc}")
        if not status:
            all_modules_ok = False
    
    # Overall status
    print("\n" + "=" * 60)
    if all_deps_ok and all_modules_ok:
        print("SYSTEM READY! All checks passed.")
        print("\nNext steps:")
        print("1. Run: jupyter notebook notebooks/mag7_results.ipynb")
        print("2. Or test the system: python -c 'from src.backtest import BacktestEngine; print(\"Ready to run backtests!\")'")
        return True
    else:
        print("SETUP ISSUES DETECTED!")
        if not all_deps_ok:
            print("- Install missing dependencies: pip install -r requirements.txt")
        if not all_modules_ok:
            print("- Check source module imports and file structure")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)