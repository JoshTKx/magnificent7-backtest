"""
Configuration constants for the Magnificent 7 RSI backtesting system.

This module contains all configurable parameters and constants used throughout
the backtesting framework to ensure consistency and easy maintenance.
"""

from typing import Dict


class TradingConstants:
    """Trading strategy configuration constants."""
    
    # RSI Technical Indicator Configuration
    RSI_WINDOW: int = 14
    RSI_BUY_THRESHOLD: int = 35
    RSI_SELL_THRESHOLD: int = 65
    
    # Portfolio Configuration
    DEFAULT_INITIAL_CASH: int = 1_000_000
    DEFAULT_COMMISSION: float = 0.001  # 0.1%
    DEFAULT_SLIPPAGE: float = 0.0002   # 0.02%
    MIN_TRANSACTION_SIZE: int = 10
    
    # Portfolio Rebalancing
    REBALANCE_TOLERANCE: float = 0.02  # 2% tolerance before rebalancing
    
    # Data Configuration
    DEFAULT_START_DATE: str = "1981-01-01"
    DEFAULT_END_DATE: str = "2023-12-31"
    DEFAULT_INTERVAL: str = "1d"
    
    # Magnificent 7 Stocks with IPO Dates
    TARGET_STOCKS: Dict[str, str] = {
        'AAPL': '1980-12-12',  # Apple Inc.
        'MSFT': '1986-03-13',  # Microsoft Corporation
        'GOOG': '2004-08-19',  # Alphabet Inc. (Google)
        'AMZN': '1997-05-15',  # Amazon.com Inc.
        'TSLA': '2010-06-29',  # Tesla Inc.
        'META': '2012-05-18',  # Meta Platforms Inc. (Facebook)
        'NVDA': '1999-01-22',  # NVIDIA Corporation
    }
    
    # Benchmark Configuration
    BENCHMARK_SYMBOL: str = '^GSPC'  # S&P 500 Index
    
    # Data Validation
    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # File Output Configuration
    RESULTS_DIR: str = "results"
    CHARTS_SUBDIR: str = "charts"
    METRICS_SUBDIR: str = "metrics" 
    DATA_SUBDIR: str = "data"
    
    # Chart Configuration
    DEFAULT_DPI: int = 300
    DEFAULT_FIGSIZE: tuple = (20, 12)
    DEFAULT_FACECOLOR: str = 'white'


class ValidationConstants:
    """Constants for data validation and error handling."""
    
    MIN_DATA_POINTS: int = 30  # Minimum data points required for analysis
    MAX_MISSING_DATA_RATIO: float = 0.1  # Maximum 10% missing data allowed
    MIN_PRICE_VALUE: float = 0.01  # Minimum valid price value
    MAX_PRICE_CHANGE_RATIO: float = 10.0  # Maximum 1000% daily change allowed


class DisplayConstants:
    """Constants for output formatting and display."""
    
    DECIMAL_PLACES: int = 2
    PERCENTAGE_DECIMAL_PLACES: int = 1
    CURRENCY_FORMAT: str = "${:,.2f}"
    PERCENTAGE_FORMAT: str = "{:.1f}%"
    
    # Output separators
    SECTION_SEPARATOR: str = "=" * 60
    SUB_SECTION_SEPARATOR: str = "-" * 40
    
