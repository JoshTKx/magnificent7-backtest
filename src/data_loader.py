"""
Data loading module for the Magnificent 7 RSI backtesting system.

This module provides functionality to fetch and validate historical stock price data
from Yahoo Finance for the Magnificent 7 stocks and benchmark data.
"""

import yfinance as yf
from functools import reduce
from typing import Dict, Optional, List
import pandas as pd

try:
    from .constants import TradingConstants, ValidationConstants
except ImportError:
    from src.constants import TradingConstants, ValidationConstants


class DataLoader:
    """
    Data loader for fetching and validating historical stock price data.
    
    This class handles fetching historical price data from Yahoo Finance,
    validating data quality, and managing IPO dates for the Magnificent 7 stocks.
    
    Attributes:
        ipo_dates (Dict[str, str]): IPO dates for each stock symbol
    """
    
    def __init__(self) -> None:
        """Initialize the DataLoader with Magnificent 7 IPO dates."""
        self.ipo_dates = TradingConstants.TARGET_STOCKS.copy()

    def fetch_benchmark(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch S&P 500 benchmark data for the specified period.
        
        Args:
            start (str): Start date in 'YYYY-MM-DD' format
            end (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: S&P 500 historical price data
        """
        spy = yf.Ticker(TradingConstants.BENCHMARK_SYMBOL)
        data = spy.history(start=start, end=end, auto_adjust=True)
        return data

    def fetch_single_stock(self, 
                          symbol: str, 
                          interval: str = TradingConstants.DEFAULT_INTERVAL, 
                          start: str = TradingConstants.DEFAULT_START_DATE, 
                          end: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data for a single stock symbol.
        
        Args:
            symbol (str): Stock ticker symbol (must be in Magnificent 7)
            interval (str): Data interval (default: '1d')
            start (str): Start date in 'YYYY-MM-DD' format
            end (Optional[str]): End date in 'YYYY-MM-DD' format
            
        Returns:
            Optional[pd.DataFrame]: Validated historical price data or None if fetch fails
            
        Raises:
            ValueError: If symbol not supported or invalid date range
        """
        if symbol not in self.ipo_dates:
            raise ValueError(f"Ticker {symbol} not found in IPO dates.")
        
        if end is not None and start >= end:
            raise ValueError("Start date must be earlier than end date.")
        
        try: 
            ticker = yf.Ticker(symbol)
            ipo_date = self.ipo_dates.get(symbol)
            start_date = max(start, ipo_date)
            data = ticker.history(start=start_date, end=end, interval=interval, auto_adjust=True)
            print(f"Fetched {len(data)} rows for {ticker.ticker} starting from {start_date} to {end if end else 'present'}")
            validated_data = self.validate_data(data, ticker.ticker)
            return validated_data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def validate_data(self, data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if data is None or data.empty:
            print(f"No data available for {symbol}")
            return None
        
        for col in required_columns:
            if col not in data.columns:
                print(f"Missing column: {col}")
                return None
            
        original_length = len(data)
        data = data.dropna(subset=required_columns)
        cleaned_length = len(data)

        if cleaned_length < original_length:
            print(f"Dropped {original_length - cleaned_length} rows with missing values for {symbol}")
        
        if data['Close'].le(0).any():
            print(f"Invalid close prices found for {symbol}")
            data = data[data['Close'] > 0]

        if len(data) < 100:
            print(f"Insufficient data after cleaning for {symbol}")
            return None
        
        return data
        

        

    def load_all_stocks(self, interval = '1d', start = '1981-01-01', end = None):
        raw_data = {}
        for symbol in self.ipo_dates:
            data = self.fetch_single_stock(symbol, interval, start, end)
            if data is not None:
                raw_data[symbol] = data
        
        all_indexes = [df for df in raw_data.values()]
        master_calendar = pd.concat(all_indexes, axis=0).index.unique().sort_values()
        final_data = {}
        for symbol, df in raw_data.items():
            
            reindexed_df = df.reindex(master_calendar)
            final_data[symbol] = reindexed_df
        
        print(f"Loaded data for {len(final_data)} stocks with unified calendar from {start} to {end if end else 'present'}.")
        return final_data
    
    
if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_all_stocks(interval='1d', start='1981-01-01', end='2023-12-31')
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} rows")
        
    
