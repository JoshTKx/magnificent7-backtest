
"""
RSI-based backtesting engine for the Magnificent 7 stocks.

This module provides a comprehensive backtesting framework that implements
RSI momentum strategy on the Magnificent 7 technology stocks over extended
time periods with detailed performance analysis.
"""

from datetime import date
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

try:
    from .constants import TradingConstants, ValidationConstants
    from .data_loader import DataLoader
    from .portfolio import Portfolio
    from .indicator import TechnicalIndicator
except ImportError:
    from src.constants import TradingConstants, ValidationConstants
    from src.data_loader import DataLoader
    from src.portfolio import Portfolio
    from src.indicator import TechnicalIndicator


class BacktestEngine:
    """
    RSI-based backtesting engine for the Magnificent 7 stocks.
    
    This class implements a complete backtesting framework with RSI signal generation,
    portfolio management, and performance evaluation capabilities for the Magnificent 7
    technology stocks over extended time periods.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format. Defaults to '1981-01-01'.
        end_date (str): End date in 'YYYY-MM-DD' format. Defaults to '2023-12-31'.
        initial_cash (int): Starting capital in USD. Defaults to $1,000,000.
        commission (float): Commission rate per trade (e.g., 0.001 = 0.1%). Defaults to 0.1%.
        slippage (float): Slippage rate per trade (e.g., 0.0002 = 0.02%). Defaults to 0.02%.
        min_transaction (int): Minimum transaction size in shares. Defaults to 10.
        
    Attributes:
        portfolio (Portfolio): Portfolio management instance
        historical_data (Dict[str, pd.DataFrame]): Raw historical price data
        historical_data_with_signals (Dict[str, pd.DataFrame]): Price data with RSI signals
        sp500_data (pd.DataFrame): S&P 500 benchmark data
    """
    
    def __init__(self, 
                 start_date: str = TradingConstants.DEFAULT_START_DATE, 
                 end_date: str = TradingConstants.DEFAULT_END_DATE, 
                 initial_cash: int = TradingConstants.DEFAULT_INITIAL_CASH, 
                 commission: float = TradingConstants.DEFAULT_COMMISSION, 
                 slippage: float = TradingConstants.DEFAULT_SLIPPAGE, 
                 min_transaction: int = TradingConstants.MIN_TRANSACTION_SIZE) -> None:
        """Initialize the backtesting engine with configuration parameters."""
        self.portfolio = Portfolio(initial_cash, commission, slippage, min_transaction)
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.min_transaction = min_transaction
        self.current_cash = initial_cash
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.historical_data_with_signals: Dict[str, pd.DataFrame] = {}
        self.sp500_data: Optional[pd.DataFrame] = None

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load historical stock data and generate trading signals.
        
        Fetches historical price data for all Magnificent 7 stocks within the
        specified date range and generates RSI-based trading signals.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping stock symbols to DataFrames
                containing historical prices and RSI trading signals.
                
        Raises:
            ValueError: If no historical data is available for the specified period.
        """
        # Load historical data from the specified data source
        data_loader = DataLoader()
        self.historical_data = data_loader.load_all_stocks(start=self.start_date, end=self.end_date)
        print(f"Loaded historical data for {len(self.historical_data)} stocks.")
        
        self.historical_data_with_signals = TechnicalIndicator.generate_signals(self.historical_data)
        print("Generated trading signals based on RSI.")
        return self.historical_data_with_signals
        print(f"Loaded historical data for {len(self.historical_data)} stocks.")
        
        self.historical_data_with_signals = TechnicalIndicator.generate_signals(self.historical_data)
        print("Generated trading signals based on RSI.")
        return self.historical_data_with_signals
    
    def extract_daily_prices(self, date: str, price_type: str = 'Close') -> Dict[str, float]:
        """
        Extract daily prices for all stocks on a given date.
        
        Args:
            date (str): Date to extract prices for
            price_type (str): Type of price to extract ('Open', 'High', 'Low', 'Close')
            
        Returns:
            Dict[str, float]: Dictionary mapping stock symbols to their prices
        """
        daily_prices = {}
        for symbol, df in self.historical_data_with_signals.items():
            if date in df.index:
                daily_prices[symbol] = df.loc[date, price_type]
        return daily_prices

    def extract_daily_signals(self, date: str) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Extract daily trading signals and RSI values for all stocks.
        
        Args:
            date (str): Date to extract signals for
            
        Returns:
            Tuple containing:
                - Dict[str, int]: Trading signals (1=buy, -1=sell, 0=hold)
                - Dict[str, float]: RSI values for each stock
        """
        daily_signals = {}
        daily_rsi = {}
        for symbol, df in self.historical_data_with_signals.items():
            if date in df.index:
                daily_signals[symbol] = df.loc[date, 'Signal']
                daily_rsi[symbol] = df.loc[date, 'RSI']
        return daily_signals, daily_rsi

    def run_backtest(self) -> None:
        stock_data_dict = self.load_data()
        if not stock_data_dict:
            print("No historical data available for backtesting.")
            return None

        master_calendar = stock_data_dict[next(iter(stock_data_dict))].index
        self.trading_days = len(master_calendar)

        for i, current_date in enumerate(master_calendar):
            if i + 1 < len(master_calendar):
                next_date = master_calendar[i + 1]
            else:
                break
            
            daily_closing_prices = self.extract_daily_prices(current_date, price_type='Close')
            daily_signals, daily_rsi = self.extract_daily_signals(current_date)
            

            has_trades = self.portfolio.generate_pending_trades(daily_closing_prices, daily_signals, daily_rsi, current_date, stock_data_dict)

            
            daily_opening_prices = self.extract_daily_prices(next_date, price_type='Open')
            self.portfolio.execute_pending_trades(daily_opening_prices, next_date)





    def evaluate_performance(self):
        metrics = self.portfolio.evaluate_performance()

        print(f"Backtest completed from {self.start_date} to {self.end_date}.")
        print(f"Initial Portfolio Value: ${metrics['Initial Value']:.2f}")
        print(f"Final Portfolio Value: ${metrics['Final Value']:.2f}")
        print(f"Total Return: {metrics['Total Return']*100:.2f}%")
        print(f"Annualized Return: {metrics['Annualized Return']*100:.2f}%")
        print(f"Annualized Volatility: {metrics['Annualized Volatility']*100:.2f}%")
        print(f"Maximum Drawdown: {metrics['Maximum Drawdown']*100:.2f}%")
        print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        print(f"Total Trades Executed: {metrics['Total Trades']}")
        print(f"Average Return per Trade: {metrics['Avg Return per Trade']*100:.2f}%")
        print(f"Win Rate: {metrics['Win Rate']*100:.2f}%")
        return metrics
    
    def best_month(self):
        best_month_returns, best_month, best_return, monthly_returns = self.portfolio.analyse_monthly_returns()
        print(f"Best Month: {best_month.strftime('%Y-%m')} with Return: {best_return * 100:.2f}%")
        print(f"Apple: {best_month_returns.get('AAPL', 'N/A')}")
        print(f"Microsoft: {best_month_returns.get('MSFT', 'N/A')}")
        print(f"Google: {best_month_returns.get('GOOG', 'N/A')}")
        print(f"Amazon: {best_month_returns.get('AMZN', 'N/A')}")
        print(f"Tesla: {best_month_returns.get('TSLA', 'N/A')}")
        print(f"Meta: {best_month_returns.get('META', 'N/A')}")
        print(f"NVIDIA: {best_month_returns.get('NVDA', 'N/A')}")

        return best_month_returns, best_month, best_return, monthly_returns
    

    def load_benchmark(self):
        data_loader = DataLoader()
        benchmark_data = data_loader.fetch_benchmark(start=self.start_date, end=self.end_date)
        if benchmark_data is None or benchmark_data.empty:
            print("No benchmark data available.")
            return None
        benchmark_data = benchmark_data[['Close']].rename(columns={'Close': 'Benchmark Close'})
        return benchmark_data

    def calculate_benchmark_performance(self):
        benchmark_data = self.load_benchmark()
        if benchmark_data is None:
            return None

        initial_value = self.initial_cash
        initial_position = initial_value / benchmark_data['Benchmark Close'].iloc[0]
        benchmark_data['Portfolio Value'] = benchmark_data['Benchmark Close'] * initial_position
        benchmark_data['Daily Return'] = benchmark_data['Portfolio Value'].pct_change().fillna(0)
        final_value = benchmark_data['Portfolio Value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        num_years = (benchmark_data.index[-1] - benchmark_data.index[0]).days / 252
        annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
        annualized_volatility = benchmark_data['Daily Return'].std() * (252 ** 0.5)

        benchmark_data['Cumulative Return'] = (1 + benchmark_data['Daily Return']).cumprod() - 1
        rolling_max = benchmark_data['Portfolio Value'].cummax()
        drawdown = (benchmark_data['Portfolio Value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else 0
        print(f"Benchmark Performance from {self.start_date} to {self.end_date}:")
        print(f"Initial Value: ${initial_value:.2f}")
        print(f"Final Value: ${final_value:.2f}")
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Annualized Return: {annualized_return*100:.2f}%")
        print(f"Annualized Volatility: {annualized_volatility*100:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")  

        return benchmark_data, {
            'Initial Value': initial_value,
            'Final Value': final_value,
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Maximum Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio
        }

if __name__ == "__main__":
    backtest = BacktestEngine(start_date="1981-01-01", end_date="2023-12-31", initial_cash=1000000)
    backtest.run_backtest()
    performance_metrics = backtest.evaluate_performance()
    best_month_returns, best_month_date, best_month_return, monthly_returns = backtest.best_month()
    benchmark_data, benchmark_metrics = backtest.calculate_benchmark_performance()
