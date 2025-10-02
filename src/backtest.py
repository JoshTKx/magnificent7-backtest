
from src.data_loader import DataLoader
from src.portfolio import Portfolio
from src.indicator import TechnicalIndicator
import pandas as pd

class BacktestEngine:
    def __init__(self, start_date = "1981-01-01", end_date = "2023-12-31", initial_cash = 1000000, commission = 0.001, slippage = 0.0002, min_transaction = 10):
        self.portfolio = Portfolio(initial_cash, commission, slippage, min_transaction)
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.min_transaction = min_transaction
        self.current_cash = initial_cash
        self.historical_data = {}
        self.historical_data_with_signals = {}

    def load_data(self):
        # Load historical data from the specified data source
        data_loader = DataLoader()
        self.historical_data = data_loader.load_all_stocks(start=self.start_date, end=self.end_date)
        print(f"Loaded historical data for {len(self.historical_data)} stocks.")
        self.historical_data_with_signals = TechnicalIndicator.generate_signals(self.historical_data)
        print("Generated trading signals based on RSI.")
        return self.historical_data_with_signals
    
    def extract_daily_prices(self, date, price_type='Close'):
        daily_prices = {}
        for symbol, df in self.historical_data_with_signals.items():
            if date in df.index:
                daily_prices[symbol] = df.loc[date, price_type]
        return daily_prices

    def extract_daily_signals(self, date):
        daily_signals = {}
        daily_rsi = {}
        for symbol, df in self.historical_data_with_signals.items():
            if date in df.index:
                daily_signals[symbol] = df.loc[date, 'Signal']
                daily_rsi[symbol] = df.loc[date, 'RSI']
        return daily_signals, daily_rsi

    def run_backtest(self):
        stock_data_dict = self.load_data()
        if not stock_data_dict:
            print("No historical data available for backtesting.")
            return None

        master_calendar = stock_data_dict[next(iter(stock_data_dict))].index

        for i, current_date in enumerate(master_calendar):
            if i + 1 < len(master_calendar):
                next_date = master_calendar[i + 1]
            else:
                break
            
            daily_closing_prices = self.extract_daily_prices(current_date, price_type='Close')
            daily_signals, daily_rsi = self.extract_daily_signals(current_date)
            

            has_trades = self.portfolio.generate_pending_trades(daily_closing_prices, daily_signals, daily_rsi, current_date, stock_data_dict)

            if has_trades:
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

if __name__ == "__main__":
    backtest = BacktestEngine(start_date="2023-01-01", end_date="2023-12-31", initial_cash=1000000)
    backtest.run_backtest()
    print("First 3 portfolio values:")
    print(backtest.portfolio.portfolio_value_history[0:3])

    print("\nFirst trade record:")
    print(backtest.portfolio.trades[0])

    print("\nSecond trade record:")
    print(backtest.portfolio.trades[1])

    print(f"Number of portfolio value records: {len(backtest.portfolio.portfolio_value_history)}")
    print(f"First date: {backtest.portfolio.portfolio_value_history[0][1]}")
    print(f"Last date: {backtest.portfolio.portfolio_value_history[-1][1]}")
    print(f"Days between: {(backtest.portfolio.portfolio_value_history[-1][1] - backtest.portfolio.portfolio_value_history[0][1]).days}")
    performance_metrics = backtest.evaluate_performance()