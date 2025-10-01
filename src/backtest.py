from src.data_loader import DataLoader
from src.portfolio import Portfolio
from src.indicator import TechnicalIndicator

class BacktestEngine:
    def __init__(self, portfolio, start_date = "1981-01-01", end_date = "2023-12-31", initial_cash = 1000000, commission = 0.001, slippage = 0.0002, min_transaction = 10):
        self.portfolio = portfolio
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.min_transaction = min_transaction
        self.current_cash = initial_cash

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
        pass



    def evaluate_performance(self):
        pass