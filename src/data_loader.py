import yfinance as yf



class DataLoader():
    def __init__(self):
        self.ipo_dates ={
            'AAPL': '1980-12-12',  
            'MSFT': '1986-03-13',   
            'GOOG': '2004-08-19', 
            'AMZN': '1997-05-15',
            'TSLA': '2010-06-29',
            'META': '2012-05-18',
            'NVDA': '1999-01-22',
        }

    def fetch_single_stock(self, symbol , interval = '1d', start = '1981-01-01', end = None):
        if symbol not in self.ipo_dates:
            raise ValueError(f"Ticker {symbol} not found in IPO dates.")
        
        if end is not None and start >= end:
            raise ValueError("Start date must be earlier than end date.")
        
        try: 
            ticker = yf.Ticker(symbol)
            ipo_date = self.ipo_dates.get(ticker.ticker)
            start_date = max(start, ipo_date)
            data = ticker.history(start=start_date, end = end, interval=interval, auto_adjust=True)
            print(f"Fetched {len(data)} rows for {ticker.ticker} starting from {start_date} to {end if end else 'present'}")
            validated_data = self.validate_data(data, ticker.ticker)
            return validated_data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    

    def validate_data(self, data, symbol):
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if data is None or data.empty:
            print(f"No data available for {symbol}")
            return None
        
        for col in required_columns:
            if col not in data.columns:
                print(f"Missing column: {col}")
                return None
            
        orginal_length = len(data)
        data = data.dropna(subset=required_columns)
        cleaned_length = len(data)

        if cleaned_length < orginal_length:
            print(f"Dropped {orginal_length - cleaned_length} rows with missing values for {symbol}")
        
        if data['Close'].le(0).any():
            print(f"Invalid close prices found for {symbol}")
            data = data[data['Close'] > 0]

        if len(data) < 100:
            print(f"Insufficient data after cleaning for {symbol}")
            return None
        
        return data
        

        

    def load_all_stocks(self, interval = '1d', start = '1981-01-01', end = None):
        all_data = {}
        for symbol in self.ipo_dates:
            data = self.fetch_single_stock(symbol, interval, start, end)
            if data is not None:
                all_data[symbol] = data
        return all_data
    
    
if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_all_stocks(interval='1d', start='2000-01-01', end='2023-12-31')
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} rows")
        
    
