class TechnicalIndicator:

    @staticmethod
    def calculate_rsi(data, window=14):
        if data['Close'].isnull().any():
            print("Data contains null values in 'Close' column. Cannot compute RSI.")
            return data

        new_data = data.copy()

        price_changes = new_data['Close'].diff()

        alpha = 1/ window
        gains = price_changes.where(price_changes > 0, 0.0)
        losses = -price_changes.where(price_changes < 0, 0.0)

        avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        rsi = rsi.where(avg_loss != 0, 100)
        rsi = rsi.where(avg_gain != 0, 0)

        new_data['RSI'] = rsi
        return new_data
        
    @staticmethod
    def add_rsi_to_all( stock_data_dict):

        dict_with_rsi = {}
        for symbol, data in stock_data_dict.items():
            data_with_rsi = TechnicalIndicator.calculate_rsi(data)
            dict_with_rsi[symbol] = data_with_rsi
        
        return dict_with_rsi
    
    @staticmethod
    def generate_signals(data, rsi_buy_threshold=35, rsi_sell_threshold=65):
        if 'RSI' not in data.columns:
            print("RSI column not found in data. Cannot generate signals.")
            return data

        new_data = data.copy()
        new_data['Signal'] = 0

        new_data.loc[new_data['RSI'] < rsi_buy_threshold, 'Signal'] = 1
        new_data.loc[new_data['RSI'] > rsi_sell_threshold, 'Signal'] = -1

        return new_data

