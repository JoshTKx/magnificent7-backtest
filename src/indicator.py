import pandas as pd
import numpy as np

class TechnicalIndicator:
    # Class-level configurable thresholds
    RSI_BUY_THRESHOLD = 35
    RSI_SELL_THRESHOLD = 65

    @staticmethod
    def calculate_rsi(data, window=14):

        new_data = data.copy()
        
        valid_data = new_data.dropna(subset=['Close'])

        if valid_data.empty:
            import warnings
            warnings.warn("No valid data for RSI calculation.", RuntimeWarning)
            new_data['RSI'] = pd.NA 
            return new_data

        price_changes = valid_data['Close'].diff()

        if price_changes.empty:
            new_data['RSI'] = pd.NA
            return new_data
        # Alpha determines the smoothing factor for the exponential moving average in RSI calculation
        alpha = 1/ window
        gains = price_changes.where(price_changes > 0, 0.0)
        losses = -price_changes.where(price_changes < 0, 0.0)
        losses = -price_changes.where(price_changes < 0, 0.0)

        avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Handle division by zero explicitly:
        rsi = rsi.where(avg_loss != 0, 100)
        rsi = rsi.where(avg_gain != 0, 0)
        # Set RSI to pd.NA when both avg_gain and avg_loss are zero
        both_zero = (avg_gain == 0) & (avg_loss == 0)
        rsi = rsi.where(~both_zero, pd.NA)
        # If both are zero, RSI is undefined; set to pd.NA.
        rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100)
        rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0)
        rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), pd.NA)
        if not rsi.index.equals(new_data.index):
            new_data['RSI'] = rsi.reindex(new_data.index, fill_value=pd.NA)
        else:
            new_data['RSI'] = rsi
        return new_data
    
        new_data['RSI'] = rsi.reindex(new_data.index, fill_value=pd.NA)
        return new_data
        
    @staticmethod
    def add_rsi_to_all(stock_data_dict):

        dict_with_rsi = {}
        for symbol, data in stock_data_dict.items():
            data_with_rsi = TechnicalIndicator.calculate_rsi(data)
            dict_with_rsi[symbol] = data_with_rsi
        
        return dict_with_rsi
    
    @staticmethod
    def generate_signals(stock_data_dict, rsi_buy_threshold=None, rsi_sell_threshold=None):
        # Use class-level thresholds if not provided
        if rsi_buy_threshold is None:
            rsi_buy_threshold = TechnicalIndicator.RSI_BUY_THRESHOLD
        if rsi_sell_threshold is None:
            rsi_sell_threshold = TechnicalIndicator.RSI_SELL_THRESHOLD

        dict_with_rsi = TechnicalIndicator.add_rsi_to_all(stock_data_dict)

        final_signal_dict = {}
        for symbol, data in dict_with_rsi.items():
            if 'RSI' not in data.columns:
                final_signal_dict[symbol] = data
                continue

            new_data = data.copy()
            new_data['Signal'] = 0

            new_data.loc[new_data['RSI'] < rsi_buy_threshold, 'Signal'] = 1
            new_data.loc[new_data['RSI'] > rsi_sell_threshold, 'Signal'] = -1

            final_signal_dict[symbol] = new_data

        return final_signal_dict
        return final_signal_dict

