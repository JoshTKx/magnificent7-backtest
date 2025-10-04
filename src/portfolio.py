import heapq
import pandas as pd
from datetime import date, timedelta
import math

class Portfolio:
    def __init__(self, initial_cash=1000000, commission=0.001, slippage = 0.0002, min_shares = 10):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {} # symbol -> number of shares
        self.share_cost = {} # symbol -> total cost of shares held

        self.commission = commission
        self.slippage = slippage
        self.min_shares = min_shares

        self.target_stocks = {
            'AAPL': '1980-12-12',  
            'MSFT': '1986-03-13',   
            'GOOG': '2004-08-19', 
            'AMZN': '1997-05-15',
            'TSLA': '2010-06-29',
            'META': '2012-05-18',
            'NVDA': '1999-01-22',
        }
        

        self.pending_trade = {} # symbol -> shares to trade

        self.trades = [] # list of trade records
        self.portfolio_value_history = [] # list of (total portfolio value, date)
        self.positions_value_history = [] # list of (positions value dict, date)

    def get_investable_stocks(self, date, price_data_dict):
        investable = []
        
        for symbol in self.target_stocks:

            if symbol not in price_data_dict:
                continue
            
            if date in price_data_dict[symbol].index:
                close_price = price_data_dict[symbol].loc[date, 'Close']
                if pd.notna(close_price) and close_price > 0:
                    investable.append(symbol)  
                    
        return investable
    
    def calculate_target_weights(self, investable=None):
        
        num_investable = len(investable)
        if num_investable == 0:
            return None
        return 1.0 / num_investable
        


    def calculate_portfolio_value(self, current_prices):
        total_value = self.cash
        for symbol, shares in self.positions.items():
            price = current_prices.get(symbol, 0)
            if price is not None and pd.notna(price) and price > 0 and shares > 0:
                total_value += shares * price
        return total_value

    def calculate_current_weights(self, current_prices):
        total_value = self.calculate_portfolio_value(current_prices)
        weights = {}
        for symbol in self.target_stocks:
            shares = self.positions.get(symbol, 0)
            price = current_prices.get(symbol, 0)
            position_value = shares * price
            weights[symbol] = position_value / total_value if total_value > 0 else 0
        return weights
    
    def determine_action(self, current_weight, rsi_signal, target_weight):
        """
        - At Target: Hold
        - Underweight + Buy: BUY to target 
        - Underweight + No Signal: Hold
        - Underweight + Sell: Hold 
        - Overweight + Sell: SELL to target
        - Overweight + No Signal: SELL to target 
        - Overweight + Buy: Hold 
        """
        
        tolerance = 0.02  # 2% tolerance band around target weight

        if abs(current_weight - target_weight) <= tolerance:
            return 0  # At Target: Hold
        elif current_weight < target_weight - tolerance:
            if rsi_signal == -1:
                return 0  # Underweight + Sell: Hold
            else:
                return 1  # Underweight + No Signal or Buy: BUY to target
        elif current_weight > target_weight + tolerance:
            if rsi_signal == 1:
                return 0  # Overweight + Buy: Hold
            else:
                return -1  # Overweight + No Signal / Sell : SELL to target
        return 0  # Default to Hold
    
    def calculate_trade_priority(self, current_weights, rsi_signals, rsi_values, target_weight, investable_stocks=None):
        priority_queue = []
        for symbol in investable_stocks:
            current_weight = current_weights.get(symbol, 0)
            rsi_signal = rsi_signals.get(symbol, 0)
            rsi_value = rsi_values.get(symbol, 50)  # Default RSI to 50 if not available
            action = self.determine_action(current_weight, rsi_signal, target_weight)

            if action == 0:
                continue

            rsi_urgency = 0 

            if action == 1 and rsi_value < 50:  # Buy
                rsi_urgency = (50 - rsi_value)/ 50
            elif action == -1 and rsi_value >= 50:  # Sell
                rsi_urgency = (rsi_value - 50) / 50 

            weight_diff = abs(current_weight - target_weight)
            priority = 0.6 * weight_diff + 0.4 * rsi_urgency

            heapq.heappush(priority_queue, (action, -priority, symbol, weight_diff))

        return priority_queue
    
    
    def generate_pending_trades(self, closing_prices, rsi_signals, rsi_values, date, price_data_dict):

        investable_stocks = self.get_investable_stocks(date, price_data_dict)
        target_weight = self.calculate_target_weights(investable_stocks)
        if not investable_stocks or target_weight == 0:
            self.pending_trade = {}
            return False
        current_weights = self.calculate_current_weights(closing_prices)
        total_portfolio_value = self.calculate_portfolio_value(closing_prices)

        trade_priority_queue = self.calculate_trade_priority(current_weights, rsi_signals, rsi_values, target_weight, investable_stocks)
        total_portfolio_value = self.calculate_portfolio_value(closing_prices)

        trade_priority_queue = self.calculate_trade_priority(current_weights, rsi_signals, rsi_values, target_weight, investable_stocks)

        pending_values_to_trade = {}
        avail_cash = self.cash

        while trade_priority_queue:

            action, _, symbol, weight_diff = heapq.heappop(trade_priority_queue)
            value_to_trade = weight_diff * total_portfolio_value * action # Positive for buy, negative for sell

            
            trade_cost = abs(value_to_trade) * (self.commission + self.slippage)
            if action == 1 and abs(value_to_trade) + trade_cost > avail_cash:
                value_to_trade = (avail_cash / (1 + self.commission + self.slippage))

            pending_values_to_trade[symbol] = value_to_trade
            if action == 1:  
                avail_cash -= value_to_trade * (1 + self.commission + self.slippage)
            
        self.pending_trade = pending_values_to_trade

        return len(self.pending_trade) > 0
    

    def calculate_trade_cash_flow(self, price, shares):
        if shares == 0 or price <= 0:
            return 0
        trade_value = abs(price * shares)
        trade_cost = trade_value * (self.commission + self.slippage)
        if shares > 0:  # Buy
            return - trade_value - trade_cost
        elif shares < 0:  # Sell
            return trade_value - trade_cost
    def calculate_shares(self,symbol, current_price, value_to_trade):
        if current_price <= 0:
            return 0
        if value_to_trade == 0:
            return 0
        
        if value_to_trade > 0:
            num_shares = math.floor(value_to_trade / current_price)
        else:
            num_shares = math.ceil(value_to_trade / current_price)

        if abs(num_shares) < self.min_shares:
            return 0
        if num_shares < 0:  # Sell
            held_shares = max(self.positions.get(symbol, 0), 0)
            num_shares = -min(abs(num_shares), held_shares)
        
        return num_shares
    
    def execute_pending_trades(self, opening_prices, date):
        trades_to_execute = self.pending_trade
        trade_record = {}
        trade_record['Date'] = date

        portfolio_values = {}

        for symbol, value in trades_to_execute.items():
            current_price = opening_prices.get(symbol, 0)
            shares = self.calculate_shares(symbol, current_price, value)
            if shares == 0:
                continue
            self.positions[symbol] = self.positions.get(symbol, 0) + shares
            # Guard against negative positions due to bugs
            if self.positions[symbol] < 0:
                self.positions[symbol] = 0
            cash_flow = self.calculate_trade_cash_flow(current_price, shares)
            self.cash += cash_flow

            if shares > 0:
                self.share_cost[symbol] = self.share_cost.get(symbol, 0) + (shares * current_price * (1 + self.commission + self.slippage)) 

            elif shares < 0 and symbol in self.share_cost:
                shares_before_trade = self.positions.get(symbol, 0) - shares  # Calculate shares before trade
                if shares_before_trade > 0:
                    avg_cost_per_share = self.share_cost[symbol] / shares_before_trade
                    if symbol not in trade_record:
                        trade_record[symbol] = {}
                    trade_record[symbol]['P&L'] = (current_price - avg_cost_per_share) * abs(shares)
                    self.share_cost[symbol] -= (abs(shares) * avg_cost_per_share) 
                if self.positions[symbol] == 0:
                    del self.share_cost[symbol]

            if symbol not in trade_record:
                trade_record[symbol] = {}
            trade_record[symbol]['Action'] = 'BUY' if shares > 0 else 'SELL'
            trade_record[symbol]['Shares'] = shares
            trade_record[symbol]['Price'] = current_price
            trade_record[symbol]['Value'] = shares * current_price
            trade_record[symbol]['Cash_Flow'] = cash_flow
            portfolio_values[symbol] = self.positions[symbol] * current_price
        

        if not portfolio_values:
            portfolio_values = {symbol: self.positions.get(symbol, 0) * opening_prices.get(symbol, 0) for symbol in self.target_stocks}


        self.pending_trade = {}
        self.trades.append(trade_record)
        self.portfolio_value_history.append((self.calculate_portfolio_value(opening_prices), date))
        self.positions_value_history.append((portfolio_values, date))

    def evaluate_performance(self):
        if not self.portfolio_value_history:
            return {}
        
        value_df = pd.DataFrame(self.portfolio_value_history, columns=['Total Value', 'Date'])
        value_df['Date'] = pd.to_datetime(value_df['Date'])
        value_df = value_df.drop_duplicates(subset=['Date'], keep='last').set_index('Date')['Total Value']



        # Calculate returns
        initial_value = self.initial_cash
        final_value = self.portfolio_value_history[-1][0]
        total_return = (final_value / initial_value) - 1 if initial_value > 0 else 0

        # Annualization 
        total_trading_days = len(value_df) 
        # print(f"positions_history: {self.positions_history}")
        # print(f"Total trading days in backtest: {total_trading_days}")
        # print(f"portfolio value history length: {len(self.portfolio_value_history)}")
        # print(f"Portfolio value history sample:\n{value_df}")
        years = total_trading_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Risk metrics
        daily_returns = value_df.pct_change().dropna()
        annualized_volatility = daily_returns.std() * (252 ** 0.5) if len(daily_returns) > 0 else 0
        
        # Drawdown
        cumulative_max = value_df.cummax()
        drawdown = (value_df / cumulative_max) - 1
        maximum_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Sharpe Ratio
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0

        # Trade statistics
        total_num_trades = sum(
            1 for trade in self.trades for symbol in trade if symbol != 'Date'
        )
        
        all_trade_pnls = []
        for trade in self.trades:
            for symbol, details in trade.items():
                if symbol != 'Date' and isinstance(details, dict) and 'P&L' in details:
                    all_trade_pnls.append(details['P&L'])
        
        realized_trades = len(all_trade_pnls)
        
        if realized_trades > 0:
            winning_trades = [pnl for pnl in all_trade_pnls if pnl > 0]
            win_rate = len(winning_trades) / realized_trades
        else:
            win_rate = 0.0

        # Average return per trade (geometric mean contribution)
        avg_return_per_trade = ((1 + total_return) ** (1 / total_num_trades) - 1) if total_num_trades > 0 else 0
        
        
        return {
        'Initial Value': initial_value,
        'Final Value': final_value,
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Maximum Drawdown': maximum_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Total Trades': total_num_trades,
        'Avg Return per Trade': avg_return_per_trade,
        'Win Rate': win_rate
    }

    def analyse_monthly_returns(self):
        if not self.portfolio_value_history:
            return {}
        
        value_df = pd.DataFrame(self.portfolio_value_history, columns=['Value', 'Date'])
        value_df['Date'] = pd.to_datetime(value_df['Date'])
        value_df = value_df.set_index('Date')

        monthly_values = value_df.resample('ME').last()
        monthly_returns = monthly_values.pct_change()

        best_month = monthly_returns['Value'].idxmax()
        best_return = monthly_returns.loc[best_month, 'Value']
        prev_month_end = best_month - pd.offsets.MonthEnd(1)

        # print(f"Best Month: {best_month.strftime('%Y-%m')} with Return: {best_return*100:.2f}%")
        # print(f"Previous Month End: {prev_month_end.strftime('%Y-%m-%d')}")
        # print(f"Positions Value History Length: {len(self.positions_value_history)}")
        

        position_df = pd.DataFrame(self.positions_value_history, columns=['Positions', 'Date'])
        position_df['Date'] = pd.to_datetime(position_df['Date'])
        position_df = position_df.drop_duplicates(subset=['Date'], keep='last').set_index('Date')['Positions']

        monthly_positions = position_df.resample('ME').last()

        
        try:
            best_month_positions = monthly_positions.loc[best_month]
        except KeyError:
            best_month_positions = None
        try:
            prev_month_position = monthly_positions.loc[prev_month_end]
        except KeyError:
            prev_month_position = None

        # print(f"Best Month Positions on {best_month.strftime('%Y-%m-%d')}: {best_month_positions}")
        # print(f"Previous Month Positions on {prev_month_end.strftime('%Y-%m-%d')}: {prev_month_position}")

        best_month_returns = {}

        for symbol in self.target_stocks:
            prev_val = prev_month_position.get(symbol, 0) if prev_month_position is not None else 0
            best_val = best_month_positions.get(symbol, 0) if best_month_positions is not None else 0

            returns = best_val - prev_val
        
            best_month_returns[symbol] = returns
        
        return best_month_returns, best_month, best_return, monthly_returns

            

