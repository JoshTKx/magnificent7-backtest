import heapq
import pandas as pd
from datetime import date, timedelta

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
        self.portfolio_value_history = [] # list of (date, total_value)

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
            return 0
        return 1.0 / num_investable
        


    def calculate_portfolio_value(self, current_prices):
        total_value = self.cash
        for symbol, shares in self.positions.items():
            if symbol in current_prices:
                price = current_prices[symbol]
                if pd.notna(price) and price > 0:
                    total_value += shares * current_prices[symbol]
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
        current_weights = self.calculate_current_weights(closing_prices)
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
        return 0
    
    
    def calculate_shares(self,symbol, current_price, value_to_trade):
        if current_price <= 0:
            return 0
        if value_to_trade == 0:
            return 0
        
        num_shares = int(value_to_trade / current_price)

        if abs(num_shares) < self.min_shares:
            return 0

        if num_shares < 0:  # Sell
            num_shares = -min(abs(num_shares), self.positions.get(symbol, 0))
        
        return num_shares

    
    def execute_pending_trades(self, opening_prices, date):
        trades_to_execute = self.pending_trade
        trade_record = {}
        trade_record['Date'] = date


        
        for symbol, value in trades_to_execute.items():
            current_price = opening_prices.get(symbol, 0)
            shares = self.calculate_shares(symbol, current_price, value)
            if shares == 0:
                continue

            trade_record[symbol] = {}
            shares_before_trade = self.positions.get(symbol, 0)
            
            self.positions[symbol] = self.positions.get(symbol, 0) + shares
            cash_flow = self.calculate_trade_cash_flow(current_price, shares)
            self.cash += cash_flow

            if shares > 0:
                self.share_cost[symbol] = self.share_cost.get(symbol, 0) + (shares * current_price * (1 + self.commission + self.slippage)) 

            elif shares < 0 and symbol in self.share_cost:
                if shares_before_trade > 0:
                    avg_cost_per_share = self.share_cost[symbol] / shares_before_trade
                    trade_record[symbol]['P&L'] = (current_price - avg_cost_per_share) * abs(shares)
                    self.share_cost[symbol] -= (abs(shares) * avg_cost_per_share) 
                if self.positions[symbol] == 0:
                    del self.share_cost[symbol]


            trade_record[symbol]['Action'] = 'BUY' if shares > 0 else 'SELL'
            trade_record[symbol]['Shares'] = shares
            trade_record[symbol]['Price'] = current_price
            trade_record[symbol]['Value'] = shares * current_price
            trade_record[symbol]['Cash_Flow'] = cash_flow


        self.pending_trade = {}
        self.trades.append(trade_record)
        self.portfolio_value_history.append((self.calculate_portfolio_value(opening_prices), date))

    def evaluate_performance(self):
        if not self.portfolio_value_history:
            return 0, 0
        
        initial_value = self.portfolio_value_history[0][0]
        final_value = self.portfolio_value_history[-1][0]
        total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0
        num_days = (self.portfolio_value_history[-1][1] - self.portfolio_value_history[0][1]).days
        annualized_return = (1 + total_return) ** (252 / num_days) - 1 if num_days > 0 else 0
        annualized_volatility = 0
        maximum_drawdown = 0
        sharpe_ratio = 0
        
        total_num_trades = sum(len(trade) - 1 for trade in self.trades)  
        avg_return_per_trade = total_return / total_num_trades if total_num_trades > 0 else 0
        win_rate = 0

        metrics = {
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
        return metrics





            

        
if __name__ == "__main__":
    
    # --- 1. SETUP: INITIALIZE PORTFOLIO AND SIMULATE START DATE ---
    
    # Start Date: A date when some stocks are NOT investable (e.g., 2000-01-01)
    # IPOs available by 2000-01-01: AAPL, MSFT, AMZN, NVDA (4 stocks)
    # IPOs NOT available: GOOG, TSLA, META (3 stocks)
    START_DATE = date(2000, 1, 1)
    
    portfolio = Portfolio(initial_cash=700000, commission=0.001, slippage=0.0002, min_shares=10)
    
    # Prices at the CLOSE of Day 1 (Execution Price for Buy is Day 2 OPEN)
    day1_close_prices = {
        'AAPL': 50.00, 'MSFT': 100.00, 'GOOG': 120.00, 'AMZN': 20.00, 
        'META': 250.00, 'TSLA': 20.00, 'NVDA': 10.00
    }
    
    # Simulate a universal BUY signal for all investable stocks
    rsi_signals = {'AAPL': 1, 'MSFT': 1, 'GOOG': 1, 'AMZN': 1, 
                   'META': 1, 'TSLA': 1, 'NVDA': 1}
    rsi_values = {'AAPL': 30, 'MSFT': 30, 'GOOG': 30, 'AMZN': 30, 
                  'META': 30, 'TSLA': 30, 'NVDA': 30}

    # Prices at the OPEN of Day 2 (Execution Price)
    day2_open_prices = {
        'AAPL': 50.50, 'MSFT': 101.00, 'GOOG': 120.00, 'AMZN': 20.20, 
        'META': 250.00, 'TSLA': 20.00, 'NVDA': 10.10
    }
    
    print("="*70)
    print(f"TESTING TARGET WEIGHT BALANCING: START DATE {START_DATE}")
    print("="*70)

    # --- 2. VERIFY INVESTABLE UNIVERSE & TARGET WEIGHT ---
    
    investable_stocks = portfolio.get_investable_stocks(START_DATE)
    calculated_target_weight = portfolio.calculate_target_weights(investable_stocks)
    
    print(f"1. Investable Stocks (as of {START_DATE}): {investable_stocks}")
    
    # Expected: 4 stocks (AAPL, MSFT, AMZN, NVDA)
    if len(investable_stocks) != 4:
        print(f"   ERROR: Expected 4 investable stocks, found {len(investable_stocks)}")
        # NOTE: Check your IPO dates if this fails.
        
    # Expected Target Weight: 1 / 4 = 0.25 (25%)
    print(f"   Calculated Target Weight: {calculated_target_weight:.4f} (Expected: 0.2500)")
    
    # --- 3. GENERATE PENDING TRADES (DAY 1 SIGNAL) ---
    
    # The trade generation must calculate the required dollar value to allocate 25% to the 4 stocks.
    
    # Expected Initial Trade Value per stock: $700,000 / 4 = $175,000 (Gross Value)
    
    portfolio.generate_pending_trades(day1_close_prices, rsi_signals, rsi_values, START_DATE)
    
    print("\n2. Generating Pending Trades (Intent):")
    
    trade_summary = {}
    for symbol, value in portfolio.pending_trade.items():
        trade_summary[symbol] = round(value, 2)
        
    print(f"   Trade Intent Symbols: {list(trade_summary.keys())}")
    print(f"   Trade Intent Values: {trade_summary}")

    # --- 4. EXECUTE TRADES (DAY 2 EXECUTION) ---

    EXECUTION_DATE = START_DATE + timedelta(days=1)
    portfolio.execute_pending_trades(day2_open_prices, EXECUTION_DATE)
    
    # --- 5. VERIFY FINAL WEIGHTS ---
    
    final_weights = portfolio.calculate_current_weights(day2_open_prices)
    final_value = portfolio.calculate_portfolio_value(day2_open_prices)

    print("\n3. Verification (Post-Execution):")
    print(f"   Final Portfolio Value: ${final_value:,.2f}")
    print(f"   Remaining Cash: ${portfolio.cash:,.2f}")

    print("\n   --- Final Positions and Weights ---")
    
    all_weights_ok = True
    
    for symbol in portfolio.target_stocks.keys():
        weight = final_weights.get(symbol, 0)
        
        if symbol in investable_stocks:
            # Check investable stocks: weight should be close to 25% (0.25)
            target = calculated_target_weight
            
            status = "OK" if abs(weight - target) < 0.005 else "FAIL"
            if status == "FAIL": all_weights_ok = False
            
            shares = portfolio.positions.get(symbol, 0)
            print(f"   {symbol:<5}: Shares: {shares:<5} | Final Weight: {weight:.4f} (Target: {target:.4f}) -> {status}")
            
        else:
            # Check un-investable stocks: weight should be 0%
            status = "OK" if abs(weight) < 0.0001 else "FAIL"
            if status == "FAIL": all_weights_ok = False
            
            print(f"   {symbol:<5}: Shares: 0     | Final Weight: {weight:.4f} (Target: 0.0000) -> {status} (Un-investable)")
            
    print("\n" + "="*70)
    if all_weights_ok:
        print("SUCCESS: Target Weight Balancing and IPO Filtering Passed.")
    else:
        print("FAILURE: Weights are not correctly balanced or IPO filter failed.")
    print("="*70)
