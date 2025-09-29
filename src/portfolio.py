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

        self.target_stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA']
        self.target_weight = 1.0 / len(self.target_stocks)

        self.pending_trade = {} # symbol -> shares to trade

        self.trades = [] # list of trade records
        self.portfolio_value_history = [] # list of (date, total_value)

    def calculate_portfolio_value(self, current_prices):
        total_value = self.cash
        for symbol, shares in self.positions.items():
            if symbol in current_prices:
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
    
    def determine_action(self, current_weight, rsi_signal):
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

        if abs(current_weight - self.target_weight) <= tolerance:
            return 0  # At Target: Hold
        elif current_weight < self.target_weight - tolerance:
            if rsi_signal == -1:
                return 0  # Underweight + Sell: Hold
            else:
                return 1  # Underweight + No Signal or Buy: BUY to target
        elif current_weight > self.target_weight + tolerance:
            if rsi_signal == 1:
                return 0  # Overweight + Buy: Hold
            else:
                return -1  # Overweight + No Signal / Sell : SELL to target
        return 0  # Default to Hold
    
    def calculate_trade_priority(self, current_weights, rsi_signals, rsi_values):
        priority_queue = []
        for symbol in self.target_stocks:
            current_weight = current_weights.get(symbol, 0)
            rsi_signal = rsi_signals.get(symbol, 0)
            rsi_value = rsi_values.get(symbol, 50)  # Default RSI to 50 if not available
            action = self.determine_action(current_weight, rsi_signal)

            if action == 0:
                continue

            rsi_urgency = 0 

            if action == 1 and rsi_value < 50:  # Buy
                rsi_urgency = (50 - rsi_value)/ 50
            elif action == -1 and rsi_value >= 50:  # Sell
                rsi_urgency = (rsi_value - 50) / 50 

            weight_diff = abs(current_weight - self.target_weight)
            priority = 0.6 * weight_diff + 0.4 * rsi_urgency

            heapq.heappush(priority_queue, (action, -priority, symbol, weight_diff))

        return priority_queue
    
    
    def generate_pending_trades(self, closing_prices, rsi_signals, rsi_values):

        current_weights = self.calculate_current_weights(closing_prices)
        total_portfolio_value = self.calculate_portfolio_value(closing_prices)

        trade_priority_queue = self.calculate_trade_priority(current_weights, rsi_signals, rsi_values)

        pending_values_to_trade = {}
        cash = self.cash

        while trade_priority_queue:

            action, _, symbol, weight_diff = heapq.heappop(trade_priority_queue)
            value_to_trade = weight_diff * total_portfolio_value * action # Positive for buy, negative for sell

            
            trade_cost = abs(value_to_trade) * (self.commission + self.slippage)
            if action == 1 and abs(value_to_trade) + trade_cost > cash:
                value_to_trade = (cash / (1 + self.commission + self.slippage)) * action
            
            pending_values_to_trade[symbol] = value_to_trade
            cash -= value_to_trade
            
        self.pending_trade = pending_values_to_trade

        return len(pending_values_to_trade) > 0
    

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
            orginal_position = self.positions.get(symbol, 0)
            
            self.positions[symbol] = self.positions.get(symbol, 0) + shares
            cash_flow = self.calculate_trade_cash_flow(current_price, shares)
            self.cash += cash_flow

            if shares > 0:
                self.share_cost[symbol] = self.share_cost.get(symbol, 0) + (shares * current_price * (1 + self.commission + self.slippage)) 

            elif shares < 0 and symbol in self.share_cost:
                avg_cost_per_share = self.share_cost[symbol] / orginal_position
                trade_record[symbol]['P&L'] = (current_price - avg_cost_per_share) * abs(shares)
                self.share_cost[symbol] = self.share_cost.get(symbol, 0) - (abs(shares) * avg_cost_per_share) 
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



            

        
if __name__ == "__main__":
    # --- 1. SETUP: INITIAL STATE & MOCK PRICES (Day 1 & Day 2) ---
    portfolio = Portfolio(initial_cash=1000000, commission=0.001, slippage=0.0002, min_shares=10)
    
    # Day 1 Close: Initial Buy Signals (to establish positions)
    day1_close_prices = {'AAPL': 150.00, 'MSFT': 300.00, 'GOOG': 120.00, 'AMZN': 90.00, 
                         'META': 250.00, 'TSLA': 200.00, 'NVDA': 450.00}
    rsi_signals = {'AAPL': 1, 'MSFT': 0, 'GOOG': 0, 'AMZN': 0, 
                   'META': 0, 'TSLA': 0, 'NVDA': 0}
    rsi_values = {'AAPL': 30, 'MSFT': 50, 'GOOG': 50, 'AMZN': 50, 
                  'META': 50, 'TSLA': 50, 'NVDA': 50}

    # Day 2 Open: Execution of Initial Buys (Sets up positions for the Sell test)
    day2_open_prices = {'AAPL': 151.00, 'MSFT': 301.00, 'GOOG': 121.00, 'AMZN': 91.00, 
                        'META': 249.00, 'TSLA': 199.00, 'NVDA': 451.00}
    
    # --- Execute Day 1 Signal / Day 2 Execution (Initial Buy) ---
    portfolio.generate_pending_trades(day1_close_prices, rsi_signals, rsi_values)
    portfolio.execute_pending_trades(day2_open_prices, date.today())
    
    initial_shares_msft = portfolio.positions.get('MSFT', 0)
    initial_cost_msft = portfolio.share_cost.get('MSFT', 0)
    initial_avg_cost_msft = initial_cost_msft / initial_shares_msft
    
    print("="*70)
    print(f"INITIAL POSITION SETUP COMPLETE (Date: {date.today()})")
    print("="*70)
    print(f"MSFT Initial Shares: {initial_shares_msft} | Avg Cost: ${initial_avg_cost_msft:.2f}")
    print(f"Current Cash: ${portfolio.cash:,.2f}\n")


    # --- 2. TEST CASE: SELL TRADE (Day 3 & Day 4) ---

    # --- Day 3 Close: Price rises, RSI signals SELL for MSFT ---
    day3_close_prices = day2_open_prices.copy()
    # Simulate a price rise and Overbought RSI
    day3_close_prices['MSFT'] = 450.00 # Price jumps!
    
    # Simulate new RSI values (MSFT now overbought)
    rsi_signals_day3 = {'MSFT': -1} 
    rsi_values_day3 = {'MSFT': 75} 
    
    # Portfolio is now Overweight in MSFT due to price rise (and all others Underweight)
    
    # --- Generate Sell Intent (Day 3 Close) ---
    print("Generating SELL Intent (Day 3 Close)...")
    portfolio.generate_pending_trades(day3_close_prices, rsi_signals_day3, rsi_values_day3)

    print(f"MSFT Value Intent: ${portfolio.pending_trade.get('MSFT', 0):,.2f} (Negative = Sell)")
    
    # --- Day 4 Open: Execute Sell Trade ---
    day4_date = date.today() + timedelta(days=1)
    
    # MSFT execution price (a small drop from close price)
    day4_open_prices = day3_close_prices.copy()
    day4_open_prices['MSFT'] = 349.00 
    
    print("\nExecuting SELL Trade (Day 4 Open)...")
    portfolio.execute_pending_trades(day4_open_prices, day4_date)
    
    # --- 3. VERIFICATION ---

    # Get the last trade record to check P&L
    sell_record = portfolio.trades[-1].get('MSFT', {})
    
    if sell_record and sell_record.get('Action') == 'SELL':
        
        shares_sold = abs(sell_record['Shares'])
        pnl = sell_record['P&L']
        
        # CALCULATE EXPECTED P&L MANUALLY (using average cost from initialization)
        proceeds_per_share = day4_open_prices['MSFT'] * (1 - portfolio.commission - portfolio.slippage)
        cost_per_share = initial_avg_cost_msft
        
        expected_pnl = shares_sold * (day4_open_prices['MSFT'] - initial_avg_cost_msft) 

        
        print("\n" + "="*70)
        print("SELL TRADE VERIFICATION")
        print("="*70)
        print(f"Shares Sold: {shares_sold}")
        print(f"Execution Price: ${day4_open_prices['MSFT']:.2f}")
        print(f"Initial Avg Cost: ${initial_avg_cost_msft:.2f}")
        print(f"Realized P&L from Trade Record: ${pnl:,.2f}")
        print(f"Expected P&L (Manual Calc):      ${expected_pnl:,.2f}")

        # Final check on MSFT position
        final_shares_msft = portfolio.positions.get('MSFT', 0)
        final_cost_msft = portfolio.share_cost.get('MSFT', 0)
        
        print(f"\nFinal MSFT Shares: {final_shares_msft}")
        print(f"Final MSFT Cost Basis: ${final_cost_msft:,.2f}")
        print(f"Final Cash Balance: ${portfolio.cash:,.2f}")
        
        # Assertions (optional, but good for testing)
        assert abs(pnl - expected_pnl) < 0.01, "P&L calculation error!"
        assert abs(final_cost_msft - (initial_cost_msft - shares_sold * initial_avg_cost_msft)) < 0.01, "Cost basis update error!"
        print("\nVerification: P&L and Cost Basis checks PASSED.")

    else:
        print("\nERROR: Sell trade for MSFT did not execute or was not recorded correctly.")