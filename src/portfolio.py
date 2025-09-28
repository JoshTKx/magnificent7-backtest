import heapq

class Portfolio:
    def __init__(self, initial_cash=1000000, commission=0.001, slippage = 0.0002, min_shares = 10):
        self.inital_cash = initial_cash
        self.cash = initial_cash
        self.positions = {} # symbol -> number of shares

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

            if action == 1 and rsi_value <= 35:  # Buy
                rsi_urgency = (35 - rsi_value)/ 35 
            elif action == -1 and rsi_value >= 65:  # Sell
                rsi_urgency = (rsi_value - 65) / 35

            weight_diff = abs(current_weight - self.target_weight)
            priority = 0.6 * weight_diff + 0.4 * rsi_urgency

            heapq.heappush(priority_queue, (-priority, symbol, action, weight_diff))

        return priority_queue
    
    def calculate_shares(self,symbol, current_price, action, total_portfolio_value, weight_diff):
        if current_price <= 0:
            return 0
        if action == 0:
            return 0
        
        value_to_trade = weight_diff * total_portfolio_value

        if action == 1:  # Buy to target
            value_to_buy = value_to_trade

            if value_to_buy > self.cash:
                value_to_buy = self.cash

            shares_to_buy = int(value_to_buy / current_price)
            shares_to_buy = shares_to_buy if shares_to_buy >= self.min_shares else 0
            return shares_to_buy
        
        elif action == -1:  # Sell to targe t
            value_to_sell =  value_to_trade
            shares_to_sell = int(value_to_sell / current_price)
            shares_to_sell = min(shares_to_sell, self.positions.get(symbol, 0) ) if shares_to_sell >= self.min_shares else 0
            return -shares_to_sell
        
        return 0  # Hold


        
            

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


    def get_current_positions(self):
        return self.positions.copy()
    
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
            if rsi_signal == 1:
                return 1  # Underweight + Buy: BUY to target 
            else:
                return 0  # Underweight + No Signal or Sell: Hold
        elif current_weight > self.target_weight + tolerance:
            if rsi_signal == 1:
                return 0  # Overweight + Buy: Hold
            else:
                return -1  # Overweight + No Signal / Sell : SELL to target
        return 0  # Default to Hold
    
    def generate_shares_to_trade(self, trade_priority_queue, total_portfolio_value, closing_prices):
        trades_to_execute = {}
        current_prices = closing_prices
        cash = self.cash
        while trade_priority_queue:
            
            _, symbol, action, weight_diff = heapq.heappop(trade_priority_queue)
            current_price = current_prices.get(symbol, 0)

            shares = self.calculate_shares(symbol, current_price, action, total_portfolio_value, weight_diff)
            abs_share  = abs(shares)

            if shares == 0:
                continue


            if action == 1:
                cost_per_share = current_price * (1 + self.commission + self.slippage)
                total_cost = cost_per_share * abs_share
                if total_cost > cash:
                    affordable_shares = int(cash / cost_per_share)
                    shares = affordable_shares if affordable_shares > self.min_shares else 0

            elif action == -1:
                if abs_share > self.positions.get(symbol, 0):
                    shares = -self.positions.get(symbol, 0)
                if abs_share < self.min_shares:
                    shares = 0

            if shares == 0:
                continue
            
            cash_flow = self.calculate_trade_cash_flow(current_price, shares)
            
            cash += cash_flow
            trades_to_execute[symbol] = shares 
        self.pending_trade = trades_to_execute

        return cash
    
    def execute_pending_trades(self, opening_prices, cash, date):
        trades_to_execute = self.pending_trade
        current_prices = opening_prices
        
        for symbol, shares in trades_to_execute.items():
            self.positions[symbol] = self.positions.get(symbol, 0) + shares

        trade_record = trades_to_execute.copy()
        trade_record['Cash_change'] = cash - self.cash
        trade_record['Date'] = date
        self.trades.append(trade_record)

        self.cash = cash
        self.portfolio_value_history.append((self.calculate_portfolio_value(current_prices), date))
            


            

        
if __name__ == "__main__":
    # Create minimal test case
    portfolio = Portfolio(initial_cash=100000)
    prices = {'AAPL': 100, 'MSFT': 200, 'GOOG': 300, 'AMZN': 150, 
            'META': 250, 'TSLA': 180, 'NVDA': 400}
    rsi_signals = {'AAPL': 1, 'MSFT': 0, 'GOOG': 0, 'AMZN': 0,
                'META': 0, 'TSLA': 0, 'NVDA': 0}
    rsi_values = {'AAPL': 30, 'MSFT': 50, 'GOOG': 50, 'AMZN': 50,
                'META': 50, 'TSLA': 50, 'NVDA': 50}

    # Test the full workflow
    weights = portfolio.calculate_current_weights(prices)
    print("Current Weights:", weights)
    priorities = portfolio.calculate_trade_priority(weights, rsi_signals, rsi_values)
    print("Trade Priorities:", priorities)
    trades, new_cash = portfolio.calculate_shares_to_trade(priorities, 100000, prices)
    print("Trades to Execute:", trades)
    portfolio.execute_trades(trades, new_cash, prices, '2023-01-01')
    print("Positions:", portfolio.get_current_positions())
    print("Cash:", portfolio.cash)
    print("Portfolio Value:", portfolio.calculate_portfolio_value(prices))



    

        
        
    



    