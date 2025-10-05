"""
Portfolio management module for the Magnificent 7 RSI backtesting system.

This module handles portfolio construction, rebalancing, trade execution,
and performance tracking for the RSI-based trading strategy.
"""

import heapq
import pandas as pd
from datetime import date, timedelta
from typing import Dict, List, Tuple, Optional, Any
import math

try:
    from .constants import TradingConstants
except ImportError:
    from src.constants import TradingConstants


class Portfolio:
    """
    Portfolio management class for RSI-based trading strategy.
    
    This class handles portfolio construction, trade execution, rebalancing,
    and performance tracking for the Magnificent 7 stocks using equal-weight
    allocation and RSI-based entry signals.
    
    Attributes:
        initial_cash (float): Initial portfolio cash value
        cash (float): Current available cash
        positions (Dict[str, int]): Current stock positions (symbol -> shares)
        share_cost (Dict[str, float]): Total cost basis for each position
        target_stocks (Dict[str, str]): Target stocks with IPO dates
        trades (List[Dict]): Record of all executed trades
        portfolio_value_history (List[Tuple]): Historical portfolio values
        positions_value_history (List[Tuple]): Historical position values
        commission (float): Transaction commission rate
        slippage (float): Price slippage rate for market impact
        min_shares (int): Minimum transaction size in shares
        pending_trade (Dict[str, int]): Pending trades awaiting execution
    """
    
    def __init__(self, 
                 initial_cash: float = TradingConstants.DEFAULT_INITIAL_CASH, 
                 commission: float = TradingConstants.DEFAULT_COMMISSION, 
                 slippage: float = TradingConstants.DEFAULT_SLIPPAGE, 
                 min_shares: int = TradingConstants.MIN_TRANSACTION_SIZE) -> None:
        """
        Initialize Portfolio with specified parameters.
        
        Args:
            initial_cash (float): Starting cash amount for portfolio
            commission (float): Commission rate per transaction (as decimal)
            slippage (float): Price slippage rate for market impact (as decimal)
            min_shares (int): Minimum number of shares for a transaction
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {} # symbol -> number of shares
        self.share_cost = {} # symbol -> total cost of shares held

        self.commission = commission
        self.slippage = slippage
        self.min_shares = min_shares

        self.target_stocks = TradingConstants.TARGET_STOCKS
        

        self.pending_trade = {} # symbol -> shares to trade

        self.trades = [] # list of trade records
        self.portfolio_value_history = [] # list of (total portfolio value, date)
        self.positions_value_history = [] # list of (positions value dict, date)

    def get_investable_stocks(self, date: date, price_data_dict: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Get list of stocks that are investable on a given date.
        
        A stock is considered investable if:
        - It has price data available for the given date
        - The closing price is valid (not NaN and greater than 0)
        
        Args:
            date (date): Date to check for investable stocks
            price_data_dict (Dict[str, pd.DataFrame]): Price data for all stocks
            
        Returns:
            List[str]: List of stock symbols that are investable on the given date
        """
        investable = []
        
        for symbol in self.target_stocks:

            if symbol not in price_data_dict:
                continue
            
            if date in price_data_dict[symbol].index:
                close_price = price_data_dict[symbol].loc[date, 'Close']
                if pd.notna(close_price) and close_price > 0:
                    investable.append(symbol)  
                    
        return investable
    
    def calculate_target_weights(self, investable: Optional[List[str]] = None) -> Optional[float]:
        """
        Calculate target weight for each investable stock using equal-weight allocation.
        
        The strategy uses equal-weight allocation, so each investable stock
        receives the same target weight (1/N where N is number of investable stocks).
        
        Args:
            investable (Optional[List[str]]): List of investable stock symbols.
                                            If None or empty, returns None.
            
        Returns:
            Optional[float]: Target weight for each stock (as decimal), or None if no investable stocks
        """
        
        num_investable = len(investable)
        if num_investable == 0:
            return None
        return 1.0 / num_investable
        

    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value including cash and positions.
        
        Args:
            current_prices (Dict[str, float]): Current prices for all stocks
            
        Returns:
            float: Total portfolio value (cash + position values)
        """
        total_value = self.cash
        for symbol, shares in self.positions.items():
            price = current_prices.get(symbol, 0)
            if price is not None and pd.notna(price) and price > 0 and shares > 0:
                total_value += shares * price
        return total_value

    def calculate_current_weights(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate current portfolio weights for each position.
        
        Args:
            current_prices (Dict[str, float]): Current prices for all stocks
            
        Returns:
            Dict[str, float]: Current weight of each position (symbol -> weight as decimal)
        """
        total_value = self.calculate_portfolio_value(current_prices)
        weights = {}
        for symbol in self.target_stocks:
            shares = self.positions.get(symbol, 0)
            price = current_prices.get(symbol, 0)
            position_value = shares * price
            weights[symbol] = position_value / total_value if total_value > 0 else 0
        return weights
    
    def determine_action(self, current_weight: float, rsi_signal: int, target_weight: float) -> int:
        """
        Determine trading action based on current weight, RSI signal, and target weight.
        
        Trading logic:
        - At Target: Hold
        - Underweight + Buy Signal: BUY to target 
        - Underweight + No/Sell Signal: Hold
        - Overweight + Sell Signal: SELL to target
        - Overweight + No/Buy Signal: SELL to target 
        
        Args:
            current_weight (float): Current position weight as decimal
            rsi_signal (int): RSI signal (-1: sell, 0: hold, 1: buy)
            target_weight (float): Target position weight as decimal
            
        Returns:
            int: Trading action (-1: sell, 0: hold, 1: buy)
        """
        tolerance = TradingConstants.REBALANCE_TOLERANCE  # 2% tolerance band around target weight

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
    
    def calculate_trade_priority(self, 
                               current_weights: Dict[str, float], 
                               rsi_signals: Dict[str, int], 
                               rsi_values: Dict[str, float], 
                               target_weight: float, 
                               investable_stocks: Optional[List[str]] = None) -> List[Tuple[int, float, str, float]]:
        """
        Calculate trade priority queue based on weight differences and RSI urgency.
        
        Priority is calculated as a weighted combination of weight difference (60%)
        and RSI urgency (40%). Trades with higher priority are executed first.
        
        Args:
            current_weights (Dict[str, float]): Current portfolio weights
            rsi_signals (Dict[str, int]): RSI signals for each stock
            rsi_values (Dict[str, float]): Current RSI values for each stock
            target_weight (float): Target weight for each stock
            investable_stocks (Optional[List[str]]): List of investable stocks
            
        Returns:
            List[Tuple[int, float, str, float]]: Priority queue of trades (action, -priority, symbol, weight_diff)
        """
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
    
    def generate_pending_trades(self, 
                              closing_prices: Dict[str, float], 
                              rsi_signals: Dict[str, int], 
                              rsi_values: Dict[str, float], 
                              date: date, 
                              price_data_dict: Dict[str, pd.DataFrame]) -> bool:
        """
        Generate pending trades based on current positions and RSI signals.
        
        This method calculates required trades to achieve target allocation
        and prioritizes them based on weight differences and RSI urgency.
        
        Args:
            closing_prices (Dict[str, float]): Current closing prices
            rsi_signals (Dict[str, int]): RSI signals for each stock
            rsi_values (Dict[str, float]): Current RSI values
            date (date): Current trading date
            price_data_dict (Dict[str, pd.DataFrame]): Historical price data
            
        Returns:
            bool: True if trades were generated, False if no trades needed
        """

        investable_stocks = self.get_investable_stocks(date, price_data_dict)
        target_weight = self.calculate_target_weights(investable_stocks)
        if not investable_stocks or target_weight == 0:
            self.pending_trade = {}
            return False
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
    
    def calculate_trade_cash_flow(self, price: float, shares: int) -> float:
        """
        Calculate net cash flow from a trade including commission and slippage costs.
        
        Args:
            price (float): Price per share
            shares (int): Number of shares (positive for buy, negative for sell)
            
        Returns:
            float: Net cash flow (negative for outflow, positive for inflow)
        """
        if shares == 0 or price <= 0:
            return 0
        trade_value = abs(price * shares)
        trade_cost = trade_value * (self.commission + self.slippage)
        if shares > 0:  # Buy
            return - trade_value - trade_cost
        elif shares < 0:  # Sell
            return trade_value - trade_cost
            
    def calculate_shares(self, symbol: str, current_price: float, value_to_trade: float) -> int:
        """
        Calculate number of shares to trade given target dollar value.
        
        Ensures trades meet minimum share requirements and don't exceed
        current holdings for sell orders.
        
        Args:
            symbol (str): Stock symbol
            current_price (float): Current price per share
            value_to_trade (float): Target dollar value to trade (positive for buy, negative for sell)
            
        Returns:
            int: Number of shares to trade (positive for buy, negative for sell)
        """
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
    
    def execute_pending_trades(self, opening_prices: Dict[str, float], date: date) -> None:
        """
        Execute all pending trades at opening prices and update portfolio state.
        
        This method:
        - Calculates exact shares to trade based on opening prices
        - Updates positions and cash balances
        - Records trade details for performance tracking
        - Updates cost basis for tax/performance calculations
        
        Args:
            opening_prices (Dict[str, float]): Opening prices for execution
            date (date): Trading date for record keeping
        """
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

    def evaluate_performance(self) -> Dict[str, Any]:
        """
        Evaluate comprehensive portfolio performance metrics.
        
        Calculates key performance indicators including:
        - Total and annualized returns
        - Risk metrics (volatility, Sharpe ratio, maximum drawdown)
        - Trade statistics and win rates
        - Time-based performance analysis
        
        Returns:
            Dict[str, Any]: Comprehensive performance metrics dictionary
        """
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

    def analyse_monthly_returns(self) -> Tuple[Dict[str, float], pd.Timestamp, float, pd.Series]:
        """
        Analyze monthly return patterns and identify best performing month.
        
        This method:
        - Calculates monthly portfolio returns
        - Identifies the best performing month
        - Breaks down stock-specific contributions for that month
        - Returns detailed monthly performance data
        
        Returns:
            Tuple containing:
            - Dict[str, float]: Stock-specific returns for best month
            - pd.Timestamp: Date of best performing month
            - float: Return percentage for best month
            - pd.Series: All monthly returns data
        """
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

            

