"""
Sensitivity analysis module for the Magnificent 7 RSI backtesting system.

This module provides comprehensive testing of strategy parameters including
RSI thresholds, rebalancing costs, and various time period analyses to
evaluate strategy robustness and optimize performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .constants import TradingConstants
    from .backtest import BacktestEngine
except ImportError:
    from src.constants import TradingConstants
    from src.backtest import BacktestEngine


class SensitivityAnalysis:
    """
    Comprehensive sensitivity analysis for RSI trading strategy parameters.
    
    This class tests the robustness of the trading strategy across different
    parameter configurations and market conditions to identify optimal settings
    and understand performance sensitivity.
    
    Attributes:
        start_date (str): Analysis start date
        end_date (str): Analysis end date  
        initial_cash (float): Initial portfolio value
        results (List[Dict]): Collection of all test results
    """
    
    def __init__(self, 
                 start_date: str, 
                 end_date: str, 
                 initial_cash: float = TradingConstants.DEFAULT_INITIAL_CASH) -> None:
        """
        Initialize SensitivityAnalysis with date range and capital.
        
        Args:
            start_date (str): Start date for analysis period
            end_date (str): End date for analysis period
            initial_cash (float): Initial portfolio cash amount
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.results = []
    
    def test_rsi_thresholds(self) -> pd.DataFrame:
        """
        Test different RSI buy/sell threshold combinations.
        
        Tests various RSI configurations from conservative to aggressive
        to determine optimal threshold settings for different risk preferences.
        
        Returns:
            pd.DataFrame: Results comparing performance across RSI threshold configurations
        """
        print("Running RSI Threshold Sensitivity Tests...")
        
        threshold_combinations = [
            (30, 70, "Traditional RSI"),
            (35, 65, "Baseline (Current)"),
            (40, 60, "Conservative"),
            (25, 75, "Aggressive")
        ]
        
        results = []
        for buy_threshold, sell_threshold, label in threshold_combinations:
            print(f"Testing {label}: Buy<{buy_threshold}, Sell>{sell_threshold}")
            
            # Modify TechnicalIndicator class thresholds temporarily
            from src.indicator import TechnicalIndicator
            TechnicalIndicator.RSI_BUY_THRESHOLD = buy_threshold
            TechnicalIndicator.RSI_SELL_THRESHOLD = sell_threshold
            
            backtest = BacktestEngine(self.start_date, self.end_date, self.initial_cash)
            backtest.run_backtest()
            metrics = backtest.portfolio.evaluate_performance()
            
            results.append({
                'Configuration': label,
                'Buy Threshold': buy_threshold,
                'Sell Threshold': sell_threshold,
                'Total Return (%)': metrics['Total Return'] * 100,
                'Annualized Return (%)': metrics['Annualized Return'] * 100,
                'Sharpe Ratio': metrics['Sharpe Ratio'],
                'Max Drawdown (%)': metrics['Maximum Drawdown'] * 100,
                'Total Trades': metrics['Total Trades'],
                'Win Rate (%)': metrics['Win Rate'] * 100
            })
        
        # Reset to baseline
        TechnicalIndicator.RSI_BUY_THRESHOLD = 35
        TechnicalIndicator.RSI_SELL_THRESHOLD = 65
        
        return pd.DataFrame(results)
    
    def test_transaction_costs(self) -> pd.DataFrame:
        """
        Test sensitivity to different transaction cost scenarios.
        
        Evaluates strategy performance under various commission and slippage
        assumptions from optimistic institutional rates to pessimistic retail costs.
        
        Returns:
            pd.DataFrame: Performance comparison across transaction cost scenarios
        """
        print("\nRunning Transaction Cost Sensitivity Tests...")
        
        cost_scenarios = [
            (0.0005, 0.0001, "Optimistic (Institutional)"),
            (0.001, 0.0002, "Baseline (Current)"),
            (0.002, 0.0005, "Conservative"),
            (0.005, 0.001, "Pessimistic (Retail)")
        ]
        
        results = []
        for commission, slippage, label in cost_scenarios:
            print(f"Testing {label}: Commission={commission*100:.2f}%, Slippage={slippage*100:.2f}%")
            
            backtest = BacktestEngine(
                self.start_date, self.end_date, 
                self.initial_cash, commission, slippage
            )
            backtest.run_backtest()
            metrics = backtest.portfolio.evaluate_performance()
            
            results.append({
                'Scenario': label,
                'Commission (%)': commission * 100,
                'Slippage (%)': slippage * 100,
                'Total Cost (%)': (commission + slippage) * 100,
                'Annualized Return (%)': metrics['Annualized Return'] * 100,
                'Sharpe Ratio': metrics['Sharpe Ratio'],
                'Total Trades': metrics['Total Trades']
            })
        
        return pd.DataFrame(results)
    
    def test_time_periods(self) -> pd.DataFrame:
        """
        Test strategy performance across different historical time periods.
        
        Analyzes strategy robustness by evaluating performance across
        distinct market regimes and decades to identify temporal dependencies.
        
        Returns:
            pd.DataFrame: Performance breakdown by historical time periods
        """
        print("\nRunning Time Period Analysis...")
        
        periods = [
            ('1981-01-01', '1990-12-31', "1980s"),
            ('1991-01-01', '2000-12-31', "1990s"),
            ('2001-01-01', '2010-12-31', "2000s"),
            ('2011-01-01', '2020-12-31', "2010s"),
            ('2020-01-01', '2023-12-31', "2020-2023")
        ]
        
        results = []
        for start, end, label in periods:
            print(f"Testing {label}: {start} to {end}")
            
            try:
                backtest = BacktestEngine(start, end, self.initial_cash)
                backtest.run_backtest()
                metrics = backtest.portfolio.evaluate_performance()
                
                results.append({
                    'Period': label,
                    'Start': start,
                    'End': end,
                    'Annualized Return (%)': metrics['Annualized Return'] * 100,
                    'Sharpe Ratio': metrics['Sharpe Ratio'],
                    'Max Drawdown (%)': metrics['Maximum Drawdown'] * 100,
                    'Win Rate (%)': metrics['Win Rate'] * 100
                })
            except Exception as e:
                print(f"Error testing {label}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def visualize_results(self, 
                         rsi_df: pd.DataFrame, 
                         cost_df: pd.DataFrame, 
                         period_df: pd.DataFrame) -> None:
        """
        Create comprehensive visualization of sensitivity test results.
        
        Generates a multi-panel dashboard showing:
        - RSI threshold impact on returns and risk metrics
        - Transaction cost sensitivity analysis
        - Performance consistency across time periods
        
        Args:
            rsi_df (pd.DataFrame): RSI threshold test results
            cost_df (pd.DataFrame): Transaction cost test results  
            period_df (pd.DataFrame): Time period analysis results
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Strategy Robustness & Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # 1. RSI Threshold Impact on Returns
        ax1 = axes[0, 0]
        x_pos = range(len(rsi_df))
        bars = ax1.bar(x_pos, rsi_df['Annualized Return (%)'], 
                       color=['green' if i == 1 else 'lightblue' for i in range(len(rsi_df))])
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(rsi_df['Configuration'], rotation=45, ha='right')
        ax1.set_ylabel('Annualized Return (%)')
        ax1.set_title('RSI Threshold Sensitivity', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 2. RSI Impact on Sharpe Ratio
        ax2 = axes[0, 1]
        ax2.bar(range(len(rsi_df)), rsi_df['Sharpe Ratio'],
                color=['green' if i == 1 else 'lightcoral' for i in range(len(rsi_df))])
        ax2.set_xticks(range(len(rsi_df)))
        ax2.set_xticklabels(rsi_df['Configuration'], rotation=45, ha='right')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Risk-Adjusted Performance', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Transaction Cost Impact
        ax3 = axes[0, 2]
        ax3.plot(cost_df['Total Cost (%)'], cost_df['Annualized Return (%)'], 
                marker='o', linewidth=2, markersize=8)
        ax3.set_xlabel('Total Transaction Cost (%)')
        ax3.set_ylabel('Annualized Return (%)')
        ax3.set_title('Transaction Cost Sensitivity', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add labels for each point
        for idx, row in cost_df.iterrows():
            ax3.annotate(row['Scenario'], 
                        (row['Total Cost (%)'], row['Annualized Return (%)']),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        # 4. Period-by-Period Performance
        ax4 = axes[1, 0]
        periods = period_df['Period']
        x_pos = range(len(periods))
        colors = ['green' if ret > 0 else 'red' for ret in period_df['Annualized Return (%)']]
        ax4.bar(x_pos, period_df['Annualized Return (%)'], color=colors, alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(periods, rotation=45, ha='right')
        ax4.set_ylabel('Annualized Return (%)')
        ax4.set_title('Performance Across Time Periods', fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        # 5. Sharpe Ratio Across Periods
        ax5 = axes[1, 1]
        ax5.bar(x_pos, period_df['Sharpe Ratio'], color='skyblue', alpha=0.7)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(periods, rotation=45, ha='right')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.set_title('Risk-Adjusted Returns by Period', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary Heatmap
        ax6 = axes[1, 2]
        summary_data = rsi_df[['Configuration', 'Annualized Return (%)', 'Sharpe Ratio', 'Win Rate (%)']].set_index('Configuration')
        sns.heatmap(summary_data.T, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax6, cbar_kws={'label': 'Value'})
        ax6.set_title('RSI Configuration Heatmap', fontweight='bold')
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    # Run all tests
    analyzer = SensitivityAnalysis('2012-01-01', '2023-12-31')
    
    rsi_results = analyzer.test_rsi_thresholds()
    cost_results = analyzer.test_transaction_costs()
    period_results = analyzer.test_time_periods()
    
    # Display results
    print("\n" + "="*80)
    print("RSI THRESHOLD SENSITIVITY")
    print("="*80)
    print(rsi_results.to_string(index=False))
    
    print("\n" + "="*80)
    print("TRANSACTION COST SENSITIVITY")
    print("="*80)
    print(cost_results.to_string(index=False))
    
    print("\n" + "="*80)
    print("TIME PERIOD ANALYSIS")
    print("="*80)
    print(period_results.to_string(index=False))
    
    # Create visualization
    fig = analyzer.visualize_results(rsi_results, cost_results, period_results)
    plt.savefig('results/robustness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results to CSV
    rsi_results.to_csv('results/rsi_sensitivity.csv', index=False)
    cost_results.to_csv('results/cost_sensitivity.csv', index=False)
    period_results.to_csv('results/period_analysis.csv', index=False)