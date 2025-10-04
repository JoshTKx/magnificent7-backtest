# Magnificent 7 RSI Backtesting System

## Project Overview
Quantitative backtesting framework implementing RSI-based momentum strategy on the Magnificent 7 tech stocks (AAPL, MSFT, NVDA, AMZN, GOOG, META, TSLA) from 1981-2023.

## Key Results
- Annualized Return: 29.77%
- Sharpe Ratio: 0.72
- Maximum Drawdown: -76.69%
- Total Trades: 1,620
- Outperformance vs S&P 500: 7,376,851.36%

## Installation
```bash
pip install -r requirements.txt
python -m src.backtest
```

## Key Findings
1. Best performing month: 1998-06 with 44.84% return, driven by AMZN
2. Strategy outperformed S&P 500 by 19.33% annualized (29.77% vs 10.44%)
3. Robustness tests show exceptional stability across RSI parameters (29.4-29.8% returns), resilience to 10x transaction costs, and consistent profitability across all decades (1980s-2020s)

## Project Structure
- src/: Core backtesting modules
- notebooks/: Analysis and visualization
- results/: Generated charts and metrics

## Methodology

- RSI(14) with 35/65 thresholds
- Equal-weight rebalancing with 2% tolerance
- Transaction costs: 0.1% commission + 0.02% slippage
- Signal generation at close, execution at next open


## Author
Joshua Teo