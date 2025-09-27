# Magnificent 7 Quantitative Backtesting System

A comprehensive backtesting framework for the "Magnificent 7" tech stocks using RSI-based trading signals.

## Overview
This project implements a quantitative trading strategy that uses Relative Strength Index (RSI) to generate buy/sell signals across seven major technology stocks: MSFT, AAPL, NVDA, AMZN, GOOG, META, and TSLA.

## Features
- Custom RSI indicator implementation
- Equal-weight portfolio rebalancing
- Transaction cost modeling (commission + slippage)
- Comprehensive performance analytics
- Risk-adjusted return calculations

## Quick Start
```bash
pip install -r requirements.txt
python -m src.backtest_engine