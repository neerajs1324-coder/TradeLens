# TradeLens

A trading journal and strategy backtester built with Python and Streamlit.

**Live app:** https://tradelens-journal.streamlit.app/

## Features

**Trade journal.** Log buys and sells with a strategy tag and notes. Import and export your journal as a CSV.

**Portfolio.** Open positions with live prices, cost basis, unrealized P&L, and an allocation chart.

**Performance.** Realized P&L using the average cost method, win rate, profit factor, a cumulative P&L chart, and a breakdown of which strategies are actually making money.

**Backtester.** Test a moving average crossover or RSI mean reversion strategy on any ticker's price history and compare it to buy and hold. Reports total and annual return, Sharpe ratio, max drawdown, and every simulated trade. Signals are lagged by a day so the test never uses prices it wouldn't have known, and you can add trading costs.

## Built with

Python, pandas, NumPy, Plotly, Streamlit, and yfinance for market data. The journal and backtest logic are unit tested with pytest.

## Run it locally

```
pip install -r requirements.txt
streamlit run app.py
```

Run the tests with `pip install pytest` then `pytest`.

## Project layout

```
app.py          the Streamlit interface
journal.py      position tracking, P&L, and stats
backtest.py     strategy signals, backtest engine, and metrics
tests/          unit tests for journal.py and backtest.py
```
