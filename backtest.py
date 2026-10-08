import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma_crossover_signal(close, fast=20, slow=50):
    """1 when the fast moving average is above the slow one, else 0."""
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(int)


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def rsi_signal(close, period=14, buy_below=30, sell_above=70):
    """Buy when RSI drops below buy_below, hold until it rises above sell_above."""
    r = rsi(close, period)
    signal = pd.Series(np.nan, index=close.index)
    signal[r < buy_below] = 1
    signal[r > sell_above] = 0
    return signal.ffill().fillna(0).astype(int)


def run_backtest(close, signal, starting_cash=10_000, fee_pct=0.0):
    """
    Long only backtest. The signal from day t is acted on at day t's close,
    so the position only earns returns starting on day t+1. That shift is what
    keeps the test from peeking at prices it wouldn't have known yet.
    """
    position = signal.shift(1).fillna(0)
    daily_ret = close.pct_change().fillna(0)
    switches = position.diff().abs().fillna(position.iloc[0])
    strat_ret = position * daily_ret - switches * fee_pct
    return pd.DataFrame({
        "close": close,
        "position": position,
        "strategy": starting_cash * (1 + strat_ret).cumprod(),
        "buy_hold": starting_cash * (1 + daily_ret).cumprod(),
    })


def metrics(equity):
    rets = equity.pct_change().dropna()
    growth = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / TRADING_DAYS
    std = rets.std()
    return {
        "total_return": (growth - 1) * 100,
        "cagr": (growth ** (1 / years) - 1) * 100 if years > 0 else 0.0,
        "volatility": std * np.sqrt(TRADING_DAYS) * 100,
        "sharpe": rets.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else 0.0,
        "max_drawdown": (equity / equity.cummax() - 1).min() * 100,
        "final_value": equity.iloc[-1],
    }


def extract_trades(result):
    """Turn the position series into a list of round trip trades."""
    trades = []
    prev_pos, prev_close, prev_date = 0, None, None
    entry_date = entry_price = None

    for date, row in result.iterrows():
        if row.position == 1 and prev_pos == 0:
            entry_date, entry_price = prev_date, prev_close
        elif row.position == 0 and prev_pos == 1:
            trades.append(_trade(entry_date, entry_price, prev_date, prev_close))
        prev_pos, prev_close, prev_date = row.position, row.close, date

    if prev_pos == 1:
        trades.append(_trade(entry_date, entry_price, prev_date, prev_close, still_open=True))

    return pd.DataFrame(trades, columns=["entry_date", "entry_price", "exit_date", "exit_price", "return_pct", "status"])


def _trade(entry_date, entry_price, exit_date, exit_price, still_open=False):
    return {
        "entry_date": entry_date,
        "entry_price": entry_price,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "return_pct": (exit_price / entry_price - 1) * 100,
        "status": "Open" if still_open else "Closed",
    }
