import numpy as np
import pandas as pd
import pytest

from backtest import extract_trades, metrics, rsi_signal, run_backtest, sma_crossover_signal


def prices(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="B"), dtype=float)


def test_always_long_matches_buy_and_hold():
    close = prices([100, 110, 99, 120, 130])
    signal = pd.Series(1, index=close.index)
    result = run_backtest(close, signal)
    # buying at the first close catches every move after it, same as buy and hold
    assert result["strategy"].iloc[-1] == pytest.approx(13_000)
    assert result["buy_hold"].iloc[-1] == pytest.approx(13_000)


def test_no_lookahead():
    close = prices([100, 200, 200])
    signal = pd.Series([0, 1, 1], index=close.index)
    result = run_backtest(close, signal)
    # signal fires on the day of the jump, so we shouldn't capture it
    assert result["strategy"].iloc[-1] == pytest.approx(10_000)


def test_fees_reduce_returns():
    close = prices(np.linspace(100, 150, 60))
    signal = sma_crossover_signal(close, 5, 20)
    free = run_backtest(close, signal)["strategy"].iloc[-1]
    paid = run_backtest(close, signal, fee_pct=0.01)["strategy"].iloc[-1]
    assert paid < free


def test_max_drawdown():
    equity = prices([100, 120, 90, 130])
    assert metrics(equity)["max_drawdown"] == pytest.approx(-25)


def test_rsi_signal_is_binary():
    close = prices(100 + np.cumsum(np.random.default_rng(0).normal(0, 2, 300)))
    sig = rsi_signal(close)
    assert set(sig.unique()) <= {0, 1}


def test_extract_trades_round_trip():
    close = prices([100, 100, 110, 120, 120, 120])
    signal = pd.Series([0, 1, 1, 0, 0, 0], index=close.index)
    trades = extract_trades(run_backtest(close, signal))
    assert len(trades) == 1
    assert trades.iloc[0]["entry_price"] == 100
    assert trades.iloc[0]["exit_price"] == 120
    assert trades.iloc[0]["return_pct"] == pytest.approx(20)
